"""A real video as the source clip: DDIM inversion onto the sampler's own 25 timesteps.

The edit machinery needs a source track whose activations can be cached at every step.
For an uploaded video there is no generation to cache, so the video is encoded by the VAE
and inverted with deterministic DDIM (eta = 0), text-conditional only (no CFG), using the same
`alphas_cumprod` and the same 25 timesteps as `run_lockstep`. This gives one latent z_t per
step. In the edit, the source track is then *driven* along that trajectory: at step i the
UNet sees z_{t_i} (so the cached activations are the video's), and its own update is replaced
by z_{t_{i+1}}. The target arms start from Gaussian noise or a blend with z_{t_0} (demo.upload_start):
sampling from z_{t_0} itself with CFG gives blotchy, grid-patterned clips (see edit.run_arms).

Inversion step (from the cleaner level to the noisier one), with ε predicted at the
cleaner latent:
    x0 = (z_k − sqrt(1 − ā_k) ε) / sqrt(ā_k);   z_next = sqrt(ā_next) x0 + sqrt(1 − ā_next) ε
"""

from __future__ import annotations

import numpy as np
import torch

from .lockstep import encode
from .pipeline import Loaded


def shot_bounds(frames: list, factor: float = 4.0, floor: float = 20.0) -> list[tuple[int, int]]:
    """[start, end) of each shot. A cut is a jump in mean absolute grey-level difference between
    consecutive frames (64x48) above max(factor x the clip's median jump, floor). 11 of the 19
    HMDB51 clips contain one; a 16-frame window across a cut transfers the cut, not the motion."""
    import cv2

    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), (64, 48)).astype(np.float32) for f in frames]
    d = np.array([np.abs(a - b).mean() for a, b in zip(small, small[1:])])
    if len(d) == 0:
        return [(0, len(frames))]
    cuts = [i + 1 for i in np.where(d > max(factor * np.median(d), floor))[0]]
    edges = [0, *cuts, len(frames)]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def read_video(path: str, frames: int = 16, size: int = 512, max_stride: int = 2, split_shots: bool = True) -> list:
    """16 frames at stride <= 2 from the middle of the longest shot, centre square crop, resized.
    A shot shorter than `frames` is used at stride 1 and its last frame repeated."""
    import cv2
    from PIL import Image

    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    if len(out) < 2:
        raise ValueError(f"could not read frames from {path}")
    if split_shots:
        a, b = max(shot_bounds(out), key=lambda s: s[1] - s[0])
        out = out[a:b]
    n = len(out)
    stride = max(1, min(max_stride, (n - 1) // (frames - 1)))
    start = max(0, (n - stride * (frames - 1) - 1) // 2)
    idx = [min(start + i * stride, n - 1) for i in range(frames)]
    res = []
    for i in idx:
        h, w, _ = out[i].shape
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        res.append(Image.fromarray(out[i][y0:y0 + s, x0:x0 + s]).resize((size, size), Image.BICUBIC))
    return res


@torch.no_grad()
def encode_frames(loaded: Loaded, frames: list) -> torch.Tensor:
    vae = loaded.pipe.vae
    x = torch.from_numpy(np.stack([np.asarray(f, np.float32) / 127.5 - 1 for f in frames])).permute(0, 3, 1, 2)
    lat = torch.cat([vae.encode(x[i:i + 4].to(loaded.device, vae.dtype)).latent_dist.mean
                     for i in range(0, len(x), 4)]) * vae.config.scaling_factor
    return lat.permute(1, 0, 2, 3).unsqueeze(0)            # (1, 4, F, h, w)


@torch.no_grad()
def invert(loaded: Loaded, frames: list, prompt: str) -> list[torch.Tensor]:
    """Latents [z_{t_0}, ..., z_{t_24}, z_clean] (26 entries), aligned with the sampler's steps."""
    pipe, cfg = loaded.pipe, loaded.cfg
    sched = pipe.scheduler
    sched.set_timesteps(cfg.steps, device=loaded.device)
    ts = [int(t) for t in sched.timesteps]                  # noisiest first
    acp = sched.alphas_cumprod.to(loaded.device, torch.float32)
    a_clean = sched.final_alpha_cumprod.to(loaded.device, torch.float32)
    emb = encode(loaded, prompt)[cfg.frames:]                # text-conditional half only
    z = encode_frames(loaded, frames).float()
    traj = [z]
    a_cur = a_clean
    for t in reversed(ts):                                   # cleanest timestep first
        eps = loaded.unet(z.half(), t, encoder_hidden_states=emb).sample.float()
        a_next = acp[t]
        x0 = (z - (1 - a_cur).sqrt() * eps) / a_cur.sqrt()
        z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        traj.append(z)
        a_cur = a_next
    return [x.cpu() for x in reversed(traj)]                 # [z_{t_0}, ..., z_{t_24}, z_clean]
