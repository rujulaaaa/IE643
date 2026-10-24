"""Experiment 2 Phase 6: does a generated clip perform the trained motion?

Motion Fidelity Score (Yatim et al., CVPR 2024, "Space-Time Diffusion Features for Zero-Shot
Text-Driven Motion Transfer"; code: diffusion-motion-transfer/motion_fidelity_score.py). CoTracker
tracklets on a regular grid in both clips; for tracklets tau (reference) and tau~ (generated),

    corr(tau, tau~) = mean_k  <v_k, v~_k> / (|v_k| |v~_k|)        v_k = frame-k displacement

    MFS_official  = mean_{tau~} max_tau corr        (the released code's direction)
    MFS_symmetric = (mean_{tau~} max_tau corr + mean_tau max_{tau~} corr) / 2   (the paper's two terms, halved)

Displacements are normalised with an epsilon (the released code divides by |v| and gives NaN for a
point that does not move). The released code also subtracts an identity matrix from the
similarity matrix, which is only meaningful when a clip is compared with itself; not copied.
"""

from __future__ import annotations

import numpy as np
import torch

_COTRACKER: dict = {}


def cotracker(device: str = "cuda"):
    if device not in _COTRACKER:
        _COTRACKER[device] = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device).eval()
    return _COTRACKER[device]


@torch.no_grad()
def tracklets(frames, grid: int = 30, device: str = "cuda") -> torch.Tensor:
    """(N, F, 2) CoTracker tracks of a grid x grid point grid placed on the first frame."""
    v = torch.from_numpy(np.stack([np.asarray(f.convert("RGB")) for f in frames])).permute(0, 3, 1, 2)
    tracks, _ = cotracker(device)(v[None].float().to(device), grid_size=grid)
    return tracks[0].permute(1, 0, 2).float().cpu()


def track_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """(N, F, 2), (M, F, 2) -> (N, M) mean cosine of per-frame displacements."""
    da, db = a[:, 1:] - a[:, :-1], b[:, 1:] - b[:, :-1]
    da = da / (da.norm(dim=-1, keepdim=True) + eps)
    db = db / (db.norm(dim=-1, keepdim=True) + eps)
    return torch.einsum("ntc,mtc->nmt", da, db).mean(-1)


def motion_fidelity_score(gen: torch.Tensor, ref: torch.Tensor) -> dict:
    s = track_similarity(gen, ref)
    g2r, r2g = float(s.max(1).values.mean()), float(s.max(0).values.mean())
    return {"mfs_official": g2r, "mfs_symmetric": (g2r + r2g) / 2}
