"""Clip-level metrics for the pilots. Ported from the E2 Kaggle notebook unchanged
where E2 defined them, so local reruns are comparable with the numbers of record.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLIP_ID = "openai/clip-vit-base-patch32"


class ClipScorer:
    """CLIP ViT-B/32 image embeddings, unit-normalised, in the joint space."""

    def __init__(self, device: str = "cuda", model_id: str = CLIP_ID):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_id)
        self.proj_dim = self.model.config.projection_dim
        self.hidden_dim = self.model.config.vision_config.hidden_size

    def _to_embedding(self, out, projection) -> torch.Tensor:
        # transformers has moved this return type around; decide by shape, not version.
        if torch.is_tensor(out):
            cand = out
        else:
            cand = next((getattr(out, a) for a in ("image_embeds", "pooler_output", "last_hidden_state")
                         if torch.is_tensor(getattr(out, a, None))), None)
            if cand is None:
                raise TypeError(f"no tensor found on {type(out).__name__}")
        if cand.ndim == 3:
            cand = cand[:, 0]
        if cand.shape[-1] != self.proj_dim:
            cand = projection(cand)
        return cand

    @torch.no_grad()
    def image(self, frames: list[Image.Image]) -> torch.Tensor:
        inp = self.proc(images=frames, return_tensors="pt").to(self.device)
        emb = self._to_embedding(self.model.get_image_features(**inp), self.model.visual_projection)
        return F.normalize(emb.float(), dim=-1)


def appearance_distance(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """d_c (E2): 1 - mean per-frame CLIP image-image cosine similarity."""
    return 1.0 - float((emb_a * emb_b).sum(-1).mean())


def motion_signature(frames: list[Image.Image], grid: int = 32) -> np.ndarray:
    """E2's pixel motion proxy: unit-normalised frame-to-frame difference maps, (F-1, grid*grid)."""
    a = np.stack([np.asarray(f.convert("L").resize((grid, grid)), np.float32) / 255 for f in frames])
    d = np.diff(a, axis=0).reshape(len(a) - 1, -1)
    return d / np.clip(np.linalg.norm(d, axis=1, keepdims=True), 1e-8, None)


def motion_distance(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """d_m (E2): 1 - mean cosine similarity of frame-difference maps."""
    return 1.0 - float((sig_a * sig_b).sum(-1).mean())


def layout_ssim(frame_a: Image.Image, frame_b: Image.Image) -> float:
    """Composition check on frame 0: SSIM of the greyscale frames."""
    from skimage.metrics import structural_similarity

    ga = np.asarray(frame_a.convert("L"), np.float32) / 255
    gb = np.asarray(frame_b.convert("L"), np.float32) / 255
    return float(structural_similarity(ga, gb, data_range=1.0))


def frames_to_array(frames: list[Image.Image]) -> np.ndarray:
    return np.stack([np.asarray(f.convert("RGB"), np.float32) / 255 for f in frames])


def psnr(frames_a: list[Image.Image], frames_b: list[Image.Image]) -> float:
    """Decoded-clip PSNR in dB over all frames (D-11 null-patch criterion: > 45 dB)."""
    mse = float(np.mean((frames_to_array(frames_a) - frames_to_array(frames_b)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def latent_mad(lat_a: torch.Tensor, lat_b: torch.Tensor) -> float:
    """Mean absolute difference of final latents (D-11 criterion: < 1e-3)."""
    return float((lat_a.float() - lat_b.float()).abs().mean())





def contact_sheet(rows: dict[str, list[Image.Image]], pick=(0, 5, 10, 15), thumb: int = 160) -> Image.Image:
    """One row per clip, a few sampled frames per row (the qualitative grid)."""
    sheet = Image.new("RGB", (thumb * len(pick), thumb * len(rows)), "white")
    for r, frames in enumerate(rows.values()):
        for c, i in enumerate(pick):
            sheet.paste(frames[min(i, len(frames) - 1)].resize((thumb, thumb)), (c * thumb, r * thumb))
    return sheet


# ---------------------------------------------------------------------------
# Edit evaluation (Deliverable 4, roadmap §5.4)
#
# The roadmap named CoTracker point tracks (RAFT as second opinion) for motion fidelity and
# DINO/CLIP for content. Substituted 2026-10-20 to avoid new model downloads in the last five
# days: dense Farneback optical flow (OpenCV) for motion, CLIP ViT-B/32 (already cached) for
# content and for the VBench-style subject-consistency proxy.
# ---------------------------------------------------------------------------

def flow_fields(frames: list[Image.Image], size: int = 128) -> np.ndarray:
    """Dense optical flow between consecutive frames, (F-1, size, size, 2), in pixels at `size`."""
    import cv2

    g = [cv2.cvtColor(np.asarray(f.convert("RGB").resize((size, size))), cv2.COLOR_RGB2GRAY) for f in frames]
    return np.stack([cv2.calcOpticalFlowFarneback(g[i], g[i + 1], None, pyr_scale=0.5, levels=3, winsize=15,
                                                  iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                     for i in range(len(g) - 1)])


def motion_fidelity(flow_a: np.ndarray, flow_b: np.ndarray) -> float:
    """s_m: mean over frame pairs of the cosine similarity of the two flattened flow fields, in [-1, 1].
    1 = every pixel moves the same way in both clips; 0 = unrelated; -1 = opposite."""
    a = flow_a.reshape(len(flow_a), -1)
    b = flow_b.reshape(len(flow_b), -1)
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return float(np.mean(num / np.maximum(den, 1e-8)))


def motion_magnitude(flow: np.ndarray) -> float:
    """Mean flow speed (pixels per frame at the flow resolution): is anything moving at all?"""
    return float(np.linalg.norm(flow, axis=-1).mean())


def content_fidelity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    """s_c: mean per-frame CLIP image-image cosine similarity (= 1 - appearance_distance)."""
    return float((emb_a * emb_b).sum(-1).mean())


def subject_consistency(emb: torch.Tensor) -> float:
    """VBench-style proxy: mean of each frame's CLIP cosine to the first and to the previous frame."""
    first = (emb[1:] * emb[0:1]).sum(-1)
    prev = (emb[1:] * emb[:-1]).sum(-1)
    return float(((first + prev) / 2).mean())


def temporal_flicker(frames: list[Image.Image], size: int = 128) -> float:
    """VBench-style temporal flickering score: 1 - mean absolute difference of consecutive frames
    (pixel values in [0, 1]); higher = steadier. VBench computes it on static clips; on moving
    clips it also falls with motion, so it is only compared between arms of the same pair."""
    a = np.stack([np.asarray(f.convert("RGB").resize((size, size)), np.float32) / 255 for f in frames])
    return float(1.0 - np.abs(np.diff(a, axis=0)).mean())


def minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def disentanglement(sc_target, sm_target, sc_other, sm_other, sc_pool, sm_pool):
    """D (roadmap §5.4, D-14) for edits whose output should carry the content of one reference and
    the motion of another:
        D = 1/2 [sc(out, content ref) + sm(out, motion ref)] - 1/2 [sc(out, motion ref) + sm(out, content ref)]
    with s_c and s_m min-max normalised over the whole evaluation pool (sc_pool / sm_pool hold every
    raw value that enters any D), so the two axes weigh equally and D lies in [-1, 1]."""
    sc_lo, sc_hi = np.nanmin(sc_pool), np.nanmax(sc_pool)
    sm_lo, sm_hi = np.nanmin(sm_pool), np.nanmax(sm_pool)
    nc = lambda v: (np.asarray(v) - sc_lo) / max(sc_hi - sc_lo, 1e-12)
    nm = lambda v: (np.asarray(v) - sm_lo) / max(sm_hi - sm_lo, 1e-12)
    return 0.5 * (nc(sc_target) + nm(sm_target)) - 0.5 * (nc(sc_other) + nm(sm_other))
