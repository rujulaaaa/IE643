"""Experiment 2 quality gate (Phase 1): is each model cell's video coherent, moving and on-prompt?

Per clip, from the sweep outputs (GIF frames, flows.pt, clip_emb.pt):
  subject_consistency  VBench (Huang et al., CVPR 2024): DINO ViT-B/16 CLS features, for each
                       frame t >= 2 the mean of max(0, cos(f_t, f_1)) and max(0, cos(f_t, f_{t-1})),
                       averaged over frames. 1 = the subject looks the same throughout.
  dynamic_degree       VBench: a frame pair "moves" if the mean of its top 5% flow magnitudes exceeds
                       6 * (short side / 256) px; the clip is dynamic if >= round(4 * pairs / 16) pairs
                       move. Here on RAFT flow at 256 px pooled to 64 x 64 (pooling lowers peaks, so this
                       is a conservative version of VBench's).
  flow_top5            the mean top-5% magnitude itself (continuous)
  clip_t               CLIP text-image alignment: mean over frames of cos(frame, prompt)
  verb_prob/verb_hit   X-CLIP zero-shot (microsoft/xclip-base-patch16-zero-shot) probability of the
                       clip's verb among the probing verbs, and whether it is the argmax
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageSequence

DINO_ID = "facebook/dino-vitb16"
XCLIP_ID = "microsoft/xclip-base-patch16-zero-shot"


def gif_frames(path) -> list[Image.Image]:
    return [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(path))]


class Dino:
    def __init__(self, device="cpu"):
        from transformers import AutoImageProcessor, ViTModel

        self.device = device
        self.proc = AutoImageProcessor.from_pretrained(DINO_ID)
        self.model = ViTModel.from_pretrained(DINO_ID, add_pooling_layer=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, frames) -> torch.Tensor:
        inp = self.proc(images=frames, return_tensors="pt").to(self.device)
        return F.normalize(self.model(**inp).last_hidden_state[:, 0].float(), dim=-1)


def subject_consistency(feats: torch.Tensor) -> float:
    sims = [(max(0.0, float(feats[t] @ feats[t - 1])) + max(0.0, float(feats[t] @ feats[0]))) / 2
            for t in range(1, len(feats))]
    return float(np.mean(sims))


def dynamic_degree(flow: np.ndarray, side: int = 256) -> tuple[bool, float]:
    """flow: (P, h, w, 2) in pixels at `side`. Returns (is_dynamic, mean top-5% magnitude)."""
    mag = np.linalg.norm(flow.astype(np.float32), axis=-1).reshape(len(flow), -1)
    k = max(1, int(round(0.05 * mag.shape[1])))
    top = np.sort(mag, axis=1)[:, -k:].mean(1)
    moving = (top > 6.0 * side / 256).sum()
    return bool(moving >= round(4 * len(flow) / 16)), float(top.mean())


class XClipVerbs:
    """Zero-shot verb recognition. X-CLIP's zero-shot model takes 32 frames: the 16 are repeated."""

    def __init__(self, verbs: list[str], device="cpu"):
        from transformers import AutoProcessor, XCLIPModel

        self.verbs, self.device = list(verbs), device
        self.proc = AutoProcessor.from_pretrained(XCLIP_ID)
        self.model = XCLIPModel.from_pretrained(XCLIP_ID).to(device).eval()
        self.n = self.model.config.vision_config.num_frames

    @torch.no_grad()
    def __call__(self, frames, subject: str) -> np.ndarray:
        idx = np.linspace(0, len(frames) - 1, self.n).round().astype(int)
        texts = [f"a video of a {subject} {v}" for v in self.verbs]
        inp = self.proc(text=texts, videos=[[np.asarray(frames[i]) for i in idx]], return_tensors="pt",
                        padding=True).to(self.device)
        return self.model(**inp).logits_per_video.softmax(-1)[0].float().cpu().numpy()
