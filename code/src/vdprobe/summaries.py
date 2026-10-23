"""Compact per-coordinate summaries (D-09) and the paired distance behind S(L, t) (D-01).

Every activation is first put in canonical form x of shape (F, C, S): frames,
channels, spatial positions. It then splits into a static and a dynamic part:

    x_bar   = mean over frames of x, broadcast back over F   (static component)
    x_tilde = x - x_bar                                      (frame-mean-removed residual)

H3 is the claim that x_bar dominates raw activations even in temporal layers, so
any motion measurement that does not remove it will mostly measure content.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

FP32_BYTES = 4


def split_static_dynamic(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """x: (F, C, S) -> (x_bar of shape (C, S), x_tilde of shape (F, C, S))."""
    x_bar = x.mean(dim=0)
    return x_bar, x - x_bar.unsqueeze(0)


def channel_stats(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """The unpaired summary stored for every (coordinate, step): 2C + F floats.

    mean    (C,)  channel mean over frames and space
    var     (C,)  channel variance over frames and space
    resnorm (F,)  L2 norm of the frame-mean-removed residual x_tilde, per frame:
                  how much each frame departs from the clip's static component
    """
    x = x.float()
    mean = x.mean(dim=(0, 2))
    var = x.var(dim=(0, 2), unbiased=False)
    _, x_tilde = split_static_dynamic(x)
    resnorm = x_tilde.flatten(1).norm(dim=1)
    return {"mean": mean, "var": var, "resnorm": resnorm}


def summary_floats(channels: int, frames: int) -> int:
    return 2 * channels + frames


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """1 - cosine similarity on the flattened tensors (D-01)."""
    a, b = a.float().flatten(), b.float().flatten()
    return float(1.0 - F.cosine_similarity(a, b, dim=0, eps=1e-12))


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a - b).norm() / (a.norm() + b.norm()).clamp_min(1e-12))


def paired_distances(xa: torch.Tensor, xb: torch.Tensor, per_channel: bool = False) -> dict:
    """D-01 distances between two clips at one coordinate, both in (F, C, S) form.

    d_static  on x_bar   (content channel of the selectivity map)
    d_dynamic on x_tilde (motion channel)
    d_raw     on x       (what you would get without the split; the H3 control)
    Optionally the per-channel dynamic distance used by D-10's channel score.
    """
    xa, xb = xa.float(), xb.float()
    a_bar, a_til = split_static_dynamic(xa)
    b_bar, b_til = split_static_dynamic(xb)
    out = {
        "d_static": cosine_distance(a_bar, b_bar),
        "d_dynamic": cosine_distance(a_til, b_til),
        "d_raw": cosine_distance(xa, xb),
        # D-01 robustness check (appendix): relative L2, ||a - b|| / (||a|| + ||b||), in [0, 1]
        "l2_static": relative_l2(a_bar, b_bar),
        "l2_dynamic": relative_l2(a_til, b_til),
        # Share of activation energy in the static part. H3 predicts this is large.
        "static_energy_a": float(a_bar.pow(2).sum() * xa.shape[0] / xa.pow(2).sum().clamp_min(1e-12)),
        "static_energy_b": float(b_bar.pow(2).sum() * xb.shape[0] / xb.pow(2).sum().clamp_min(1e-12)),
    }
    if per_channel:
        ta = a_til.permute(1, 0, 2).flatten(1)  # (C, F*S)
        tb = b_til.permute(1, 0, 2).flatten(1)
        out["d_dynamic_per_channel"] = (1 - F.cosine_similarity(ta, tb, dim=1, eps=1e-12)).cpu()
    return out


def selectivity(d_content: float, d_motion: float) -> float:
    """S = (d_c - d_m) / (d_c + d_m) in [-1, 1]. Positive: tracks content. Negative: tracks motion."""
    den = d_content + d_motion
    return 0.0 if den <= 0 else (d_content - d_motion) / den


def resize_map(m: torch.Tensor, grid: int) -> torch.Tensor:
    """(F, h, w) -> (F, grid, grid). Area-average when shrinking, nearest when growing."""
    m = m.float().unsqueeze(1)
    if m.shape[-1] >= grid and m.shape[-2] >= grid:
        m = F.adaptive_avg_pool2d(m, grid)
    else:
        m = F.interpolate(m, size=(grid, grid), mode="nearest")
    return m.squeeze(1)
