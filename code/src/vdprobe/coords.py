"""The coordinate system: every hookable module of AnimateDiff's UNetMotionModel.

Temporal path: the motion modules (AnimateDiffTransformer3D) inside each down block,
up block and the mid block, plus the two self-attention layers inside each (both
attend across frames). Spatial path: attn1 (self) and attn2 (cross, to the text) of
every spatial transformer block inherited from SD1.5.

For SD1.5 + mm_sd_v15_v2 this is 95 coordinates (E3):
    temporal_module 21 · temporal_attn1 21 · temporal_attn2 21 · spatial_attn1 16 · spatial_attn2 16

A measurement is addressed by the triple (module path, denoising step, clip id).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch

KINDS = ("spatial_attn1", "spatial_attn2", "temporal_attn1", "temporal_attn2", "temporal_module")
D07_STEPS = (0, 4, 8, 12, 16, 20, 24)  # full tensors and patching are limited to these (D-07)

_MOTION_MODULE = re.compile(r"motion_modules\.\d+$")
_BLOCK = re.compile(r"^(down_blocks|up_blocks)\.(\d+)")


@dataclass(frozen=True)
class Coordinate:
    name: str    # module path inside the UNet, e.g. "up_blocks.1.motion_modules.0"
    kind: str    # one of KINDS
    block: str   # "down_blocks.0" ... "mid_block" ... "up_blocks.3"
    level: int   # resolution level; the latent side length is divided by 2**level

    @property
    def path(self) -> str:
        return "temporal" if self.kind.startswith("temporal") else "spatial"

    @property
    def is_attention(self) -> bool:
        return "attn" in self.kind

    @property
    def channel_dim(self) -> int:
        """Channel axis of the raw output tensor."""
        return 1 if self.kind == "temporal_module" else -1

    @property
    def short(self) -> str:
        b = self.block.replace("down_blocks.", "down").replace("up_blocks.", "up").replace("mid_block", "mid")
        idx = re.findall(r"(attentions|motion_modules)\.(\d+)", self.name)
        tail = {"temporal_module": "mm", "temporal_attn1": "mm.a1", "temporal_attn2": "mm.a2",
                "spatial_attn1": "sp.a1", "spatial_attn2": "sp.a2"}[self.kind]
        return f"{b}.{tail}{idx[-1][1] if idx else ''}"


def classify(name: str) -> str | None:
    inside_motion = ".motion_modules." in name
    if _MOTION_MODULE.search(name):
        return "temporal_module"
    if name.endswith(".attn1"):
        return "temporal_attn1" if inside_motion else "spatial_attn1"
    if name.endswith(".attn2"):
        return "temporal_attn2" if inside_motion else "spatial_attn2"
    return None


def _block_and_level(name: str, n_levels: int) -> tuple[str, int]:
    m = _BLOCK.match(name)
    if m:
        i = int(m.group(2))
        return f"{m.group(1)}.{i}", i if m.group(1) == "down_blocks" else n_levels - 1 - i
    if name.startswith("mid_block"):
        return "mid_block", n_levels - 1
    raise ValueError(f"cannot place {name} in the U-Net")


def enumerate_coordinates(unet: torch.nn.Module) -> list[Coordinate]:
    n_levels = len(unet.config.block_out_channels)
    coords = []
    for name, _ in unet.named_modules():
        kind = classify(name)
        if kind:
            block, level = _block_and_level(name, n_levels)
            coords.append(Coordinate(name, kind, block, level))
    return coords


def count_by_kind(coords: list[Coordinate]) -> dict[str, int]:
    c = Counter(x.kind for x in coords)
    return {k: c.get(k, 0) for k in KINDS}


def block_order(coords: list[Coordinate]) -> list[str]:
    """Blocks in forward order: down_blocks.0 .. mid_block .. up_blocks.N."""
    seen = []
    for c in coords:
        if c.block not in seen:
            seen.append(c.block)
    key = lambda b: (0, int(b.split(".")[1])) if b.startswith("down") else \
                    (1, 0) if b == "mid_block" else (2, int(b.split(".")[1]))
    return sorted(seen, key=key)


def _module_index(c: Coordinate, group: str) -> int:
    m = re.findall(rf"{group}\.(\d+)", c.name)
    return int(m[0]) if m else -1


def coarse_coordinates(coords: list[Coordinate]) -> list[Coordinate]:
    """D-08 coarse pass, with D-25's choice of representative: per block, the last
    motion module (9 temporal) and attn2 of the last spatial transformer (7 spatial)."""
    out = []
    for block in block_order(coords):
        in_block = [c for c in coords if c.block == block]
        mms = [c for c in in_block if c.kind == "temporal_module"]
        if mms:
            out.append(max(mms, key=lambda c: _module_index(c, "motion_modules")))
        cross = [c for c in in_block if c.kind == "spatial_attn2"]
        if cross:
            out.append(max(cross, key=lambda c: _module_index(c, "attentions")))
    return out


def by_name(coords: list[Coordinate]) -> dict[str, Coordinate]:
    return {c.name: c for c in coords}


# ---------------------------------------------------------------------------
# Tensor layouts. Raw outputs arrive in three layouts; everything downstream works
# on the canonical (branches, F, C, S) form. With classifier-free guidance the
# batch is [unconditional, conditional].
#
#   spatial_attn*    (B*F, S, C)   frames folded into batch, tokens = spatial positions
#   temporal_attn*   (B*S, F, C)   spatial positions folded into batch, tokens = frames
#   temporal_module  (B*F, C, H, W)
# ---------------------------------------------------------------------------

def canonical(t: torch.Tensor, kind: str, n_branches: int, frames: int) -> torch.Tensor:
    if kind.startswith("spatial_attn"):
        _, s, c = t.shape
        return t.reshape(n_branches, frames, s, c).permute(0, 1, 3, 2)
    if kind.startswith("temporal_attn"):
        bs, f, c = t.shape
        return t.reshape(n_branches, bs // n_branches, f, c).permute(0, 2, 3, 1)
    if kind == "temporal_module":
        _, c, h, w = t.shape
        return t.reshape(n_branches, frames, c, h * w)
    raise ValueError(kind)


def frame_mean_raw(t: torch.Tensor, kind: str, n_branches: int, frames: int) -> torch.Tensor:
    """x̄ broadcast back to the raw layout of `t` (same shape): the mean over frames, per CFG branch."""
    if kind.startswith("temporal_attn"):          # (B*S, F, C): frames are already an axis
        return t.mean(dim=1, keepdim=True).expand_as(t)
    if kind.startswith("spatial_attn") or kind == "temporal_module":   # frames folded into batch
        v = t.reshape(n_branches, frames, *t.shape[1:])
        return v.mean(dim=1, keepdim=True).expand_as(v).reshape(t.shape)
    raise ValueError(kind)


def branch_rows(n_rows: int, n_branches: int, branch: int) -> slice:
    """Rows of a raw tensor (any of the three layouts) that belong to one CFG branch."""
    per = n_rows // n_branches
    return slice(branch * per, (branch + 1) * per)


def spatial_hw(level: int, latent_hw: tuple[int, int]) -> tuple[int, int]:
    h, w = latent_hw
    f = 2 ** level
    return -(-h // f), -(-w // f)
