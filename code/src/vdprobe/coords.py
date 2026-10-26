"""Defines the coordinate system used to probe AnimateDiff's UNetMotionModel.

1. Identifies temporal motion modules and their two attention layers across the
   down, middle, and up blocks.
2. Identifies spatial self-attention (attn1) and text cross-attention (attn2)
   layers inherited from the base model (Stable Diffusion 1.5).
3. Enumerates 95 probe coordinates for the SD1.5 and mm_sd_v15_v2 setup:
   21 temporal modules, 21 temporal attn1 layers, 21 temporal attn2 layers,
   16 spatial attn1 layers, and 16 spatial attn2 layers.
4. Represents each measurement by its module path, denoising step, and clip ID
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

import torch

# Main probe types: spatial self-attention (attn1) and text cross-attention (attn2) from SD1.5,
# and AnimateDiff's temporal motion modules with their two temporal attention layers
# These five kinds give the 95 coordinates that every probe, patch and selectivity map refers to
# Experiment 1 (the probing sweep, patching and edits) used only these
KINDS = ("spatial_attn1", "spatial_attn2", "temporal_attn1", "temporal_attn2", "temporal_module")

# Optional extra kind, added for Experiment 2's refined pass: the 22 spatial ResNet blocks, so that all
# 117 coordinates (95 + 22) could be sketched and compared. Their output has the same (B*F, C, H, W)
# layout as a motion module. Only included when enumerate_coordinates(..., resnets=True)
RESNET = "resnet"

# Selected steps from the 25 denoising steps to be probed
D07_STEPS = (0, 4, 8, 12, 16, 20, 24)  # full tensors and patching are limited to these (D-07)

_MOTION_MODULE = re.compile(r"motion_modules\.\d+$")
_RESNET = re.compile(r"(down_blocks\.\d+|up_blocks\.\d+|mid_block)\.resnets\.\d+$")
_BLOCK = re.compile(r"^(down_blocks|up_blocks)\.(\d+)")


@dataclass(frozen=True)
class Coordinate:
    name: str   # U-Net module path for this probe location.
    kind: str   # Probe type, such as spatial attention or a temporal module
    block: str  # U-Net block containing the module
    level: int  # Spatial downsampling level meaning how many times the original video resolution has been halved 

    @property
    # Returns whether this probe location belongs to the temporal or spatial path
    def path(self) -> str:
        return "temporal" if self.kind.startswith("temporal") else "spatial"

    @property
    # Returns whether this probe location is an attention layer
    # Since attention maps can be recorded and visualized to see what the model is attending to at each step
    def is_attention(self) -> bool:
        return "attn" in self.kind

    @property
    # Returns the channel axis in the raw output tensor for this probe type. Used for channel-based
    # patching (copying only selected channels across activations): position 1 for motion modules
    # and ResNets, (B*F, C, H, W); position -1 for attention layers, (tokens, C)
    def channel_dim(self) -> int:
        """Channel axis of the raw output tensor."""
        return 1 if self.kind in ("temporal_module", RESNET) else -1

    @property
    def short(self) -> str:
        b = self.block.replace("down_blocks.", "down").replace("up_blocks.", "up").replace("mid_block", "mid")
        idx = re.findall(r"(attentions|motion_modules|resnets)\.(\d+)", self.name)
        tail = {"temporal_module": "mm", "temporal_attn1": "mm.a1", "temporal_attn2": "mm.a2",
                "spatial_attn1": "sp.a1", "spatial_attn2": "sp.a2", RESNET: "rn"}[self.kind]
        return f"{b}.{tail}{idx[-1][1] if idx else ''}"

# Classifies U-Net modules by type and names them as spatial or temporal probe locations
def classify(name: str) -> str | None:
    inside_motion = ".motion_modules." in name
    if _MOTION_MODULE.search(name):
        return "temporal_module"
    if name.endswith(".attn1"):
        return "temporal_attn1" if inside_motion else "spatial_attn1"
    if name.endswith(".attn2"):
        return "temporal_attn2" if inside_motion else "spatial_attn2"
    return None

# Returns a module's U-Net block and spatial resolution level from its name
def _block_and_level(name: str, n_levels: int) -> tuple[str, int]:
    m = _BLOCK.match(name)
    if m:
        i = int(m.group(2))
        return f"{m.group(1)}.{i}", i if m.group(1) == "down_blocks" else n_levels - 1 - i
    if name.startswith("mid_block"):
        return "mid_block", n_levels - 1
    raise ValueError(f"cannot place {name} in the U-Net")


def enumerate_coordinates(unet: torch.nn.Module, resnets: bool = False) -> list[Coordinate]:
    """Every hookable coordinate in module order; `resnets` adds the spatial ResNet blocks."""
    n_levels = len(unet.config.block_out_channels)
    coords = []
    for name, _ in unet.named_modules():
        kind = classify(name) or (RESNET if resnets and _RESNET.search(name) else None)
        if kind:
            block, level = _block_and_level(name, n_levels)
            coords.append(Coordinate(name, kind, block, level))
    return coords


def count_by_kind(coords: list[Coordinate]) -> dict[str, int]:
    c = Counter(x.kind for x in coords)
    return {k: c.get(k, 0) for k in KINDS + ((RESNET,) if RESNET in c else ())}


def in_blocks(coords: list[Coordinate], blocks) -> list[Coordinate]:
    """Every coordinate inside the given blocks (the refined pass), in module order."""
    blocks = set(blocks)
    return [c for c in coords if c.block in blocks]


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
    if kind in ("temporal_module", RESNET):
        _, c, h, w = t.shape
        return t.reshape(n_branches, frames, c, h * w)
    raise ValueError(kind)


def frame_mean_raw(t: torch.Tensor, kind: str, n_branches: int, frames: int) -> torch.Tensor:
    """x̄ broadcast back to the raw layout of `t` (same shape): the mean over frames, per CFG branch."""
    if kind.startswith("temporal_attn"):          # (B*S, F, C): frames are already an axis
        return t.mean(dim=1, keepdim=True).expand_as(t)
    if kind.startswith("spatial_attn") or kind in ("temporal_module", RESNET):   # frames folded into batch
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
