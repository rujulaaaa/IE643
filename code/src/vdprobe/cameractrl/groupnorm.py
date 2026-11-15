"""Group normalisation over frames: making diffusers' `UNetMotionModel` follow CameraCtrl's U-Net.

Both code bases hold the same weights, but the GroupNorm at the entry of every motion module takes
its statistics over different axes (found with the layer-by-layer trace of
`scripts/cameractrl_parity.py`, 2026-11-15):

    authors' UNet3DConditionModel    per frame (`InflatedGroupNorm`), as AnimateDiff v3 was trained
    diffusers UNetMotionModel        all frames of the clip jointly (the norm is applied to the 5D tensor)

Every other norm (ResNets, spatial transformers, the output norm) is per frame in both. Left as
diffusers has it, the noise prediction differs from the authors' by about 1 % (relative L2) and
activations inside the U-Net by up to 40 %. The graft switches the 20 layers to per-frame.
Parameters and state-dict keys do not change.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

_MOTION_NORM = re.compile(r"^(?:(?:down_blocks|up_blocks)\.\d+|mid_block)\.motion_modules\.\d+\.(?:temporal_transformer\.)?norm$")


class _SwitchedGroupNorm(nn.GroupNorm):
    @property
    def _active(self) -> bool:
        return self.clip.get("authors", True)        # "authors": False restores diffusers' behaviour (diagnostics)


class PerFrameGroupNorm(_SwitchedGroupNorm):
    """For an input (batch, C, frames, h, w): statistics over h and w of each frame separately."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or not self._active:
            return super().forward(x)
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        y = F.group_norm(y, self.num_groups, self.weight, self.bias, self.eps)
        return y.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)


def authors_groupnorm(unet) -> int:
    """Switch the input norm of every motion module to per-frame statistics. Returns the number of
    layers switched; calling it twice is harmless. `unet._vd_clip["authors"] = False` turns it off."""
    clip = getattr(unet, "_vd_clip", None)
    if clip is None:
        clip = unet._vd_clip = {"authors": True}
    n = 0
    for name, m in unet.named_modules():
        if isinstance(m, nn.GroupNorm) and _MOTION_NORM.match(name):
            m.__class__, m.clip = PerFrameGroupNorm, clip
            n += 1
    return n
