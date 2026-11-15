"""Group normalisation over frames: making diffusers' `UNetMotionModel` follow CameraCtrl's U-Net.

Both code bases hold the same weights, but two kinds of GroupNorm layer take their statistics over
different axes (found with the layer-by-layer trace of `scripts/cameractrl_parity.py`, 2026-11-15):

    layer                               authors' UNet3DConditionModel        diffusers UNetMotionModel
    motion module input norm            per frame (InflatedGroupNorm)        all frames jointly (5D tensor)
    output norm (conv_norm_out)         all frames jointly (5D tensor)       per frame
    ResNet norms, spatial transformer   per frame                            per frame

Left as diffusers has them, the noise prediction differs from the authors' by about 1 % (relative
L2) and activations inside the U-Net by up to 40 %. CameraCtrl was trained with the left column, so
the graft switches the two kinds of layer. Parameters and state-dict keys do not change.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

_MOTION_NORM = re.compile(r"^(?:(?:down_blocks|up_blocks)\.\d+|mid_block)\.motion_modules\.\d+\.(?:temporal_transformer\.)?norm$")
_OUTPUT_NORM = "conv_norm_out"


class _SwitchedGroupNorm(nn.GroupNorm):
    @property
    def _active(self) -> bool:
        return self.clip.get("authors", True)        # "authors": False restores diffusers' behaviour (diagnostics)


class CrossFrameGroupNorm(_SwitchedGroupNorm):
    """For an input (batch * frames, C, h, w), batch-major: statistics over frames, h and w jointly."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.clip["frames"]
        if f <= 1 or x.ndim != 4 or not self._active:
            return super().forward(x)
        bf, c, h, w = x.shape
        if bf % f:
            raise ValueError(f"batch of {bf} is not a multiple of {f} frames")
        y = x.reshape(bf // f, f, c, h, w).permute(0, 2, 1, 3, 4)
        y = F.group_norm(y, self.num_groups, self.weight, self.bias, self.eps)
        return y.permute(0, 2, 1, 3, 4).reshape(bf, c, h, w)


class PerFrameGroupNorm(_SwitchedGroupNorm):
    """For an input (batch, C, frames, h, w): statistics over h and w of each frame separately."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or not self._active:
            return super().forward(x)
        b, c, f, h, w = x.shape
        y = x.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        y = F.group_norm(y, self.num_groups, self.weight, self.bias, self.eps)
        return y.reshape(b, f, c, h, w).permute(0, 2, 1, 3, 4)


def authors_groupnorm(unet) -> dict:
    """Switch the motion-module input norms to per-frame and the output norm to joint statistics.

    The frame count is read from every U-Net call. Returns the number of layers switched per kind;
    calling it twice is harmless. `unet._vd_clip["authors"] = False` turns both off again.
    """
    clip = getattr(unet, "_vd_clip", None)
    if clip is None:
        clip = unet._vd_clip = {"frames": 1, "authors": True}

        def read_frames(module, args, kwargs):
            sample = args[0] if args else kwargs["sample"]
            clip["frames"] = sample.shape[2]

        unet.register_forward_pre_hook(read_frames, with_kwargs=True)
    n = {"motion_module_norm": 0, "output_norm": 0}
    for name, m in unet.named_modules():
        if not isinstance(m, nn.GroupNorm):
            continue
        if _MOTION_NORM.match(name):
            m.__class__, m.clip = PerFrameGroupNorm, clip
            n["motion_module_norm"] += 1
        elif name == _OUTPUT_NORM:
            m.__class__, m.clip = CrossFrameGroupNorm, clip
            n["output_norm"] += 1
    return n
