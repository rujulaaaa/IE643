"""Group normalisation over all frames jointly, as CameraCtrl's U-Net does it.

The authors' `UNet3DConditionModel` keeps the video as (batch, C, frames, h, w) through its ResNet
blocks and its output norm and applies a plain `nn.GroupNorm` there, so the statistics of a group
are taken over every frame of the clip. diffusers' `UNetMotionModel` folds the frames into the
batch and normalises each frame on its own. The weights are the same; the outputs differ by about
1 % (relative L2 of the noise prediction; parity run of 2026-11-15). CameraCtrl was trained with
the joint version, so the graft switches the same layers to it. The spatial transformers' and the
motion modules' own norms are per frame in both code bases and are left alone.
"""

from __future__ import annotations

import re

import torch
import torch.nn.functional as F
from torch import nn

_TARGET = re.compile(r"^(?:(?:down_blocks|up_blocks)\.\d+|mid_block)\.resnets\.\d+\.norm[12]$|^conv_norm_out$")


class CrossFrameGroupNorm(nn.GroupNorm):
    """`nn.GroupNorm` for an input (batch * frames, C, h, w), batch-major: statistics over frames, h and w."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.clip["frames"]
        if f <= 1 or x.ndim != 4:
            return super().forward(x)
        bf, c, h, w = x.shape
        if bf % f:
            raise ValueError(f"batch of {bf} is not a multiple of {f} frames")
        y = x.reshape(bf // f, f, c, h, w).permute(0, 2, 1, 3, 4)
        y = F.group_norm(y, self.num_groups, self.weight, self.bias, self.eps)
        return y.permute(0, 2, 1, 3, 4).reshape(bf, c, h, w)


def cross_frame_groupnorm(unet) -> int:
    """Switch the ResNet norms and the output norm of a `UNetMotionModel` to joint statistics over frames.

    The layers keep their parameters and state-dict keys (only their class changes). The frame count is
    read from every U-Net call. Returns the number of layers switched; calling it twice is harmless.
    """
    clip = getattr(unet, "_vd_clip", None)
    if clip is None:
        clip = unet._vd_clip = {"frames": 1}

        def read_frames(module, args, kwargs):
            sample = args[0] if args else kwargs["sample"]
            clip["frames"] = sample.shape[2]

        unet.register_forward_pre_hook(read_frames, with_kwargs=True)
    n = 0
    for name, m in unet.named_modules():
        if _TARGET.match(name) and isinstance(m, nn.GroupNorm):
            m.__class__ = CrossFrameGroupNorm
            m.clip = clip
            n += 1
    return n
