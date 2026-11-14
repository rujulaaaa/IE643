"""The CameraCtrl pose encoder: Plücker embedding -> one feature map per U-Net resolution.

Re-implementation of `CameraPoseEncoder` (cameractrl/models/pose_adaptor.py) for the released
configuration only (`ksize=1, sk=True, use_conv=False`, one temporal self-attention per block,
sinusoidal positions). Module names are kept so that `pose_encoder_state_dict` of
`CameraCtrl.ckpt` loads with `strict=True`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention import FeedForward


class ResnetBlock(nn.Module):
    """[average-pool /2] -> [1x1 conv if the width changes] -> 3x3 conv, ReLU, 1x1 conv, plus skip."""

    def __init__(self, in_c: int, out_c: int, down: bool):
        super().__init__()
        self.down = down
        self.in_conv = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else None
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.block2 = nn.Conv2d(out_c, out_c, 1)

    def forward(self, x):
        if self.down:
            x = F.avg_pool2d(x, 2, 2)
        if self.in_conv is not None:
            x = self.in_conv(x)
        return self.block2(F.relu(self.block1(x))) + x


class _Positions(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div)
        pe[0, :, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class _TemporalSelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, max_len: int):
        super().__init__()
        self.heads = heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim), nn.Dropout(0.0)])
        self.pos_encoder = _Positions(dim, max_len)

    def forward(self, x):                                   # (N, F, C)
        x = self.pos_encoder(x)
        n, f, c = x.shape
        q, k, v = (m(x).view(n, f, self.heads, c // self.heads).transpose(1, 2) for m in (self.to_q, self.to_k, self.to_v))
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(n, f, c)
        return self.to_out[0](out)


class TemporalTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, max_len: int):
        super().__init__()
        self.attention_blocks = nn.ModuleList([_TemporalSelfAttention(dim, heads, max_len)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim)])
        self.ff = FeedForward(dim, dropout=0.0, activation_fn="geglu")
        self.ff_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.attention_blocks[0](self.norms[0](x)) + x
        return self.ff(self.ff_norm(x)) + x


class CameraPoseEncoder(nn.Module):
    def __init__(self, channels=(320, 640, 1280, 1280), nums_rb: int = 2, cin: int = 384, downscale_factor: int = 8,
                 heads: int = 8, max_len: int = 16):
        super().__init__()
        if cin != 6 * downscale_factor ** 2:
            raise ValueError("cin must be 6 * downscale_factor ** 2 (a pixel-unshuffled Plücker embedding)")
        self.channels = tuple(channels)
        self.downscale_factor = downscale_factor
        self.unshuffle = nn.PixelUnshuffle(downscale_factor)
        self.encoder_conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        self.encoder_down_conv_blocks = nn.ModuleList()
        self.encoder_down_attention_blocks = nn.ModuleList()
        for i, ch in enumerate(channels):
            convs, attns = nn.ModuleList(), nn.ModuleList()
            for j in range(nums_rb):
                first = j == 0
                convs.append(ResnetBlock(channels[max(i - 1, 0)] if first else ch, ch, down=first and i != 0))
                attns.append(TemporalTransformerBlock(ch, heads, max_len))
            self.encoder_down_conv_blocks.append(convs)
            self.encoder_down_attention_blocks.append(attns)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """x: (B, 6, F, H, W). Returns one (B, C_i, F, h_i, w_i) per level, h_0 = H / downscale_factor."""
        b, _, f, _, _ = x.shape
        x = self.encoder_conv_in(self.unshuffle(x.transpose(1, 2).flatten(0, 1)))        # (B*F, C, h, w)
        features = []
        for convs, attns in zip(self.encoder_down_conv_blocks, self.encoder_down_attention_blocks):
            for conv, attn in zip(convs, attns):
                x = conv(x)
                c, h, w = x.shape[1:]
                t = x.view(b, f, c, h, w).permute(0, 3, 4, 1, 2).reshape(b * h * w, f, c)
                x = attn(t).view(b, h, w, f, c).permute(0, 3, 4, 1, 2).reshape(b * f, c, h, w)
            features.append(x.view(b, f, *x.shape[1:]).transpose(1, 2))
        return features
