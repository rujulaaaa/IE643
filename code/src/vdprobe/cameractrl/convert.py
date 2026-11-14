"""Weights: `CameraCtrl.ckpt` into the adapter, and AnimateDiff's v3 domain-adapter LoRA into the U-Net."""

from __future__ import annotations

import re

import torch

from .adapter import CameraAdapter, merge_key

_PROC = re.compile(r"^(.+\.motion_modules\.\d+)\.temporal_transformer\.transformer_blocks\.0\.attention_blocks\.0"
                   r"\.processor\.qkv_merge\.(weight|bias)$")
_LORA = re.compile(r"^(.+)\.processor\.(to_q|to_k|to_v|to_out)_lora\.(down|up)\.weight$")


def site_of(processor_key: str) -> tuple[str, str]:
    """Authors' processor key -> (diffusers attention path, 'weight' | 'bias')."""
    m = _PROC.match(processor_key)
    if not m:
        raise KeyError(f"not a CameraCtrl attention-processor key: {processor_key}")
    return f"{m.group(1)}.transformer_blocks.0.attn1", m.group(2)


def split_checkpoint(ckpt: dict) -> tuple[dict, dict]:
    """(`encoder.state_dict()`, `adapter.merges.state_dict()`) from a loaded `CameraCtrl.ckpt`."""
    merges = {}
    for k, v in ckpt["attention_processor_state_dict"].items():
        site, leaf = site_of(k)
        merges[f"{merge_key(site)}.{leaf}"] = v
    return dict(ckpt["pose_encoder_state_dict"]), merges


def load_into(adapter: CameraAdapter, ckpt: dict) -> None:
    encoder_sd, merge_sd = split_checkpoint(ckpt)
    adapter.encoder.load_state_dict(encoder_sd, strict=True)
    adapter.merges.load_state_dict(merge_sd, strict=True)
    adapter.clear_cache()


@torch.no_grad()
def merge_domain_lora(unet, lora_sd: dict, scale: float = 1.0) -> int:
    """`W += scale * up @ down` on the spatial attention projections (the authors' tools/merge_lora2unet.py).

    Returns the number of merged matrices. Every LoRA pair must find its target.
    """
    params = dict(unet.named_parameters())
    pairs: dict[tuple[str, str], dict] = {}
    for k, v in lora_sd.items():
        m = _LORA.match(k)
        if not m:
            raise KeyError(f"not a domain-adapter LoRA key: {k}")
        pairs.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = v
    for (path, proj), ab in pairs.items():
        target = f"{path}.{proj}.0.weight" if proj == "to_out" else f"{path}.{proj}.weight"
        if "motion_modules" in path or target not in params:
            raise KeyError(f"LoRA pair {path}.{proj} has no spatial-attention target in this U-Net")
        if set(ab) != {"down", "up"}:
            raise KeyError(f"LoRA pair {path}.{proj} is incomplete")
        w = params[target]
        delta = ab["up"].to(w.device, torch.float32) @ ab["down"].to(w.device, torch.float32)
        w.copy_((w.float() + scale * delta).to(w.dtype))
    return len(pairs)
