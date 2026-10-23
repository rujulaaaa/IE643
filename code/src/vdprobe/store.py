"""On-disk format for one clip: summary.safetensors + manifest.json.

summary.safetensors holds, per coordinate name `c` (module path):
    c::mean     (T, C)          fp32
    c::var      (T, C)          fp32
    c::resnorm  (T, F)          fp32
    c::attn     (T, F, G, G)    fp32, attention coordinates only
with T the number of recorded denoising steps (25). Every later stage queries by
the triple (module path, step index, clip id): the file is the clip, the key prefix
is the module, the leading axis is the step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def save_clip(out_dir: str | Path, tensors: dict[str, torch.Tensor], manifest: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    st_path = out_dir / "summary.safetensors"
    save_file({k: v.contiguous() for k, v in tensors.items()}, str(st_path),
              metadata={"clip_id": str(manifest.get("clip_id", "")), "format": "vdprobe-summary-v1"})
    payload_bytes = sum(v.numel() * v.element_size() for v in tensors.values())
    sizes = {"payload_bytes": payload_bytes, "file_bytes": os.path.getsize(st_path),
             "n_tensors": len(tensors)}
    manifest = {**manifest, "storage": sizes}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return sizes


def load_clip(clip_dir: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    clip_dir = Path(clip_dir)
    return load_file(str(clip_dir / "summary.safetensors")), json.loads((clip_dir / "manifest.json").read_text())


def query(tensors: dict[str, torch.Tensor], module: str, field: str, step: int | None = None) -> torch.Tensor:
    t = tensors[f"{module}::{field}"]
    return t if step is None else t[step]
