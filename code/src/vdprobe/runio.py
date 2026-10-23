"""Shared plumbing for the experiment scripts: CLI flags, result records, artifacts."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import torch

from .config import REPO_ROOT, GenerationConfig
from .pipeline import PROFILES, Loaded, build_pipeline, env_info


def default_profile() -> str:
    """Everything on the GPU where it fits. Measured peaks for the frozen config: 10.2 GB on a
    T4 (sm_75), 6.4 GB on an RTX 4060 Laptop (sm_89, which has flash attention)."""
    if not torch.cuda.is_available():
        raise SystemExit("a CUDA GPU is required")
    p = torch.cuda.get_device_properties(0)
    vram = p.total_memory / 2 ** 30
    return "gpu" if vram >= 12 or (p.major >= 8 and vram >= 7.5) else "lowvram"


def parser(description: str, experiment: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--profile", choices=PROFILES, default=None,
                   help="memory profile; default: gpu where it fits (>= 12 GB, or >= 7.5 GB on sm_80+), else lowvram")
    p.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "local" / experiment,
                   help="output directory")
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "generation.yaml")
    return p


def load(args) -> Loaded:
    cfg = GenerationConfig.from_yaml(args.config)
    profile = args.profile or default_profile()
    print(f"building pipeline (profile={profile}) ...")
    loaded = build_pipeline(cfg, profile=profile)
    print(f"  base {loaded.base_id}@{loaded.revisions['base_model']['commit'][:10]}, "
          f"adapter @{loaded.revisions['motion_adapter']['commit'][:10]}, {loaded.load_s:.1f} s")
    return loaded


def record(experiment: str, loaded: Loaded, **fields) -> dict:
    c = loaded.cfg
    return {"experiment": experiment, "date": datetime.date.today().isoformat(), **env_info(),
            "profile": loaded.profile, "revisions": loaded.revisions,
            "frames": c.frames, "resolution": f"{c.height}x{c.width}", "steps": c.steps,
            "scheduler": "DDIM", "guidance_scale": c.guidance_scale, **fields}


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(f"wrote {path}")


def save_gif(frames, path: Path, size: int = 256, fps: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    small = [f.resize((size, size)) for f in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0)
