"""Loading CameraCtrl onto the project's pipeline, and choosing each lockstep track's camera.

    loaded = load_cameractrl()                       # Loaded, with loaded.extras["camera"] = CameraAdapter
    tracks = [Track("off", prompt), Track("pan", prompt, meta={"camera": "pan-left"})]
    run_lockstep(loaded, coords, tracks, noise, before_track=camera_switcher(loaded))

`Track.meta["camera"]` is None / "off" (no camera input), a trajectory name, a `CameraSpec`, or
a function `step -> one of those` (camera switched during denoising).
"""

from __future__ import annotations

import torch

from ..config import CONFIG_DIR, GenerationConfig
from ..pipeline import Loaded, build_pipeline
from .adapter import CameraAdapter, CameraSpec
from .convert import load_into, merge_domain_lora
from .pose_encoder import CameraPoseEncoder
from .rays import CAMERAS, Trajectory, trajectory

OFF = (None, "off", "none")


def cameractrl_config() -> GenerationConfig:
    return GenerationConfig.from_yaml(CONFIG_DIR / "cameractrl.yaml")


def _load_ckpt(repo: str, filename: str) -> dict:
    from huggingface_hub import hf_hub_download

    return torch.load(hf_hub_download(repo, filename), map_location="cpu", weights_only=True)


def attach_adapter(loaded: Loaded, ckpt: dict | None = None, encoder=None) -> CameraAdapter:
    """Build the adapter on `loaded.unet`, load `ckpt` if given, hook it in, and register it in `loaded.extras`."""
    adapter = CameraAdapter(loaded.unet, encoder or CameraPoseEncoder()).eval()
    if ckpt is not None:
        load_into(adapter, ckpt)
    p = next(loaded.unet.parameters())
    adapter.to(p.device, p.dtype).requires_grad_(False)
    adapter.attach()
    loaded.extras["camera"] = adapter
    return adapter


def load_cameractrl(cfg: GenerationConfig | None = None, profile: str = "gpu", device: str = "cuda") -> Loaded:
    """SD1.5 + AnimateDiff v3, the v3 domain-adapter LoRA merged in, and the CameraCtrl adapter attached (camera off)."""
    cfg = cfg or cameractrl_config()
    x = cfg.extra
    loaded = build_pipeline(cfg, profile=profile, device=device)
    lora = _load_ckpt(x["domain_lora_repo"], x["domain_lora_file"])
    n = merge_domain_lora(loaded.unet, lora.get("state_dict", lora), float(x.get("domain_lora_scale", 1.0)))
    attach_adapter(loaded, _load_ckpt(x["cameractrl_repo"], x["cameractrl_file"]))
    loaded.revisions["cameractrl"] = {"repo_id": x["cameractrl_repo"], "file": x["cameractrl_file"]}
    loaded.revisions["domain_lora"] = {"repo_id": x["domain_lora_repo"], "file": x["domain_lora_file"],
                                       "scale": float(x.get("domain_lora_scale", 1.0)), "merged_matrices": n}
    return loaded


def named_trajectory(cfg, name: str) -> Trajectory:
    """A named trajectory with the magnitudes of the config (`trajectory_step`, `trajectory_roll_deg`)."""
    x = getattr(cfg, "extra", {}) or {}
    return trajectory(name, cfg.frames, step=float(x.get("trajectory_step", 0.1)),
                      roll_deg=float(x.get("trajectory_roll_deg", 2.0)))


def resolve(cfg, camera) -> CameraSpec | None:
    """Turn a track's camera entry (name, Trajectory, CameraSpec, None / "off") into a `CameraSpec` with config magnitudes."""
    if camera in OFF:
        return None
    if isinstance(camera, (str, Trajectory)):
        camera = CameraSpec(camera)
    if not isinstance(camera, CameraSpec):
        raise TypeError(f"camera must be a name, a Trajectory, a CameraSpec or None, got {type(camera).__name__}")
    fix = lambda t: named_trajectory(cfg, t) if isinstance(t, str) else t
    return CameraSpec(fix(camera.trajectory), camera.scale, dict(camera.block_scales),
                      {k: fix(v) for k, v in camera.block_trajectories.items()})


def camera_switcher(loaded: Loaded):
    """`before_track(step, track)` for `run_lockstep`: sets the adapter from `track.meta["camera"]`.

    A track without the entry runs with the camera off, so state never carries over from the
    previous track. Call `loaded.extras["camera"].set(None)` after the run if the pipeline is
    reused outside a lockstep loop.
    """
    adapter: CameraAdapter = loaded.extras["camera"]
    cfg = loaded.cfg

    def before_track(step: int, track) -> None:
        camera = track.meta.get("camera")
        if callable(camera):
            camera = camera(step)
        adapter.set(resolve(cfg, camera), cfg.height, cfg.width, cfg.frames)

    return before_track


def on_until(camera, tau: int):
    """Camera on for steps < tau, off from step tau."""
    return lambda step: camera if step < tau else None


def on_from(camera, tau: int):
    """Camera off for steps < tau, on from step tau."""
    return lambda step: camera if step >= tau else None


__all__ = ["CAMERAS", "CameraSpec", "attach_adapter", "camera_switcher", "cameractrl_config", "load_cameractrl",
           "named_trajectory", "on_from", "on_until", "resolve"]
