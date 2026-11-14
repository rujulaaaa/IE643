"""The two camera models behind one interface, so every camera script runs on both.

    lora   SD1.5 + AnimateDiff v2 motion module + the eight camera MotionLoRAs (Part A, D-40).
           The camera is a weight change; configs/generation.yaml, configs/camera_set.yaml.
    cc     CameraCtrl (Part B, D-41): the camera is an input trajectory;
           configs/cameractrl.yaml, configs/cc_set.yaml.

In both, a track names its camera in `Track.meta["camera"]` (a name of the model's camera set,
the set's anchor camera for "no camera", or a function step -> name | None) and
`CameraModel.before_track` applies it inside `run_lockstep`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from . import camlora
from .camera_set import CameraSet
from .config import CONFIG_DIR, GenerationConfig
from .pipeline import Loaded

MODELS = ("lora", "cc")
CONFIGS = {"lora": ("generation.yaml", "camera_set.yaml"), "cc": ("cameractrl.yaml", "cc_set.yaml")}
# camera names of the two sets by motion type, in the order (positive, negative) of the set's own naming
FAMILIES = {
    "lora": {"pan": ("pan-left", "pan-right"), "tilt": ("tilt-up", "tilt-down"), "zoom": ("zoom-in", "zoom-out"),
             "roll": ("rolling-clockwise", "rolling-anticlockwise")},
    "cc": {"pan": ("pan-left", "pan-right"), "tilt": ("tilt-up", "tilt-down"), "zoom": ("zoom-in", "zoom-out"),
           "roll": ("roll-cw", "roll-ccw")},
}
SIG_INDEX = {"pan": 0, "tilt": 1, "zoom": 2, "roll": 3}      # component of metrics.camera_signature


def family_of(model: str, camera: str) -> str | None:
    """Motion type of a camera name ("pan", "tilt", "zoom", "roll"); None for off / static."""
    return next((f for f, names in FAMILIES[model].items() if camera in names), None)


@dataclass
class CameraModel:
    name: str                       # "lora" | "cc"
    loaded: Loaded
    cameras: CameraSet
    before_track: Callable          # run_lockstep's before_track
    reset: Callable[[], None]       # camera off, for calls outside a lockstep loop

    @property
    def off(self) -> str:
        return self.cameras.anchor_camera

    @property
    def flow_size(self) -> tuple[int, int]:
        """RAFT input size: 256 px high, the model's aspect ratio, sides divisible by 8."""
        c = self.loaded.cfg
        return 256, max(8, round(256 * c.width / c.height / 8) * 8)

    def block_only(self, camera: str, block: str) -> dict:
        """Track metadata that keeps the camera in the modules under `block` and removes it elsewhere."""
        return self.block_meta(camera, {"": 0.0, block: 1.0})

    def block_without(self, camera: str, block: str) -> dict:
        return self.block_meta(camera, {block: 0.0})

    def block_meta(self, camera: str, scales: dict[str, float]) -> dict:
        if self.name == "lora":
            return {"camera": camera, "camera_block_scales": scales}
        from .cameractrl.adapter import CameraSpec
        return {"camera": CameraSpec(camera, block_scales=scales)}


def default_config(model: str):
    return CONFIG_DIR / CONFIGS[model][0]


def load_camera_model(model: str, config=None, profile: str | None = None, cameras=None) -> CameraModel:
    """Build the pipeline of `model`. `cameras` limits which MotionLoRAs are attached (lora only)."""
    if model not in MODELS:
        raise ValueError(f"model must be one of {MODELS}")
    cs = CameraSet(CONFIG_DIR / CONFIGS[model][1])
    cfg = GenerationConfig.from_yaml(config or default_config(model))
    if model == "cc":
        from .cameractrl.model import camera_switcher, load_cameractrl
        loaded = load_cameractrl(cfg, profile=profile or "gpu")
        adapter = loaded.extras["camera"]
        return CameraModel(model, loaded, cs, camera_switcher(loaded), lambda: adapter.set(None))
    from .pipeline import build_pipeline
    from .runio import default_profile
    loaded = build_pipeline(cfg, profile=profile or default_profile())
    names = camlora.load_camera_loras(loaded.pipe, cameras or cs.moving_cameras, cs.lora_repo)
    loaded.revisions["camera_loras"] = {"repo_template": cs.lora_repo, "cameras": names}
    unet = loaded.unet
    return CameraModel(model, loaded, cs, camlora.camera_switcher(unet), lambda: camlora.use_camera(unet, None))
