"""The camera probing set: static scenes filmed by a moving camera.

A clip is (scene, camera, seed). Content is the scene prompt; motion is the camera, which never
appears in the prompt: it is a MotionLoRA (configs/camera_set.yaml) or a CameraCtrl trajectory
(configs/cc_set.yaml). A camera pair holds scene, seed and prompt fixed and changes only the
camera; a content pair holds the camera and changes the scene.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import CONFIG_DIR

DEFAULT_TEMPLATE = "{scene}, high quality, detailed"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


@dataclass(frozen=True)
class CameraClip:
    scene: str
    camera: str
    seed: int

    @property
    def id(self) -> str:
        return f"{self.seed}__{_slug(self.scene)}__{self.camera}"

    def prompt(self, cfg=None) -> str:
        """The scene prompt. A config whose template has other fields (the frozen subject/verb one) is not used."""
        template = getattr(cfg, "prompt_template", None) or DEFAULT_TEMPLATE
        fields = {name for _, name, _, _ in string.Formatter().parse(template) if name}
        return (template if fields == {"scene"} else DEFAULT_TEMPLATE).format(scene=self.scene)

    @property
    def phrases(self) -> dict[str, str]:
        return {"scene": self.scene}


@dataclass(frozen=True)
class CameraPair:
    kind: str          # "motion" (camera swap) | "content" (scene swap), the names analyze_sweep.py expects
    a: CameraClip      # the anchor clip
    b: CameraClip

    @property
    def swapped(self) -> tuple[str, str]:
        return (self.a.camera, self.b.camera) if self.kind == "motion" else (self.a.scene, self.b.scene)


class CameraSet:
    def __init__(self, path: str | Path = CONFIG_DIR / "camera_set.yaml"):
        raw = yaml.safe_load(Path(path).read_text())
        self.raw = raw
        self.scenes: list[str] = raw["scenes"]
        self.cameras: list[str] = [str(c) for c in raw["cameras"]]
        self.anchor_scene: str = raw["anchor_scene"]
        self.anchor_camera: str = str(raw["anchor_camera"])
        self.seeds: list[int] = raw["seeds"]
        self.heldout_seed: int = raw["heldout_seed"]
        self.lora_repo: str | None = raw.get("lora_repo")
        self.expected: dict[str, int] = raw.get("expected_counts", {})
        if self.anchor_scene not in self.scenes or self.anchor_camera not in self.cameras:
            raise AssertionError("the anchor scene and anchor camera must be members of the set")
        if len(set(self.scenes)) != len(self.scenes) or len(set(self.cameras)) != len(self.cameras):
            raise AssertionError("duplicate scene or camera")
        self.check_counts()

    @property
    def moving_cameras(self) -> list[str]:
        return [c for c in self.cameras if c != self.anchor_camera]

    def clips(self, seed: int) -> list[CameraClip]:
        return [CameraClip(s, c, seed) for s in self.scenes for c in self.cameras]

    def camera_pairs(self, seed: int) -> list[CameraPair]:
        """Same scene and seed: the anchor camera against every other camera."""
        return [CameraPair("motion", CameraClip(s, self.anchor_camera, seed), CameraClip(s, c, seed))
                for s in self.scenes for c in self.moving_cameras]

    def content_pairs(self, seed: int) -> list[CameraPair]:
        """Same camera and seed: the anchor scene against every other scene."""
        return [CameraPair("content", CameraClip(self.anchor_scene, c, seed), CameraClip(s, c, seed))
                for c in self.cameras for s in self.scenes if s != self.anchor_scene]

    def all_clips(self) -> list[CameraClip]:
        return [c for seed in self.seeds for c in self.clips(seed)]

    def check_counts(self) -> dict[str, int]:
        seed = self.seeds[0]
        counts = {"clips_per_seed": len(self.clips(seed)), "camera_pairs": len(self.camera_pairs(seed)),
                  "content_pairs": len(self.content_pairs(seed))}
        for k, v in self.expected.items():
            if counts.get(k) != v:
                raise AssertionError(f"camera set drifted from the frozen protocol: {k}={counts.get(k)}, expected {v}")
        return counts
