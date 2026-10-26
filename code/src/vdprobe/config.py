"""Loads and provides the project's video generation settings as set in configs/generation.yaml"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass(frozen=True) # frozen to prevent changing of config values for consistent generation
class GenerationConfig:
    adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    base_candidates: tuple[str, ...] = (
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "botp/stable-diffusion-v1-5",
        "emilianJR/epiCRealism",
    )
    frames: int = 16
    height: int = 512
    width: int = 512
    steps: int = 25
    guidance_scale: float = 7.5
    negative_prompt: str = "bad quality, worse quality, blurry"
    prompt_template: str = "a {subject} {verb} {scene}, high quality, detailed"
    noise_device: str = "cpu"
    noise_dtype: str = "float16"
    extra: dict = field(default_factory=dict)

    @classmethod
    # Loads generation settings from a YAML file into a config object
    def from_yaml(cls, path: str | Path = CONFIG_DIR / "generation.yaml") -> "GenerationConfig":
        raw = yaml.safe_load(Path(path).read_text())
        known = {f for f in cls.__dataclass_fields__ if f != "extra"}
        kwargs = {k: v for k, v in raw.items() if k in known}
        if "base_candidates" in kwargs:
            kwargs["base_candidates"] = tuple(kwargs["base_candidates"])
        return cls(**kwargs, extra={k: v for k, v in raw.items() if k not in known})

    # Builds the video-generation prompt from a subject, action and scene
    def prompt(self, subject: str, verb: str, scene: str) -> str:
        return self.prompt_template.format(subject=subject, verb=verb, scene=scene)

    @property
    #Returns the shape of the initial noise tensor-(batch, channels, frames, h, w) for the configured video
    # The Stable Diffusion's VAE latent space uses one-eighth of the video resolution, with a batch size of 1 and 4 channels
    def latent_shape(self) -> tuple[int, int, int, int, int]:
        return (1, 4, self.frames, self.height // 8, self.width // 8)

    # Returns the video-generation settings as a dictionary for reusing elsewhere like when saving the settings with experimental results
    def to_dict(self) -> dict:
        d = asdict(self)
        d["base_candidates"] = list(self.base_candidates)
        return d
