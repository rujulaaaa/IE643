"""Building the AnimateDiff pipeline, the noise policy, and one generation.

Two memory profiles, neither of which changes the computation:
  gpu      everything resident on the GPU + VAE slicing (E1's Kaggle T4 configuration, ~10.2 GB)
  lowvram  model CPU offload + VAE slicing, for 8 GB cards (D-27)
"""

from __future__ import annotations

import gc
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from .config import GenerationConfig

SD_FILES = ["model_index.json", "scheduler/*", "tokenizer/*", "feature_extractor/*",
            "text_encoder/config.json", "unet/config.json", "vae/config.json"]
SD_WEIGHTS = {"fp16": ["text_encoder/model.fp16.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors",
                       "vae/diffusion_pytorch_model.fp16.safetensors"],
              None: ["text_encoder/model.safetensors", "unet/diffusion_pytorch_model.safetensors",
                     "vae/diffusion_pytorch_model.safetensors"]}
ADAPTER_FILES = ["config.json", "diffusion_pytorch_model.safetensors"]
PROFILES = ("gpu", "lowvram")


@dataclass
class Loaded:
    pipe: object
    cfg: GenerationConfig
    profile: str
    base_id: str
    revisions: dict
    load_s: float
    device: str = "cuda"
    extras: dict = field(default_factory=dict)

    @property
    def unet(self):
        return self.pipe.unet


def _snapshot(repo_id: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, allow_patterns=patterns))
    except Exception:
        # offline: fall back to whatever is already cached
        return Path(snapshot_download(repo_id, allow_patterns=patterns, local_files_only=True))


def _resolve_base(cfg: GenerationConfig) -> tuple[str, Path, str | None]:
    last = None
    for repo in cfg.base_candidates:
        for variant in ("fp16", None):
            try:
                path = _snapshot(repo, SD_FILES + SD_WEIGHTS[variant])
                if all((path / f).exists() for f in SD_WEIGHTS[variant]):
                    return repo, path, variant
            except Exception as e:  # repo withdrawn, gated or offline
                last = e
        print(f"  [skip] {repo}: {type(last).__name__ if last else 'weights missing'}")
    raise RuntimeError(f"no SD1.5 mirror resolved; last error: {last}")


def build_pipeline(cfg: GenerationConfig, profile: str = "gpu", device: str = "cuda",
                   vae_slicing: bool = True) -> Loaded:
    from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
    from diffusers.utils import logging as dlog

    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}")
    dlog.set_verbosity_error()
    t0 = time.perf_counter()

    adapter_path = _snapshot(cfg.adapter_id, ADAPTER_FILES)
    adapter = MotionAdapter.from_pretrained(adapter_path, torch_dtype=torch.float16)
    base_id, base_path, variant = _resolve_base(cfg)
    pipe = AnimateDiffPipeline.from_pretrained(base_path, motion_adapter=adapter,
                                               torch_dtype=torch.float16, variant=variant)
    # AnimateDiff reference scheduler settings, not diffusers' defaults.
    pipe.scheduler = DDIMScheduler.from_pretrained(base_path, subfolder="scheduler", clip_sample=False,
                                                   timestep_spacing="linspace", beta_schedule="linear",
                                                   steps_offset=1)
    pipe.set_progress_bar_config(disable=True)
    if profile == "lowvram":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if vae_slicing:  # decode frames one at a time; E1 measured -4 GB peak on a T4
        pipe.vae.enable_slicing()

    revisions = {  # D-20: the snapshot directory name is the resolved commit hash
        "base_model": {"repo_id": base_id, "commit": base_path.name, "variant": variant},
        "motion_adapter": {"repo_id": cfg.adapter_id, "commit": adapter_path.name},
    }
    return Loaded(pipe, cfg, profile, base_id, revisions, time.perf_counter() - t0, device)


def make_noise(cfg: GenerationConfig, seed: int) -> torch.Tensor:
    """The fixed noise tensor for a seed: drawn on the CPU in fp16, identical on any GPU (E2, D-04)."""
    g = torch.Generator(cfg.noise_device).manual_seed(seed)
    return torch.randn(cfg.latent_shape, generator=g, dtype=getattr(torch, cfg.noise_dtype),
                       device=cfg.noise_device)


@dataclass
class Generation:
    latents: torch.Tensor          # final latents (1, 4, F, h, w), CPU
    frames: list | None            # decoded PIL frames
    seconds: float                 # denoising wall clock (excludes decode)
    decode_seconds: float = 0.0
    denoise_peak_alloc_gb: float = 0.0
    peak_alloc_gb: float = 0.0     # whole call including decode
    peak_reserved_gb: float = 0.0


@torch.no_grad()
def decode(loaded: Loaded, latents: torch.Tensor) -> list:
    pipe = loaded.pipe
    video = pipe.decode_latents(latents.to(loaded.device, torch.float16), decode_chunk_size=16)
    return pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]


@torch.no_grad()
def generate(loaded: Loaded, prompt: str, noise: torch.Tensor, decode_frames: bool = True) -> Generation:
    cfg, pipe = loaded.cfg, loaded.pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = pipe(prompt=prompt, negative_prompt=cfg.negative_prompt, num_frames=cfg.frames,
               height=cfg.height, width=cfg.width, num_inference_steps=cfg.steps,
               guidance_scale=cfg.guidance_scale, latents=noise.clone().to(loaded.device),
               output_type="latent")
    torch.cuda.synchronize()
    gen = Generation(latents=out.frames.detach().float().cpu(), frames=None, seconds=time.perf_counter() - t0)
    gen.denoise_peak_alloc_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    if decode_frames:
        t1 = time.perf_counter()
        gen.frames = decode(loaded, out.frames)
        torch.cuda.synchronize()
        gen.decode_seconds = time.perf_counter() - t1
    gen.peak_alloc_gb = torch.cuda.max_memory_allocated() / 2 ** 30
    gen.peak_reserved_gb = torch.cuda.max_memory_reserved() / 2 ** 30
    return gen


def phrase_token_positions(tokenizer, prompt: str, phrases: dict[str, str]) -> dict[str, list[int]]:
    """Token positions of each phrase inside the 77-token prompt (for cross-attention maps).
    `_all` holds every prompt token between BOS and EOS."""
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length,
                    truncation=True).input_ids
    out = {"_all": list(range(1, ids.index(tokenizer.eos_token_id)))}
    for name, phrase in phrases.items():
        pid = tokenizer(phrase, add_special_tokens=False).input_ids
        hit = next((i for i in range(len(ids) - len(pid) + 1) if ids[i:i + len(pid)] == pid), None)
        if hit is None:
            raise ValueError(f"phrase '{phrase}' not found in prompt '{prompt}'")
        out[name] = list(range(hit, hit + len(pid)))
    return out


def env_info() -> dict:
    import diffusers
    import transformers

    info = {"python": platform.python_version(), "torch": torch.__version__,
            "diffusers": diffusers.__version__, "transformers": transformers.__version__}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_total_vram_gb=round(p.total_memory / 2 ** 30, 2),
                    capability=f"sm_{p.major}{p.minor}")
    return info
