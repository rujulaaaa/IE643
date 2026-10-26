"""This file:
1. Loads the AnimateDiff pipeline from the configured base model and motion adapter.
2. Prepares the initial noise tensor used for video generation.
3. Generates video clips and decodes latent tensors into frames.

Memory profiles:
- gpu: Keeps the model on the GPU and uses VAE slicing (using about 10.2 GB on Kaggle T4 GPU).
- lowvram: Offloads the model to the CPU and uses VAE slicing for 8 GB GPUs.

Both profiles preserve the same generation computation.
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


# Data container for the loaded AnimateDiff pipeline, its config, and metadata about the load
@dataclass
class Loaded:
    pipe: object                 # Loaded AnimateDiff pipeline
    cfg: GenerationConfig        # Video-generation settings used by the pipeline
    profile: str                 # Memory profile used to load the model (CPU or GPU)
    base_id: str                 # Hugging Face ID of the selected base model
    revisions: dict              # Model repository revisions used for this run (versions of base model and motion adapter)
    load_s: float                # Time taken to load the model, in seconds
    device: str = "cuda"          # Device used for model inference
    extras: dict = field(default_factory=dict)  # Additional model-loading details that dont have their own field as such

    @property
    # Returns the U-Net component of the loaded pipeline 
    def unet(self):
        """Return the U-Net component of the loaded pipeline."""
        return self.pipe.unet

# Downloads the requested model files from Hugging Face or uses the local cache if downloading fails
def _snapshot(repo_id: str, patterns: list[str]) -> Path:
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo_id, allow_patterns=patterns))
    except Exception:
        # offline: fall back to whatever is already cached
        return Path(snapshot_download(repo_id, allow_patterns=patterns, local_files_only=True))

# Checks the configured Hugging Face repositories for a usable Stable Diffusion 1.5 model
def _resolve_base(cfg: GenerationConfig) -> tuple[str, Path, str | None]:
    last = None
    # Checks each base model repository in the configured order
    for repo in cfg.base_candidates:
        # Prefers fp16 weights and falls back to full-precision weights
        for variant in ("fp16", None):
            try:
                path = _snapshot(repo, SD_FILES + SD_WEIGHTS[variant])
                if all((path / f).exists() for f in SD_WEIGHTS[variant]):
                    return repo, path, variant
            except Exception as e:  # repository may be withdrawn, unavailable or offline
                last = e
        print(f"  [skip] {repo}: {type(last).__name__ if last else 'weights missing'}")
    raise RuntimeError(f"no SD1.5 mirror resolved; last error: {last}")

# Aseembles the complete pipeline = pretrained Stable Diffusion base model + pretrained AnimateDiff motion adapter
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
    # weights loaded in float16 (half precision): half the memory of float32, which matters on our 8 GB laptop GPU and the T4

    base_id, base_path, variant = _resolve_base(cfg)
    # Clubs and makes the pipeline using
    # 1. Base model (which includes the text encoder, U-Net and VAE) 
    # and 2. the motion adapter - this gives pretrained AnimateDiff motion weights (adds temporal processing to Unets)
    # Model weights here again taken in low precision format
    pipe = AnimateDiffPipeline.from_pretrained(base_path, motion_adapter=adapter,
                                              
                                               torch_dtype=torch.float16, variant=variant)
    # The Diffusers' (software library used to load and run model) default options was discarded since it was not optimal here, instead we set specific DDIM scheduler options
    # So, DDIM scheduler controls how model moves from the noisy latent space toward the final video
    # The below options show how the pipeline moves through its denoising steps and how noise is added to the latent tensor at each step
    pipe.scheduler = DDIMScheduler.from_pretrained(base_path, subfolder="scheduler", clip_sample=False,
                                                   timestep_spacing="linspace", beta_schedule="linear",
                                                   steps_offset=1)

    """To explain these options further: all are configured this way to match the AnimateDiff's original scheduler values
    1. clip_sample = False: disabled clipping the predicted image tensor values to a fixed range (to match AnimateDiff's actual scheduler values)
    2. timestep_spacing = makes sure that the denoising timesteps are evenly spaced 
    3. beta_schedule = linear schedule used for noise levels called beta values (meaning added noise across steps will increase linearly)
                     - kept this way since motin adapter was trained to work with this process 
    """
    
    pipe.set_progress_bar_config(disable=True)
    if profile == "lowvram":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    # VAE = variational autoencoder
    # - used to convert the latent video tensor (that U-Net works with when generating videos) into viewable frames

    # Decoding here involved the VAE to create full resolution frame from the smaller latents 
    # This expansion needs GPU memory for intermediate computations
    # Used majorly for memory optimization, slice the input into smaller batches instead of taking it as one batch
    if vae_slicing:  
        pipe.vae.enable_slicing()

    # Records the model versions and precision used for experiment traceability.
    revisions = {  
        "base_model": {"repo_id": base_id, "commit": base_path.name, "variant": variant},
        "motion_adapter": {"repo_id": cfg.adapter_id, "commit": adapter_path.name},
    }
    return Loaded(pipe, cfg, profile, base_id, revisions, time.perf_counter() - t0, device)


# Generates seeded fp16 noise on the CPU for consistent starting latents across GPUs for inference part
def make_noise(cfg: GenerationConfig, seed: int) -> torch.Tensor:
    g = torch.Generator(cfg.noise_device).manual_seed(seed)
    return torch.randn(cfg.latent_shape, generator=g, dtype=getattr(torch, cfg.noise_dtype),
                       device=cfg.noise_device)


# Stores the results (outputs and performance metrics) of a video generation run
@dataclass
class Generation:
    latents: torch.Tensor              # Final video latents with shape (1, 4, F, h, w) stored on the CPU
    frames: list | None                # Decoded video frames as PIL images, or None if decoding is skipped

    # Denoising = repeateadly updating the noisy latent tensors to move towards the final latent video representation
    # Decoding = VAE used to convert the final representation into viewable image frames
    seconds: float                     # Denoising time in seconds, excluding decoding
    decode_seconds: float = 0.0        # Time spent decoding the latents into video frames
    denoise_peak_alloc_gb: float = 0.0 # Peak GPU memory allocated during denoising, in GB
    peak_alloc_gb: float = 0.0         # Peak GPU memory allocated during the full call, including decoding, in GB
    peak_reserved_gb: float = 0.0      # Peak GPU memory reserved by PyTorch's caching allocator, in GB

@torch.no_grad()
# Decodes the final video latents with the pretrained VAE and returns the frames as images
def decode(loaded: Loaded, latents: torch.Tensor) -> list:
    pipe = loaded.pipe
    video = pipe.decode_latents(latents.to(loaded.device, torch.float16), decode_chunk_size=16)
    return pipe.video_processor.postprocess_video(video=video, output_type="pil")[0]


@torch.no_grad() # gradient tracking disabled since only inference is being done here, not training
# Generates a video from the given prompt and initial noise tensor, returning the results and performance metrics
def generate(loaded: Loaded, prompt: str, noise: torch.Tensor, decode_frames: bool = True) -> Generation:
    cfg, pipe = loaded.cfg, loaded.pipe
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    #This returns the final video latents first after each denoising step 
    #VAE decoding is postponed until later to save GPU memory since decoding is memory intensive
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

# Maps prompt phrases to token positions for cross-attention analysis
# The `_all` entry contains all tokens between the beginning and end markers
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

# Collects software +GPU details for recording the experiment environment
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
