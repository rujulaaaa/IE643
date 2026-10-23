"""Activation patching over the 25 denoising steps, via paired runs (Deliverable 3).

A paired run is two generations from the same noise tensor:

  source run (clip A)   TensorCache records A's activations at (coordinates × steps)
  target run (clip B)   a probe reads that cache while B is generated:
                          Patcher       splices A's activation into B's forward pass
                                        (resampling corruption from a real clip, D-12)
                          PairDistance  scores B against A with the D-01 distances,
                                        keeping only scalars (D-26)

Patch granularities implemented here (roadmap §5.3): 1 whole tensor; 2 channel subset;
4 scaled injection out + α·(cached − out). Granularity 3 (temporal attention weights
only) is an attention-processor swap, in `edit.py`.

Component patches (D-29): `component="dynamic"` transplants only the source's
frame-mean-removed part x̃ and keeps the target's static part x̄; `component="static"`
does the reverse. These are the D-01 decomposition used as an intervention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .coords import Coordinate, branch_rows, frame_mean_raw
from .harness import Harness, Probe
from .metrics import latent_mad, psnr
from .pipeline import Generation, Loaded, generate
from .summaries import paired_distances

# D-11: the null patch must reproduce the unpatched run to this tolerance.
NULL_PATCH_MAX_LATENT_MAD = 1e-3
NULL_PATCH_MIN_PSNR_DB = 45.0


def window(name: str, n_steps: int = 25, center: int | None = None) -> tuple[int, ...]:
    """Patch windows over the denoising trajectory (D-13): 'one', 'five', 'first_third', 'all'."""
    if name == "all":
        return tuple(range(n_steps))
    if name == "first_third":
        return tuple(range(math.ceil(n_steps / 3)))
    if center is None:
        raise ValueError(f"window '{name}' needs a center step")
    if name == "one":
        return (center,)
    if name == "five":
        return tuple(i for i in range(center - 2, center + 3) if 0 <= i < n_steps)
    raise ValueError(name)


@dataclass(frozen=True)
class PatchSpec:
    coords: tuple[str, ...]
    steps: tuple[int, ...]
    alpha: float = 1.0                       # granularity 4 when < 1
    branches: str = "all"                    # "all" CFG branches, or "cond" only
    channels: dict = field(default_factory=dict)  # coord name -> (C,) mask; granularity 2
    component: str = "all"                   # "all" | "dynamic" (x̃ only) | "static" (x̄ only), D-29

    def __post_init__(self):
        if self.branches not in ("all", "cond"):
            raise ValueError("branches must be 'all' or 'cond'")
        if self.component not in ("all", "dynamic", "static"):
            raise ValueError("component must be 'all', 'dynamic' or 'static'")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")


class TensorCache(Probe):
    """Full activations at selected (coordinate, step), moved to `device` (CPU by default).

    mode="raw"   the tensor as it flows, all CFG branches (what Patcher needs)
    mode="cond"  text-conditional branch in canonical (F, C, S) form (what PairDistance needs)
    """

    def __init__(self, coords, steps, mode: str = "raw", device: str = "cpu"):
        super().__init__(coords, steps)
        if mode not in ("raw", "cond"):
            raise ValueError(mode)
        self.mode, self.device = mode, device
        self.store: dict[tuple[str, int], torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        t = ctx.cond(output, coord.kind) if self.mode == "cond" else output
        self.store[(coord.name, ctx.step)] = t.detach().to(self.device, copy=True)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.store.values())


class Patcher(Probe):
    """Replace the output at (coord, step) with the source clip's cached activation.

    `step_offset` reads the cache at step + offset. It exists for the negative control:
    a mis-indexed null patch must FAIL the D-11 criterion, or the check has no power.
    """

    def __init__(self, source: TensorCache, spec: PatchSpec, step_offset: int = 0):
        super().__init__(spec.coords, spec.steps)
        if source.mode != "raw":
            raise ValueError("patching needs a raw-mode TensorCache")
        self.source, self.spec, self.step_offset = source, spec, step_offset
        self.applied: list[tuple[str, int]] = []

    @torch.no_grad()
    def __call__(self, ctx, coord: Coordinate, module, output, args, kwargs):
        key = (coord.name, ctx.step + self.step_offset)
        if key not in self.source.store:
            raise KeyError(f"source cache has no activation for {key}")
        cached = self.source.store[key]
        src = cached.to(output.device, output.dtype)
        if src is cached:  # never hand the cache itself to the network
            src = src.clone()
        if src.shape != output.shape:
            raise ValueError(f"{coord.name}: cached {tuple(src.shape)} vs live {tuple(output.shape)}")
        self.applied.append((coord.name, ctx.step))
        spec = self.spec
        if spec.component != "all":
            # x = x̄ + x̃: keep one part of the live output, take the other from the source.
            # Written as out + (part of δ), δ = src − out, so a self-patch (δ = 0) is exact in fp16.
            delta = src.float() - output.float()
            fm = frame_mean_raw(delta, coord.kind, ctx.n_branches, ctx.frames)
            src = (output.float() + (delta - fm if spec.component == "dynamic" else fm)).to(output.dtype)
        mask = spec.channels.get(coord.name)
        if spec.alpha == 1.0 and spec.branches == "all" and mask is None:
            return src
        # weight = alpha · channel mask · branch mask, broadcast to the output.
        # lerp is exact at both ends: weight 0 returns the live value, weight 1 the source.
        weight = torch.full([1] * output.ndim, spec.alpha, device=output.device)
        if mask is not None:
            shape = [1] * output.ndim
            shape[coord.channel_dim] = -1
            weight = weight * mask.to(output.device, torch.float32).reshape(shape)
        if spec.branches == "cond":
            keep = torch.zeros(output.shape[0], device=output.device)
            keep[branch_rows(output.shape[0], ctx.n_branches, ctx.cond_branch)] = 1
            weight = weight * keep.reshape(-1, *[1] * (output.ndim - 1))
        return torch.lerp(output.float(), src.float(), weight.expand(output.shape)).to(output.dtype)


class PairDistance(Probe):
    """D-01 distances of the live (target) clip against a cached reference clip."""

    def __init__(self, reference: TensorCache, per_channel: bool = False):
        if reference.mode != "cond":
            raise ValueError("PairDistance needs a cond-mode TensorCache")
        super().__init__({k[0] for k in reference.store}, {k[1] for k in reference.store})
        self.reference, self.per_channel = reference, per_channel
        self.rows: list[dict] = []

    def wants(self, coord, step):
        return (coord.name, step) in self.reference.store

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        xa = self.reference.store[(coord.name, ctx.step)].to(output.device)
        d = paired_distances(xa, ctx.cond(output, coord.kind), per_channel=self.per_channel)
        self.rows.append({"coord": coord.name, "short": coord.short, "kind": coord.kind,
                          "block": coord.block, "level": coord.level, "step": ctx.step, **d})


# ---------------------------------------------------------------------------
# Paired-run orchestration
# ---------------------------------------------------------------------------

def run_with(loaded: Loaded, coords: list[Coordinate], probes: list[Probe], prompt: str,
             noise: torch.Tensor, token_index: dict | None = None, decode_frames: bool = True) -> Generation:
    with Harness(loaded.unet, coords, probes, token_index=token_index) as ctx:
        gen = generate(loaded, prompt, noise, decode_frames=decode_frames)
    if ctx.step + 1 != loaded.cfg.steps:
        raise RuntimeError(f"expected {loaded.cfg.steps} UNet calls, saw {ctx.step + 1}")
    return gen


def null_patch_verdict(reference: Generation, patched: Generation) -> dict:
    mad = latent_mad(reference.latents, patched.latents)
    db = psnr(reference.frames, patched.frames)
    return {"latent_mad": mad, "psnr_db": db,
            "pass": bool(mad < NULL_PATCH_MAX_LATENT_MAD and db > NULL_PATCH_MIN_PSNR_DB),
            "criterion": f"latent MAD < {NULL_PATCH_MAX_LATENT_MAD:g} and PSNR > {NULL_PATCH_MIN_PSNR_DB:g} dB (D-11)"}
