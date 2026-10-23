"""Lockstep sampling: several clips from one noise tensor, denoised one step at a time together.

`AnimateDiffPipeline.__call__` runs one clip through all 25 steps. A paired measurement
then needs the first clip's activations cached for every step it will be compared at,
which for the 16 coarse coordinates is ~7.6 GB of host memory per anchor clip (fp16,
text-conditional branch). Here the denoising loop is unrolled across clips instead:

    for step t:
        for clip in order:            anchors first, then the clips scored against them
            UNet(clip, t)             hooks see (clip, t); probes record / score / patch
            DDIM step

so a probe only ever needs the step-t activations of the clips that ran before it in the
same step. Memory holds one step instead of 25, the D-01 distances stay on the full
flattened tensors (D-26), and every clip is generated exactly once. Patching uses the same
loop: the source clip runs first at step t and the target reads its fresh activation.

The loop replicates `AnimateDiffPipeline.__call__` of diffusers 0.40 (CFG batch
[negative, positive], prompt embeddings repeated per frame, `scale_model_input`, guidance,
stateless DDIM step with eta = 0). `scripts/lockstep_check.py` verifies that a clip comes
out the same as from `pipe(...)`.
"""

from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field

import torch

from .coords import Coordinate
from .harness import Harness, Probe
from .pipeline import Loaded
from .summaries import paired_distances, split_static_dynamic


@dataclass
class Track:
    """One clip inside a lockstep group."""
    id: str
    prompt: str
    probes: list = field(default_factory=list)
    embeds: torch.Tensor | None = None    # (2F, 77, D): [negative, positive], repeated per frame
    latents: torch.Tensor | None = None   # (1, 4, F, h, w), current state
    start: torch.Tensor | None = None     # this track's own starting latents instead of the shared noise
    meta: dict = field(default_factory=dict)


class LockstepHarness(Harness):
    """Harness whose step index is set by the sampler instead of counted per UNet call."""

    def _step_hook(self, module, args, kwargs):
        sample = args[0] if args else kwargs["sample"]
        self.ctx.n_branches, self.ctx.frames = sample.shape[0], sample.shape[2]
        self.ctx.latent_hw = tuple(sample.shape[-2:])

    def __enter__(self):
        # Hook every coordinate once; probes are swapped per track, so the set of wanted
        # coordinates is not known at install time.
        modules = dict(self.unet.named_modules())
        self._handles.append(self.unet.register_forward_pre_hook(self._step_hook, with_kwargs=True))
        for c in self.coords:
            self._handles.append(modules[c.name].register_forward_hook(self._make_hook(c), with_kwargs=True))
        return self.ctx


@torch.no_grad()
def encode(loaded: Loaded, prompt: str) -> torch.Tensor:
    pipe, cfg = loaded.pipe, loaded.cfg
    pos, neg = pipe.encode_prompt(prompt, loaded.device, 1, True, cfg.negative_prompt)
    return torch.cat([neg, pos]).repeat_interleave(cfg.frames, dim=0)


@torch.no_grad()
def run_lockstep(loaded: Loaded, coords: list[Coordinate], tracks: list[Track], noise: torch.Tensor,
                 after_track=None, after_step=None, before_track=None, log_every: int = 0) -> dict:
    """Denoise every track from `noise`, in list order within each step.

    `before_track(step, track)` runs just before a track's UNet call (e.g. to switch an
    attention processor between recording and injecting); `after_track(step, track)` and
    `after_step(step)` let callers free per-step caches.
    Returns timing; final latents are left on each track (CPU, fp32).
    """
    pipe, cfg = loaded.pipe, loaded.cfg
    dev = loaded.device
    sched = pipe.scheduler
    sched.set_timesteps(cfg.steps, device=dev)
    timesteps = sched.timesteps
    for tr in tracks:
        if tr.embeds is None:
            tr.embeds = encode(loaded, tr.prompt)
        tr.latents = (noise if tr.start is None else tr.start).clone().to(dev) * sched.init_noise_sigma

    harness = LockstepHarness(loaded.unet, coords, [])
    t0 = time.perf_counter()
    with harness as ctx:
        ctx.timesteps = [float(t) for t in timesteps]
        for i, t in enumerate(timesteps):
            ctx.step = i
            for tr in tracks:
                harness.probes = tr.probes
                if before_track:
                    before_track(i, tr)
                inp = sched.scale_model_input(torch.cat([tr.latents] * 2), t)
                pred = loaded.unet(inp, t, encoder_hidden_states=tr.embeds).sample
                uncond, text = pred.chunk(2)
                pred = uncond + cfg.guidance_scale * (text - uncond)
                tr.latents = sched.step(pred, t, tr.latents, eta=0.0).prev_sample
                if after_track:
                    after_track(i, tr)
            if after_step:
                after_step(i)
            if log_every and (i + 1) % log_every == 0:
                el = time.perf_counter() - t0
                print(f"    step {i + 1:>2}/{len(timesteps)}  {el:7.1f} s  "
                      f"({el / (i + 1) / len(tracks):.2f} s per clip-step)", flush=True)
        harness.probes = []
    for tr in tracks:
        tr.latents = tr.latents.float().cpu()
    torch.cuda.synchronize()
    return {"seconds": time.perf_counter() - t0, "n_tracks": len(tracks)}


# ---------------------------------------------------------------------------
# Probes for the probing sweep
# ---------------------------------------------------------------------------

class StepCache(Probe):
    """Text-conditional activations of one clip at the current step, canonical (F, C, S), fp16 on CPU."""

    def __init__(self, coords, device: str = "cpu"):
        super().__init__(coords, None)
        self.device = device
        self.store: dict[str, torch.Tensor] = {}
        self.step = -1

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        if ctx.step != self.step:
            self.store.clear()
            self.step = ctx.step
        x = ctx.cond(output, coord.kind).to(torch.float16)
        self.store[coord.name] = x.to(self.device, non_blocking=False, copy=True)

    def clear(self):
        self.store.clear()


class PairScorer(Probe):
    """D-01 distances of the live clip against each reference clip that ran earlier in the same step.

    refs: list of (pair_index, StepCache). Rows hold scalars only; per-channel dynamic
    distances (D-10) are kept for `per_channel_steps`.
    """

    def __init__(self, coords, refs: list[tuple[int, StepCache]], per_channel_steps=()):
        super().__init__(coords, None)
        self.refs = refs
        self.per_channel_steps = set(per_channel_steps)
        self.rows: list[dict] = []
        self.per_channel: dict[tuple[int, str, int], torch.Tensor] = {}

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        xb = ctx.cond(output, coord.kind)
        pc = ctx.step in self.per_channel_steps
        for pair_idx, cache in self.refs:
            if cache.step != ctx.step or coord.name not in cache.store:
                raise RuntimeError(f"reference for pair {pair_idx} has no step-{ctx.step} activation at {coord.name}")
            xa = cache.store[coord.name].to(xb.device)
            d = paired_distances(xa, xb, per_channel=pc)
            if pc:
                self.per_channel[(pair_idx, coord.name, ctx.step)] = d.pop("d_dynamic_per_channel").half()
            self.rows.append({"pair": pair_idx, "coord": coord.name, "step": ctx.step, **d})


class Sketcher(Probe):
    """Count-sketch of the flattened x_tilde and x_bar of one clip, for the linear probes (D-21).

    A count sketch (random bucket h(i) and sign s(i) per element, summed per bucket) keeps
    inner products in expectation, so PCA + logistic regression on the sketch approximates
    the same pipeline on the full flattened activation, which is far too large to store
    (21M values per step at 64x64). Buckets and signs are a fixed function of the coordinate
    and the tensor kind, identical for every clip.
    """

    def __init__(self, coords, steps, dim: int = 512):
        super().__init__(coords, steps)
        self.dim = dim
        self.out: dict[tuple[str, int], torch.Tensor] = {}

    def _sketch(self, x: torch.Tensor, key: str) -> torch.Tensor:
        x = x.float().flatten()
        g = torch.Generator(x.device).manual_seed(zlib.crc32(key.encode()))
        idx = torch.randint(0, self.dim, (x.numel(),), generator=g, device=x.device)
        sign = torch.randint(0, 2, (x.numel(),), generator=g, device=x.device, dtype=torch.int8)
        x = torch.where(sign.bool(), x, -x)
        return torch.zeros(self.dim, device=x.device).index_add_(0, idx, x)

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        x_bar, x_til = split_static_dynamic(ctx.cond(output, coord.kind).float())
        self.out[(coord.name, ctx.step)] = torch.stack(
            [self._sketch(x_til, f"{coord.name}|tilde"), self._sketch(x_bar, f"{coord.name}|bar")]).half().cpu()
