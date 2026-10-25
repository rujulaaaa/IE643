"""Deliverables 3-4: the intervention framework and the two editing directions.

An edit is a lockstep pair from one noise tensor:

    source track   clip A, generated normally; its activations at the chosen coordinates
                   (and, for granularity 3, its temporal attention weights) are recorded at
                   each step t as it runs
    target track   the edit prompt; at step t it receives A's step-t values at the same
                   coordinates (resampling corruption from a real clip, D-12)

Directions (roadmap §5.4):
    "motion"   keep A's motion, change the content: target prompt = (new subject, A's verb),
               transplant A's motion-selective activations / temporal attention
    "content"  keep A's content, change the motion: target prompt = (A's subject, new verb),
               transplant A's content-selective activations
The same target prompt without any transplant is the prompt-only baseline, which every
edit has to beat (D-16).

Granularities (roadmap §5.3):
    1  whole tensor                         PatchSpec(alpha=1)
    2  channel subset (top-k% by S_c, D-10) PatchSpec(channels={coord: mask})
    3  temporal attention weights only      AttnSpec: the target's temporal attention
       (our training-free baseline in place     probabilities are replaced by the source's,
       of MotionClone, D-17 as amended)         P_t <- (1 - a) P_t + a P_s, then P_t @ V_t
    4  scaled injection                     PatchSpec(alpha < 1)

Attention-level operators from the editing literature (Stage 6b, D-37), added after the
component patches failed D-16:
    spatial K/V injection                   KVSpec: the target's spatial self-attention (attn1)
       (MasaCtrl, Cao et al. ICCV 2023;         queries the source's keys and values, frame by
        UniEdit's content branch)                frame. Appearance comes from the source, layout
                                                 and motion stay the target's own (its queries).
    temporal attention at all steps          AttnSpec over the whole trajectory (UniEdit's
       (UniEdit, Bai et al. 2024)               motion branch: SA-T Q/K from the motion source)
Any earlier track can be the source of either operator (AttnSpec.source / KVSpec.source), so
one target can take its appearance from clip A and its motion from clip B ("dual source").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .coords import Coordinate
from .lockstep import Track, run_lockstep
from .patching import Patcher, PatchSpec, TensorCache
from .pipeline import Loaded, decode


# ---------------------------------------------------------------------------
# Granularity 3: temporal attention-weight injection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AttnSpec:
    modules: tuple[str, ...]        # temporal attention module paths (…motion_modules.k….attn1/attn2)
    steps: tuple[int, ...]
    alpha: float = 1.0
    source: str = "source"          # id of the track whose probabilities are injected


@dataclass(frozen=True)
class KVSpec:
    """Spatial self-attention K/V injection (MasaCtrl): target queries attend to the source's keys/values."""
    modules: tuple[str, ...]        # spatial attn1 module paths
    steps: tuple[int, ...]
    source: str = "source"
    frames: str = "framewise"       # "framewise": frame f reads source frame f; "first": every frame reads frame 0
    keep_own: bool = False          # also keep the target's own keys/values (extended attention)

    def __post_init__(self):
        if self.frames not in ("framewise", "first"):
            raise ValueError(self.frames)


class AttnController:
    """Shared state for the injecting processors: which track is running and what was recorded.

    Recordings are keyed by (track id, module), so several tracks can act as sources."""

    def __init__(self, spec: AttnSpec | KVSpec | None, store_device: str | None = None):
        self.spec = spec
        self.store_device = store_device    # None: keep recorded probabilities where they are
        self.mode: str | None = None    # None | "record" | "inject"
        self.step = -1
        self.track = "source"           # id of the running track
        self.frames = 16                # frames per CFG branch (KVSpec frames="first")
        self.store: dict[tuple[str, str], torch.Tensor] = {}
        self.applied = 0

    def active(self, name: str) -> bool:
        return self.spec is not None and self.mode is not None and name in self.spec.modules \
            and self.step in self.spec.steps


class InjectingProcessor:
    """`AttnProcessor2_0` with the probabilities made explicit so they can be recorded or replaced.

    Both modes return the stock SDPA output; injection adds a·W_out((P_src − P) V) to it.
    Probabilities are stored in fp32 (≈1 GB per step for all 42 modules at 16 frames).

    When the controller is inactive for this module it defers to `AttnProcessor2_0`, so an
    edit with no AttnSpec runs exactly the stock computation.
    """

    def __init__(self, name: str, ctl: AttnController, default):
        self.name, self.ctl, self.default = name, ctl, default

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        if not self.ctl.active(self.name):
            return self.default(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)
        if attn.spatial_norm is not None or attn.group_norm is not None or attention_mask is not None \
                or hidden_states.ndim != 3:
            raise NotImplementedError("temporal attention in AnimateDiff has none of these")
        residual = hidden_states
        b = hidden_states.shape[0]
        ehs = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q, k, v = attn.to_q(hidden_states), attn.to_k(ehs), attn.to_v(ehs)
        d = k.shape[-1] // attn.heads
        q, k, v = (x.view(b, -1, attn.heads, d).transpose(1, 2) for x in (q, k, v))
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        probs = (torch.matmul(q, k.transpose(-1, -2)).float() * d ** -0.5).softmax(-1)
        stock = self.default(attn, residual, encoder_hidden_states, attention_mask, temb, **kw)
        if self.ctl.mode == "record":
            # recording is read-only: keep a copy of the probabilities, return the stock output
            p = probs if self.ctl.store_device is None else probs.to(self.ctl.store_device)
            self.ctl.store[(self.ctl.track, self.name)] = p
            return stock
        # lerp(P, P_src, a) @ V = P @ V + a (P_src - P) @ V: add only the change to the stock
        # output, so injecting a clip's own probabilities adds exactly zero (D-11)
        p_src = self.ctl.store[(self.ctl.spec.source, self.name)]
        d_probs = self.ctl.spec.alpha * (p_src.to(probs.device) - probs)
        self.ctl.applied += 1
        delta = torch.matmul(d_probs.to(v.dtype), v).transpose(1, 2).reshape(b, -1, attn.heads * d)
        delta = F.linear(delta, attn.to_out[0].weight)        # out-projection without its bias
        return stock + delta / attn.rescale_output_factor


class KVInjectingProcessor:
    """`AttnProcessor2_0` for spatial self-attention whose keys/values can come from another track.

    record: store this track's K and V, return the stock output.
    inject: Q from this track, K/V from the source track (MasaCtrl's mutual self-attention), per
            CFG branch and frame; `frames="first"` gives every frame the source's frame 0, so no
            frame-to-frame (motion) information travels with the appearance; `keep_own` appends the
            target's own K/V (extended attention). Injecting a track's own K/V framewise reproduces
            the stock output exactly.
    """

    def __init__(self, name: str, ctl: AttnController, default):
        self.name, self.ctl, self.default = name, ctl, default

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        if not self.ctl.active(self.name):
            return self.default(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)
        if encoder_hidden_states is not None or attn.spatial_norm is not None or attn.group_norm is not None \
                or attention_mask is not None or hidden_states.ndim != 3:
            raise NotImplementedError("spatial self-attention in SD1.5 has none of these")
        if self.ctl.mode == "record":
            k, v = attn.to_k(hidden_states), attn.to_v(hidden_states)
            if self.ctl.store_device is not None:
                k, v = k.to(self.ctl.store_device), v.to(self.ctl.store_device)
            self.ctl.store[(self.ctl.track, self.name)] = (k, v)
            return self.default(attn, hidden_states, encoder_hidden_states, attention_mask, temb, **kw)
        spec = self.ctl.spec
        b = hidden_states.shape[0]
        q = attn.to_q(hidden_states)
        k, v = (x.to(q.device) for x in self.ctl.store[(spec.source, self.name)])
        if spec.frames == "first":
            f = self.ctl.frames
            k, v = (x.view(b // f, f, *x.shape[1:])[:, :1].expand(-1, f, -1, -1).reshape(x.shape) for x in (k, v))
        if spec.keep_own:
            k = torch.cat([attn.to_k(hidden_states), k], dim=1)
            v = torch.cat([attn.to_v(hidden_states), v], dim=1)
        d = k.shape[-1] // attn.heads
        q, k, v = (x.view(b, -1, attn.heads, d).transpose(1, 2) for x in (q, k, v))
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, attn.heads * d).to(q.dtype)
        out = attn.to_out[1](attn.to_out[0](out))
        if attn.residual_connection:
            out = out + hidden_states
        self.ctl.applied += 1
        return out / attn.rescale_output_factor


class attention_injection:
    """Context manager installing injecting processors on the named modules, restoring on exit.

    `processor` is InjectingProcessor (temporal attention weights) or KVInjectingProcessor
    (spatial self-attention keys/values)."""

    def __init__(self, unet, ctl: AttnController, processor=None):
        self.unet, self.ctl, self.saved = unet, ctl, {}
        self.processor = processor or InjectingProcessor

    def __enter__(self):
        from diffusers.models.attention_processor import AttnProcessor2_0

        if self.ctl.spec is None:
            return self.ctl
        mods = dict(self.unet.named_modules())
        for name in self.ctl.spec.modules:
            m = mods[name]
            self.saved[name] = m.processor
            m.set_processor(self.processor(name, self.ctl, AttnProcessor2_0()))
        return self.ctl

    def __exit__(self, *exc):
        mods = dict(self.unet.named_modules())
        for name, p in self.saved.items():
            mods[name].set_processor(p)
        return False


# ---------------------------------------------------------------------------
# One edit
# ---------------------------------------------------------------------------

@dataclass
class EditSpec:
    patch: PatchSpec | None = None      # granularities 1, 2, 4
    attn: AttnSpec | None = None        # granularity 3
    label: str = ""
    kv: KVSpec | None = None            # spatial K/V injection (D-37)

    @property
    def is_baseline(self) -> bool:
        return self.patch is None and self.attn is None and self.kv is None


@dataclass
class EditResult:
    source_latents: torch.Tensor
    target_latents: torch.Tensor
    source_frames: list | None = None
    target_frames: list | None = None
    seconds: float = 0.0
    n_patched: int = 0
    n_attn_injected: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Arm:
    """One target track of a multi-arm edit: a prompt plus what it receives from the source."""
    id: str
    prompt: str
    spec: EditSpec
    step_offset: int = 0        # Patcher reads the source at step + offset (mis-indexed control)
    lora: str | None = None     # name of a LoRA adapter already attached to the UNet, active
    lora_scale: float = 1.0     # for this track only (the trained baseline, D-19)


@dataclass
class ArmsResult:
    source_latents: torch.Tensor
    latents: dict                       # arm id -> final latents
    frames: dict = field(default_factory=dict)   # "source" / arm id -> PIL frames
    seconds: float = 0.0
    n_patched: dict = field(default_factory=dict)
    n_attn_injected: dict = field(default_factory=dict)
    n_kv_injected: dict = field(default_factory=dict)


def _union_modules(specs, source: str = "source") -> AttnSpec | None:
    specs = [s for s in specs if s is not None]
    if not specs:
        return None
    mods = tuple(sorted({m for s in specs for m in s.modules}))
    steps = tuple(sorted({t for s in specs for t in s.steps}))
    return AttnSpec(mods, steps, source=source)


def _union_attn(arms: list[Arm], source: str = "source", kind: str = "attn") -> AttnSpec | None:
    """What `source` must record: the union of the modules/steps of every arm that reads it."""
    specs = [getattr(a.spec, kind) for a in arms]
    return _union_modules([s for s in specs if s is not None and s.source == source], source)


def _recording_specs(arms: list[Arm], kind: str) -> dict:
    """{track id: spec to record} for every track some arm reads with this operator. A source must
    be "source" or an unedited arm that runs before every arm reading it."""
    order = {a.id: i for i, a in enumerate(arms)}
    out = {}
    for a in arms:
        s = getattr(a.spec, kind)
        if s is None or s.source in out:
            continue
        if s.source != "source":
            src = next((x for x in arms if x.id == s.source), None)
            if src is None or order[src.id] >= order[a.id]:
                raise ValueError(f"arm {a.id} reads {s.source}, which must be an earlier arm")
            if not src.spec.is_baseline:
                raise ValueError(f"{s.source} is a source, so it cannot itself be edited")
        out[s.source] = _union_attn(arms, s.source, kind)
    return out


@torch.no_grad()
def run_arms(loaded: Loaded, coords: list[Coordinate], source_prompt: str, arms: list[Arm],
             noise: torch.Tensor | None, decode_frames: bool = True, cache_device: str = "cuda",
             log_every: int = 0, source_trajectory: list | None = None,
             extra_probes: dict | None = None) -> ArmsResult:
    """One source clip and N target arms in one lockstep group, sharing the source's cache.

    At step t the source runs first: its raw activations at the union of every arm's patch
    coordinates are cached (and its temporal attention probabilities recorded, for arms with
    an AttnSpec). Each arm then runs and takes what its own spec asks for. A sweep over N
    settings for one source therefore costs N + 1 generations, not 2N. The cache holds one
    step (two when an arm uses a negative step offset).

    `extra_probes` ({"source" or arm id: [probes]}) adds recorders to individual tracks, e.g.
    cross-attention maps for subject masks.

    `source_trajectory` (from `inversion.invert`, 26 latents) drives the source along a real
    video's inverted trajectory instead of letting it denoise. The arms start from `noise`; with
    noise=None they start from the inverted noise z_{t_0} instead. That second option looks
    natural but fails: inverted latents are not Gaussian, and sampling from them with CFG 7.5
    gives blotchy, grid-patterned clips even without any patch (checked 2026-10-21).
    """
    if source_trajectory is not None and noise is None:
        noise = source_trajectory[0].half()
    patch_specs = [a.spec.patch for a in arms if a.spec.patch is not None]
    cache = None
    if patch_specs:
        cache = TensorCache({c for p in patch_specs for c in p.coords},
                            {t + a.step_offset for a in arms if a.spec.patch for t in a.spec.patch.steps},
                            mode="raw", device=cache_device)
    keep = 1 + max([-a.step_offset for a in arms] + [0])
    extra_probes = extra_probes or {}
    tracks = [Track("source", source_prompt, ([cache] if cache else []) + list(extra_probes.get("source", [])),
                    start=None if source_trajectory is None else source_trajectory[0].half())]
    patchers = {}
    for a in arms:
        probes = []
        if a.spec.patch is not None:
            patchers[a.id] = Patcher(cache, a.spec.patch, step_offset=a.step_offset)
            probes.append(patchers[a.id])
        tracks.append(Track(a.id, a.prompt, probes + list(extra_probes.get(a.id, []))))
    # per operator: what each source track records, and what each arm injects
    record = {"attn": _recording_specs(arms, "attn"), "kv": _recording_specs(arms, "kv")}
    own = {k: {a.id: getattr(a.spec, k) for a in arms} for k in ("attn", "kv")}
    ctls = {k: AttnController(_union_modules(own[k].values()), store_device=cache_device)  # processors on the union
            for k in own}                                                                    # of every arm's modules;
    ctls["kv"].frames = loaded.cfg.frames
    applied = {a.id: 0 for a in arms}
    applied_kv = {a.id: 0 for a in arms}

    use_lora = any(a.lora for a in arms)
    lora_of = {a.id: (a.lora, a.lora_scale) for a in arms}

    def before_track(step, tr):
        if use_lora:
            name, scale = lora_of.get(tr.id, (None, 0.0))
            if name:
                loaded.unet.enable_adapters()
                loaded.unet.set_adapters([name], weights=[scale])
            else:
                loaded.unet.disable_adapters()
        for k, ctl in ctls.items():            # then each track activates only its own
            ctl.step, ctl.track = step, tr.id
            if tr.id in record[k]:
                ctl.spec, ctl.mode = record[k][tr.id], "record"
            else:
                ctl.spec = own[k].get(tr.id)
                ctl.mode = "inject" if ctl.spec else None

    def after_track(step, tr):
        if tr.id == "source" and source_trajectory is not None:
            tr.latents = source_trajectory[step + 1].to(tr.latents.device, tr.latents.dtype)
        if tr.id != "source":
            applied[tr.id] += ctls["attn"].applied
            applied_kv[tr.id] += ctls["kv"].applied
        for ctl in ctls.values():
            ctl.applied = 0

    def after_step(step):
        if cache is not None:
            for k in [k for k in cache.store if k[1] <= step - keep + 1]:
                del cache.store[k]
        for ctl in ctls.values():
            ctl.store.clear()

    try:
        with attention_injection(loaded.unet, ctls["attn"]), \
                attention_injection(loaded.unet, ctls["kv"], KVInjectingProcessor):
            t = run_lockstep(loaded, coords, tracks, noise, before_track=before_track,
                             after_track=after_track, after_step=after_step, log_every=log_every)
    finally:
        if use_lora:
            loaded.unet.disable_adapters()
    res = ArmsResult(tracks[0].latents, {tr.id: tr.latents for tr in tracks[1:]}, seconds=t["seconds"],
                     n_patched={k: len(p.applied) for k, p in patchers.items()}, n_attn_injected=applied,
                     n_kv_injected=applied_kv)
    if decode_frames:
        for tr in tracks:
            res.frames[tr.id] = decode(loaded, tr.latents)
    return res


@torch.no_grad()
def run_edit(loaded: Loaded, coords: list[Coordinate], source_prompt: str, target_prompt: str,
             noise: torch.Tensor, spec: EditSpec, decode_frames: bool = True,
             cache_device: str = "cuda") -> EditResult:
    """Source and target denoised in lockstep; the target receives the source's values per `spec`.

    With an empty spec this is the prompt-only baseline (the source still runs, which keeps the
    cost and the numerics of every arm identical).
    """
    r = run_arms(loaded, coords, source_prompt, [Arm("target", target_prompt, spec)], noise,
                 decode_frames=decode_frames, cache_device=cache_device)
    return EditResult(r.source_latents, r.latents["target"], r.frames.get("source"), r.frames.get("target"),
                      seconds=r.seconds, n_patched=r.n_patched.get("target", 0),
                      n_attn_injected=r.n_attn_injected.get("target", 0))


def temporal_attention_modules(coords: list[Coordinate], blocks=None) -> tuple[str, ...]:
    """Temporal attn1/attn2 module paths, optionally restricted to some blocks."""
    return tuple(c.name for c in coords if c.kind.startswith("temporal_attn") and (blocks is None or c.block in blocks))


def channel_mask(scores: torch.Tensor, top_frac: float, most_negative: bool = True) -> torch.Tensor:
    """Granularity 2 (D-10): a (C,) 0/1 mask of the top `top_frac` channels by S_c.
    Motion-selective channels have the most negative S_c."""
    k = max(1, int(round(top_frac * scores.numel())))
    order = torch.argsort(scores, descending=not most_negative)
    mask = torch.zeros_like(scores, dtype=torch.float32)
    mask[order[:k]] = 1.0
    return mask
