"""The activation-extraction harness.

`Harness` installs one forward hook per coordinate plus a pre-hook on the UNet that
counts denoising steps (one UNet call per step; the CFG branches are batched). Each
hook hands the module output to a list of probes, in order. A probe either records
(returns None) or intervenes (returns a replacement tensor, which later probes and
the rest of the network then see).

Probes defined here:
  ShapeAccountant   bytes a full fp16 dump would take (the baseline E3 measured)
  SummaryRecorder   the compact per-clip store (D-09 / D-24), 44.6 MiB per clip
Patching and paired-distance probes live in vdprobe.patching.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import torch

from . import attention as A
from .coords import Coordinate, branch_rows, canonical, spatial_hw
from .summaries import channel_stats, resize_map


@dataclass
class RunContext:
    step: int = -1
    n_branches: int = 1
    frames: int = 1
    latent_hw: tuple[int, int] = (0, 0)
    timesteps: list[float] = field(default_factory=list)
    token_index: dict[str, list[int]] = field(default_factory=dict)

    @property
    def cond_branch(self) -> int:
        # Pipelines concatenate [negative, positive] under CFG.
        if self.n_branches not in (1, 2):
            raise ValueError(f"expected 1 or 2 CFG branches, got {self.n_branches}")
        return self.n_branches - 1

    def cond(self, t: torch.Tensor, kind: str) -> torch.Tensor:
        """Text-conditional branch of a raw output, as (F, C, S)."""
        return canonical(t, kind, self.n_branches, self.frames)[self.cond_branch]

    def cond_rows(self, t: torch.Tensor) -> torch.Tensor:
        return t[branch_rows(t.shape[0], self.n_branches, self.cond_branch)]


class Probe:
    """Base class. `coords=None` means every coordinate; `steps=None` every step."""

    def __init__(self, coords=None, steps=None):
        self.coords = None if coords is None else {c if isinstance(c, str) else c.name for c in coords}
        self.steps = None if steps is None else set(steps)

    def wants(self, coord: Coordinate, step: int) -> bool:
        return (self.coords is None or coord.name in self.coords) and (self.steps is None or step in self.steps)

    def __call__(self, ctx: RunContext, coord: Coordinate, module, output: torch.Tensor,
                 args: tuple, kwargs: dict) -> torch.Tensor | None:
        raise NotImplementedError


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            t = _first_tensor(e)
            if t is not None:
                return t
    return getattr(x, "sample", None)


class Harness:
    """Context manager: `with Harness(unet, coords, [probes]) as ctx: pipe(...)`."""

    def __init__(self, unet: torch.nn.Module, coords: list[Coordinate], probes: list[Probe],
                 token_index: dict[str, list[int]] | None = None):
        self.unet = unet
        self.coords = coords
        self.probes = probes
        self.ctx = RunContext(token_index=dict(token_index or {}))
        self._handles = []

    def _step_hook(self, module, args, kwargs):
        sample = args[0] if args else kwargs["sample"]
        t = args[1] if len(args) > 1 else kwargs.get("timestep")
        ctx = self.ctx
        ctx.step += 1
        ctx.n_branches, ctx.frames = sample.shape[0], sample.shape[2]
        ctx.latent_hw = tuple(sample.shape[-2:])
        ctx.timesteps.append(float(t.flatten()[0]) if torch.is_tensor(t) else float(t))

    def _make_hook(self, coord: Coordinate):
        def hook(module, args, kwargs, output):
            tensor = _first_tensor(output)
            if tensor is None:
                return None
            replaced = False
            for p in self.probes:
                if p.wants(coord, self.ctx.step):
                    r = p(self.ctx, coord, module, tensor, args, kwargs)
                    if r is not None:
                        tensor, replaced = r, True
            if not replaced:
                return None
            return (tensor, *output[1:]) if isinstance(output, tuple) else tensor
        return hook

    def __enter__(self) -> RunContext:
        modules = dict(self.unet.named_modules())
        self._handles.append(self.unet.register_forward_pre_hook(self._step_hook, with_kwargs=True))
        for c in self.coords:
            if any(p.coords is None or c.name in p.coords for p in self.probes):
                self._handles.append(modules[c.name].register_forward_hook(self._make_hook(c), with_kwargs=True))
        return self.ctx

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False


# ---------------------------------------------------------------------------
# Recorders
# ---------------------------------------------------------------------------

class ShapeAccountant(Probe):
    """What a full fp16 dump of every hooked tensor would cost (E3's baseline)."""

    BYTES = 2  # fp16

    def __init__(self, coords=None, steps=None):
        super().__init__(coords, steps)
        self.records = []

    def __call__(self, ctx, coord, module, output, args, kwargs):
        self.records.append({"name": coord.name, "kind": coord.kind, "step": ctx.step,
                             "shape": tuple(output.shape), "numel": output.numel(),
                             "branches": ctx.n_branches})

    def bytes_by_kind(self, cond_only: bool = False) -> dict[str, int]:
        out = defaultdict(int)
        for r in self.records:
            out[r["kind"]] += r["numel"] * self.BYTES // (r["branches"] if cond_only else 1)
        return dict(out)


class SummaryRecorder(Probe):
    """The compact store (D-24). Per (coordinate, step), text-conditional branch:
    channel mean (C), channel variance (C), per-frame residual norm (F); and for
    attention coordinates one head-averaged (F, grid, grid) map."""

    def __init__(self, coords=None, steps=None, grid: int = 16, cross_phrases=("subject",),
                 attention_maps: bool = True):
        super().__init__(coords, steps)
        self.grid = grid
        self.cross_phrases = tuple(cross_phrases)
        self.attention_maps = attention_maps
        self.data: dict[str, dict[str, list[torch.Tensor]]] = defaultdict(lambda: defaultdict(list))

    def _cross_tokens(self, ctx: RunContext, phrase: str) -> list[int]:
        idx = ctx.token_index.get(phrase) or ctx.token_index.get("_all")
        if not idx:
            raise KeyError(f"no token positions for phrase '{phrase}'; pass token_index to the Harness")
        return idx

    def _movie(self, ctx, coord, module, args, kwargs) -> dict[str, torch.Tensor]:
        hs = args[0] if args else kwargs["hidden_states"]
        hs = ctx.cond_rows(hs)
        if coord.kind == "spatial_attn1":
            maps = {"attn": A.spatial_received(module, hs)}
        elif coord.kind == "spatial_attn2":
            ehs = ctx.cond_rows(kwargs["encoder_hidden_states"])
            maps = {}
            for i, phrase in enumerate(self.cross_phrases):
                key = "attn" if i == 0 else f"attn_{phrase}"
                maps[key] = A.cross_token_mass(module, hs, ehs, self._cross_tokens(ctx, phrase))
        else:  # temporal_attn1 / temporal_attn2
            maps = {"attn": A.temporal_cross_frame_mass(module, hs)}
        h, w = spatial_hw(coord.level, ctx.latent_hw)
        return {k: resize_map(m.reshape(m.shape[0], h, w), self.grid) for k, m in maps.items()}

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        rec = self.data[coord.name]
        for k, v in channel_stats(ctx.cond(output, coord.kind)).items():
            rec[k].append(v.cpu())
        if self.attention_maps and coord.is_attention:
            for k, v in self._movie(ctx, coord, module, args, kwargs).items():
                rec[k].append(v.cpu())

    def tensors(self) -> dict[str, torch.Tensor]:
        """{"<module path>::<field>": tensor with a leading step axis}."""
        return {f"{name}::{k}": torch.stack(vals) for name, rec in self.data.items() for k, vals in rec.items()}
