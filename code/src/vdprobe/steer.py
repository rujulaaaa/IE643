"""Steering probes for a U-Net site: add a direction, or ablate SAE features (novelty phase, Part C).

Both act on every position, frame and CFG branch of the module's output, at the steps given.
"""

from __future__ import annotations

import torch

from .harness import Probe
from .sae import Normalizer, TopKSAE, splice


def _channels_last(t: torch.Tensor, channel_dim: int):
    """(tokens, C) view of a module tensor and the function that undoes it."""
    moved = t.movedim(channel_dim, -1)
    return moved.reshape(-1, moved.shape[-1]), lambda flat: flat.reshape(moved.shape).movedim(-1, channel_dim)


class AddDirection(Probe):
    """Add `strength * direction` (a (C,) vector in activation units) to the module output."""

    def __init__(self, coord: str, steps, direction: torch.Tensor, strength: float):
        super().__init__([coord], steps)
        self.direction, self.strength = direction.detach().float(), float(strength)
        self.applied = 0

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        if self.strength == 0.0:
            return None                                   # exact no-op: the output object is not replaced
        if output.shape[coord.channel_dim] != self.direction.numel():
            raise ValueError(f"{coord.name}: direction has {self.direction.numel()} channels, "
                             f"the output {output.shape[coord.channel_dim]}")
        shape = [1] * output.ndim
        shape[coord.channel_dim] = -1
        self.applied += 1
        return (output.float() + self.strength * self.direction.to(output.device).reshape(shape)).to(output.dtype)


class FeatureAblate(Probe):
    """Zero the listed SAE features in the module's **update** (output minus input) and put the
    edited update back: input + splice(update). The SAE's reconstruction error is kept, so an
    empty feature list changes nothing (up to float rounding)."""

    def __init__(self, coord: str, steps, sae: TopKSAE, normalizer: Normalizer, features: list[int]):
        super().__init__([coord], steps)
        self.sae, self.normalizer, self.features = sae, normalizer, list(features)
        self.applied = 0
        self.removed_norm = 0.0        # running sum of the L2 norm of what was removed (0 if the features never fire)

    def _edit(self, z: torch.Tensor) -> torch.Tensor:
        if not self.features:
            return z
        z = z.clone()
        z[:, self.features] = 0
        return z

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        inp = args[0] if args else kwargs.get("hidden_states")
        if inp is None or inp.shape != output.shape:
            raise ValueError(f"{coord.name}: FeatureAblate needs a module whose output has its input's shape")
        upd, back = _channels_last(output.float() - inp.float(), coord.channel_dim)
        sae = self.sae.to(upd.device)
        new = splice(sae, self.normalizer, upd, self._edit)
        self.applied += 1
        self.removed_norm += float((upd - new).norm())
        return (inp.float() + back(new)).to(output.dtype)
