"""Attribution patching (Part D1): one backward pass estimates the effect of patching every site at once.

For a clean clip A and a corrupted clip B at one denoising step, with noise predictions eps_A, eps_B and
d = eps_A - eps_B, take L(eps) = <eps, stop_grad(d)>. Replacing B's prediction by A's raises L by |d|^2.
Patching site n (B's activation a_n replaced by A's) raises L, to first order, by

    atp(n) = sum((a_n[A] - a_n[B]) * dL/da_n |_B)                 (Nanda 2023; AtP*, Kramar et al., arXiv:2403.00745)

and `atp(n) / |d|^2` is the estimated fraction of the A-B gap that site closes (1 = the whole prediction moves to A).
Nothing is differentiated through the sampler; each step is screened on its own.

Only the text-conditional branch is run with gradients (batch 1), which halves the memory: the estimate is that of
patching the conditional branch only.
"""

from __future__ import annotations

from typing import Callable

import torch


def _first_tensor(x):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (tuple, list)):
        for e in x:
            t = _first_tensor(e)
            if t is not None:
                return t
    return getattr(x, "sample", None)


class GradCache:
    """Context manager that keeps the output of each named module of `model`; with `grad=True` the outputs stay in
    the autograd graph (`retain_grad`), so after `backward` their gradients are in `.grads`."""

    def __init__(self, model: torch.nn.Module, names, grad: bool = False):
        self.model, self.names, self.grad = model, list(names), grad
        self.acts: dict[str, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        modules = dict(self.model.named_modules())
        missing = [n for n in self.names if n not in modules]
        if missing:
            raise KeyError(f"no such modules: {missing[:5]}")
        for n in self.names:
            def hook(module, args, output, n=n):
                t = _first_tensor(output)
                if t is None:
                    return
                if self.grad and t.requires_grad:
                    t.retain_grad()
                    self.acts[n] = t
                else:
                    self.acts[n] = t.detach()
            self._handles.append(modules[n].register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        return False

    @property
    def grads(self) -> dict[str, torch.Tensor]:
        return {n: t.grad for n, t in self.acts.items() if t.grad is not None}


def atp_scores(acts_a: dict, acts_b: dict, grads_b: dict) -> dict[str, float]:
    """sum((a_A - a_B) * dL/da_B) per site present in all three dicts; shapes must agree."""
    out = {}
    for n, g in grads_b.items():
        if n not in acts_a or n not in acts_b:
            continue
        a, b = acts_a[n], acts_b[n]
        if a.shape != b.shape or b.shape != g.shape:
            raise ValueError(f"{n}: shapes {tuple(a.shape)}, {tuple(b.shape)}, grad {tuple(g.shape)}")
        out[n] = float(((a.float() - b.float()) * g.float()).sum())
    return out


def atp_step(model: torch.nn.Module, names, call_a: Callable[[], torch.Tensor], call_b: Callable[[], torch.Tensor]) -> dict:
    """One step of the screen. `call_a()` / `call_b()` run the model on A's / B's inputs and return the prediction
    that L is taken on. Returns {"scores": {site: atp}, "gap": |d|^2, "frac": {site: atp / |d|^2}}."""
    with torch.no_grad(), GradCache(model, names) as ca:
        eps_a = call_a().detach()
    with torch.enable_grad(), GradCache(model, names, grad=True) as cb:
        eps_b = call_b()
        d = (eps_a - eps_b).detach().float()
        loss = (eps_b.float() * d).sum()
        loss.backward()
        scores = atp_scores(ca.acts, {n: t.detach() for n, t in cb.acts.items()}, cb.grads)
    model.zero_grad(set_to_none=True)
    gap = float((d * d).sum())
    return {"scores": scores, "gap": gap, "frac": {n: s / gap if gap > 0 else float("nan") for n, s in scores.items()}}


def spearman(x, y) -> float:
    """Spearman rank correlation (average ranks for ties); NaN for fewer than 3 pairs."""
    import numpy as np
    from scipy.stats import spearmanr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() >= 3 else float("nan")
