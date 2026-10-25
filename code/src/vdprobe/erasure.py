"""Concept erasure as a necessity test (Experiment 2 Phase 5c).

LEACE (Belrose et al., NeurIPS 2023), the closed-form least-squares eraser that makes a concept Z
linearly unpredictable from X while changing X as little as possible:

    r(x) = x - W^+ P W (x - mu),    W = (Sigma_XX^{1/2})^+,   P = orthogonal projection onto
                                                               the column space of W Sigma_XZ

Control (amnesic probing, Elazar et al. TACL 2021): a random eraser of the same rank, drawn in
the same whitened space. "Erasing Z matters" only if LEACE changes the output more than it.
Here X = the frame-varying component x_tilde of one coordinate, one sample per (frame, position),
channel vectors; Z = the clip's verb (one-hot).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .coords import frame_mean_raw
from .harness import Probe


@dataclass
class Eraser:
    mu: torch.Tensor       # (C,)
    A: torch.Tensor        # (C, C) = W^+ P W, applied as x - (x - mu) A^T
    rank: int

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., C) channel-last."""
        A, mu = self.A.to(x.device, torch.float32), self.mu.to(x.device, torch.float32)
        xf = x.float()
        return (xf - (xf - mu) @ A.T).to(x.dtype)


def _sqrt_and_pinv(S: np.ndarray, eps: float = 1e-6):
    w, V = np.linalg.eigh(S)
    keep = w > eps * w.max()
    sq = (V[:, keep] * np.sqrt(w[keep])) @ V[:, keep].T
    isq = (V[:, keep] / np.sqrt(w[keep])) @ V[:, keep].T
    return sq, isq


def fit_leace(X: np.ndarray, Z: np.ndarray) -> Eraser:
    """X: (n, C) samples, Z: (n, K) concept (e.g. one-hot)."""
    X, Z = X.astype(np.float64), Z.astype(np.float64)
    mu = X.mean(0)
    Xc, Zc = X - mu, Z - Z.mean(0)
    Sxx = Xc.T @ Xc / len(X)
    Sxz = Xc.T @ Zc / len(X)
    sq, W = _sqrt_and_pinv(Sxx)            # W = Sigma^{-1/2} (pseudo), sq = Sigma^{1/2}
    U, s, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, s > 1e-8 * max(s.max(), 1e-30)]
    P = U @ U.T
    A = sq @ P @ W
    return Eraser(torch.from_numpy(mu).float(), torch.from_numpy(A).float(), U.shape[1])


def random_eraser(X: np.ndarray, rank: int, seed: int = 0) -> Eraser:
    """Same-rank control: projection onto a random `rank`-dim subspace of the whitened space."""
    X = X.astype(np.float64)
    mu = X.mean(0)
    Xc = X - mu
    sq, W = _sqrt_and_pinv(Xc.T @ Xc / len(X))
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.normal(size=(X.shape[1], rank)))
    A = sq @ (U @ U.T) @ W
    return Eraser(torch.from_numpy(mu).float(), torch.from_numpy(A).float(), rank)


class EraseProbe(Probe):
    """Applies an Eraser to the frame-varying component of a coordinate's raw output, at chosen
    steps, in every CFG branch: out = x_bar + r(x_tilde), r(x_tilde) re-centred over frames."""

    def __init__(self, coord, eraser: Eraser, steps):
        super().__init__([coord], steps)
        self.coord, self.eraser = coord, eraser
        self.applied = 0

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        bar = frame_mean_raw(output, coord.kind, ctx.n_branches, ctx.frames)
        til = output - bar
        cd = coord.channel_dim % til.ndim
        er = self.eraser(til.movedim(cd, -1)).movedim(-1, cd)
        # the eraser is affine (the -mu term): re-centre over frames so x_bar is left exactly as it was
        er = er - frame_mean_raw(er, coord.kind, ctx.n_branches, ctx.frames)
        self.applied += 1
        return bar + er


class TildeSampler(Probe):
    """Collects x_tilde channel vectors of the text-conditional branch: `n_pos` random (frame,
    position) samples per call, for fitting an eraser."""

    def __init__(self, coord, steps, n_pos: int = 256, seed: int = 0):
        super().__init__([coord], steps)
        self.n_pos, self.g = n_pos, torch.Generator().manual_seed(seed)
        self.samples: list[torch.Tensor] = []

    @torch.no_grad()
    def __call__(self, ctx, coord, module, output, args, kwargs):
        from .summaries import split_static_dynamic

        _, til = split_static_dynamic(ctx.cond(output, coord.kind).float())     # (F, C, S)
        v = til.permute(0, 2, 1).reshape(-1, til.shape[1])                       # (F*S, C)
        idx = torch.randperm(v.shape[0], generator=self.g)[: self.n_pos]
        self.samples.append(v[idx.to(v.device)].half().cpu())
