"""Experiment 2 analysis primitives (docs/experiment2.md; reports/Content motion probing experiment design.md).

  sign_flip_test      exact / Monte-Carlo sign-flip test over subjects, per cell, with a max-|t|
                      family-wise p over the whole map (Nichols & Holmes 2002; the subject is the unit)
  subject_means       per-subject means of the D-28 selectivity units over a set of cells
  anova_fractions     balanced-design variance fractions of a representation by factor
                      (Kobak et al. 2016 marginalisation; exact for a full factorial)
  flow_rdm            motion RDM from camera-compensated, subject-masked flow
  content_rdm         content RDM from CLIP frame embeddings
  cosine_rdm          layer RDM from sketches
  decoding_gain       within-prompt flow decoding: cross-validated R^2 of a layer minus that of the
                      noisy latent at the same step (the latent-only control)
  channel_selectivity per-channel verb eta^2 of the dynamic energy (MOFT-style channel selection)
"""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd

from . import metrics as M


# ---------------------------------------------------------------------------
# Inference over subjects
# ---------------------------------------------------------------------------

def sign_flip_test(X: np.ndarray, n_flip: int = 10000, rng_seed: int = 643, alternative: str = "two-sided") -> dict:
    """X: (n_subjects, cells) per-subject effects (NaN-free). One-sample t per cell; null by flipping
    each subject's whole row (keeps the dependence between cells). Exact over all 2^n patterns when
    2^n <= n_flip. Returns t, per-cell p and the max-statistic FWER p."""
    n = X.shape[0]
    if 2 ** n <= n_flip:
        signs = np.array(list(product((1, -1), repeat=n)), float)
    else:
        rng = np.random.default_rng(rng_seed)
        signs = np.vstack([np.ones(n), rng.choice((1.0, -1.0), size=(n_flip - 1, n))])

    def tstat(Y):           # Y: (k, n, cells)
        m = Y.mean(1)
        sd = Y.std(1, ddof=1)
        return m / np.maximum(sd / np.sqrt(n), 1e-12)

    t_obs = tstat(X[None])[0]
    null = np.concatenate([tstat(signs[i:i + 512, :, None] * X[None]) for i in range(0, len(signs), 512)])
    if alternative == "less":
        stat_obs, stat_null = -t_obs, -null
    elif alternative == "greater":
        stat_obs, stat_null = t_obs, null
    else:
        stat_obs, stat_null = np.abs(t_obs), np.abs(null)
    p = (stat_null >= stat_obs[None] - 1e-12).mean(0)
    p_fwer = (stat_null.max(1)[:, None] >= stat_obs[None] - 1e-12).mean(0)
    return {"t": t_obs, "mean": X.mean(0), "p": p, "p_fwer": p_fwer, "n_patterns": len(signs), "n": n}


def subject_means(units: pd.DataFrame, chan: str, coords: list[str], steps) -> tuple[np.ndarray, list[str]]:
    """units: analysis.selectivity_units output (+ a 'subject' column). Returns (n_subjects, coords*steps)
    per-subject mean of S_<chan> over its units, and the subject order."""
    u = units[units.coord.isin(coords) & units.step.isin(list(steps))]
    piv = u.pivot_table(index="subject", columns=["coord", "step"], values=f"S_{chan}", aggfunc="mean")
    piv = piv.reindex(columns=pd.MultiIndex.from_product([coords, list(steps)]))
    return piv.to_numpy(), list(piv.index)


# ---------------------------------------------------------------------------
# Representation-level analyses
# ---------------------------------------------------------------------------

def anova_fractions(Z: np.ndarray, design: pd.DataFrame, factors: list[str], pairs=()) -> dict:
    """Z: (n, d) one vector per clip of a balanced factorial design (columns of `design`). Sum of
    squares of each factor's marginal means (and of the listed two-way interactions), as a fraction
    of the total sum of squares about the grand mean. Biased upward by noise at few levels; compare
    fractions between layers and cells, not to zero."""
    Z = Z - Z.mean(0)
    total = float((Z ** 2).sum())
    if total <= 1e-12 * max(1, Z.size):   # identical clips (e.g. text-blind early layers at step 0)
        return {k: float("nan") for k in [*factors, *[f"{a}x{b}" for a, b in pairs], "residual"]}
    out = {}
    for f in factors:
        g = pd.DataFrame(Z).groupby(design[f].to_numpy()).transform("mean").to_numpy()
        out[f] = float((g ** 2).sum()) / total
    for a, b in pairs:
        key = design[a].astype(str).to_numpy() + "|" + design[b].astype(str).to_numpy()
        gab = pd.DataFrame(Z).groupby(key).transform("mean").to_numpy()
        ga = pd.DataFrame(Z).groupby(design[a].to_numpy()).transform("mean").to_numpy()
        gb = pd.DataFrame(Z).groupby(design[b].to_numpy()).transform("mean").to_numpy()
        out[f"{a}x{b}"] = float(((gab - ga - gb) ** 2).sum()) / total
    out["residual"] = 1.0 - sum(out.values())
    return out


def cosine_rdm(V: np.ndarray) -> np.ndarray:
    """(n, d) -> (n, n) 1 - cosine."""
    U = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    return 1.0 - U @ U.T


def content_rdm(emb: np.ndarray) -> np.ndarray:
    """emb: (n, F, D) CLIP frame embeddings -> 1 - mean over frames of the frame-wise cosine."""
    E = emb / np.maximum(np.linalg.norm(emb, axis=-1, keepdims=True), 1e-12)
    return 1.0 - np.einsum("ifd,jfd->ij", E, E) / E.shape[1]


def flow_rdm(flows: np.ndarray, masks: np.ndarray | None = None) -> np.ndarray:
    """flows: (n, P, h, w, 2); masks: (n, F, H, W) subject masks or None. Camera motion removed per clip,
    then 1 - masked flow cosine between every pair (metrics.masked_motion_fidelity)."""
    n = len(flows)
    comp = [M.remove_camera_motion(flows[i], None if masks is None else masks[i]) for i in range(n)]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = M.masked_motion_fidelity(comp[i], comp[j], None if masks is None else masks[i],
                                         None if masks is None else masks[j])
            D[i, j] = D[j, i] = 1.0 - s
    return D


def flow_descriptor(flow: np.ndarray, mask: np.ndarray | None, grid: int = 8) -> np.ndarray:
    """Camera-compensated, subject-masked flow pooled to grid x grid x 2 and averaged over frame pairs."""
    import torch

    f = M.remove_camera_motion(flow, mask)
    if mask is not None:
        pm = M._pair_masks(mask, f)
        f = f * pm[..., None]
    t = torch.from_numpy(np.ascontiguousarray(f.mean(0))).permute(2, 0, 1)[None].float()
    return torch.nn.functional.adaptive_avg_pool2d(t, grid)[0].flatten().numpy()


def within_prompt_residual(X: np.ndarray, prompt_ids: np.ndarray) -> np.ndarray:
    """Subtract each prompt's mean over seeds: what varies across noise draws of one prompt."""
    df = pd.DataFrame(X)
    return X - df.groupby(prompt_ids).transform("mean").to_numpy()


# up to 1e6: with ~240 clips and 512 features the ridge must be able to shrink to "predict the mean" (R^2 ~ 0);
# the first grid stopped at 1e3 and overfit to R^2 << 0 (session 6 of the log)
RIDGE_ALPHAS = tuple(np.logspace(-1, 6, 15))


def _ridge_path(Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray, alphas) -> np.ndarray:
    """Predictions on Xte for every alpha, from one SVD of the (centred, standardised) training data:
    W(a) = V diag(s / (s^2 + a)) U' Y.  Returns (n_alphas, n_test, n_targets)."""
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my = Ytr.mean(0)
    U, sv, Vt = np.linalg.svd((Xtr - mx) / sx, full_matrices=False)
    UtY = U.T @ (Ytr - my)
    Z = ((Xte - mx) / sx) @ Vt.T                                  # test data in the right-singular basis
    return np.stack([Z @ ((sv / (sv ** 2 + a))[:, None] * UtY) + my for a in alphas])


def _cv_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, alphas=RIDGE_ALPHAS, inner_folds: int = 4) -> float:
    """Held-out R^2 of ridge regression, leave-one-group-out. The ridge strength is chosen *inside* each
    training set by grouped K-fold over its groups — never by leave-one-clip-out: the within-prompt residuals
    come in exact +/- pairs (the two seeds of a prompt, same subject), so a clip's partner left in the training
    set predicts it perfectly and leave-one-out would always pick the weakest ridge (session 6 of the log)."""
    pred = np.zeros_like(Y, dtype=float)
    for g in np.unique(groups):
        te = groups == g
        Xtr, Ytr, gtr = X[~te], Y[~te], groups[~te]
        ug = np.unique(gtr)
        err = np.zeros(len(alphas))
        for k in range(min(inner_folds, len(ug))):
            iv = np.isin(gtr, ug[k::inner_folds])
            P = _ridge_path(Xtr[~iv], Ytr[~iv], Xtr[iv], alphas)
            err += ((P - Ytr[iv][None]) ** 2).sum(axis=(1, 2))
        best = alphas[int(np.argmin(err))]
        pred[te] = _ridge_path(Xtr, Ytr, X[te], [best])[0]
    ss = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ((Y - pred) ** 2).sum() / max(ss, 1e-12))


def decoding_gain(X_layer: np.ndarray, X_latent: np.ndarray, Y: np.ndarray, prompt_ids: np.ndarray,
                  groups: np.ndarray) -> dict:
    """Within-prompt residual flow Y decoded from the layer and from the noisy latent, leave-one-group-out
    (groups = subjects). G = max(R^2(layer), 0) - max(R^2(latent), 0); G > 0: the layer carries generated
    motion beyond what is already in its input latent."""
    Xl, Xz, Yr = (within_prompt_residual(a, prompt_ids) for a in (X_layer, X_latent, Y))
    r_layer, r_lat = _cv_r2(Xl, Yr, groups), _cv_r2(Xz, Yr, groups)
    # a held-out R^2 below 0 means "no better than the mean": both are floored at 0 before the difference
    return {"r2_layer": r_layer, "r2_latent": r_lat, "gain": max(r_layer, 0.0) - max(r_lat, 0.0)}


def channel_selectivity(energy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """energy: (n_clips, C) per-channel dynamic energy; eta^2 of `labels` per channel."""
    energy = np.asarray(energy, dtype=np.float64)     # stored fp16: squares over- and underflow, the floor rounds to 0
    e = energy - energy.mean(0)
    g = pd.DataFrame(e).groupby(labels).transform("mean").to_numpy()
    return (g ** 2).sum(0) / np.maximum((e ** 2).sum(0), 1e-12)
