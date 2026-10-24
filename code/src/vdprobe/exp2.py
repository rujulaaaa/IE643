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


def _cv_r2(X: np.ndarray, Y: np.ndarray, groups: np.ndarray, alphas=(0.1, 1, 10, 100, 1000)) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    pred = np.zeros_like(Y)
    for g in np.unique(groups):
        te = groups == g
        sc = StandardScaler().fit(X[~te])
        m = RidgeCV(alphas=alphas).fit(sc.transform(X[~te]), Y[~te])
        pred[te] = m.predict(sc.transform(X[te]))
    ss = ((Y - Y.mean(0)) ** 2).sum()
    return float(1 - ((Y - pred) ** 2).sum() / max(ss, 1e-12))


def decoding_gain(X_layer: np.ndarray, X_latent: np.ndarray, Y: np.ndarray, prompt_ids: np.ndarray,
                  groups: np.ndarray) -> dict:
    """Within-prompt residual flow Y decoded from the layer and from the noisy latent, leave-one-group-out
    (groups = subjects). G = R^2(layer) - R^2(latent); G > 0: the layer carries generated motion beyond
    what is already in its input latent."""
    Xl, Xz, Yr = (within_prompt_residual(a, prompt_ids) for a in (X_layer, X_latent, Y))
    r_layer, r_lat = _cv_r2(Xl, Yr, groups), _cv_r2(Xz, Yr, groups)
    return {"r2_layer": r_layer, "r2_latent": r_lat, "gain": r_layer - r_lat}


def channel_selectivity(energy: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """energy: (n_clips, C) per-channel dynamic energy; eta^2 of `labels` per channel."""
    e = energy - energy.mean(0)
    g = pd.DataFrame(e).groupby(labels).transform("mean").to_numpy()
    return (g ** 2).sum(0) / np.maximum((e ** 2).sum(0), 1e-12)
