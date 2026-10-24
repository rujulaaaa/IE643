"""Stage 3-4 analysis: selectivity maps, the pre-registered tests for H1-H3, linear probes.

Input: the per-group outputs of scripts/probe_sweep.py (distances.csv, pairs.json,
sketches.pt, manifest.json), one group per (seed, scene).

Selectivity units (D-28). S compares a content distance with a motion distance, which come
from two different pairs. They are matched by the clip they share: for every clip
B = (s, v, scene) with s not its family anchor and v not the anchor verb,

    content pair  (anchor(s), v, scene) -> B      subject swapped, verb held
    motion pair   (s, running, scene)   -> B      verb swapped, subject held

and S_unit = (d_c - d_m) / (d_c + d_m) for that clip, at every (coordinate, step), on x_tilde
(S_dyn) and on x_bar (S_stat). 7 x 3 x 2 Q units + 6 x 3 x 2 H units = 78 per seed. The map is
the median over units within a seed, with a 95% bootstrap CI over units, then averaged over
seeds (D-04). The ratio of medians, (med d_c - med d_m) / (med d_c + med d_m), is kept as a
robustness check.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

N_BOOT = 1000
RNG_SEED = 643


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Group:
    path: Path
    seed: int
    scene: str
    manifest: dict
    pairs: pd.DataFrame
    dist: pd.DataFrame


def find_groups(roots: list[Path]) -> list[Path]:
    out = []
    for r in roots:
        out += [p.parent for p in Path(r).rglob("manifest.json") if (p.parent / "distances.csv").exists()]
    return sorted(set(out))


def load_group(path: Path) -> Group:
    m = json.loads((path / "manifest.json").read_text())
    pairs = pd.DataFrame(json.loads((path / "pairs.json").read_text()))
    dist = pd.read_csv(path / "distances.csv")
    return Group(path, m["seed"], m["scene"], m, pairs, dist)


def load_all(roots: list[Path]) -> tuple[list[Group], pd.DataFrame]:
    """All groups, plus one long table: (seed, scene, pair kind/ids, coord, step, distances)."""
    groups = [load_group(p) for p in find_groups(roots)]
    if not groups:
        raise SystemExit(f"no sweep groups under {roots}")
    frames = []
    for g in groups:
        d = g.dist.merge(g.pairs, on="pair")
        d["seed"], d["scene"] = g.seed, g.scene
        frames.append(d)
    return groups, pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Matched selectivity units (D-28)
# ---------------------------------------------------------------------------

def selectivity_units(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (seed, scene, unit clip B, coord, step) with d_c, d_m and S on both channels."""
    keys = ["seed", "scene", "b", "coord", "step"]
    cols = ["d_static", "d_dynamic", "d_raw", "l2_static", "l2_dynamic"]
    c = long[long.kind == "content"][keys + cols + ["family_b"]]
    m = long[long.kind == "motion"][keys + cols]
    u = c.merge(m, on=keys, suffixes=("_c", "_m"))
    for chan, col in (("dyn", "d_dynamic"), ("stat", "d_static"), ("raw", "d_raw"),
                      ("l2dyn", "l2_dynamic"), ("l2stat", "l2_static")):
        dc, dm = u[f"{col}_c"], u[f"{col}_m"]
        u[f"S_{chan}"] = np.where(dc + dm > 0, (dc - dm) / (dc + dm), 0.0)
    return u.rename(columns={"b": "unit", "family_b": "family"})


def _boot_medians(values: np.ndarray, rng, n_boot=N_BOOT) -> np.ndarray:
    """values: (n_units, ...) -> (n_boot, ...) medians of unit resamples."""
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    return np.median(values[idx], axis=1)


def selectivity_map(units: pd.DataFrame, chan: str, coords: list[str], n_steps: int = 25,
                    n_boot: int = N_BOOT) -> dict:
    """Median-over-units map per seed, averaged over seeds, with bootstrap CI.

    Returns {"median": (L, T), "lo": (L, T), "hi": (L, T), "boot": (n_boot, L, T),
             "per_seed": {seed: (L, T)}, "ratio_of_medians": (L, T)}.
    """
    rng = np.random.default_rng(RNG_SEED)
    boots, meds, rom = [], {}, []
    for seed, us in units.groupby("seed"):
        piv = us.pivot_table(index=["scene", "unit"], columns=["coord", "step"], values=f"S_{chan}")
        piv = piv.reindex(columns=pd.MultiIndex.from_product([coords, range(n_steps)]))
        arr = piv.to_numpy().reshape(len(piv), len(coords), n_steps)
        meds[seed] = np.median(arr, axis=0)
        boots.append(_boot_medians(arr, rng, n_boot))
        col = {"dyn": "d_dynamic", "stat": "d_static", "raw": "d_raw",
               "l2dyn": "l2_dynamic", "l2stat": "l2_static"}[chan]
        dc = us.pivot_table(index=["scene", "unit"], columns=["coord", "step"], values=f"{col}_c") \
            .reindex(columns=piv.columns).to_numpy().reshape(arr.shape)
        dm = us.pivot_table(index=["scene", "unit"], columns=["coord", "step"], values=f"{col}_m") \
            .reindex(columns=piv.columns).to_numpy().reshape(arr.shape)
        mc, mm = np.median(dc, 0), np.median(dm, 0)
        rom.append((mc - mm) / np.maximum(mc + mm, 1e-12))
    boot = np.mean(boots, axis=0)                  # average over seeds, per bootstrap draw
    return {"median": np.mean(list(meds.values()), axis=0),
            "lo": np.percentile(boot, 2.5, axis=0), "hi": np.percentile(boot, 97.5, axis=0),
            "boot": boot, "per_seed": meds, "ratio_of_medians": np.mean(rom, axis=0)}


# ---------------------------------------------------------------------------
# Pre-registered tests (D-22, D-23; statements in docs/hypotheses.md)
# ---------------------------------------------------------------------------

def smallest_mass_set(counts: dict, mass: float = 0.95) -> list:
    total = sum(counts.values())
    out, acc = [], 0
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        out.append(k)
        acc += v
        if acc >= mass * total:
            break
    return out


def test_h1(smap_dyn: dict, coord_meta: list[dict]) -> dict:
    """H1: motion selectivity M(b) = -mean_t median S_dyn peaks at a mid-resolution temporal block."""
    temporal = [i for i, c in enumerate(coord_meta) if c["kind"] == "temporal_module"]
    blocks = [coord_meta[i]["block"] for i in temporal]
    levels = {coord_meta[i]["block"]: coord_meta[i]["level"] for i in temporal}
    n_levels = max(levels.values()) + 1
    M = -smap_dyn["median"][temporal].mean(axis=1)
    Mb = -smap_dyn["boot"][:, temporal].mean(axis=2)          # (n_boot, n_blocks)
    arg = np.argmax(Mb, axis=1)
    counts = {blocks[k]: int((arg == k).sum()) for k in range(len(blocks))}
    set95 = smallest_mass_set({k: v for k, v in counts.items() if v > 0})
    extremes = [b for b in set95 if levels[b] in (0, n_levels - 1)]
    return {"M_by_block": dict(zip(blocks, M.round(4).tolist())),
            "M_ci": {b: [float(np.percentile(Mb[:, k], 2.5)), float(np.percentile(Mb[:, k], 97.5))]
                     for k, b in enumerate(blocks)},
            "argmax_block": blocks[int(np.argmax(M))], "bootstrap_argmax_counts": counts,
            "set95": set95, "levels": levels, "survives": len(extremes) == 0,
            "rule": "95% bootstrap argmax set excludes every level-0 and level-3 block"}


def _changepoint(m: np.ndarray) -> tuple[int, float, float]:
    """Least-squares two-segment step fit; tau = first index of the second segment, in 1..T-1."""
    best = None
    for tau in range(1, len(m)):
        a, b = m[:tau], m[tau:]
        sse = ((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum()
        if best is None or sse < best[0]:
            best = (sse, tau, a.mean(), b.mean())
    return best[1], best[2], best[3]


def test_h2(smap_dyn: dict, coord_names: list[str], block_coord: str) -> dict:
    """H2: on m(t) = -median S_dyn(L*, t), the changepoint lies in the first half and the
    early segment is the more motion-selective one."""
    i = coord_names.index(block_coord)
    m = -smap_dyn["median"][i]
    tau, early, late = _changepoint(m)
    taus = np.array([_changepoint(-b[i])[0] for b in smap_dyn["boot"]])
    lo, hi = np.percentile(taus, [2.5, 97.5])
    n = len(m)
    return {"coord": block_coord, "m_t": m.round(4).tolist(), "tau": int(tau), "tau_ci": [float(lo), float(hi)],
            "mean_early": float(early), "mean_late": float(late),
            "survives": bool(hi <= n // 2 and early > late),
            "rule": f"tau CI within steps <= {n // 2} and mean m(t<tau) > mean m(t>=tau)"}


def test_h3(smap_dyn: dict, smap_stat: dict, coord_meta: list[dict], steps=(0, 4, 8, 12, 16, 20, 24)) -> dict:
    """H3 (D-23): f_stat >= 0.75 and n_flip >= 1 over the coarse coordinates x D-07 steps."""
    s = list(steps)
    stat_pos = smap_stat["lo"][:, s] > 0
    dyn_neg = smap_dyn["hi"][:, s] < 0
    temporal = np.array([c["kind"].startswith("temporal") for c in coord_meta])
    f_stat = float(stat_pos.mean())
    flips = stat_pos & dyn_neg & temporal[:, None]
    n_flip = int(flips.sum())
    where = [(coord_meta[i]["short"], s[j]) for i, j in zip(*np.nonzero(flips))]
    return {"f_stat": f_stat, "n_flip": n_flip, "flips": where,
            "f_dyn_motion": float(dyn_neg.mean()),
            "survives": bool(f_stat >= 0.75 and n_flip >= 1), "rule": "f_stat >= 0.75 and n_flip >= 1"}


# ---------------------------------------------------------------------------
# Linear probes (D-21)
# ---------------------------------------------------------------------------

def load_sketches(groups: list[Group]) -> tuple[dict, pd.DataFrame, list[str]]:
    """{clip id: (L, T, 2, dim)} and a clip table (seed, scene, subject, verb, family)."""
    feats, rows, coords = {}, [], None
    for g in groups:
        if (g.path / "sketches.pt").exists():
            sk = torch.load(g.path / "sketches.pt")
        else:       # Experiment 2 sweeps: the coarse coordinates are a slice of sketches_all.pt
            idx = [g.manifest["all_coords"].index(n) for n in g.manifest["coords"]]
            sk = {cid: t[idx] for cid, t in torch.load(g.path / "sketches_all.pt").items()}
        coords = coords or g.manifest["coords"]
        if g.manifest["coords"] != coords:
            raise ValueError("groups disagree on the coordinate list")
        lab = {r.a: (r.subject_a, r.verb_a, r.family_a) for r in g.pairs.itertuples()}
        lab.update({r.b: (r.subject_b, r.verb_b, r.family_b) for r in g.pairs.itertuples()})
        for cid, t in sk.items():
            feats[cid] = t.float().numpy()
            s, v, f = lab[cid]
            rows.append({"clip": cid, "seed": g.seed, "scene": g.scene, "subject": s, "verb": v, "family": f})
    return feats, pd.DataFrame(rows), coords


def probe_accuracy(feats: dict, clips: pd.DataFrame, target: str, channel: int,
                   n_coords: int, n_steps: int, max_pca: int = 256, C: float = 1.0) -> dict:
    """Scene-split probe (D-21): train on one scene, test on the other, swap, average; per seed,
    then averaged over seeds. channel 0 = x_tilde sketch, 1 = x_bar sketch.
    Returns {"acc": (L, T), "per_seed": {seed: (L, T)}, "chance": float}."""
    res = probe_scores(feats, clips, target, channel, n_coords, n_steps, probe="linear",
                       control=False, max_pca=max_pca, C=C)
    return {"acc": res["acc"], "per_seed": res["per_seed"], "chance": res["chance"]}


# ---------------------------------------------------------------------------
# Probe controls (D-34): control task, non-linear probe, permutation null
#
# Control task (Hewitt & Liang 2019). The "type" of a clip is its prompt minus the scene,
# (subject, verb) within a seed: the same type appears once in each scene, so a scene-split probe can score
# well either because the layer carries the target or because it identifies the type and the
# probe memorises type -> label. The control task keeps the second route and removes the
# first: every type gets a label drawn by permuting the real labels over types (class
# frequencies kept), fixed per seed. selectivity = acc(real) - acc(control).
#
# Permutation null (Ojala & Garriga 2010, test 1): training labels permuted, test labels kept;
# p = (1 + #{null >= observed}) / (1 + R), on the seed-averaged accuracy. The permutation is
# restricted (verbs shuffled within each subject, family labels across subjects) and defined
# on types, so both scenes get the same relabelling; p_fwer uses the max over the map.
# Scaler + PCA are unsupervised, so they are fitted once per fold and shared by the real,
# control and permuted fits.
# ---------------------------------------------------------------------------

PROBES = ("linear", "mlp")


def _classifier(probe: str, C: float, hidden: int, alpha: float, seed: int):
    if probe == "linear":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(C=C, max_iter=2000)
    if probe == "mlp":
        from sklearn.neural_network import MLPClassifier
        # Hewitt & Liang: small hidden layers + weight decay keep MLP probes selective.
        # lbfgs suits n ~ 60; 500 iterations: shuffled labels never converge and would otherwise
        # run to the cap (2000 iterations cost ~7x as much for no change in held-out accuracy)
        return MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, solver="lbfgs", max_iter=500,
                             random_state=seed)
    raise ValueError(f"probe must be one of {PROBES}")


def control_labels(cs: pd.DataFrame, target: str, rng, type_cols=("subject", "verb")) -> np.ndarray:
    """Hewitt-Liang control labels for one seed's clips: a fixed random type -> label map."""
    keys = [tuple(r) for r in cs[list(type_cols)].itertuples(index=False)]
    first = dict(zip(keys[::-1], cs[target].to_numpy()[::-1]))     # one label per type
    types = list(first)
    mapping = dict(zip(types, rng.permutation([first[k] for k in types])))
    return np.array([mapping[k] for k in keys])


def permuted_labels(cs: pd.DataFrame, target: str, rng, type_cols=("subject", "verb"),
                    block: str = "subject") -> np.ndarray:
    """One draw of the restricted permutation null for one seed's clips. Labels are permuted at
    the type level (so both scenes get the same relabelling) and within `block` when the target
    varies inside it (verb: shuffle each subject's 4 verbs), else across blocks (family: shuffle
    the subjects' labels). Exchangeable under "the layer carries subject identity but no target"."""
    keys = [tuple(r) for r in cs[list(type_cols)].itertuples(index=False)]
    lab = dict(zip(keys[::-1], cs[target].to_numpy()[::-1]))
    bpos = list(type_cols).index(block)
    by_block: dict = {}
    for k in lab:
        by_block.setdefault(k[bpos], []).append(k)
    if all(len({lab[k] for k in ks}) == 1 for ks in by_block.values()):     # target is a block property
        blocks = list(by_block)
        new = dict(zip(blocks, rng.permutation([lab[by_block[b][0]] for b in blocks])))
        mapping = {k: new[k[bpos]] for k in lab}
    else:
        mapping = {}
        for ks in by_block.values():
            mapping.update(zip(ks, rng.permutation([lab[k] for k in ks])))
    return np.array([mapping[k] for k in keys])


def _probe_cell(X: np.ndarray, folds: list, labels: list[tuple[np.ndarray, np.ndarray]], probe: str,
                max_pca: int, C: float, hidden: int, alpha: float) -> np.ndarray:
    """Held-out accuracy of one (seed, coordinate, step) cell for each (train labels, test labels)
    pair, averaged over the scene folds: real (y, y), control (c, c), permutation draw r (pi_r, y)."""
    import warnings

    from sklearn.decomposition import PCA
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    out = np.zeros(len(labels))
    # Matrices this small run ~30x slower with a multi-threaded BLAS; parallelism is over cells.
    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for tr in folds:
            sc = StandardScaler().fit(X[tr])
            k = min(max_pca, int(tr.sum()) - 1, X.shape[1])
            pca = PCA(n_components=k, random_state=0).fit(sc.transform(X[tr]))
            z_tr, z_te = pca.transform(sc.transform(X[tr])), pca.transform(sc.transform(X[~tr]))
            for j, (y_fit, y_eval) in enumerate(labels):
                clf = _classifier(probe, C, hidden, alpha, seed=j).fit(z_tr, y_fit[tr])
                out[j] += (clf.predict(z_te) == y_eval[~tr]).mean()
    return out / len(folds)


def probe_scores(feats: dict, clips: pd.DataFrame, target: str, channel: int, n_coords: int, n_steps: int,
                 probe: str = "linear", control: bool = True, n_perm: int = 0, max_pca: int = 256,
                 C: float = 1.0, hidden: int = 10, alpha: float = 1.0, n_jobs: int = 1,
                 rng_seed: int = RNG_SEED) -> dict:
    """Scene-split probe of `target` with the D-34 controls, per seed, averaged over seeds.

    The permutation draws are shared by every cell of a seed, so the maximum over the map is a
    valid null for the whole map: p_fwer controls the family-wise error over all (L, T) cells.

    Returns {"acc", "control_acc", "selectivity", "p_perm", "p_fwer": (L, T) or None,
    "per_seed": {seed: acc}, "chance"}: control_acc / selectivity are None without `control`,
    p_perm / p_fwer None without n_perm."""
    from joblib import Parallel, delayed

    per_seed, ctrl_seed, null_seed = {}, {}, {}
    for seed, cs in clips.groupby("seed"):
        cs = cs.reset_index(drop=True)
        scenes = sorted(cs.scene.unique())
        if len(scenes) < 2:
            continue
        rng = np.random.default_rng([rng_seed, int(seed)])
        y = cs[target].to_numpy()
        labels = [(y, y)]
        if control:
            c = control_labels(cs, target, rng)
            labels.append((c, c))
        labels += [(permuted_labels(cs, target, rng), y) for _ in range(n_perm)]
        folds = [(cs.scene == s).to_numpy() for s in scenes]
        X_all = np.stack([feats[c] for c in cs["clip"]])        # (n, L, T, 2, dim)
        res = Parallel(n_jobs=n_jobs)(
            delayed(_probe_cell)(X_all[:, li, t, channel], folds, labels, probe, max_pca, C, hidden, alpha)
            for li in range(n_coords) for t in range(n_steps))
        res = np.array(res).reshape(n_coords, n_steps, -1)
        per_seed[seed] = res[..., 0]
        if control:
            ctrl_seed[seed] = res[..., 1]
        if n_perm:
            null_seed[seed] = res[..., 1 + control:]
    acc = np.mean(list(per_seed.values()), axis=0)
    out = {"acc": acc, "per_seed": per_seed, "chance": 1.0 / clips[target].nunique(),
           "control_acc": None, "selectivity": None, "p_perm": None, "p_fwer": None}
    if control:
        out["control_acc"] = np.mean(list(ctrl_seed.values()), axis=0)
        out["selectivity"] = acc - out["control_acc"]
    if n_perm:
        null = np.mean(list(null_seed.values()), axis=0)       # (L, T, R), seed-averaged per draw
        eps = 1e-12
        out["p_perm"] = (1 + (null >= acc[..., None] - eps).sum(-1)) / (1 + n_perm)
        null_max = null.reshape(-1, n_perm).max(0)              # max over the map, per draw
        out["p_fwer"] = (1 + (null_max[None, None, :] >= acc[..., None] - eps).sum(-1)) / (1 + n_perm)
    return out


# ---------------------------------------------------------------------------
# Layer comparison for Experiment 2 (docs/experiment2.md)
# ---------------------------------------------------------------------------

def _hsic_unbiased(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al., JMLR 2012), used by debiased CKA:
    [tr(K~L~) + (1'K~1)(1'L~1)/((n-1)(n-2)) - 2/(n-2) 1'K~L~1] / (n(n-3)), K~ = K with zero diagonal."""
    n = K.shape[0]
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0)
    np.fill_diagonal(L, 0)
    one_k, one_l = K.sum(0), L.sum(0)
    return float(((K * L).sum() + one_k.sum() * one_l.sum() / ((n - 1) * (n - 2))
                  - 2 / (n - 2) * one_k @ one_l) / (n * (n - 3)))


def debiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Debiased linear CKA between two (n, p) representations of the same n clips. Unlike the
    plain estimator (which gives ~0.9 for two *independent* random 60 x 512 matrices), it is ~0 under
    independence at small n; it can be slightly negative, and is reported as is."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    K, L = X @ X.T, Y @ Y.T
    den = np.sqrt(max(_hsic_unbiased(K, K), 0) * max(_hsic_unbiased(L, L), 0))
    return _hsic_unbiased(K, L) / den if den > 0 else 0.0


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    res = y - X1 @ beta
    return float(1 - res @ res / ((y - y.mean()) @ (y - y.mean())))


def rsa_commonality(d_layer: np.ndarray, predictors: dict[str, np.ndarray], rank: bool = True) -> dict:
    """Multiple-regression RSA with commonality analysis on vectorised RDMs (one entry per clip
    pair): R^2 of the layer distances on every subset of the model RDMs (e.g. CLIP content
    distance, masked-flow motion distance, scene), the unique share of each predictor
    (R^2_all - R^2_without_it), and the shared remainder (R^2_all - sum of uniques, negative under
    suppression). Rank-transformed by default (robust to the scale of cosine distances)."""
    from itertools import combinations

    from scipy.stats import rankdata

    tf = (lambda v: rankdata(v)) if rank else (lambda v: np.asarray(v, float))
    y = tf(d_layer)
    names = list(predictors)
    cols = {k: tf(v) for k, v in predictors.items()}
    r2 = {}
    for r in range(1, len(names) + 1):
        for sub in combinations(names, r):
            r2[sub] = _r2(y, np.column_stack([cols[k] for k in sub]))
    full = r2[tuple(names)]
    unique = {k: full - (r2[tuple(n for n in names if n != k)] if len(names) > 1 else 0.0) for k in names}
    return {"r2": full, "unique": unique, "shared": full - sum(unique.values()),
            "r2_single": {k: r2[(k,)] for k in names}}


def rsa_permutation(D_layer: np.ndarray, preds: dict[str, np.ndarray], n_perm: int, rng,
                    groups: np.ndarray | None = None, rank: bool = True) -> dict:
    """Permutation p-values for rsa_commonality's unique terms. D_layer and preds are square
    (N, N) RDMs over the same N clips; rows and columns of the layer RDM are permuted together
    (within `groups`, e.g. seed, when given), the predictors are kept, and the regression refitted.
    p = (1 + #{null >= observed}) / (1 + R)."""
    iu = np.triu_indices(len(D_layer), 1)
    vec = {k: v[iu] for k, v in preds.items()}
    obs = rsa_commonality(D_layer[iu], vec, rank)
    null = {k: [] for k in preds}
    for _ in range(n_perm):
        if groups is None:
            perm = rng.permutation(len(D_layer))
        else:
            perm = np.arange(len(D_layer))
            for g in np.unique(groups):
                idx = np.flatnonzero(groups == g)
                perm[idx] = rng.permutation(idx)
        res = rsa_commonality(D_layer[np.ix_(perm, perm)][iu], vec, rank)
        for k in preds:
            null[k].append(res["unique"][k])
    obs["p_unique"] = {k: (1 + sum(v >= obs["unique"][k] - 1e-12 for v in null[k])) / (1 + n_perm) for k in preds}
    return obs


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)
