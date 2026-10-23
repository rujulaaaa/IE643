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
        sk = torch.load(g.path / "sketches.pt")
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
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    per_seed = {}
    for seed, cs in clips.groupby("seed"):
        scenes = sorted(cs.scene.unique())
        if len(scenes) < 2:
            continue
        X_all = np.stack([feats[c] for c in cs["clip"]])        # (n, L, T, 2, dim)
        y_all = cs[target].to_numpy()
        acc = np.zeros((n_coords, n_steps))
        for li in range(n_coords):
            for t in range(n_steps):
                X = X_all[:, li, t, channel]
                fold = []
                for tr_scene in scenes:
                    tr = (cs.scene == tr_scene).to_numpy()
                    k = min(max_pca, tr.sum() - 1, X.shape[1])
                    clf = make_pipeline(StandardScaler(), PCA(n_components=k, random_state=0),
                                        LogisticRegression(C=C, max_iter=2000))
                    clf.fit(X[tr], y_all[tr])
                    fold.append(float((clf.predict(X[~tr]) == y_all[~tr]).mean()))
                acc[li, t] = np.mean(fold)
        per_seed[seed] = acc
    chance = 1.0 / clips[target].nunique()
    return {"acc": np.mean(list(per_seed.values()), axis=0), "per_seed": per_seed, "chance": chance}


def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)
