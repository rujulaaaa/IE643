"""Stages 5-6 · Score saved edit runs (edit_sweep.py outputs) and, for dev runs, pick the arm.

For every pair directory: decode each arm's latents, then per arm
    sc_A, sc_B   content fidelity (mean per-frame CLIP image cosine) to clip A / to B_ref
    sm_A, sm_B   motion fidelity (cosine of Farneback flow fields) to clip A / to B_ref
    consistency  subject consistency (VBench-style proxy) and its ratio to source A
    flicker      temporal-flicker proxy;  magnitude  mean flow speed
D (roadmap §5.4, D-14), s_c and s_m min-max normalised over every value in the run:
    motion direction  (A's motion, B's content):  D = ½[ŝc(out,B) + ŝm(out,A)] − ½[ŝc(out,A) + ŝm(out,B)]
    content direction (A's content, B's motion):  D = ½[ŝc(out,A) + ŝm(out,B)] − ½[ŝc(out,B) + ŝm(out,A)]
Per arm and direction: the paired gain over the prompt-only baseline of the same pair,
ΔD = D(arm) − D(base); D-16 statistic = median D(arm) − median D(base), 95% paired bootstrap
CI over pairs (10,000 resamples); consistency guard: median consistency ratio ≥ 0.9.

Stage 5 selection rule (D-31, fixed before the dev run was scored): per direction, among the
component-patch arms, the largest median ΔD over the dev pairs whose consistency guard holds;
ties broken by fewer patched modules. Written to <run>/selection.json for --arms selected.

    python scripts/score_edits.py results/kaggle/edit-dev/out --select
    python scripts/score_edits.py results/kaggle/vdprobe-lora/out --with results/kaggle/vdprobe-test/out

--with RUN (repeatable) adds the cached scores.csv of runs on the same pairs to this run's, so
s_c and s_m are normalised over one pool and every arm is compared with the same prompt-only
baseline. A (pair, arm) present in both keeps the --with run's row (the reference clips are
regenerated from the same noise; the regeneration is checked separately).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vdprobe import runio
from vdprobe.metrics import (ClipScorer, content_fidelity, disentanglement, flow_fields, motion_fidelity,
                             motion_magnitude, subject_consistency, temporal_flicker)

GUARD = 0.9


def pair_dirs(root: Path):
    return sorted(p.parent for p in root.rglob("manifest.json") if (p.parent / "latents.pt").exists())


def score_run(root: Path, loaded, clip: ClipScorer) -> pd.DataFrame:
    from vdprobe.pipeline import decode

    rows = []
    for d in pair_dirs(root):
        lat = torch.load(d / "latents.pt")
        meta = json.loads((d / "arms.json").read_text())
        feats = {}
        for k, x in lat.items():
            fr = decode(loaded, x.float())
            feats[k] = {"emb": clip.image(fr).cpu(), "flow": flow_fields(fr), "flicker": temporal_flicker(fr)}
        A, B = feats["source"], feats["B_ref"]
        cons_src = subject_consistency(A["emb"])
        for k, f in feats.items():
            rows.append({"pair": d.name, "family_seed": meta["pair"]["seed"], "arm": k,
                         "direction": k.split(".")[0] if "." in k else "ref",
                         "sc_A": content_fidelity(f["emb"], A["emb"]), "sc_B": content_fidelity(f["emb"], B["emb"]),
                         "sm_A": motion_fidelity(f["flow"], A["flow"]), "sm_B": motion_fidelity(f["flow"], B["flow"]),
                         "consistency": subject_consistency(f["emb"]),
                         "consistency_ratio": subject_consistency(f["emb"]) / cons_src,
                         "flicker": f["flicker"], "magnitude": motion_magnitude(f["flow"])})
        print(f"scored {d.name}: {len(feats)} clips", flush=True)
    return pd.DataFrame(rows)


def add_d(df: pd.DataFrame) -> pd.DataFrame:
    e = df[df.direction.isin(["motion", "content"])]
    sc_pool = np.concatenate([e.sc_A, e.sc_B])
    sm_pool = np.concatenate([e.sm_A, e.sm_B])
    df = df.copy()
    m = df.direction == "motion"
    c = df.direction == "content"
    df["D"] = np.nan
    df.loc[m, "D"] = disentanglement(df.sc_B[m], df.sm_A[m], df.sc_A[m], df.sm_B[m], sc_pool, sm_pool)
    df.loc[c, "D"] = disentanglement(df.sc_A[c], df.sm_B[c], df.sc_B[c], df.sm_A[c], sc_pool, sm_pool)
    df.attrs["pool"] = {"sc": [float(sc_pool.min()), float(sc_pool.max())],
                        "sm": [float(sm_pool.min()), float(sm_pool.max())]}
    return df


def boot_median_diff(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
    return float(np.median(a) - np.median(b)), [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for d in ("motion", "content"):
        base = df[df.arm == f"{d}.base"].set_index("pair")
        for arm, g in df[df.direction == d].groupby("arm"):
            g = g.set_index("pair").loc[base.index]
            stat, ci = boot_median_diff(g.D.values, base.D.values)
            out.append({"direction": d, "arm": arm, "n_pairs": len(g),
                        "median_D": float(g.D.median()), "median_D_base": float(base.D.median()),
                        "median_dD": float((g.D - base.D).median()), "frac_dD_pos": float(((g.D - base.D) > 0).mean()),
                        "d16_stat": stat, "d16_ci_lo": ci[0], "d16_ci_hi": ci[1],
                        "ci_excludes_0": bool(ci[0] > 0),
                        "consistency_ratio": float(g.consistency_ratio.median()),
                        "guard_ok": bool(g.consistency_ratio.median() >= GUARD),
                        "sc_A": float(g.sc_A.median()), "sc_B": float(g.sc_B.median()),
                        "sm_A": float(g.sm_A.median()), "sm_B": float(g.sm_B.median()),
                        "flicker": float(g.flicker.median()), "magnitude": float(g.magnitude.median())})
    return pd.DataFrame(out).sort_values(["direction", "median_dD"], ascending=[True, False])


def select(summary: pd.DataFrame, root: Path) -> dict:
    """D-31: best component arm per direction, from dev pairs."""
    arms = json.loads(next(root.rglob("arms.json")).read_text())["arms"]
    n_mods = {a["id"]: len(a.get("patch", {}).get("coords", [])) for a in arms}
    sel = {}
    for d, comp in (("motion", "dynamic"), ("content", "static")):
        s = summary[(summary.direction == d) & summary.arm.str.startswith(f"{d}.{comp}.") & summary.guard_ok].copy()
        s["n_mods"] = s.arm.map(n_mods)
        best = s.sort_values(["median_dD", "n_mods"], ascending=[False, True]).iloc[0]
        _, _, block, w = re.match(r"(\w+)\.(\w+)\.(.+)\.(\w+)$", best.arm).groups()
        sel[d] = {"arm": best.arm, "blocks": "all" if block == "allmm" else [block], "window": w,
                  "median_dD_dev": float(best.median_dD)}
    return sel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--with", dest="with_runs", type=Path, action="append", default=[],
                    help="earlier scored run(s) on the same pairs to pool with this one")
    ap.add_argument("--profile", default=None)
    ap.add_argument("--config", type=Path, default=runio.REPO_ROOT / "configs" / "generation.yaml")
    args = ap.parse_args()
    cache = args.run / "scores.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        loaded = runio.load(args)
        df = score_run(args.run, loaded, ClipScorer())
        df.to_csv(cache, index=False)
    if args.with_runs:
        df = pd.concat([*(pd.read_csv(r / "scores.csv") for r in args.with_runs), df])
        df = df.drop_duplicates(["pair", "arm"], keep="first").reset_index(drop=True)
    df = add_d(df)
    df.to_csv(args.run / "scores_D.csv", index=False)
    summary = summarise(df)
    summary.to_csv(args.run / "summary.csv", index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    print(summary[["direction", "arm", "median_D", "median_dD", "frac_dD_pos", "d16_ci_lo", "d16_ci_hi",
                   "consistency_ratio", "sm_A", "sc_B", "sm_B", "sc_A"]].round(3).to_string(index=False))
    res = {"pool": df.attrs["pool"], "n_pairs": int(df.pair.nunique()), "with": [str(r) for r in args.with_runs]}
    if args.select:
        res["selection"] = select(summary, args.run)
        (args.run / "selection.json").write_text(json.dumps(res["selection"], indent=1))
        print("selection:", json.dumps(res["selection"]))
    runio.write_json(args.run / "score_summary.json", res)


if __name__ == "__main__":
    main()
