"""Experiment 2 Step 8 · the encodes / uses verdict per candidate (design report, decision rule).

    python scripts/exp2_verdict.py --analysis results/exp2/<primary cell> --candidates results/exp2/selection/candidates.json \
        --patching results/kaggle/vdprobe-exp2-patch-*/out --erasure results/kaggle/vdprobe-exp2-erase/out \
        --sweep results/kaggle/vdprobe-exp2-<primary>/out/<primary> --out results/exp2/verdict

Criteria (the report's table; δ = 0.1 fixed in advance):
  E presence    motion candidates: RSA unique motion share with max-over-map p_FWER < 0.05 somewhere in the
                candidate's window, and the flow-decoding gain over the latent > 0 there; content candidate: the
                same for the unique content share. Reported with the Hewitt-Liang selectivity of a linear
                verb (motion) or subject (content) probe at the candidate's coordinates and window.
  S sufficiency PE of the candidate's own channel (motion pairs: eff_m; content pairs: eff_c) on A→B patches:
                subject-level bootstrap 95% CI > 0, and the paired difference to the unrelated-clip control > 0.
  N necessity   NE on B→A patches with CI > 0, or (motion candidates) LEACE verb erasure changes motion more
                than the same-rank random eraser (paired CI > 0).
  D dissociation the other channel's PE on the other pair type within [-δ, δ] (TOST: 90% CI inside), for the
                motion locus and the content locus, plus the interaction (PE_m - PE_c)_X - (PE_m - PE_c)_Y > 0.
Labels: "motion locus" / "content locus" (D holds), "uses (sufficient / necessary / both)", "encodes but does not
use" (E holds, S and N fail), "not shown". The redundancy label needs a joint group patch, which was not run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DELTA = 0.1
N_BOOT = 10000


def subject_boot(df: pd.DataFrame, col: str, unit: str = "subject_b", n: int = N_BOOT, seed: int = 643,
                 alpha: float = 0.05) -> dict:
    """Mean over subjects of per-subject means; percentile CI from resampling subjects."""
    g = df.groupby(unit)[col].mean().dropna()
    if len(g) < 2:
        return {"mean": float(g.mean()) if len(g) else float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(g)}
    rng = np.random.default_rng(seed)
    v = g.to_numpy()
    boots = v[rng.integers(0, len(v), (n, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.percentile(boots, 100 * alpha / 2)),
            "hi": float(np.percentile(boots, 100 * (1 - alpha / 2))), "n": int(len(v))}


def presence(an_dir: Path, cand: dict) -> dict:
    m = np.load(an_dir / "maps.npz")
    coords = [str(c) for c in m["all_coords"]]
    idx = [coords.index(c) for c in cand["coords"] if c in coords]
    steps = cand["steps"]
    ch = 0 if cand["component"] == "static" else 1          # rsa channel: 0 content, 1 motion
    p = m["rsa_p_fwer"][np.ix_(idx, steps)][..., ch]
    u = m["rsa_unique"][np.ix_(idx, steps)][..., ch]
    out = {"rsa_unique_max": float(u.max()), "rsa_p_fwer_min": float(p.min()), "rsa_significant": bool((p < .05).any())}
    if "decode_gain" in m and ch == 1:
        g = m["decode_gain"][np.ix_(idx, steps)]
        out["decode_gain_max"] = float(g.max())
        out["decode_positive"] = bool(((p < .05) & (g > 0)).any())
        out["E"] = out["decode_positive"]
    else:
        out["E"] = out["rsa_significant"]
    return out


def probe_selectivity(sweep: Path, cand: dict) -> dict | None:
    """Hewitt-Liang selectivity of a linear probe at the candidate (D-34 controls, no permutations)."""
    import torch

    from vdprobe import analysis as A

    groups = A.find_groups([sweep])
    if not groups:
        return None
    feats, rows = {}, []
    for g in map(A.load_group, groups):
        sk = torch.load(g.path / "sketches_all.pt")
        names = g.manifest["all_coords"]
        idx = [names.index(c) for c in cand["coords"] if c in names]
        for cid, t in sk.items():
            feats[cid] = t[idx][:, cand["steps"]].float().numpy()            # (L, T, 2, dim)
            rows.append({"clip": cid, "seed": g.seed, "scene": g.scene, **g.manifest["clip_meta"][cid]})
    clips = pd.DataFrame(rows)
    target = "subject" if cand["component"] == "static" else "verb"
    part = 1 if cand["component"] == "static" else 0
    L, T = next(iter(feats.values())).shape[:2]
    r = A.probe_scores(feats, clips, target, part, L, T, probe="linear", n_jobs=-1)
    return {"target": target, "acc_max": float(r["acc"].max()), "control_at_max": float(r["control_acc"].flat[r["acc"].argmax()]),
            "selectivity_max": float(r["selectivity"].max()), "chance": r["chance"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--patching", type=Path, nargs="+", required=True, help="directories holding rows.csv (any depth)")
    ap.add_argument("--erasure", type=Path, nargs="*", default=[], help="directories holding erasure rows.csv + manifest")
    ap.add_argument("--sweep", type=Path, default=None, help="primary cell sweep (for probe selectivity)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cands = json.loads(args.candidates.read_text())["candidates"]
    rows = pd.concat([pd.read_csv(f, keep_default_na=False, na_values=["", "NaN", "nan"])   # "null" is a name, not NA
                      for d in args.patching for f in Path(d).rglob("rows.csv")], ignore_index=True)
    rows = rows.drop_duplicates(["pair", "candidate", "direction"])
    er = {}
    for d in args.erasure:
        for f in Path(d).rglob("manifest.json"):
            man = json.loads(f.read_text())
            if man.get("experiment") == "exp2_erasure":
                er[man["coord"]] = (pd.read_csv(f.parent / "rows.csv"), man)

    res = {"delta": DELTA, "n_pairs": int(rows.pair.nunique()), "candidates": {}}
    pe = {}
    for cand in cands:
        nm = cand["name"]
        own, own_pairs = ("eff_c", "content") if cand["component"] == "static" else ("eff_m", "motion")
        other, other_pairs = ("eff_m", "motion") if own == "eff_c" else ("eff_c", "content")
        r = rows[rows.candidate == nm]
        suff = r[(r.direction == "suff") & (r.kind == own_pairs)]
        ctrl = r[(r.direction == "ctrl") & (r.kind == own_pairs)]
        nec = r[(r.direction == "nec") & (r.kind == own_pairs)]
        cross = r[(r.direction == "suff") & (r.kind == other_pairs)]
        S = subject_boot(suff, own)
        both = suff.merge(ctrl, on=["pair", "subject_b"], suffixes=("", "_ctrl"))
        both["diff"] = both[own] - both[f"{own}_ctrl"]
        S_ctrl = subject_boot(both, "diff")
        N = subject_boot(nec, own)
        cross_ci = subject_boot(cross, other, alpha=0.10)          # 90% CI for TOST
        out = {"own_channel": own_pairs, "S": S, "S_minus_control": S_ctrl, "N": N, "other_channel_PE": cross_ci}
        out["S_pass"] = bool(S["lo"] > 0 and S_ctrl["lo"] > 0)
        out["N_pass"] = bool(N["lo"] > 0)
        coord = cand["coords"][0] if len(cand["coords"]) == 1 else None
        if coord in er and own == "eff_m":
            e, man = er[coord]
            a = e[e.eraser == "leace_verb"].set_index("clip")
            b = e[e.eraser == "random_verb"].set_index("clip")
            d = pd.DataFrame({"diff": a.motion_change - b.motion_change, "subject_b": a.subject})
            out["erasure_leace_minus_random"] = subject_boot(d, "diff")
            out["N_pass"] = out["N_pass"] or bool(out["erasure_leace_minus_random"]["lo"] > 0)
        out["other_within_delta"] = bool(-DELTA < cross_ci["lo"] and cross_ci["hi"] < DELTA)
        try:
            out["presence"] = presence(args.analysis, cand)
        except Exception as ex:          # e.g. a candidate outside the analysed coordinates
            out["presence"] = {"E": False, "error": str(ex)}
        if args.sweep is not None:
            try:
                out["probe"] = probe_selectivity(args.sweep, cand)
            except Exception as ex:
                out["probe"] = {"error": str(ex)}
        E, Sp, Np = out["presence"].get("E", False), out["S_pass"], out["N_pass"]
        if Sp or Np:
            out["label"] = "uses (" + " and ".join(x for x, ok in (("sufficient", Sp), ("necessary", Np)) if ok) + ")"
        elif E:
            out["label"] = "encodes but does not use"
        else:
            out["label"] = "not shown"
        pe[nm] = (suff, cross, own)
        res["candidates"][nm] = out
        print(f"{nm:20s} {out['label']:32s} S {S['mean']:+.3f} [{S['lo']:+.3f}, {S['hi']:+.3f}]  "
              f"N {N['mean']:+.3f} [{N['lo']:+.3f}, {N['hi']:+.3f}]  E {E}", flush=True)

    # double dissociation: every motion candidate against the content candidate
    content = [c["name"] for c in cands if c["component"] == "static"]
    motion = [c["name"] for c in cands if c["component"] != "static"]
    res["dissociation"] = {}
    for y in content:
        for x in motion:
            sx, cx, _ = pe[x]
            sy, cy, _ = pe[y]
            gx = pd.concat([sx.assign(v=sx.eff_m), cx.assign(v=-cx.eff_c)])     # PE_m - PE_c for X, per subject
            gy = pd.concat([sy.assign(v=-sy.eff_c), cy.assign(v=cy.eff_m)])     # (PE_m - PE_c) for Y = -(PE_c - PE_m)
            dx, dy = gx.groupby("subject_b").v.mean(), gy.groupby("subject_b").v.mean()
            both = pd.DataFrame({"v": dx - dy}).dropna().reset_index()
            inter = subject_boot(both, "v")
            X, Y = res["candidates"][x], res["candidates"][y]
            ok = bool(X["S_pass"] and X["other_within_delta"] and Y["S_pass"] and Y["other_within_delta"] and inter["lo"] > 0)
            res["dissociation"][f"{x} vs {y}"] = {"interaction": inter, "double_dissociation": ok}
            if ok:
                X["label"], Y["label"] = "motion locus", "content locus"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verdict.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: v["label"] for k, v in res["candidates"].items()}, indent=1))
    print("dissociation:", {k: v["double_dissociation"] for k, v in res["dissociation"].items()})


if __name__ == "__main__":
    main()
