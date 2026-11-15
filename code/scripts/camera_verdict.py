"""Verdict of scripts/camera_patch.py: bootstrap over pairs, then the rules of D-40 (HC2, HC3) or D-41 (HK3).

Per candidate and direction: mean of `eff_cam` and `eff_content` with a 95 % percentile CI over
pairs (10,000 draws), and for the named candidates the paired difference sufficiency - control.

    HC2 (lora)  candidate up1_motion_modules_all: CI lower bound of sufficiency > 0.3 and CI lower
                bound of (sufficiency - control) > 0
    HC3 (lora)  the same candidate, other scene: |mean eff_content| <= 0.1
    HK3 (cc)    candidate mm_<--block> (the best block of HK2): CI lower bound of transfer > 0.3
                and |mean eff_content| <= 0.1

    python scripts/camera_verdict.py results/kaggle/vdprobe-cam-patch --out results/camera/patch_lora
    python scripts/camera_verdict.py results/kaggle/vdprobe-cc-patch --out results/camera/patch_cc --block up_blocks.1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 10_000
MIN_EFFECT, MAX_LEAK = 0.3, 0.1
HC_CANDIDATE = "up1_motion_modules_all"


def load_rows(roots) -> pd.DataFrame:
    files = sorted({f for r in roots for f in Path(r).rglob("rows.csv") if (f.parent / "manifest.json").exists()})
    if not files:
        raise SystemExit(f"no finished pairs under {roots}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def boot(values, rng, n_boot=N_BOOT) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    draws = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)),
            "n": int(len(v))}


def summarise(df: pd.DataFrame, block: str | None = None, n_boot=N_BOOT, seed=643) -> dict:
    rng = np.random.default_rng(seed)
    model = str(df.model.iloc[0])
    table = {}
    for (cand, direction), g in df.groupby(["candidate", "direction"], sort=False):
        e = table.setdefault(cand, {})
        e[direction] = {"eff_cam": boot(g.eff_cam, rng, n_boot), "eff_cam_cos": boot(g.eff_cam_cos, rng, n_boot)}
        if direction in ("transfer", "ctrl"):
            e[direction]["eff_content"] = boot(g.eff_content, rng, n_boot)
    for cand, g in df.groupby("candidate", sort=False):      # paired: sufficiency minus control, per pair
        w = g.pivot_table(index="pair", columns="direction", values="eff_cam")
        if {"suff", "ctrl"} <= set(w.columns):
            table[cand]["suff_minus_ctrl"] = boot((w.suff - w.ctrl).values, rng, n_boot)
    out = {"model": model, "n_pairs": int(df.pair.nunique()), "n_boot": n_boot,
           "weak_pairs": sorted(df[df.eff_cam.isna()].pair.unique().tolist()), "candidates": table,
           "per_camera": {cam: {f"{c}|{d}": float(g.eff_cam.mean()) for (c, d), g in x.groupby(["candidate", "direction"])}
                          for cam, x in df.groupby("camera")},
           "verdicts": {}}
    lower = lambda b: b["lo"] is not None and b["lo"] > MIN_EFFECT
    if model == "lora" and HC_CANDIDATE in table:
        t = table[HC_CANDIDATE]
        s, diff = t.get("suff"), t.get("suff_minus_ctrl")
        if s and diff:
            out["verdicts"]["HC2"] = {"candidate": HC_CANDIDATE, "sufficiency": s["eff_cam"], "suff_minus_ctrl": diff,
                                      "pass": bool(lower(s["eff_cam"]) and diff["lo"] is not None and diff["lo"] > 0)}
        if "transfer" in t:
            leak = t["transfer"]["eff_content"]
            out["verdicts"]["HC3"] = {"candidate": HC_CANDIDATE, "transfer": t["transfer"]["eff_cam"], "leak": leak,
                                      "exp2_object_motion_leak": 0.173,
                                      "pass": bool(leak["mean"] is not None and abs(leak["mean"]) <= MAX_LEAK)}
    if model == "cc" and block and f"mm_{block}" in table and "transfer" in table[f"mm_{block}"]:
        t = table[f"mm_{block}"]["transfer"]
        leak = t["eff_content"]
        out["verdicts"]["HK3"] = {"candidate": f"mm_{block}", "transfer": t["eff_cam"], "leak": leak,
                                  "pass": bool(lower(t["eff_cam"]) and leak["mean"] is not None
                                               and abs(leak["mean"]) <= MAX_LEAK)}
    return out


def markdown(s: dict) -> str:
    f = lambda b: "–" if not b or b["mean"] is None else f"{b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]"
    lines = [f"| candidate ({s['model']}, {s['n_pairs']} pairs) | sufficiency | transfer | leak (content) | necessity | control | S − control |",
             "|---|---|---|---|---|---|---|"]
    for cand, t in s["candidates"].items():
        g = lambda d, k="eff_cam": f(t.get(d, {}).get(k))
        lines.append(f"| `{cand}` | {g('suff')} | {g('transfer')} | {g('transfer', 'eff_content')} | {g('nec')} | "
                     f"{g('ctrl')} | {f(t.get('suff_minus_ctrl'))} |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/camera/patch"))
    p.add_argument("--block", default=None, help="cc: the block of HK2 whose mm_<block> candidate is tested for HK3")
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args()
    df = load_rows(args.roots)
    s = summarise(df, args.block, args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verdict.json").write_text(json.dumps(s, indent=2))
    (args.out / "table.md").write_text(markdown(s) + "\n")
    df.to_csv(args.out / "rows.csv", index=False)
    print(markdown(s))
    print(json.dumps(s["verdicts"], indent=2))


if __name__ == "__main__":
    main()
