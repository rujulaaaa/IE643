"""Verdict of D-44 (HR1-HR3): Part A per-head / per-block temporal attention transfer on new scenes.

    python scripts/d44_verdict.py [--root results/kaggle/vdprobe-d44-cam] [--out results/camera/d44]

Rules (docs/protocol.md, D-44), all pairs, drop only |d| <= 1e-4 (rows with NaN eff_cam), 95 % pair bootstrap:
    HR1  transfer(attn_first_third) CI lo > 0.3  and  leak CI hi < 0.1
    HR2  per pair transfer(attn_down_blocks.2) - 0.5 transfer(attn_first_third): CI lo > 0
    HR3  the same with attn_h6
Also writes the knock-out (necessity mode) table for the original pairs next to the D-43 mean ablation.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from camera_verdict import boot, load_rows

ROOT = Path(__file__).resolve().parents[1]
ALL = "attn_first_third"


def fmt(b):
    return "–" if b is None or b.get("mean") is None else f"{b['mean']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]"


def verdict(df, rng) -> dict:
    t = df[df.direction == "transfer"]
    w = t.pivot_table(index="pair", columns="candidate", values="eff_cam")
    leak = t[t.candidate == ALL].eff_content
    allh = boot(w[ALL], rng)
    out = {"n_pairs": int(w[ALL].notna().sum()), "all_heads": allh, "leak": boot(leak, rng),
           "median_all": float(w[ALL].median())}
    out["HR1"] = bool(allh["lo"] > 0.3 and out["leak"]["hi"] < 0.1)
    for h, cand in (("HR2", "attn_down_blocks.2"), ("HR3", "attn_h6")):
        diff = boot((w[cand] - 0.5 * w[ALL]).values, rng)
        out[h] = bool(diff["lo"] > 0)
        out[h + "_detail"] = {"candidate": cand, "transfer": boot(w[cand], rng), "minus_half_all": diff,
                              "median": float(w[cand].median())}
    out["per_candidate"] = {c: {"transfer": boot(w[c], rng), "median": float(w[c].median()),
                                "leak": boot(t[t.candidate == c].eff_content, rng)} for c in w.columns}
    sc = t[t.candidate == ALL].assign(scene=lambda d: d.scene_a).groupby("scene").eff_cam
    out["per_scene_all_heads"] = {k: float(v) for k, v in sc.mean().items()}
    return out


def knockout(df, rng) -> dict:
    n = df[df.direction == "nec"]
    return {c: {"knockout": boot(g.eff_cam, rng), "median": float(g.eff_cam.median()), "n": int(g.pair.nunique())}
            for c, g in n.groupby("candidate", sort=False)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT / "results/kaggle/vdprobe-d44-cam")
    p.add_argument("--out", type=Path, default=ROOT / "results/camera/d44")
    args = p.parse_args()
    rng = np.random.default_rng(644)
    df = load_rows([args.root])
    heads = df[df.candidate.str.startswith("attn_")]
    v = verdict(heads, rng)
    ko = knockout(df[~df.candidate.str.startswith("attn_")], rng)
    L = [f"**D-44 verdict** (n = {v['n_pairs']} pairs, scenes {', '.join(v['per_scene_all_heads'])})", "",
         f"- HR1 {'PASS' if v['HR1'] else 'FAIL'}: all-heads transfer {fmt(v['all_heads'])} (median "
         f"{v['median_all']:.2f}), leak {fmt(v['leak'])}"]
    for h in ("HR2", "HR3"):
        d = v[h + "_detail"]
        L.append(f"- {h} {'PASS' if v[h] else 'FAIL'}: {d['candidate']} transfer {fmt(d['transfer'])} (median "
                 f"{d['median']:.2f}); minus half of all heads {fmt(d['minus_half_all'])}")
    L += ["", "| arm | transfer | median | leak |", "|---|---|---|---|"]
    L += [f"| {c} | {fmt(e['transfer'])} | {e['median']:.2f} | {fmt(e['leak'])} |" for c, e in v["per_candidate"].items()]
    L += ["", "Per-scene mean all-heads transfer: " + ", ".join(f"{k} {x:.2f}" for k, x in v["per_scene_all_heads"].items())]
    if ko:
        L += ["", "**Knock-out (A<-B), original Part A pairs**", "", "| candidate | knock-out | median | n |",
              "|---|---|---|---|"] + [f"| {c} | {fmt(e['knockout'])} | {e['median']:.2f} | {e['n']} |" for c, e in ko.items()]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verdict.json").write_text(json.dumps({"verdict": v, "knockout": ko}, indent=1))
    (args.out / "verdict.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
