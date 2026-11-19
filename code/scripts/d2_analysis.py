"""Analysis of Part D2 and the D1 follow-up (D-43): per-head attention, mean ablation, single sites.

    python scripts/d2_analysis.py        -> results/camera/d2/{summary.json, tables.md}

Inputs (downloaded kernels): vdprobe-d2-cc, vdprobe-d2-cam (this phase); vdprobe-cc-patch, vdprobe-cam-patch
(knock-out necessity of the same candidates, D-40 / D-41); vdprobe-cc-atp (AtP fracs). Bootstrap over pairs as
in scripts/camera_verdict.py. No pass rules (D-43): numbers are reported as observations.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from camera_verdict import boot, load_rows

ROOT = Path(__file__).resolve().parents[1]
K = ROOT / "results" / "kaggle"
OUT = ROOT / "results" / "camera" / "d2"
ATP_STEPS = (0, 2, 4)          # the screened steps inside the patch window 0-7 (as attribution_screen.block_sums)


def fmt(b):
    return "–" if b is None or b.get("mean") is None else f"{b['mean']:.2f} [{b['lo']:.2f}, {b['hi']:.2f}]"


def mode_rows(df, mode):
    if mode == "heads":
        return df[df.candidate.str.startswith("attn_")]
    if mode == "sites":
        return df[df.candidate.str.startswith("site_")]
    return df[df.direction == "mean_abl"]


def per_candidate(df, rng):
    out = {}
    for cand, g in df.groupby("candidate", sort=False):
        e = out.setdefault(cand, {})
        for d, h in g.groupby("direction"):
            e[d] = {"eff_cam": boot(h.eff_cam, rng), "median": float(h.eff_cam.median())}
            if d == "transfer":
                e[d]["eff_content"] = boot(h.eff_content, rng)
    return out


def heads(df, rng):
    t = per_candidate(mode_rows(df, "heads"), rng)
    heads_ = sorted([c for c in t if c.startswith("attn_h")], key=lambda c: int(c[6:]))
    allh = t["attn_first_third"]["transfer"]["eff_cam"]["mean"]
    best = max(heads_, key=lambda c: t[c]["transfer"]["eff_cam"]["mean"])
    # per-pair share of the best head (paired), and how many single heads beat all heads together
    w = mode_rows(df, "heads").query("direction == 'transfer'").pivot_table(index="pair", columns="candidate",
                                                                            values="eff_cam")
    n_beat = int(sum(t[c]["transfer"]["eff_cam"]["mean"] > allh for c in heads_))
    med = {c: float(w[c].median()) for c in heads_ + ["attn_first_third"]}
    best_med = max(heads_, key=med.get)
    return {"table": t, "heads": heads_, "best_head": best, "best_head_by_median": best_med,
            "median_share": med[best_med] / med["attn_first_third"] if med["attn_first_third"] else None,
            "best_share_of_all_heads": t[best]["transfer"]["eff_cam"]["mean"] / allh if allh else None,
            "best_minus_all": boot((w[best] - w["attn_first_third"]).values, rng),
            "n_heads_above_all_heads": n_beat}


def ablate(df, knock, rng):
    t = per_candidate(mode_rows(df, "ablate"), rng)
    nec = knock[knock.direction == "nec"]
    rows = {}
    for cand in t:
        m = mode_rows(df, "ablate").query("candidate == @cand").set_index("pair").eff_cam
        k = nec[nec.candidate == cand].set_index("pair").eff_cam
        both = m.index.intersection(k.index)
        rows[cand] = {"mean_ablation": t[cand]["mean_abl"]["eff_cam"], "median_mean": float(m.median()),
                      "knockout_same_pairs": boot(k[both], rng) if len(both) else None,
                      "mean_minus_knockout": boot((m[both] - k[both]).values, rng) if len(both) else None,
                      "n_paired": int(len(both))}
    return rows


def sites(df, atp, rng):
    from scipy.stats import spearmanr
    s = mode_rows(df, "sites")
    t = per_candidate(s, rng)
    keys = {(c, sc) for c, sc in zip(s.camera, s.scene_a)}
    a = atp[atp.step.isin(ATP_STEPS) & [(c, sc) in keys for c, sc in zip(atp.camera, atp.scene)]]
    frac = a.groupby("site").frac.mean()
    out = {}
    for cand, e in t.items():
        site = cand[len("site_"):]
        out[site] = {**e, "atp_frac": float(frac.get(site, np.nan))}
    names = list(out)
    x = [out[n]["atp_frac"] for n in names]
    res = {"sites": out, "n_atp_pairs": int(a.pair.nunique())}
    for d in ("nec", "suff", "transfer"):
        y = [out[n][d]["eff_cam"]["mean"] for n in names]
        res[f"spearman_atp_vs_{d}"] = float(spearmanr(x, y).statistic)
    return res


def md(model, h, ab, st) -> str:
    L = [f"### {model}", "", "**Per-head temporal attention (first third), B2<-A transfer and B<-A sufficiency**", "",
         "| arm | sufficiency | transfer | content leak | median transfer |", "|---|---|---|---|---|"]
    for c, e in h["table"].items():
        L.append(f"| {c} | {fmt(e.get('suff', {}).get('eff_cam'))} | {fmt(e['transfer']['eff_cam'])} | "
                 f"{fmt(e['transfer']['eff_content'])} | {e['transfer']['median']:.2f} |")
    L += ["", f"- best head {h['best_head']}: share of all-heads transfer {h['best_share_of_all_heads']:.2f}; "
              f"best − all (paired) {fmt(h['best_minus_all'])}; heads above all-heads: {h['n_heads_above_all_heads']}"
              f"/{len(h['heads'])}",
          f"- by medians: best head {h['best_head_by_median']}, share of the all-heads median "
              f"{h['median_share']:.2f}", "",
          "**Mean ablation vs knock-out (A<-B, D-40 / D-41 runs, same pairs)**", "",
          "| candidate | mean ablation | median | knock-out | mean − knock-out | n |", "|---|---|---|---|---|---|"]
    for c, e in ab.items():
        L.append(f"| {c} | {fmt(e['mean_ablation'])} | {e['median_mean']:.2f} | {fmt(e['knockout_same_pairs'])} | "
                 f"{fmt(e['mean_minus_knockout'])} | {e['n_paired']} |")
    if st:
        L += ["", "**Single sites (D1 follow-up)**", "",
              "| site | AtP frac | sufficiency | transfer | leak | necessity |", "|---|---|---|---|---|---|"]
        for s, e in st["sites"].items():
            L.append(f"| {s} | {e['atp_frac']:.3f} | {fmt(e['suff']['eff_cam'])} | {fmt(e['transfer']['eff_cam'])} | "
                     f"{fmt(e['transfer']['eff_content'])} | {fmt(e['nec']['eff_cam'])} |")
        L += ["", f"- Spearman AtP frac vs necessity {st['spearman_atp_vs_nec']:.2f}, vs sufficiency "
                  f"{st['spearman_atp_vs_suff']:.2f}, vs transfer {st['spearman_atp_vs_transfer']:.2f} "
                  f"({len(st['sites'])} sites; AtP over {st['n_atp_pairs']} matching pairs)"]
    return "\n".join(L) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(643)
    atp = pd.concat([pd.read_csv(f) for f in (K / "vdprobe-cc-atp").rglob("atp.csv")])
    summary, text = {}, []
    for model, new, old in (("cc", "vdprobe-d2-cc", "vdprobe-cc-patch"), ("lora", "vdprobe-d2-cam", "vdprobe-cam-patch")):
        df, knock = load_rows([K / new]), load_rows([K / old])
        h, ab = heads(df, rng), ablate(df, knock, rng)
        st = sites(df, atp, rng) if model == "cc" else None
        summary[model] = {"n_pairs": {m: int(mode_rows(df, m).pair.nunique()) for m in ("heads", "ablate", "sites")},
                          "heads": h, "ablate": ab, "sites": st}
        text.append(md(model, h, ab, st))
        if model == "lora":     # post hoc, as for HC3: the city-street pairs (smallest camera effects) drive the CIs
            city = df.scene_a.str.startswith("a city street")
            hp = heads(df[~city], rng)
            summary["lora_posthoc_no_city"] = {"n_pairs": int(df[~city].pair.nunique()), "heads": hp}
            text.append("Post hoc, not part of D-43: without the city-street pairs\n\n" +
                        md("lora (no city street)", hp, ablate(df[~city], knock, rng), None))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "tables.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
