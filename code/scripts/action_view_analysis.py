"""D-45 analysis: sweep sensitivities (HV1) and patching verdict (HV2).

    python scripts/action_view_analysis.py sweep  <sweep out dir>  --out results/action_view
    python scripts/action_view_analysis.py patch  <patch out dir>  --out results/action_view
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EXP1_CELL = ("up_blocks.0.motion_modules.2", (0, 1, 2))


def block_of(coord: str) -> str:
    parts = coord.split(".")
    return parts[0] if parts[0] == "mid_block" else ".".join(parts[:2])


def sweep(dirs, out):
    cfg = yaml.safe_load((ROOT / "configs" / "av_set.yaml").read_text())
    kind_of = {s: k for k, ss in cfg["pair_kinds"].items() for s in ss}
    frames = []
    for d in dirs:
        for g in sorted(Path(d).glob("s*__camera")):
            pairs = json.loads((g / "pairs.json").read_text())
            pk = {p["pair"]: ("camera" if p["kind"] == "motion" else kind_of[p["scene_b"]]) for p in pairs}
            cam = {p["pair"]: p["camera_b"] if p["kind"] == "motion" else p["camera_a"] for p in pairs}
            df = pd.read_csv(g / "distances.csv")
            df["kind"], df["camera"], df["group"] = df["pair"].map(pk), df["pair"].map(cam), g.name
            frames.append(df)
    df = pd.concat(frames)
    res = {}
    for col in ("d_raw", "d_dynamic", "d_static"):
        m = df.groupby(["coord", "step", "kind"])[col].mean().unstack("kind")
        A, Vc, Vt = m["action"] / m["subject"], m["camera"] / m["subject"], m["view"] / m["subject"]
        r_ac, r_at = spearmanr(A, Vc).statistic, spearmanr(A, Vt).statistic
        top = A.sort_values(ascending=False).head(10)
        rank = lambda s: s.rank(ascending=False)
        e1 = [(EXP1_CELL[0], s) for s in EXP1_CELL[1] if (EXP1_CELL[0], s) in A.index]
        blocks = m.assign(A=A, V_cam=Vc, V_txt=Vt).reset_index()
        blocks["block"] = blocks["coord"].map(block_of)
        res[col] = {
            "spearman_A_Vcam": float(r_ac), "spearman_A_Vtxt": float(r_at),
            "spearman_Vcam_Vtxt": float(spearmanr(Vc, Vt).statistic),
            "HV1_pass": bool(r_ac < 0.5 and r_at < 0.5) if col == "d_raw" else None,
            "top10_action_cells": [{"coord": c, "step": int(s), "A": float(A[(c, s)]),
                                    "rank_Vcam": int(rank(Vc)[(c, s)]), "rank_Vtxt": int(rank(Vt)[(c, s)])}
                                   for c, s in top.index],
            "exp1_cell": [{"coord": c, "step": int(s), "A": float(A[(c, s)]), "V_cam": float(Vc[(c, s)]),
                           "V_txt": float(Vt[(c, s)]), "rank_A": int(rank(A)[(c, s)]),
                           "rank_Vcam": int(rank(Vc)[(c, s)]), "n_cells": len(A)} for c, s in e1],
            "blocks": blocks.groupby("block")[["A", "V_cam", "V_txt", "action", "subject", "view", "camera"]]
                            .mean().round(4).to_dict(orient="index"),
            "steps": blocks.groupby("step")[["A", "V_cam", "V_txt"]].mean().round(4).to_dict(orient="index"),
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps(res, indent=1))
    df.groupby(["coord", "step", "kind"])["d_raw"].mean().unstack("kind").to_csv(out / "sweep_cells.csv")
    r = res["d_raw"]
    print(f"HV1 (d_raw): rho(A, V_cam) = {r['spearman_A_Vcam']:.3f}, rho(A, V_txt) = {r['spearman_A_Vtxt']:.3f} -> "
          f"{'pass' if r['HV1_pass'] else 'fail'}")
    for c in ("d_dynamic", "d_static"):
        print(f"  {c}: rho(A, V_cam) = {res[c]['spearman_A_Vcam']:.3f}, rho(A, V_txt) = {res[c]['spearman_A_Vtxt']:.3f}")
    print(pd.DataFrame(r["blocks"]).T[["A", "V_cam", "V_txt"]])
    print("exp1 cell:", r["exp1_cell"])
    plot_sweep(df, out)


def plot_sweep(df, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = df.groupby(["coord", "step", "kind"])["d_raw"].mean().unstack("kind").reset_index()
    m["block"] = m["coord"].map(block_of)
    order = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3", "mid_block",
             "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, (k, lab) in zip(axes, [("action", "action / subject"), ("camera", "camera trajectory / subject"),
                                   ("view", "text viewpoint / subject")]):
        g = m.assign(r=m[k] / m["subject"]).groupby(["block", "step"])["r"].mean().unstack("step")
        g = g.reindex([b for b in order if b in g.index])
        from matplotlib.colors import LogNorm
        im = ax.imshow(g.values, aspect="auto", cmap="viridis", norm=LogNorm() if k == "camera" else None)
        ax.set_title(lab + (" (log)" if k == "camera" else "")), ax.set_xlabel("denoising step")
        ax.set_yticks(range(len(g))), ax.set_yticklabels(g.index)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "sweep_sensitivity.png", dpi=150)


def boot(x, n=10000, seed=643):
    x = np.asarray([v for v in x if np.isfinite(v)])
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(x)


def patch(dirs, out):
    frames = []
    for d in dirs:
        for f in Path(d).glob("*/rows.csv"):
            df = pd.read_csv(f)
            sig = json.loads((f.parent / "sigs.json").read_text())
            t = np.array(sig["T"])
            # camera kept: cosine of the patched clip's camera signature with the target's (moving target cameras)
            df["cam_cos_T"] = [float(np.dot(np.array(sig["T<-S|" + c]), t) / (np.linalg.norm(sig["T<-S|" + c]) * np.linalg.norm(t) + 1e-12))
                               if ct != "off" else float("nan") for c, ct in zip(df.candidate, df.camera_t)]
            frames.append(df)
    rows = pd.concat(frames)
    checks = [json.loads(f.read_text()) for d in dirs for f in Path(d).glob("*/manifest.json")]
    n_fail = sum(not c["passed"] for c in checks)
    table = []
    for cand, g in rows.groupby("candidate"):
        moving = g[g.camera_s != g.camera_t]
        r = {"candidate": cand}
        extra = [("act_src", g)] if "act_src" in g else []
        for col, src in [("act_flow", g), ("act_clip", g), ("leak_subj", g), ("leak_cam", moving), ("cam_cos_T", g)] + extra:
            r[col], r[col + "_lo"], r[col + "_hi"], r[col + "_n"] = boot(src[col])
        r["act_flow_still_cam"] = boot(g[g.camera_s == g.camera_t]["act_flow"])[0]
        r["act_flow_moving_cam"] = boot(moving["act_flow"])[0]
        r["HV2_pass"] = bool(r["act_flow_lo"] > 0.3 and abs(r["leak_subj"]) <= 0.1 and abs(r["leak_cam"]) <= 0.1)
        r["HV4_pass"] = bool(r["act_flow_lo"] > 0.1 and abs(r["leak_subj"]) <= 0.1 and abs(r["leak_cam"]) <= 0.1) if "@" in cand else None
        if "act_src" in g:
            r["act_src_median"] = float(g["act_src"].median())
            r["HV6_pass"] = bool(r["act_src_lo"] > 0.2 and abs(r["leak_subj"]) <= 0.1)
        table.append(r)
    t = pd.DataFrame(table).sort_values("act_src" if "act_src" in rows else "act_flow", ascending=False)
    out.mkdir(parents=True, exist_ok=True)
    tag = "" if out.name == "action_view" else ""
    t.to_csv(out / "patch_verdict.csv", index=False)
    (out / "patch_verdict.json").write_text(json.dumps({"n_pairs": rows["pair"].nunique(), "n_pairs_failing_checks": n_fail,
                                                        "HV2_pass": bool(t["HV2_pass"].any()),
                                                        "table": t.to_dict(orient="records")}, indent=1))
    pd.set_option("display.width", 200)
    print(f"{rows['pair'].nunique()} pairs, {n_fail} failing checks; HV2 {'pass' if t['HV2_pass'].any() else 'fail'}")
    cols = ["candidate", "act_flow", "act_flow_lo", "act_flow_hi", "leak_subj", "leak_cam", "cam_cos_T",
            "act_flow_still_cam", "act_flow_moving_cam"]
    cols += [c for c in ("act_src", "act_src_lo", "act_src_hi", "act_src_median", "HV4_pass", "HV6_pass") if c in t]
    print(t[cols].round(3).to_string(index=False))


def mv_sweep(dirs, out):
    df = pd.concat([pd.read_csv(f) for d in dirs for f in Path(d).glob("s*/distances.csv")])
    m = df.groupby(["site", "step", "kind"])["d"].mean().unstack("kind")
    A, Vc, Vt = m["action"] / m["subject"], m["camera"] / m["subject"], m["view"] / m["subject"]
    r_ac, r_at = spearmanr(A, Vc).statistic, spearmanr(A, Vt).statistic
    sites = list(dict.fromkeys(df["site"]))
    blk = m.assign(A=A, V_cam=Vc, V_txt=Vt).groupby("site")[["A", "V_cam", "V_txt", "action", "subject", "view", "camera"]].mean()
    blk = blk.reindex(sites)
    res = {"spearman_A_Vcam": float(r_ac), "spearman_A_Vtxt": float(r_at),
           "spearman_Vcam_Vtxt": float(spearmanr(Vc, Vt).statistic), "HV1_MV_pass": bool(r_ac < 0.5 and r_at < 0.5),
           "site_rank_spearman_A_Vcam": float(spearmanr(blk["A"], blk["V_cam"]).statistic),
           "top10_action_cells": [{"site": c, "step": int(s_), "A": float(A[(c, s_)]),
                                   "rank_Vcam": int(Vc.rank(ascending=False)[(c, s_)])}
                                  for c, s_ in A.sort_values(ascending=False).head(10).index],
           "sites": blk.round(4).to_dict(orient="index")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "mv_sweep.json").write_text(json.dumps(res, indent=1))
    print(f"HV1-MV: rho(A, V_cam) = {r_ac:.3f}, rho(A, V_txt) = {r_at:.3f} -> {'pass' if res['HV1_MV_pass'] else 'fail'}")
    print(blk[["A", "V_cam", "V_txt"]].round(3).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (s_, lab) in zip(axes, [(A, "action / subject"), (Vc, "camera input / subject"), (Vt, "text viewpoint / subject")]):
        g = s_.unstack("step").reindex(sites)
        im = ax.imshow(g.values, aspect="auto", cmap="viridis")
        ax.set_title(lab), ax.set_xlabel("denoising step")
        ax.set_yticks(range(len(sites))), ax.set_yticklabels(sites, fontsize=7)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "mv_sweep_sensitivity.png", dpi=150)


def mv_patch(dirs, out):
    rows = pd.concat([pd.read_csv(f) for d in dirs for f in Path(d).glob("*/rows.csv")])
    man = [json.loads(f.read_text()) for d in dirs for f in Path(d).glob("*/manifest.json")]
    table = []
    for cand, g in rows.groupby("candidate"):
        moving = g[g.view_s != g.view_t]
        r = {"candidate": cand}
        for col, src in (("act_clip", g), ("leak_subj", g), ("leak_view", moving)):
            r[col], r[col + "_lo"], r[col + "_hi"], r[col + "_n"] = boot(src[col])
        r["act_clip_median"] = float(g["act_clip"].median())
        r["HV3_pass"] = bool(r["act_clip_lo"] > 0.3 and abs(r["leak_subj"]) <= 0.1 and abs(r["leak_view"]) <= 0.1)
        table.append(r)
    t = pd.DataFrame(table).sort_values("act_clip", ascending=False)
    out.mkdir(parents=True, exist_ok=True)
    t.to_csv(out / "mv_patch_verdict.csv", index=False)
    (out / "mv_patch_verdict.json").write_text(json.dumps({"n_pairs": rows["pair"].nunique(),
        "n_pairs_failing_checks": sum(not m_["passed"] for m_ in man), "HV3_pass": bool(t["HV3_pass"].any()),
        "table": t.to_dict(orient="records")}, indent=1))
    print(f"{rows['pair'].nunique()} pairs; HV3 {'pass' if t['HV3_pass'].any() else 'fail'}")
    print(t[["candidate", "act_clip", "act_clip_lo", "act_clip_hi", "act_clip_median", "leak_subj", "leak_view",
             "HV3_pass"]].round(3).to_string(index=False))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("sweep", "patch", "mvsweep", "mvpatch"))
    p.add_argument("dirs", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=ROOT / "results" / "action_view")
    a = p.parse_args()
    {"sweep": sweep, "patch": patch, "mvsweep": mv_sweep, "mvpatch": mv_patch}[a.what](a.dirs, a.out)


if __name__ == "__main__":
    main()
