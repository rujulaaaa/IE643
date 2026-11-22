#!/usr/bin/env python3
"""Figures of the novelty phase (poster and report: A1 portrait, three columns of 17.7 cm).

Every figure is written as a vector PDF at its final printed size, so the type sizes below are the type sizes on
the paper (body text on the poster is 16.4 pt, captions 13 pt). All numbers come from the project's result files:

    results/analysis/maps.npz                    Experiment 1 selectivity (S_dyn medians, 16 sites x 25 steps)
    results/action_view/sweep_cells.csv           CameraCtrl viewpoint sweep (D-45)
    results/autoresearch/mv_sweep/s*/distances.csv  MVDream sweep (D-46)
    results/autoresearch/{av_patch,av_late,av_matched,av_combo}/*/rows.csv   patching runs (D-45..D-49)

    python scripts/novelty_figures.py            # -> results/action_view/figs/
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, TwoSlopeNorm
from PIL import Image, ImageSequence

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "results")
OUT = os.path.join(R, "action_view", "figs")
os.makedirs(OUT, exist_ok=True)
COLW = 17.7 / 2.54          # column width, inches

INK, NAVY, MUTED, GRID = "#14202E", "#12314F", "#6A7480", "#D6DBE0"
ACT, CAM, TXT, SUBJ, OURS, BAD = "#B4553C", "#2E6F95", "#C9963F", "#6A7480", "#2E6B4F", "#B4553C"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans", "font.size": 12.5, "axes.titlesize": 13.5, "axes.titleweight": "bold",
    "axes.titlepad": 6, "axes.labelsize": 12.5, "axes.labelcolor": INK, "axes.edgecolor": "#9AA3AC",
    "axes.linewidth": 0.9, "axes.grid": False, "xtick.labelsize": 11.2, "ytick.labelsize": 11.2,
    "xtick.color": INK, "ytick.color": INK, "legend.fontsize": 11.2, "legend.frameon": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})

BLOCKS = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3", "mid_block",
          "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"]
SHORT = {b: b.replace("down_blocks.", "down ").replace("up_blocks.", "up ").replace("mid_block", "mid") for b in BLOCKS}


def block_of(c):
    p = c.split(".")
    return p[0] if p[0] == "mid_block" else ".".join(p[:2])


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ Experiment 1 map
def f_exp1():
    # the project's own analysis output (trusted); coords are stored as an object array
    z = np.load(os.path.join(R, "analysis", "maps.npz"), allow_pickle=True)
    coords = [str(c) for c in z["coords"]]
    S = z["S_dyn_median"]
    lab = []
    for c in coords:
        b = SHORT[block_of(c)]
        lab.append(f"{b} · {'temporal' if 'motion_modules' in c else 'cross-attn'}")
    order = sorted(range(len(coords)), key=lambda i: (BLOCKS.index(block_of(coords[i])), "motion" in coords[i]))
    S, lab = S[order], [lab[i] for i in order]
    fig, ax = plt.subplots(figsize=(COLW, 4.9))
    lim = np.nanmax(np.abs(S))
    im = ax.imshow(S, aspect="auto", cmap="RdBu", norm=TwoSlopeNorm(0, -0.25, lim), interpolation="nearest")
    ax.set_yticks(range(len(lab)), lab, fontsize=9.6)
    ax.set_xlabel("denoising step")
    i = lab.index("up 0 · temporal")
    ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 3, 1, fill=False, ec="#C01F1F", lw=2.2))
    ax.annotate("motion-selective:\nup 0 temporal, steps 0–2", (2.6, i), xytext=(8, i - 3.2), fontsize=10.5,
                color="#8E1B1B", arrowprops=dict(arrowstyle="-|>", color="#8E1B1B", lw=1.2))
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.035)
    cb.set_label("selectivity S  (+ content, − motion)", fontsize=10.5)
    cb.ax.tick_params(labelsize=9.6)
    save(fig, "f1_exp1_map.pdf")


# ------------------------------------------------------------------ sensitivity maps (CameraCtrl, MVDream)
def heat3(M, rows, title, name, steps_label="denoising step"):
    fig, axes = plt.subplots(1, 3, figsize=(COLW, 4.0), sharey=True, gridspec_kw={"wspace": 0.08})
    specs = [("action", "Action / subject", "Reds", None), ("camera", "Camera / subject", "Blues", "log"),
             ("view", "Text view / subject", "YlOrBr", None)]
    for ax, (k, t, cm, sc) in zip(axes, specs):
        g = M[k]
        im = ax.imshow(g.values, aspect="auto", cmap=cm, interpolation="nearest",
                       norm=LogNorm(vmin=max(g.values.min(), 0.3), vmax=g.values.max()) if sc and g.values.max() / max(g.values.min(), 0.3) > 10 else None)
        ax.set_title(t, fontsize=12)
        ax.set_xticks([0, 10, 20])
        ax.set_xlabel(steps_label, fontsize=11)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.25, fraction=0.06, aspect=18)
        cb.ax.tick_params(labelsize=9)
    axes[0].set_yticks(range(len(rows)), rows, fontsize=10.2)
    save(fig, name)


def f_cc_maps():
    df = pd.read_csv(os.path.join(R, "action_view", "sweep_cells.csv"))
    df["block"] = df["coord"].map(block_of)
    M = {}
    for k in ("action", "camera", "view"):
        g = df.assign(r=df[k] / df["subject"]).groupby(["block", "step"])["r"].mean().unstack("step")
        M[k] = g.reindex([b for b in BLOCKS if b in g.index])
    heat3(M, [SHORT[b] for b in M["action"].index], "CameraCtrl", "f2_cc_maps.pdf")
    return df


MV_GROUPS = [("in 0–3", [f"input_blocks.{i}" for i in range(4)]), ("in 4–6", [f"input_blocks.{i}" for i in range(4, 7)]),
             ("in 7–9", [f"input_blocks.{i}" for i in range(7, 10)]), ("in 10–11", ["input_blocks.10", "input_blocks.11"]),
             ("middle", ["middle_block"]), ("out 0–2", [f"output_blocks.{i}" for i in range(3)]),
             ("out 3–5", [f"output_blocks.{i}" for i in range(3, 6)]), ("out 6–8", [f"output_blocks.{i}" for i in range(6, 9)]),
             ("out 9–11", [f"output_blocks.{i}" for i in range(9, 12)])]


def f_mv_maps():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "autoresearch", "mv_sweep", "s*", "distances.csv"))])
    m = d.groupby(["site", "step", "kind"])["d"].mean().unstack("kind").reset_index()
    gmap = {s: g for g, ss in MV_GROUPS for s in ss}
    m["grp"] = m["site"].map(gmap)
    M = {}
    for k in ("action", "camera", "view"):
        g = m.assign(r=m[k] / m["subject"]).groupby(["grp", "step"])["r"].mean().unstack("step")
        M[k] = g.reindex([g_ for g_, _ in MV_GROUPS])
    heat3(M, [g for g, _ in MV_GROUPS], "MVDream", "f3_mv_maps.pdf")


# ------------------------------------------------------------------ action vs camera per cell (CameraCtrl)
def f_scatter(df):
    c = df.assign(A=df["action"] / df["subject"], V=df["camera"] / df["subject"])
    fig, ax = plt.subplots(figsize=(COLW * 0.49, 2.75))
    mm = c["coord"].str.contains("motion_modules")
    ax.scatter(c.loc[~mm, "V"], c.loc[~mm, "A"], s=9, c="#B9C2CB", label="cross-attention", lw=0)
    ax.scatter(c.loc[mm, "V"], c.loc[mm, "A"], s=9, c="#7C93A8", label="temporal (motion module)", lw=0)
    e = c[(c["coord"] == "up_blocks.0.motion_modules.2") & (c["step"] <= 2)]
    ax.scatter(e["V"], e["A"], s=60, c=ACT, ec="white", lw=0.8, zorder=5, label="Exp. 1 motion cell")
    ax.set_xscale("log")
    ax.set_xlabel("camera sensitivity  (× subject)")
    ax.set_ylabel("action sensitivity  (× subject)")
    ax.legend(loc="lower right", fontsize=9.4, handletextpad=0.2, borderaxespad=0.3, frameon=True, framealpha=0.95, edgecolor="none")
    ax.grid(True, color=GRID, lw=0.6)
    save(fig, "f4_scatter.pdf")


# ------------------------------------------------------------------ camera leak by strategy
def rows(run):
    return pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(R, "autoresearch", run, "*", "rows.csv"))])


def boot(x, n=10000, seed=643):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    mm = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(mm, 2.5), np.percentile(mm, 97.5)


def f_leak():
    p, l = rows("av_patch"), rows("av_late")
    mov = lambda r: r[r.camera_s != r.camera_t]
    items = [("Exp. 1 site, 0–2", mov(p)[mov(p).candidate == "exp1_region"].leak_cam),
             ("up 1, 0–7", mov(p)[mov(p).candidate == "mm_up_blocks.1"].leak_cam),
             ("all MM, 0–7", mov(l)[mov(l).candidate == "mm_all@0-7"].leak_cam),
             ("all MM, 3–10", mov(l)[mov(l).candidate == "mm_all@3-10"].leak_cam),
             ("attn, 3–24", mov(l)[mov(l).candidate == "attn@3-24"].leak_cam)]
    fig, ax = plt.subplots(figsize=(COLW * 0.49, 2.75))
    ys = np.arange(len(items))[::-1]
    for y, (lab, x) in zip(ys, items):
        m, lo, hi = boot(x)
        ax.barh(y, m, color=CAM, height=0.62)
        ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], color=INK, lw=1, capsize=2.5)
    ax.barh(-1.15, 0.004, color=OURS, height=0.62)
    ax.text(0.03, -1.15, "matched donor: 0", va="center", fontsize=10.2, color=OURS, weight="bold")
    ax.set_yticks(list(ys) + [-1.15], [i[0] for i in items] + [""], fontsize=10.2)
    ax.axvline(0.1, color=MUTED, ls="--", lw=0.9)
    ax.text(0.12, -1.15, "rule ≤ 0.1", fontsize=9.4, color=MUTED, va="center") if False else None
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("donor camera carried over")
    ax.grid(True, axis="x", color=GRID, lw=0.6)
    save(fig, "f5_leak.pdf")


# ------------------------------------------------------------------ transfer vs identity leak
def f_tradeoff():
    t = []
    for run in ("av_matched", "av_combo"):
        r = rows(run)
        for c, g in r.groupby("candidate"):
            if (run == "av_combo" and c in ("mm_up_blocks.1", "mm_up23@0-7", "mm_up123@0-7")) or (run == "av_matched" and c == "mm_up_blocks.3"):
                continue    # duplicates; up-block 3 = full noise-prediction hand-off (not a localisation)
            a, s = boot(g.act_src), boot(g.leak_subj)
            t.append((c, *a, *s))
    t = pd.DataFrame(t, columns=["c", "a", "alo", "ahi", "s", "slo", "shi"])
    names = {"mm_up_blocks.1": "up 1", "mm_up_blocks.2": "up 2", "mm_up_blocks.0": "up 0",
             "mm_up123@0-12": "up 1–3, steps 0–12", "mm_up123_dyn@0-12": "up 1–3 dyn, 0–12",
             "mm_up123_dyn@0-24": "up 1–3 dyn, 0–24", "mm_all_dyn@3-24": "all dyn, 3–24",
             "attn_first_third": "temporal attn", "exp1_region": "Exp. 1 site"}
    fig, ax = plt.subplots(figsize=(COLW, 3.45))
    ax.axvspan(-0.1, 0.1, color=OURS, alpha=0.08, lw=0)
    ax.text(0.0, -0.44, "identity kept", ha="center", fontsize=10.2, color=OURS, weight="bold")
    t = t[~t.c.isin(["mm_up_blocks.3"])]
    for _, x in t.iterrows():
        good = abs(x.s) <= 0.1 and x.alo > 0.1
        col = OURS if good else (BAD if x.s > 0.2 else "#9AA3AC")
        ax.errorbar(x.s, x.a, xerr=[[x.s - x.slo], [x.shi - x.s]], yerr=[[x.a - x.alo], [x.ahi - x.a]], fmt="o",
                    color=col, ms=5 if col != "#9AA3AC" else 3.5, capsize=1.8, lw=1)
    # explicit label positions (data coordinates) so nothing collides
    place = {"mm_up_blocks.1": (0.15, 0.52, "up-block 1"), "mm_up_blocks.2": (0.15, 0.04, "up-block 2"),
             "mm_up123@0-12": (0.40, 0.66, "up 1–3, steps 0–12"), "mm_up123_dyn@0-12": (0.30, 0.14, "up 1–3 dynamic, 0–12"),
             "mm_all_dyn@3-24": (0.52, 0.44, "all modules dynamic, 3–24"),
             "mm_up123_dyn@0-24": (0.60, 0.16, "up 1–3 dynamic, 0–24"), "exp1_region": (0.17, -0.32, "Exp. 1 site")}
    for c, (lx, ly, lab) in place.items():
        row = t[t.c == c]
        if row.empty:
            continue
        x = row.iloc[0]
        col = OURS if c.startswith("mm_up_blocks") else (MUTED if c == "exp1_region" else BAD)
        ax.annotate(lab, (x.s, x.a), xytext=(lx, ly), fontsize=10.4, color=col, va="center",
                    arrowprops=dict(arrowstyle="-", color=col, lw=0.7, shrinkA=0, shrinkB=4))
    ax.axhline(0, color=INK, lw=0.6)
    ax.set_xlim(-0.12, 1.0)
    ax.set_ylim(-0.5, 0.75)
    ax.set_xlabel("identity leak  (0 = target's subject kept, 1 = donor's)")
    ax.set_ylabel("motion transfer\n(0 = target, 1 = donor)")
    ax.grid(True, color=GRID, lw=0.6)
    save(fig, "f6_tradeoff.pdf")


# ------------------------------------------------------------------ frames for the qualitative strips
def frames():
    pair = os.path.join(R, "autoresearch", "av_combo", "zoom-in_to_zoom-in__deer_to_horse__jumping", "gifs")
    for key, g in [("S", "S"), ("T", "T"), ("I", "I"), ("P1", "T_from_S_mm_up_blocks.1"), ("P123", "T_from_S_mm_up123@0-12")]:
        fr = [f.convert("RGB").copy() for f in ImageSequence.Iterator(Image.open(os.path.join(pair, g + ".gif")))]
        for j, i in enumerate((0, 5, 10, 15)):
            fr[i].save(os.path.join(OUT, f"q_{key}_{j}.jpg"), quality=92)
    sheet = Image.open(os.path.join(R, "autoresearch", "mv_matched_sheets", "az0_to_az0__horse_to_tiger__jumping.png"))
    w, h = sheet.size
    rh = h // 5
    for k, name in enumerate(["S", "T", "I", "P"]):
        sheet.crop((150, k * rh, w, (k + 1) * rh)).save(os.path.join(OUT, f"mv_{name}.jpg"), quality=92)
    print("wrote frames")


if __name__ == "__main__":
    f_exp1()
    df = f_cc_maps()
    f_mv_maps()
    f_scatter(df)
    f_leak()
    f_tradeoff()
    frames()
