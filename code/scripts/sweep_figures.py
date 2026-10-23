"""Stage 4 figures from the probing-sweep analysis (scripts/analyze_sweep.py outputs).

Blue = content, orange = motion, as in the deck and make_figures.py. Sign convention of S
(roadmap §5.2): S > 0 means a content swap moves the coordinate more than a motion swap
(content-selective), S < 0 the reverse.

    python scripts/sweep_figures.py results/analysis

Writes to results/figures/:
  sweep_selectivity_maps.png   S_dyn and S_stat, 16 coordinates x 25 steps, significant cells dotted
  sweep_h1_blocks.png          M(b) per temporal block with its 95% bootstrap CI (H1)
  sweep_h2_timestep.png        m(t) on the H1 block with the changepoint and its CI (H2)
  sweep_static_energy.png      static-energy share F||x̄||²/||x||² per coordinate (H3 context)
  sweep_probes.png             scene-split linear-probe accuracy over steps, x̃ vs x̄ (D-21)
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
CONTENT, MOTION = "#2A78D6", "#EB6834"
INK, INK2, INK3, RULE, MID = "#101820", "#55636E", "#8493A0", "#DDE4E9", "#F0EFEC"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white"})

# diverging: motion pole, neutral grey midpoint, content pole (equal steps per arm)
DIVERGING = LinearSegmentedColormap.from_list("motion_content", [MOTION, "#F5B89C", MID, "#9EC5F4", CONTENT])
BLOCKS = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3", "mid_block",
          "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"]
SHORT = {b: b.replace("_blocks.", "").replace("_block", "") for b in BLOCKS}


def selectivity_maps(maps, coords, out):
    labels = [c["short"] for c in coords]
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.4)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    for ax, key, title in zip(axes, ("dyn", "stat"), ("S_dyn  (frame-varying part x̃)", "S_stat  (frame mean x̄)")):
        med, lo, hi = maps[f"S_{key}_median"], maps[f"S_{key}_lo"], maps[f"S_{key}_hi"]
        im = ax.imshow(med, cmap=DIVERGING, norm=norm, aspect="auto", interpolation="nearest")
        # 95% bootstrap CI excludes 0: hollow ring = content-significant, filled = motion-significant
        r, c = np.where(lo > 0)
        ax.scatter(c, r, s=6, facecolor="none", edgecolor="white", linewidth=0.5)
        r, c = np.where(hi < 0)
        ax.scatter(c, r, s=14, color=INK, edgecolor="white", linewidth=0.6)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlabel("denoising step (0 = pure noise)")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cb.set_ticks([-0.1, -0.05, 0, 0.1, 0.2, 0.3, 0.4])
    cb.set_label("median S   (< 0 motion-selective,  > 0 content-selective)", color=INK2)
    cb.outline.set_visible(False)
    fig.text(0.0, -0.02, "Black dots: 95% CI below 0 (motion-selective).  White rings: CI above 0 "
             "(content-selective).  Colour scale has unequal arms (−0.1 to 0.4) so the small motion "
             "values stay visible.", fontsize=7, color=INK3)
    fig.savefig(out / "sweep_selectivity_maps.png")
    plt.close(fig)


def h1_blocks(h1, out):
    m = np.array([h1["M_by_block"][b] for b in BLOCKS])
    ci = np.array([h1["M_ci"][b] for b in BLOCKS])
    x = np.arange(len(BLOCKS))
    fig, ax = plt.subplots(figsize=(5.6, 2.9))
    ax.axhline(0, color=INK3, linewidth=0.8, zorder=1)
    colors = [MOTION if v > 0 else CONTENT for v in m]
    ax.vlines(x, ci[:, 0], ci[:, 1], color=colors, linewidth=2, zorder=2)
    ax.scatter(x, m, s=40, color=colors, edgecolor="white", linewidth=1.5, zorder=3)
    best = int(np.argmax(m))
    ax.annotate(f"{SHORT[BLOCKS[best]]}: argmax in {h1['bootstrap_argmax_counts'][BLOCKS[best]]}/1000 resamples",
                (best, m[best]), xytext=(8, 14), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[b] for b in BLOCKS])
    ax.set_ylabel("M(b) = − mean_t median S_dyn")
    ax.text(-0.4, ax.get_ylim()[1], "motion-leaning ↑", fontsize=7, color=INK3, va="top")
    ax.text(-0.4, ax.get_ylim()[0], "content-leaning ↓", fontsize=7, color=INK3, va="bottom")
    fig.savefig(out / "sweep_h1_blocks.png")
    plt.close(fig)


def h2_timestep(h2, out):
    m = np.array(h2["m_t"])
    t = np.arange(len(m))
    lo, hi = h2["tau_ci"]
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    ax.axvspan(lo - 0.5, hi - 0.5, color=RULE, alpha=0.6, linewidth=0, zorder=0)
    ax.axhline(0, color=INK3, linewidth=0.8, zorder=1)
    ax.plot(t, m, color=MOTION, linewidth=2, zorder=2)
    ax.scatter(t, m, s=10, color=MOTION, zorder=3)
    ax.axvline(h2["tau"] - 0.5, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(h2["tau"] - 0.3, m.max() * 0.92, f"changepoint τ = {h2['tau']}  (95% CI {int(lo)}–{int(hi)})",
            fontsize=7.5, color=INK2)
    ax.set_xlabel("denoising step")
    ax.set_ylabel("m(t) = − median S_dyn")
    ax.set_title(h2["coord"].replace("_blocks.", "").replace("motion_modules.", "mm"), loc="left",
                 fontsize=9, color=INK)
    fig.savefig(out / "sweep_h2_timestep.png")
    plt.close(fig)


def static_energy(share, coords, out):
    names = [c["name"] for c in coords if c["name"] in share]
    short = {c["name"]: c["short"] for c in coords}
    kind = {c["name"]: c["kind"] for c in coords}
    v = np.array([share[n] for n in names])
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    temporal = np.array([kind[n] == "temporal_module" for n in names])
    ax.hlines(y, 0, v, color=np.where(temporal, MOTION, CONTENT), linewidth=2)
    ax.scatter(v, y, s=30, color=np.where(temporal, MOTION, CONTENT), edgecolor="white", linewidth=1.2, zorder=3)
    for yi, vi in zip(y, v):
        ax.text(vi + 0.015, yi, f"{vi:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([short[n] for n in names], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("static-energy share  F‖x̄‖² / ‖x‖²")
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], color=MOTION, label="temporal motion module")
    ax.scatter([], [], color=CONTENT, label="spatial cross-attention")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.savefig(out / "sweep_static_energy.png")
    plt.close(fig)


def probes(maps, probes_json, coords, out, show=("mid.mm0", "up0.mm2", "up2.mm2", "mid.sp.a20")):
    labels = [c["short"] for c in coords]
    fig, axes = plt.subplots(1, len(show), figsize=(9.6, 2.4), sharey=True)
    chance = probes_json["verb_tilde"]["chance"]
    for ax, name in zip(axes, show):
        i = labels.index(name)
        ax.axhline(chance, color=INK3, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.plot(maps["probe_verb_tilde"][i], color=MOTION, linewidth=2, label="x̃ (frame-varying)")
        ax.plot(maps["probe_verb_bar"][i], color=CONTENT, linewidth=2, label="x̄ (frame mean)")
        ax.set_title(name, loc="left", fontsize=9, color=INK)
        ax.set_xlabel("step")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("verb accuracy (other scene)")
    axes[0].text(24, chance + 0.02, "chance", fontsize=7, color=INK3, ha="right")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=7.5, loc="lower left", bbox_to_anchor=(0.06, 1.0), ncol=2)
    fig.savefig(out / "sweep_probes.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("analysis", type=Path, nargs="?", default=ROOT / "results" / "analysis")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    a = json.loads((args.analysis / "analysis.json").read_text())
    maps = np.load(args.analysis / "maps.npz", allow_pickle=True)
    selectivity_maps(maps, a["coords"], OUT)
    h1_blocks(a["H1"], OUT)
    h2_timestep(a["H2"], OUT)
    static_energy(a["static_energy_share"], a["coords"], OUT)
    probes(maps, a["probes"], a["coords"], OUT)
    print("wrote", *sorted(p.name for p in OUT.glob("sweep_*.png")))


if __name__ == "__main__":
    main()
