"""Figures for the README from the result JSONs. Blue = content, orange = motion, as in the deck.

    python scripts/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
# Deck palette (deck_src/theme.py), validated colour-blind-safe on light and dark surfaces.
CONTENT, MOTION = "#2A78D6", "#EB6834"
INK, INK2, RULE = "#101820", "#55636E", "#DDE4E9"

plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white"})


def e2_pixel(sources: dict[str, Path]):
    """d_c and d_m, noun swaps vs verb swaps, Kaggle record beside the local rerun."""
    fig, axes = plt.subplots(1, len(sources), figsize=(3.4 * len(sources), 2.9), sharey=True, squeeze=False)
    for ax, (label, path) in zip(axes[0], sources.items()):
        rows = json.loads(path.read_text())["rows"]
        x = np.arange(2)
        for k, (prefix, color, name) in enumerate((("noun", CONTENT, "noun swap (motion fixed)"),
                                                   ("verb", MOTION, "verb swap (subject fixed)"))):
            rs = [r for r in rows if r["clip"].startswith(prefix)]
            means = [np.mean([r["d_c"] for r in rs]), np.mean([r["d_m"] for r in rs])]
            pos = x + (k - 0.5) * 0.36
            ax.bar(pos, means, width=0.32, color=color, label=name, zorder=2)
            for j, key in enumerate(("d_c", "d_m")):
                ax.scatter([pos[j]] * len(rs), [r[key] for r in rs], s=12, color="white",
                           edgecolor=INK, linewidth=0.6, zorder=3)
            for j, m in enumerate(means):
                ax.text(pos[j], 0.012, f"{m:.3f}", ha="center", va="bottom", fontsize=7.5, color="white", zorder=4)
        ax.set_xticks(x, ["d_c  appearance\n(CLIP)", "d_m  motion proxy\n(pixel frame diffs)"])
        noun = [r for r in rows if r["clip"].startswith("noun")]
        verb = [r for r in rows if r["clip"].startswith("verb")]
        ratio = np.mean([r["d_c"] for r in noun]) / np.mean([r["d_c"] for r in verb])
        ax.set_title(f"{label}\nd_c noun/verb = {ratio:.1f}×", fontsize=9, color=INK, loc="left")
        ax.grid(axis="y", color=RULE, linewidth=0.6, zorder=0)
        ax.set_ylim(0, 0.62)
    axes[0][0].set_ylabel("distance to base clip (1 − cos)")
    axes[0][-1].legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.suptitle("E2 · one-word swaps under a fixed noise tensor: appearance vs. the pixel motion proxy",
                 fontsize=9.5, x=0.02, ha="left", color=INK, y=1.04)
    fig.savefig(OUT / "e2_pixel_confound.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = {"Kaggle T4 (record)": ROOT / "results/kaggle_t4/E2_result.json"}
    local = ROOT / "results/local/E2/E2_result.json"
    if local.exists():
        sources["RTX 4060 Laptop (local rerun)"] = local
    e2_pixel(sources)
    print("figures in", OUT)


if __name__ == "__main__":
    main()
