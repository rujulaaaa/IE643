"""Slide-sized figures that the report's figures do not cover.

    cd ~/IE643_CourseProject_Prep/code && env -u PYTHONPATH .venv/bin/python ../slides/make_slide_figures.py

Writes slides/figures/qual_wide_42.png: test pair 1 in both directions, one column per clip and
one row per frame, so the strip fits a 16:9 slide (the report's version is one row per clip).
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "code" / "results" / "kaggle" / "vdprobe-test" / "out"
OUT = ROOT / "slides" / "figures"
PAIR = "42__on-a-beach__golden-retriever-running__horse-walking"
FRAMES = (0, 8, 15)

PANELS = [
    ("Keep the dog's motion, make it a horse", "motion", [
        ("source", "A: dog running"), ("motion.base", "prompt only\n\"a horse running\""),
        ("motion.sel", r"$\tilde{x}$ patch" "\nup3, steps 0-8"), ("motion.attn", "attention\ninjection")]),
    ("Keep the dog, make it walk", "content", [
        ("source", "A: dog running"), ("content.base", "prompt only\n\"a golden retriever walking\""),
        ("content.sel", r"$\bar{x}$ patch" "\ndown3, steps 0-8")]),
]


def frames(gif: Path) -> list[np.ndarray]:
    return [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(Image.open(gif))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = next(TEST.glob(f"job*/{PAIR}"))
    s = pd.read_csv(TEST / "scores_D.csv")
    s = s[s.pair == PAIR].set_index("arm")
    ncol = sum(len(p[2]) for p in PANELS) + 1          # one spacer column between the panels
    fig = plt.figure(figsize=(1.55 * ncol, 1.55 * len(FRAMES) + 0.8))
    gs = GridSpec(len(FRAMES), ncol, figure=fig, width_ratios=[1] * 4 + [0.18] + [1] * 3,
                  left=0.035, right=0.995, top=0.82, bottom=0.01, wspace=0.04, hspace=0.04)
    col = 0
    for title, direction, arms in PANELS:
        for arm, label in arms:
            fr = frames(d / "gifs" / f"{arm}.gif")
            for i, k in enumerate(FRAMES):
                ax = fig.add_subplot(gs[i, col])
                ax.imshow(fr[k])
                ax.set_xticks([])
                ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_visible(False)
                if col == 0:
                    ax.set_ylabel(f"frame {k}", fontsize=8)
                if i == 0:
                    txt = label
                    if arm != "source":
                        r = s.loc[arm]
                        txt += (f"\nflow to A {r.sm_A:.2f}" if direction == "motion"
                                else f"\nCLIP to A {r.sc_A:.2f}")
                    ax.set_title(txt, fontsize=7.5, pad=3)
            col += 1
        x0 = fig.axes[-len(FRAMES) * len(arms)].get_position().x0
        x1 = fig.axes[-1].get_position().x1
        fig.text((x0 + x1) / 2, 0.985, title, ha="center", va="top", fontsize=9.5, weight="bold")
        col += 1                                        # skip the spacer
    fig.savefig(OUT / "qual_wide_42.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
