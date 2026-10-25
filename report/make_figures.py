"""Report figures that are not already produced by code/scripts/sweep_figures.py.

    cd ~/IE643_CourseProject_Prep/code && env -u PYTHONPATH .venv/bin/python ../report/make_figures.py

Writes report/figures/qual_<pair>.png (frame strips of one test pair, both directions,
with the per-clip scores from scores_D.csv) and report/figures/lora_val.png.
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageSequence

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = ROOT / "report" / "figures"
TEST = CODE / "results" / "kaggle" / "vdprobe-test" / "out"
FRAMES = (0, 5, 10, 15)

ROWS = {
    "motion": [("source", "A (source)"), ("B_ref", "B (reference)"), ("motion.base", "prompt only"),
               ("motion.sel", "x̃ patch, up3, steps 0-8"), ("motion.attn", "attention injection")],
    "content": [("source", "A (source)"), ("B_ref", "B (reference)"), ("content.base", "prompt only"),
                ("content.sel", "x̄ patch, down3, steps 0-8")],
}


def frames(gif: Path) -> list[np.ndarray]:
    return [np.asarray(f.convert("RGB")) for f in ImageSequence.Iterator(Image.open(gif))]


def qual(pair: str, direction: str, scores: pd.DataFrame):
    d = next(TEST.glob(f"job*/{pair}"))
    rows = ROWS[direction]
    fig, axes = plt.subplots(len(rows), len(FRAMES), figsize=(1.55 * len(FRAMES) + 1.9, 1.55 * len(rows)))
    s = scores[scores.pair == pair].set_index("arm")
    for i, (arm, label) in enumerate(rows):
        fr = frames(d / "gifs" / f"{arm}.gif")
        for j, k in enumerate(FRAMES):
            ax = axes[i, j]
            ax.imshow(fr[k])
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(f"frame {k}", fontsize=8)
        txt = label
        if arm in s.index and arm not in ("source", "B_ref"):
            r = s.loc[arm]
            txt += f"\ns_m(A) {r.sm_A:.2f}  s_c(B) {r.sc_B:.2f}" if direction == "motion" else \
                   f"\ns_c(A) {r.sc_A:.2f}  s_m(B) {r.sm_B:.2f}"
            txt += f"\nD {r.D:.3f}"
        axes[i, 0].set_ylabel(txt, fontsize=7.5, rotation=0, ha="right", va="center", labelpad=6)
    fig.subplots_adjust(left=0.21, right=0.99, top=0.95, bottom=0.01, wspace=0.03, hspace=0.06)
    fig.savefig(OUT / f"qual_{direction}_{pair.split('__', 1)[0]}.png", dpi=200)
    plt.close(fig)


def lora_val():
    """Held-out validation loss over training, old recipe (laptop) and cut-free (Kaggle, 3 seeds)."""
    colors = {"full": "#2a6fdb", "probe": "#e8692f", "anti": "#6b7280"}
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for v in colors:
        old = json.loads((CODE / "results/local/motion_lora" / v / "train_log.json").read_text())
        st, lo = zip(*old["val"])
        ax.plot(st, lo, color=colors[v], ls="--", lw=1.2, label=f"{v}, original windows")
        runs = sorted((CODE / "results/kaggle/vdprobe-lora/out").glob(f"job*/{v}_s*/{v}/train_log.json"))
        curves = [json.loads(p.read_text())["val"] for p in runs]
        if curves:
            st = [c[0] for c in curves[0]]
            lo = np.array([[c[1] for c in cu] for cu in curves])
            ax.plot(st, lo.mean(0), color=colors[v], lw=1.8, label=f"{v}, cut-free ({len(curves)} seeds)")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out loss (ε-MSE)")
    ax.set_ylim(0.060, 0.072)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2, frameon=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "lora_val.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(TEST / "scores_D.csv")
    pair = "42__on-a-beach__golden-retriever-running__horse-walking"
    qual(pair, "motion", scores)
    qual(pair, "content", scores)
    lora_val()


if __name__ == "__main__":
    main()
