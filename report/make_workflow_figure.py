"""Workflow figure for the report (Section 2).

    cd ~/IE643/code && env -u PYTHONPATH .venv/bin/python ../report/make_workflow_figure.py

Extracts frame thumbnails of test pair 1337/forest/athlete-walking/soldier-jumping and renders
the motion-module rows of the S_dyn selectivity map into report/figures/workflow/, then compiles
report/figures/workflow/workflow.tex into report/figures/workflow.pdf.
"""

import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
OUT = ROOT / "report" / "figures" / "workflow"
PAIR = (CODE / "results" / "kaggle" / "vdprobe-lit-test" / "out" / "job1" / "edit"
        / "1337__in-a-forest__athlete-walking__soldier-jumping" / "gifs")
FRAMES = (0, 7, 15)

# clip name in the edit run -> thumbnail prefix
CLIPS = {
    "source": "anchor",                    # an athlete walking
    "motion.base": "subject",              # a soldier walking (subject swapped)
    "content.base": "verb",                # an athlete jumping (verb swapped)
    "B_ref": "b",                          # a soldier jumping
    "motion.dual.up23.from2": "edit",      # B's soldier with A's walk
}

CONTENT = "#2F6DB5"
MOTION = "#E07B39"


def thumbnails():
    for clip, prefix in CLIPS.items():
        gif = Image.open(PAIR / f"{clip}.gif")
        for i, f in enumerate(FRAMES):
            gif.seek(f)
            gif.convert("RGB").resize((160, 160), Image.LANCZOS).save(OUT / f"{prefix}_{i}.png")


def selectivity_map():
    maps = np.load(CODE / "results" / "analysis" / "maps.npz")
    rows = [i for i, c in enumerate(maps["coords"]) if "motion_modules" in c]
    s = maps["S_dyn_median"][rows]  # 9 temporal blocks (down0 ... up3) x 25 steps
    cmap = LinearSegmentedColormap.from_list("cm", [MOTION, "#FFFFFF", CONTENT])
    norm = TwoSlopeNorm(vmin=-0.1, vcenter=0.0, vmax=0.4)
    fig = plt.figure(figsize=(25 / 10, 9 / 10), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(s, cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(OUT / "smap.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    thumbnails()
    selectivity_map()
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "workflow.tex"],
                   cwd=OUT, check=True, stdout=subprocess.DEVNULL)
    shutil.copy(OUT / "workflow.pdf", OUT.parent / "workflow.pdf")


if __name__ == "__main__":
    main()
