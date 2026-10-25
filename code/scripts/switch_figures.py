"""Stage 6c (D-38) figures from a scored switch run (score_edits.py on switch_sweep.py outputs).

    python scripts/switch_figures.py <scored run dir> --gifs <dir holding the pair directories> --out results/figures

Writes
  switch_curve.png        what each direction keeps and takes, against the switch step tau (medians over pairs)
  switch_gain.png         D-16 statistic (median D(arm) - median D(prompt only)) with its 95% paired-bootstrap CI
  switch_qualitative.png  frames of one pair: source A, prompt only, the switch arms, attention injection
"""

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CONTENT, MOTION = "#2A78D6", "#EB6834"
INK, INK2, INK3, RULE = "#101820", "#55636E", "#8493A0", "#DDE4E9"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": RULE, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 160, "savefig.bbox": "tight", "savefig.facecolor": "white"})
H2_TAU, H2_CI = 2, (1, 3)


def tau_of(arm: str) -> int | None:
    m = re.search(r"\.t(\d+)$", arm)
    return int(m.group(1)) if m else None


def curve(df: pd.DataFrame, out: Path):
    """Per direction: the kept channel (from A) and the taken channel (from B) against tau."""
    spec = {"motion": ("sm_A", "motion kept from A", "sc_B", "new subject (B)"),
            "content": ("sc_A", "subject kept from A", "sm_B", "new motion (B)")}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, (d, (keep, keep_lbl, take, take_lbl)) in zip(axes, spec.items()):
        g = df[df.arm.str.startswith(f"{d}.switch.t")].copy()
        g["tau"] = g.arm.map(tau_of)
        med = g.groupby("tau")[[keep, take]].median()
        base = df[df.arm == f"{d}.base"][[keep, take]].median()
        ax.axvspan(H2_CI[0], H2_CI[1], color=RULE, alpha=0.6, linewidth=0, zorder=0)
        ax.axvline(H2_TAU, color=INK2, linewidth=1, linestyle=(0, (3, 2)), zorder=1)
        ax.text(H2_TAU + 0.1, 1.0, "H2 changepoint τ = 2\n(95% CI shaded)", fontsize=7, color=INK2, va="top")
        keep_c = MOTION if keep.startswith("sm") else CONTENT
        take_c = CONTENT if keep_c == MOTION else MOTION
        for col, c, lbl, ls, mk in ((keep, keep_c, keep_lbl, "-", "o"), (take, take_c, take_lbl, (0, (4, 2)), "s")):
            ax.plot(med.index, med[col], color=c, linewidth=2, linestyle=ls, marker=mk, markersize=6,
                    markeredgecolor="white", markeredgewidth=1.2, zorder=3)
            ax.axhline(base[col], color=c, linewidth=0.8, linestyle=(0, (1, 2)), zorder=2)
            ax.annotate(lbl, (med.index[-1], med[col].iloc[-1]), xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=INK2, va="center")
        if d == "motion":
            mk = df[df.arm.str.startswith("motion.switchmask.t")].copy()
            mk["tau"] = mk.arm.map(tau_of)
            mm = mk.groupby("tau")[[keep, take]].median()
            for col, c in ((keep, keep_c), (take, take_c)):
                ax.scatter(mm.index, mm[col], s=46, facecolor="white", edgecolor=c, linewidth=1.8, zorder=4)
            ax.scatter([], [], s=46, facecolor="white", edgecolor=INK2, linewidth=1.8, label="masked switch")
            ax.legend(loc="lower right", frameon=False, fontsize=7.5)
        ax.set_xticks(sorted(g.tau.unique()))
        ax.set_xlabel("switch step τ (of 25)")
        ax.set_title({"motion": "Keep A's motion, change the subject", "content": "Keep A's subject, change the motion"}[d],
                     loc="left", fontsize=9, color=INK)
        ax.set_xlim(0.5, max(g.tau) + 2.6)
    axes[0].set_ylabel("median fidelity (cosine)")
    fig.text(0.0, -0.04, "Solid / dashed: switch arms. Dotted horizontal lines: prompt-only baseline. Motion = Farneback "
             "flow cosine; content = CLIP image cosine.", fontsize=7, color=INK3)
    fig.savefig(out / "switch_curve.png")
    plt.close(fig)


NAMES = {"motion.attn": "attention injection, first third (baseline)", "motion.sel": "dynamic-part patch (Stage 6)",
         "content.sel": "static-part patch (Stage 6)"}


def name(arm: str) -> str:
    if arm in NAMES:
        return NAMES[arm]
    t = tau_of(arm)
    if ".switchmask." in arm:
        return f"masked switch, τ = {t}"
    if ".switch." in arm:
        return f"switch, τ = {t}"
    return arm


def gain(summary: pd.DataFrame, out: Path, n_pairs: int):
    s = summary[~summary.arm.str.endswith(".base")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharex=True)
    for ax, d in zip(axes, ("motion", "content")):
        g = s[s.direction == d].sort_values("d16_stat")
        y = np.arange(len(g))
        pre = g.arm.isin([f"{d}.switch.t2", f"{d}.switchmask.t2"])
        c = MOTION if d == "motion" else CONTENT
        ax.axvline(0, color=INK3, linewidth=0.8)
        ax.hlines(y, g.d16_ci_lo, g.d16_ci_hi, color=c, linewidth=2, alpha=0.9)
        ax.scatter(g.d16_stat, y, s=40, color=np.where(pre, INK, c), edgecolor="white", linewidth=1.4, zorder=3)
        ax.set_yticks(y, [name(a) + ("  (pre-registered)" if p else "") for a, p in zip(g.arm, pre)], fontsize=7.5)
        ax.set_title({"motion": "Keep motion, change subject", "content": "Keep subject, change motion"}[d],
                     loc="left", fontsize=9, color=INK)
        ax.set_xlabel("median D(arm) − median D(prompt only), 95% CI")
    fig.text(0.0, -0.03, f"{n_pairs} held-out test pairs. A CI that excludes 0 is D-16's success criterion. Dark dots: "
             "the arms fixed in advance (D-38).", fontsize=7, color=INK3)
    fig.tight_layout()
    fig.savefig(out / "switch_gain.png")
    plt.close(fig)


def frames(gif: Path, idx=(0, 5, 10, 15), size=160):
    im = Image.open(gif)
    out = []
    for i in idx:
        im.seek(min(i, im.n_frames - 1))
        out.append(im.convert("RGB").resize((size, size)))
    return out


def qualitative(pair_dir: Path, out: Path, rows):
    gifs = pair_dir / "gifs"
    rows = [(a, l) for a, l in rows if (gifs / f"{a}.gif").exists()]
    fig, axes = plt.subplots(len(rows), 4, figsize=(6.4, 1.62 * len(rows)))
    for r, (arm, label) in enumerate(rows):
        for c, im in enumerate(frames(gifs / f"{arm}.gif")):
            ax = axes[r, c]
            ax.imshow(im)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if c == 0:
                ax.set_ylabel(label, fontsize=7.5, color=INK, rotation=0, ha="right", va="center", labelpad=6)
            if r == 0:
                ax.set_title(f"frame {(0, 5, 10, 15)[c]}", fontsize=7.5, color=INK2)
    axes[0, 0].text(0, 1.35, pair_dir.name.replace("__", " · ").replace("-", " "), transform=axes[0, 0].transAxes, fontsize=8, color=INK2)
    fig.subplots_adjust(wspace=0.03, hspace=0.06)
    fig.savefig(out / f"switch_qualitative__{pair_dir.name}.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    ap.add_argument("--gifs", type=Path, nargs="*", default=[], help="roots holding the pair directories")
    ap.add_argument("--pairs", nargs="*", default=None, help="pair ids for the qualitative sheet (default: all)")
    ap.add_argument("--out", type=Path, default=Path("results/figures"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.run / "scores_D.csv")
    summary = pd.read_csv(args.run / "summary.csv")
    curve(df, args.out)
    gain(summary, args.out, df.pair.nunique())
    motion_rows = [("source", "A (source)"), ("motion.base", "prompt only"), ("motion.switch.t2", "switch τ=2"),
                   ("motion.switchmask.t2", "masked switch τ=2"), ("motion.switch.t5", "switch τ=5"),
                   ("motion.attn", "attention injection")]
    content_rows = [("source", "A (source)"), ("content.base", "prompt only"), ("content.switch.t2", "switch τ=2"),
                    ("content.switch.t3", "switch τ=3"), ("content.switch.t5", "switch τ=5"), ("B_ref", "B (motion ref.)")]
    dirs = sorted({p.parent for root in args.gifs for p in root.rglob("manifest.json")})
    for d in dirs:
        if args.pairs is None or d.name in args.pairs:
            qualitative(d, args.out, motion_rows)
            (args.out / f"switch_qualitative__{d.name}.png").rename(args.out / f"switch_qual_motion__{d.name}.png")
            qualitative(d, args.out, content_rows)
            (args.out / f"switch_qualitative__{d.name}.png").rename(args.out / f"switch_qual_content__{d.name}.png")
    print("wrote", sorted(p.name for p in args.out.glob("switch_*.png")))


if __name__ == "__main__":
    main()
