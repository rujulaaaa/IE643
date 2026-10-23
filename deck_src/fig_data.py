"""Charts drawn from the measured pilot results, plus the two map figures.

Every number here comes from E1_result.json, E2_result.json, E3_result.json or
section 7 of the roadmap.  The only figure that is not measured is the
predicted selectivity map, and it says so on its face.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle

from figbase import (canvas, save, hx, sqbox, label, arrow, line, filmstrip,
                     brace_h, gait, F, FL, FM, FSB, FBK, FMONO, DPI, _face)
import theme as T

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "fig") + os.sep
SRC = os.path.dirname(_HERE) + os.sep

E1 = json.load(open(SRC + "E1_result.json"))
E2 = json.load(open(SRC + "E2_result.json"))
E3 = json.load(open(SRC + "E3_result.json"))


def axes_style(ax, bottom=True, left=False):
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    if bottom:
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(hx(T.RULE_STRONG))
        ax.spines["bottom"].set_linewidth(0.8)
    if left:
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_color(hx(T.RULE_STRONG))
        ax.spines["left"].set_linewidth(0.8)
    ax.tick_params(colors=hx(T.TX3), length=0, labelsize=8)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily("Lato")
        t.set_color(hx(T.TX3))


def plain_fig(w, h, bg=T.PAPER):
    fig = plt.figure(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor(hx(bg))
    return fig


# ===========================================================================
# E3 : storage
# ===========================================================================

def storage(path=OUT + "storage.png"):
    fig, ax = canvas(5.55, 2.35)

    full = E3["full_dump_gb_per_clip"]
    summ = E3["summary_store_mb_per_clip"] / 1024.0
    disk = 73.0

    x0, xw = 3.0, 68.0
    sc = xw / 100.0
    bh = 3.3

    rows = [("a full fp16 dump", full, T.MOTION, "87.9 GB"),
            ("the summaries we store", summ, T.CONTENT, "44.6 MB")]
    ys = [29.5, 20.8]

    for (name, v, col, lab), y in zip(rows, ys):
        label(ax, x0, y + bh + 1.4, name, size=8.2, font=FSB, color=T.TX2,
              ha="left", va="bottom")
        sqbox(ax, x0, y, max(v * sc, 0.26), bh, fc=col, ec=None, z=4)
        label(ax, x0 + max(v * sc, 0.26) + 1.4, y + bh / 2, lab, size=9.0,
              font=FSB, color=col, ha="left")

    dx = x0 + disk * sc
    line(ax, [dx, dx], [17.6, 35.4], color=T.TX1, lw=1.0, z=6, ls=(0, (2.6, 2.2)))
    label(ax, dx, 36.1, "one Kaggle disk, 73 GB", size=7.4, font=FSB,
          color=T.TX1, va="bottom")

    for g in (0, 25, 50, 75, 100):
        gx = x0 + g * sc
        line(ax, [gx, gx], [19.2, 19.9], color=T.RULE_STRONG, lw=0.8, z=2)
        label(ax, gx, 18.5, str(g), size=7.0, font=F, color=T.TX3, va="top")
    line(ax, [x0, x0 + xw], [19.9, 19.9], color=T.RULE_STRONG, lw=0.8, z=2)
    label(ax, x0 + xw + 2.6, 19.4, "GB per clip", size=7.0, font=F,
          color=T.TX3, ha="left", va="center")

    label(ax, x0, 10.4, "2,017x", size=25, font=FBK, color=T.TX1,
          ha="left", va="center")
    label(ax, x0 + 24.5, 11.7, "smaller, per clip", size=8.6, font=FSB,
          color=T.TX2, ha="left", va="center")
    label(ax, x0 + 24.5, 8.9, "31.6 TB -> 15.7 GB across all 360 clips",
          size=8.0, font=F, color=T.TX3, ha="left", va="center")
    label(ax, x0, 3.0,
          "Measured on one clip: 95 coordinates, 25 timesteps, fp16.",
          size=7.4, font=F, color=T.TX3, ha="left", va="center")

    save(fig, path)
    return path


# ===========================================================================
# E2 : does a fixed seed hold the scene still?
# ===========================================================================

def e2_swaps(path=OUT + "e2_swaps.png"):
    rows = {r["clip"]: r for r in E2["rows"]}
    groups = [("noun swaps", ["cat", "horse", "bird"], "noun"),
              ("verb swaps", ["walking", "jumping", "standing"], "verb")]

    fig = plain_fig(5.6, 2.85)
    ax = fig.add_axes([0.285, 0.145, 0.665, 0.635])
    axes_style(ax, bottom=False)

    ys, names = [], []
    y = 6.0
    marks = []
    for gname, keys, pref in groups:
        gy = []
        for k in keys:
            r = rows[f"{pref}:{k}"]
            ys.append(y); names.append(k); gy.append(y)
            marks.append((y, r["d_c"], r["d_m"]))
            y -= 1.0
        ax.text(-0.128, np.mean(gy), gname.replace(" ", "\n"), fontsize=7.8,
                color=hx(T.TX3), ha="center", va="center", **_face(FSB),
                clip_on=False, linespacing=1.5)
        y -= 0.65

    h = 0.30
    for yy, dc, dm in marks:
        ax.barh(yy + h / 2 + 0.035, dc, height=h, color=hx(T.CONTENT),
                edgecolor="none", zorder=3)
        ax.barh(yy - h / 2 - 0.035, dm, height=h, color=hx(T.MOTION),
                edgecolor="none", zorder=3)
        ax.text(dc + 0.011, yy + h / 2 + 0.035, f"{dc:.2f}", va="center",
                ha="left", fontsize=7.4, color=hx(T.CONTENT), **_face(FSB))
        ax.text(dm + 0.011, yy - h / 2 - 0.035, f"{dm:.2f}", va="center",
                ha="left", fontsize=7.4, color=hx(T.MOTION), **_face(FSB))

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8.4)
    for t in ax.get_yticklabels():
        t.set_fontfamily("Lato"); t.set_color(hx(T.TX1))
    ax.set_xticks([])
    ax.set_xlim(0, 0.60)
    ax.set_ylim(min(ys) - 0.75, max(ys) + 0.75)

    fig.text(0.285, 0.935, "how far each swap moves the clip",
             fontsize=8.8, color=hx(T.TX1), **_face(FSB))
    fig.text(0.285, 0.855, "appearance", fontsize=8.0, color=hx(T.CONTENT),
             **_face(FSB))
    fig.text(0.415, 0.855, "pixel motion proxy", fontsize=8.0,
             color=hx(T.MOTION), **_face(FSB))
    fig.text(0.285, 0.035,
             "Noun swaps move appearance 2.6x more than verb swaps.",
             fontsize=7.6, color=hx(T.TX2), **_face(F))
    save(fig, path)
    return path


def e2_ssim(path=OUT + "e2_ssim.png"):
    rows = {r["clip"]: r for r in E2["rows"]}
    order = [("horse", "same body plan"), ("cat", "same body plan"),
             ("bird", "different body plan")]

    fig = plain_fig(5.6, 2.30)
    ax = fig.add_axes([0.235, 0.215, 0.70, 0.545])
    axes_style(ax, bottom=False)

    vals = [rows["noun:" + k]["layout_ssim"] for k, _ in order]
    y = [2.0, 1.0, 0.0]
    ax.barh(y, vals, height=0.36, color=hx(T.CONTENT), edgecolor="none", zorder=3)
    for yy, v in zip(y, vals):
        ax.text(v + 0.011, yy, f"{v:.3f}", va="center", ha="left", fontsize=8.0,
                color=hx(T.CONTENT), **_face(FSB))
    ax.set_yticks(y)
    ax.set_yticklabels([k for k, _ in order], fontsize=8.6)
    for t in ax.get_yticklabels():
        t.set_fontfamily("Lato"); t.set_color(hx(T.TX1))
    for yy, (_, fam) in zip(y, order):
        ax.text(-0.014, yy - 0.285, fam, ha="right", va="center", fontsize=6.9,
                color=hx(T.TX3), **_face(F), clip_on=False)
    ax.set_xticks([])
    ax.set_xlim(0, 0.60)
    ax.set_ylim(-0.62, 2.62)

    fig.text(0.235, 0.895, "how much of the layout survives the swap",
             fontsize=8.8, color=hx(T.TX1), **_face(FSB))
    fig.text(0.235, 0.815, "layout SSIM against the retriever clip",
             fontsize=7.6, color=hx(T.TX3), **_face(F))
    fig.text(0.235, 0.045,
             "Layout survives only within a body plan - which is why subjects "
             "are split\ninto two families and cross-family pairs are reported "
             "separately.",
             fontsize=7.6, color=hx(T.TX2), **_face(F), linespacing=1.55)
    save(fig, path)
    return path


# ===========================================================================
# the layer x timestep plane
# ===========================================================================

COORDS = [
    ("up 3", "s"), ("up 2", "s"), ("up 1", "s"), ("mid", "s"),
    ("down 2", "s"), ("down 1", "s"), ("down 0", "s"),
    ("up 3", "t"), ("up 2", "t"), ("up 1", "t"), ("up 0", "t"), ("mid", "t"),
    ("down 3", "t"), ("down 2", "t"), ("down 1", "t"), ("down 0", "t"),
]

DIVERGING = LinearSegmentedColormap.from_list(
    "content_motion",
    [hx(T.MOTION), "#F4A984", "#F0EFEC", "#8FB9EC", hx(T.CONTENT)], N=256)


def _predicted_field(rng):
    """The pattern H1-H3 predict. Not a measurement; the figure says so."""
    nT = 25
    t = np.linspace(0, 1, nT)                    # 0 = noisy, 1 = clean
    field = np.zeros((len(COORDS), nT))
    # depth of each temporal coordinate, 0 = shallow encoder .. 1 = shallow decoder
    peak = {"up 1": 1.0, "mid": 0.72, "up 0": 0.86, "up 2": 0.62,
            "down 3": 0.55, "down 2": 0.42, "up 3": 0.30,
            "down 1": 0.26, "down 0": 0.16}
    for i, (blk, kind) in enumerate(COORDS):
        if kind == "t":
            strength = peak.get(blk, 0.3)
            # motion settles early: most negative at low t
            prof = -0.92 * strength * (1.0 - 1.0 / (1.0 + np.exp(-(t - 0.38) * 7.5)))
            prof += 0.22 * (1 - strength) * t
        else:
            # content settles late
            prof = 0.85 / (1.0 + np.exp(-(t - 0.30) * 6.0)) - 0.12
            prof *= 0.75 + 0.25 * (blk in ("up 1", "up 2", "down 2"))
        field[i] = prof
    field += rng.normal(0, 0.035, field.shape)
    return np.clip(field, -1, 1)


def selectivity_map(path=OUT + "selectivity.png"):
    rng = np.random.default_rng(643)
    field = _predicted_field(rng)
    nT = field.shape[1]

    fig = plain_fig(7.1, 4.05)
    ax = fig.add_axes([0.205, 0.175, 0.665, 0.715])
    im = ax.imshow(field, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto",
                   interpolation="nearest")

    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["t=25", "20", "13", "7", "t=1"])
    ax.set_yticks(range(len(COORDS)))
    ax.set_yticklabels([f"{b}  {'spatial' if k=='s' else 'temporal'}"
                        for b, k in COORDS])
    axes_style(ax, bottom=False)
    for t_ in ax.get_yticklabels():
        t_.set_fontsize(7.4)
    for t_ in ax.get_xticklabels():
        t_.set_fontsize(7.4)
    for i, (b, k) in enumerate(COORDS):
        ax.get_yticklabels()[i].set_color(
            hx(T.CONTENT) if k == "s" else hx(T.MOTION))

    # separate the spatial block from the temporal block
    ax.axhline(6.5, color=hx(T.PAPER), lw=2.4)
    ax.set_xlim(-0.5, nT - 0.5)
    fig.text(0.205, 0.108, "noisy", fontsize=7.0, color=hx(T.TX3),
             **_face(F), ha="left")
    fig.text(0.870, 0.108, "clean", fontsize=7.0, color=hx(T.TX3),
             **_face(F), ha="right")

    # colour key
    cax = fig.add_axes([0.885, 0.175, 0.021, 0.715])
    cb = fig.colorbar(im, cax=cax, ticks=[-1, 0, 1])
    cb.outline.set_visible(False)
    cax.set_yticklabels(["motion-\nselective", "neither", "content-\nselective"],
                        fontsize=7.2)
    for t_ in cax.get_yticklabels():
        t_.set_fontfamily("Lato")
    cax.get_yticklabels()[0].set_color(hx(T.MOTION))
    cax.get_yticklabels()[2].set_color(hx(T.CONTENT))
    cax.get_yticklabels()[1].set_color(hx(T.TX3))
    cax.tick_params(length=0, pad=5)

    fig.text(0.205, 0.940, "selectivity  S(L, t),  as H1–H3 predict it",
             fontsize=9.2, color=hx(T.TX1), **_face(FSB))
    fig.text(0.205, 0.030,
             "Drawn from the predictions, not measured. Stage 4 replaces every "
             "cell with a real value.",
             fontsize=7.2, color=hx(T.TX3), **_face(F))
    save(fig, path)
    return path


def prior_work_map(path=OUT + "prior_work.png"):
    fig, ax = canvas(6.4, 3.30)

    px, py, pw, ph = 20.0, 17.0, 76.0, 27.0
    sqbox(ax, px, py, pw, ph, fc=T.PANEL_2, ec=None, z=1)

    band_y, band_h = py + 0.40 * ph, 0.24 * ph
    sqbox(ax, px, band_y, pw, band_h, fc=T.MOTION_WASH, ec=None, z=2)
    band_x = 0.34 * pw
    sqbox(ax, px, py, band_x, ph, fc=T.MOTION_WASH, ec=None, z=2)
    sqbox(ax, px, band_y, band_x, band_h, fc=T.MOTION, ec=None, z=3)
    label(ax, px + band_x / 2, band_y + band_h / 2, "look here first",
          size=7.6, font=FSB, color="FFFFFF", z=6)

    line(ax, [px, px + pw], [py, py], color=T.RULE_STRONG, lw=0.9, z=4)
    line(ax, [px, px], [py, py + ph], color=T.RULE_STRONG, lw=0.9, z=4)

    # the layer constraint, annotated inside its own band
    label(ax, px + pw - 1.6, band_y + band_h / 2 + 1.5,
          "motion reads cleanest just after the first up-block",
          size=7.0, font=FSB, color=T.MOTION, ha="right", va="center", z=6)
    label(ax, px + pw - 1.6, band_y + band_h / 2 - 1.6,
          "Xiao et al., NeurIPS 2024", size=6.6, font=F, color=T.MOTION,
          ha="right", va="center", z=6)

    # the timestep constraint, above its band
    label(ax, px + band_x / 2, py + ph + 4.6, "motion settles in the first third",
          size=7.0, font=FSB, color=T.MOTION, va="center", z=6)
    label(ax, px + band_x / 2, py + ph + 1.9, "Baherwani et al., arXiv 2025",
          size=6.6, font=F, color=T.MOTION, va="center", z=6)

    label(ax, px, py - 2.2, "noisy", size=6.8, font=F, color=T.TX3,
          ha="left", va="center")
    label(ax, px + pw, py - 2.2, "clean", size=6.8, font=F, color=T.TX3,
          ha="right", va="center")
    label(ax, px + pw / 2, py - 2.2, "denoising timestep", size=7.2, font=FSB,
          color=T.TX2, va="center")
    label(ax, px - 8.6, py + ph / 2, "U-Net depth", size=7.2, font=FSB,
          color=T.TX2, rot=90, va="center")
    for lab, fy in (("high res", 0.94), ("mid res", 0.52), ("bottleneck", 0.06)):
        label(ax, px - 1.4, py + ph * fy, lab, size=6.8, font=F, color=T.TX3,
              ha="right", va="center")

    line(ax, [px - 14.0, px + pw], [10.6, 10.6], color=T.RULE, lw=0.8, z=1)
    label(ax, px - 14.0, 7.2,
          "And the signal sits in temporal attention, not in the features.",
          size=7.4, font=FSB, color=T.TX2, ha="left", va="center")
    label(ax, px - 14.0, 4.0, "Ling et al., ICLR 2025   ·   Yatim et al., CVPR 2024",
          size=6.8, font=F, color=T.TX3, ha="left", va="center")

    save(fig, path)
    return path


# ===========================================================================
# compute budget and the calendar
# ===========================================================================

STAGES = [
    ("0", "Sep 2-10", "scoping, reading,\npilots E1-E4", 0.0, True),
    ("—", "Sep 11-20", "midsem window\nreading only", 0.0, True),
    ("1", "Sep 21-27", "prep deck, freeze\nprotocol, run E5", 0.5, False),
    ("2", "Sep 28-Oct 4", "extraction harness\n+ 360 clips", 9.5, False),
    ("3", "Oct 5-11", "coarse selectivity\nsweep", 0.5, False),
    ("4", "Oct 12-18", "refined sweep,\nthe map", 0.5, False),
    ("5", "Oct 19-25", "patching\nframework", 14.3, False),
    ("6", "Oct 26-Nov 1", "edit demos, metrics,\nLoRA runs", 9.4, False),
    ("7", "Nov 2 -", "interface, report,\npackaging", 0.0, False),
]


def budget(path=OUT + "budget.png"):
    fig, ax = canvas(11.7, 3.50)

    n = len(STAGES)
    x0, xw = 8.0, 88.0
    slot = xw / n
    base = 11.2
    scale = 0.515
    quota = 30.0

    # the week we are in, as a full-height band
    here_i = 2
    hx_c = x0 + here_i * slot + slot / 2
    sqbox(ax, hx_c - slot * 0.48, 1.2, slot * 0.96, base + quota * scale + 1.6,
          fc=T.MOTION_WASH, ec=None, z=0, alpha=0.55)

    for i, (sid, dates, what, hrs, done) in enumerate(STAGES):
        cx_ = x0 + i * slot + slot / 2
        bw = slot * 0.40
        here = (i == here_i)
        col = T.MOTION if here else (T.RULE_STRONG if done else T.CONTENT)
        if hrs > 0:
            sqbox(ax, cx_ - bw / 2, base, bw, hrs * scale, fc=col, ec=None, z=4)
            label(ax, cx_, base + hrs * scale + 1.0, f"{hrs:g}", size=7.6,
                  font=FSB, color=col, va="bottom")
        else:
            line(ax, [cx_ - bw / 2, cx_ + bw / 2], [base, base],
                 color=T.RULE_STRONG, lw=1.8, z=4)
        label(ax, cx_, base - 2.1,
              "stage " + sid if sid != "\u2014" else "midsem",
              size=7.2, font=FSB, color=T.TX2 if not done else T.TX3, va="center", z=5)
        label(ax, cx_, base - 4.4, dates, size=6.9, font=F, color=T.TX3,
              va="center", z=5)
        label(ax, cx_, base - 6.4, what, size=6.6, font=F,
              color=T.TX3 if not done else T.RULE_STRONG, va="top", z=5)

    line(ax, [x0 - 1.0, x0 + xw], [base, base], color=T.RULE_STRONG, lw=0.9, z=3)

    qy = base + quota * scale
    line(ax, [x0 - 1.0, x0 + xw], [qy, qy], color=T.TX1, lw=1.0, z=5,
         ls=(0, (3, 2.4)))
    label(ax, x0 + xw, qy + 0.8, "Kaggle free quota, 30 GPU-h per week",
          size=7.2, font=FSB, color=T.TX1, ha="right", va="bottom")

    for g in (10, 20):
        gy = base + g * scale
        line(ax, [x0 - 1.0, x0 + xw], [gy, gy], color=T.RULE, lw=0.7, z=1)
        label(ax, x0 - 1.8, gy, str(g), size=6.8, font=F, color=T.TX3,
              ha="right", va="center")
    label(ax, x0 - 1.8, base, "0", size=6.8, font=F, color=T.TX3,
          ha="right", va="center")
    label(ax, x0 - 4.6, base + 15 * scale, "GPU-hours", size=6.9, font=FSB,
          color=T.TX3, rot=90, va="center")

    label(ax, hx_c, base + quota * scale + 2.0, "we are here", size=7.6,
          font=FSB, color=T.MOTION, va="bottom", z=6)

    save(fig, path)
    return path


if __name__ == "__main__":
    import sys
    fns = dict(storage=storage, e2_swaps=e2_swaps, e2_ssim=e2_ssim,
               selectivity=selectivity_map, prior=prior_work_map, budget=budget)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in fns.items():
        if which in ("all", k):
            print(fn())
