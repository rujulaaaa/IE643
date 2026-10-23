"""Concept diagrams: the U-Net factorisation, AnimateDiff, patching."""

import numpy as np
from figbase import *
import theme as T

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "fig") + os.sep


# ===========================================================================
# 1.  What "content" and "motion" mean, and what the target edit is
# ===========================================================================

def content_motion_matrix(path=OUT + "cm_matrix.png"):
    fig, ax = canvas(11.7, 3.55)

    cw_, ch_ = 25.0, 9.2
    gx, gy = 11.0, 4.2
    x0, y0 = 26.0, 2.2

    cols = [x0, x0 + cw_ + gx]
    rows = [y0 + ch_ + gy, y0]

    subj = [("golden retriever", T.CONTENT), ("horse", T.THIRD)]
    verb = ["running", "walking"]
    kinds = ["run", "walk"]
    tags = {(0, 0): ("clip A", "given"), (1, 1): ("clip B", "given"),
            (0, 1): ("A's content", "B's motion"),
            (1, 0): ("B's content", "A's motion")}

    for ri in range(2):
        for ci in range(2):
            x, y = cols[ci], rows[ri]
            given = (ri == ci)
            sqbox(ax, x, y, cw_, ch_, fc=T.PANEL_2 if given else T.PAPER,
                  ec=T.RULE if given else T.TX1, lw=0.9 if given else 1.15, z=2)

            n = 6
            frames = filmstrip(ax, x + 1.5, y + 2.5, cw_ - 3.0, 5.4, n=n,
                               gap=0.5, fc=T.PAPER if given else T.PANEL_2, z=3)
            g = gait(n, kinds[ci])
            col = subj[ri][1]
            pts = [(f[0] + f[1] / 2, y + 2.5 + 0.85 + g[k] * 3.7)
                   for k, f in enumerate(frames)]
            line(ax, [p[0] for p in pts], [p[1] for p in pts], color=col,
                 lw=1.0, z=4, alpha=0.45)
            for p in pts:
                ax.add_patch(Circle(p, 0.72, facecolor=hx(col),
                                    edgecolor="none", zorder=5))

            t1, t2 = tags[(ri, ci)]
            label(ax, x + cw_ / 2, y + 1.25, f"{subj[ri][0]}, {verb[ci]}",
                  size=9.0, font=FSB, color=T.TX2 if given else T.TX1)
            if given:
                label(ax, x + cw_ / 2, y + ch_ - 1.15, f"{t1}  ·  {t2}",
                      size=7.4, font=FSB, color=T.TX3)
            else:
                label(ax, x + cw_ / 2, y + ch_ - 1.15, "the edit we have to produce",
                      size=7.4, font=FSB, color=T.MOTION)

    # rows carry content, columns carry motion
    mid_gx = cols[0] + cw_ + gx / 2
    mid_gy = rows[1] + ch_ + gy / 2
    for ri, direction in ((0, 1), (1, -1)):
        y = rows[ri] + ch_ / 2
        if direction > 0:
            arrow(ax, cols[0] + cw_ + 1.2, y, cols[1] - 1.2, y,
                  color=T.CONTENT, lw=1.2, head=7)
        else:
            arrow(ax, cols[1] - 1.2, y, cols[0] + cw_ + 1.2, y,
                  color=T.CONTENT, lw=1.2, head=7)
        label(ax, mid_gx, y + 1.5, "content", size=7.0, font=FSB, color=T.CONTENT)
    for ci, direction in ((1, 1), (0, -1)):
        x = cols[ci] + cw_ / 2
        if direction > 0:
            arrow(ax, x, rows[1] + ch_ + 1.0, x, rows[0] - 1.0,
                  color=T.MOTION, lw=1.2, head=7)
        else:
            arrow(ax, x, rows[0] - 1.0, x, rows[1] + ch_ + 1.0,
                  color=T.MOTION, lw=1.2, head=7)
        label(ax, x + 1.0, mid_gy, "motion", size=7.0, font=FSB,
              color=T.MOTION, ha="left")

    # axes
    for ci in range(2):
        label(ax, cols[ci] + cw_ / 2, rows[0] + ch_ + 2.0, verb[ci],
              size=8.6, font=FSB, color=T.MOTION)
    label(ax, (cols[0] + cols[1] + cw_) / 2, rows[0] + ch_ + 4.3,
          "MOTION  —  how it moves", size=7.8, font=FSB, color=T.TX3, spacing=" ")
    for ri in range(2):
        label(ax, cols[0] - 2.0, rows[ri] + ch_ / 2, subj[ri][0],
              size=8.6, font=FSB, color=subj[ri][1], ha="right")
    label(ax, 6.2, (rows[1] + rows[0] + ch_) / 2,
          "CONTENT  —  what is depicted", size=7.8, font=FSB, color=T.TX3,
          rot=90, spacing=" ")

    save(fig, path)
    return path


# ===========================================================================
# 2.  The U-Net factorisation   (required diagram)
# ===========================================================================

def unet(path=OUT + "unet.png"):
    fig, ax = canvas(11.7, 4.42)

    SP_H, TE_H = 2.05, 1.55
    BH = SP_H + TE_H
    BW = 7.9

    lvl_y = {0: 30.0, 1: 22.8, 2: 15.6, 3: 8.4}
    res = {0: "64x64", 1: "32x32", 2: "16x16", 3: "8x8"}

    down_x = [8.0, 17.0, 26.0, 35.0]
    mid_x = 46.0
    up_x = [57.0, 66.0, 75.0, 84.0]

    # one soft lane per resolution level, so the U reads as four bands
    for lv in range(4):
        y = lvl_y[lv]
        sqbox(ax, 6.4, y - 0.55, 87.0, BH + 1.1, fc=T.PANEL_2, ec=None, z=0)
        label(ax, 5.4, y + BH / 2, res[lv], size=7.0, font=FMONO,
              color=T.TX3, ha="right", z=6)
    label(ax, 5.4, lvl_y[0] + BH + 2.6, "latent\nresolution", size=6.6, font=F,
          color=T.TX3, ha="right", va="center", z=6)

    def block(x, lv, name, temporal=True):
        y = lvl_y[lv]
        sqbox(ax, x, y + TE_H, BW, SP_H, fc=T.CONTENT_WASH, ec=None, z=4)
        label(ax, x + BW / 2, y + TE_H + SP_H / 2, "spatial",
              size=7.4, font=FSB, color=T.CONTENT, z=6)
        if temporal:
            sqbox(ax, x, y, BW, TE_H, fc=T.MOTION, ec=None, z=4)
            label(ax, x + BW / 2, y + TE_H / 2, "temporal",
                  size=6.8, font=FSB, color="FFFFFF", z=6)
        sqbox(ax, x, y, BW, BH, fc=None, ec=T.RULE_STRONG, lw=0.8, z=5)
        label(ax, x + BW / 2, y - 1.65, name, size=6.9, font=FSB, color=T.TX2, z=6)
        return (x, y)

    downs = [block(x, i, f"down {i}") for i, x in enumerate(down_x)]
    block(mid_x, 3, "mid")
    ups = [block(x, 3 - i, f"up {i}") for i, x in enumerate(up_x)]

    def step(a, b):
        arrow(ax, a[0] + BW + 0.3, a[1] + BH / 2, b[0] - 0.3, b[1] + BH / 2,
              color=T.TX3, lw=0.95, head=6, z=3)

    for i in range(3):
        step(downs[i], downs[i + 1])
    step(downs[3], (mid_x, lvl_y[3]))
    step((mid_x, lvl_y[3]), ups[0])
    for i in range(3):
        step(ups[i], ups[i + 1])

    # skip connections
    for i in range(4):
        d, u = downs[i], ups[3 - i]
        y = lvl_y[i] + BH + 1.55
        x1, x2 = d[0] + BW / 2, u[0] + BW / 2
        line(ax, [x1, x1], [lvl_y[i] + BH, y], color=T.RULE_STRONG, lw=0.8, z=1)
        line(ax, [x1, x2], [y, y], color=T.RULE_STRONG, lw=0.8, z=1,
             ls=(0, (3, 2.4)))
        arrow(ax, x2, y, x2, lvl_y[i] + BH + 0.15, color=T.RULE_STRONG,
              lw=0.8, head=5, z=1)
    label(ax, (down_x[0] + up_x[3] + BW) / 2, lvl_y[0] + BH + 2.2,
          "skip connections", size=6.9, font=F, color=T.TX3, va="bottom")

    # what each half sees
    ix, iy = 8.0, 0.2
    label(ax, ix, iy + 4.05, "what each half sees", size=7.4, font=FSB,
          color=T.TX2, ha="left")
    fr = filmstrip(ax, ix, iy + 0.9, 20.0, 2.3, n=8, gap=0.45, fc=T.PANEL, z=2)
    sqbox(ax, fr[2][0] - 0.25, iy + 0.65, fr[2][1] + 0.5, 2.8,
          fc=None, ec=T.CONTENT, lw=1.3, z=5)
    line(ax, [fr[0][0] + fr[0][1] / 2, fr[7][0] + fr[7][1] / 2],
         [iy + 2.05, iy + 2.05], color=T.MOTION, lw=1.5, z=4)
    for f in fr:
        ax.add_patch(Circle((f[0] + f[1] / 2, iy + 2.05), 0.38,
                            facecolor=hx(T.MOTION), edgecolor="none", zorder=5))
    label(ax, ix + 21.4, iy + 2.85, "spatial sees one frame",
          size=7.0, font=F, color=T.CONTENT, ha="left")
    label(ax, ix + 21.4, iy + 1.25, "temporal sees the stack",
          size=7.0, font=F, color=T.MOTION, ha="left")

    # counts + repetition, right
    rx = 62.0
    sqbox(ax, rx, iy + 0.15, 31.0, 4.4, fc=T.PANEL_2, ec=None, z=1)
    label(ax, rx + 1.3, iy + 3.25,
          "21 temporal modules  ·  16 spatial attention pairs",
          size=7.0, font=FSB, color=T.TX2, ha="left")
    label(ax, rx + 1.3, iy + 1.35,
          "and the whole pass runs again at each of 25 denoising steps",
          size=6.9, font=F, color=T.TX3, ha="left")

    save(fig, path)
    return path


# ===========================================================================
# 3.  AnimateDiff: a frozen backbone with motion modules threaded in
# ===========================================================================

def animatediff(path=OUT + "animatediff.png"):
    fig, ax = canvas(7.35, 3.5)
    H = ax.get_ylim()[1]

    n = 5
    BW, BH = 14.6, 8.4
    gap = 3.9
    x0 = 1.5
    ybase = 12.0

    for i in range(n):
        x = x0 + i * (BW + gap)
        sqbox(ax, x, ybase, BW, BH, fc=T.PAPER, ec=T.CONTENT_EDGE, lw=1.2, z=3)
        label(ax, x + BW / 2, ybase + BH / 2 + 1.1, f"SD block {i+1}",
              size=8.4, font=FSB, color=T.CONTENT, z=5)
        label(ax, x + BW / 2, ybase + BH / 2 - 1.7, "frozen",
              size=6.8, font=F, color=T.TX3, z=5)
        if i < n - 1:
            mx = x + BW + gap / 2
            sqbox(ax, mx - 3.2, ybase + 1.2, 6.4, 6.0, fc=T.MOTION,
                  ec=T.MOTION, lw=1.0, z=4)
            label(ax, mx, ybase + 4.2, "motion", size=6.4, font=FSB,
                  color="FFFFFF", z=6)
            arrow(ax, x + BW, ybase + BH / 2, mx - 3.2, ybase + BH / 2,
                  color=T.TX3, lw=0.9, head=5, z=2)
            arrow(ax, mx + 3.2, ybase + BH / 2, x + BW + gap, ybase + BH / 2,
                  color=T.TX3, lw=0.9, head=5, z=2)

    brace_h(ax, x0, x0 + n * BW + (n - 1) * gap, ybase - 1.4,
            color=T.CONTENT_EDGE, depth=1.1)
    label(ax, 50, ybase - 3.4, "image-diffusion backbone, weights never updated",
          size=7.6, font=F, color=T.CONTENT)

    label(ax, 50, ybase + BH + 5.4, "trained separately, on video",
          size=7.6, font=F, color=T.MOTION)
    arrow(ax, 50, ybase + BH + 4.3, 50, ybase + BH + 1.6,
          color=T.MOTION_EDGE, lw=1.0, head=6)

    # proportional parameter bar
    by = 2.2
    total = 860 + 417
    w_all = 97.0
    w_sd = w_all * 860 / total
    sqbox(ax, 1.5, by, w_sd, 2.5, fc=T.CONTENT_WASH, ec=T.CONTENT_EDGE, lw=0.8, z=3)
    sqbox(ax, 1.5 + w_sd, by, w_all - w_sd, 2.5, fc=T.MOTION, ec=T.MOTION, lw=0.8, z=3)
    label(ax, 1.5 + w_sd / 2, by + 1.25, "SD1.5 UNet  860M", size=7.2,
          font=FSB, color=T.CONTENT, z=5)
    label(ax, 1.5 + w_sd + (w_all - w_sd) / 2, by + 1.25, "motion  417M",
          size=7.2, font=FSB, color="FFFFFF", z=5)

    save(fig, path)
    return path


# ===========================================================================
# 4.  Activation patching   (required diagram)
# ===========================================================================

def patching(path=OUT + "patching.png"):
    fig, ax = canvas(11.7, 2.95)

    nT = 9
    x0, xw = 17.5, 40.0
    sw = (xw - 1.3 * (nT - 1)) / nT
    y_src, y_tgt = 17.6, 4.8
    bh = 4.2
    tstar = 3

    def chain(y, tint, edge, wash, lab, sub):
        label(ax, 16.2, y + bh / 2 + 0.95, lab, size=8.6, font=FSB,
              color=tint, ha="right")
        label(ax, 16.2, y + bh / 2 - 1.35, sub, size=7.0, font=F,
              color=T.TX3, ha="right")
        xs = []
        for i in range(nT):
            x = x0 + i * (sw + 1.3)
            hot = (i == tstar)
            sqbox(ax, x, y, sw, bh, fc=tint if hot else wash,
                  ec=tint if hot else edge, lw=1.2 if hot else 0.8, z=4)
            xs.append(x)
        return xs

    xs_s = chain(y_src, T.CONTENT, T.CONTENT_EDGE, T.PAPER,
                 "source run A", "a golden retriever running")
    xs_t = chain(y_tgt, T.MOTION, T.MOTION_EDGE, T.PAPER,
                 "target run B", "a horse walking")

    # the splice
    xc = xs_s[tstar] + sw / 2
    sqbox(ax, xc - 6.3, y_tgt + bh + 2.5, 12.6, 2.9, fc=T.TX1, ec=None, z=6)
    label(ax, xc, y_tgt + bh + 3.95, "cache (L, t)", size=7.2, font=FMONO,
          color="FFFFFF", z=7)
    line(ax, [xc, xc], [y_src - 0.15, y_tgt + bh + 5.4], color=T.TX1, lw=0.9, z=5)
    arrow(ax, xc, y_tgt + bh + 2.5, xc, y_tgt + bh + 0.25, color=T.TX1,
          lw=0.9, head=6, z=5)
    label(ax, xc + 7.4, y_tgt + bh + 3.95,
          "splice A's activation into B at the same coordinate",
          size=7.0, font=F, color=T.TX2, ha="left", z=7)
    label(ax, xc, y_src - 1.35, "one coordinate", size=6.6, font=F,
          color=T.TX2, va="top", z=7)

    label(ax, x0 + xw / 2, y_tgt - 1.3, "25 denoising steps,  noisy -> clean",
          size=7.0, font=F, color=T.TX3, va="top")

    # outputs, drawn with the same colour = content / trace = motion code
    ox, ow = 63.0, 15.5
    def out(y, col, kind, cap, sub, capcol):
        arrow(ax, x0 + xw + 0.8, y + bh / 2, ox - 1.0, y + bh / 2,
              color=T.RULE_STRONG, lw=0.9, head=5, z=3)
        n = 6
        fr = filmstrip(ax, ox, y, ow, bh, n=n, gap=0.4, fc=T.PANEL_2, z=3)
        g = gait(n, kind)
        pts = [(f[0] + f[1] / 2, y + 0.7 + g[k] * (bh - 1.4))
               for k, f in enumerate(fr)]
        line(ax, [p[0] for p in pts], [p[1] for p in pts], color=col,
             lw=0.9, z=4, alpha=0.45)
        for p in pts:
            ax.add_patch(Circle(p, 0.55, facecolor=hx(col),
                                edgecolor="none", zorder=5))
        label(ax, ox + ow + 1.4, y + bh / 2 + 0.9, cap, size=7.6, font=FSB,
              color=capcol, ha="left")
        label(ax, ox + ow + 1.4, y + bh / 2 - 1.2, sub, size=6.8, font=F,
              color=T.TX3, ha="left")

    out(y_src, T.CONTENT, "run", "output A", "retriever, running", T.CONTENT)
    out(y_tgt, T.THIRD, "run", "output B*", "horse -- but now running?", T.MOTION)

    # the read-out
    qy = 0.0
    sqbox(ax, 17.5, qy, 79.0, 2.5, fc=T.PANEL_2, ec=None, z=1)
    label(ax, 18.6, qy + 1.25,
          "Then ask: does B* now move like A, look like A, both, or neither?",
          size=7.4, font=FSB, color=T.TX2, ha="left", z=4)
    label(ax, 95.4, qy + 1.25,
          "one axis moves and the other does not  =  that coordinate carries that axis",
          size=7.0, font=F, color=T.MOTION, ha="right", z=4)

    save(fig, path)
    return path


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    fns = dict(matrix=content_motion_matrix, unet=unet,
               animatediff=animatediff, patching=patching)
    for k, fn in fns.items():
        if which in ("all", k):
            print(fn())


# ===========================================================================
# 5.  a quiet motif for the title slide   (dark surface)
# ===========================================================================

def title_motif(path=OUT + "title_motif.png"):
    fig, ax = canvas(5.15, 3.60, bg=T.INK)

    n, sh = 12, 5.6
    VIOLET = "8A7BE0"
    groups = [
        (62.0, "same content, different motion", T.CONTENT_D,
         [(T.CONTENT_D, "run"), (T.CONTENT_D, "walk")]),
        (33.0, "same motion, different content", T.MOTION_D,
         [(T.CONTENT_D, "run"), (VIOLET, "run")]),
    ]
    for top, title, accent, clips in groups:
        label(ax, 3.0, top + 3.4, title, size=7.4, font=FSB, color=accent,
              ha="left")
        for i, (col, kind) in enumerate(clips):
            y = top - i * 8.8 - sh
            fr = filmstrip(ax, 3.0, y, 90.0, sh, n=n, gap=0.7,
                           fc=T.PANEL_INK, z=2)
            g = gait(n, kind)
            pts = [(f[0] + f[1] / 2, y + 0.95 + g[k] * (sh - 1.9))
                   for k, f in enumerate(fr)]
            line(ax, [p[0] for p in pts], [p[1] for p in pts], color=col,
                 lw=0.9, z=4, alpha=0.4)
            for p in pts:
                ax.add_patch(Circle(p, 0.78, facecolor=hx(col),
                                    edgecolor="none", zorder=5))

    line(ax, [3.0, 93.0], [11.4, 11.4], color=T.RULE_D, lw=0.8)
    label(ax, 3.0, 7.4, "colour  =  what is depicted", size=6.9, font=F,
          color=T.CONTENT_D, ha="left")
    label(ax, 3.0, 3.2, "trace   =  how it moves", size=6.9, font=F,
          color=T.MOTION_D, ha="left")
    save(fig, path)
    return path
