"""Diagrams for the second half of the deck: the five deliverables."""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from figbase import *
import theme as T

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "fig") + os.sep

SEQ = LinearSegmentedColormap.from_list(
    "seq_content", ["#FFFFFF", "#CDE2FB", "#86B6EF", "#2A78D6", "#184F95"], N=256)

NL = chr(10)


# ===========================================================================
# the five deliverables, and what each hands to the next   (dark surface)
# ===========================================================================

def chain(path=OUT + "chain.png"):
    fig, ax = canvas(11.7, 2.45, bg=T.INK)

    items = [
        ("01", "Setup and" + NL + "probing tools", "activations, indexed"),
        ("02", "Content / motion" + NL + "analysis", "a ranked coordinate list"),
        ("03", "Patching" + NL + "framework", "a causal test per coordinate"),
        ("04", "Disentanglement" + NL + "demonstration", "the edit, both directions"),
        ("05", "The trained" + NL + "model", "does the map hold in weights?"),
    ]
    n = len(items)
    bw, gap = 16.6, 4.0
    x0 = (100 - (n * bw + (n - 1) * gap)) / 2
    y, bh = 6.6, 8.8

    for i, (num, title, passes) in enumerate(items):
        x = x0 + i * (bw + gap)
        sqbox(ax, x, y, bw, bh, fc=T.PANEL_INK, ec=None, z=2)
        sqbox(ax, x, y + bh - 0.42, bw, 0.42, fc=T.CONTENT_D, ec=None, z=3)
        label(ax, x + 1.4, y + bh - 2.3, num, size=8.6, font=FBK,
              color=T.CONTENT_D, ha="left", va="center", z=5)
        label(ax, x + bw / 2, y + bh / 2 - 0.6, title, size=8.2, font=FSB,
              color=T.TX1_D, va="center", z=5)
        label(ax, x + bw / 2, y - 1.5, passes, size=6.9, font=F,
              color=T.TX3_D, va="center", z=5)
        if i < n - 1:
            arrow(ax, x + bw + 0.6, y + bh / 2, x + bw + gap - 0.6, y + bh / 2,
                  color=T.TX3_D, lw=1.1, head=7, z=4)

    label(ax, 50, 18.6,
          "Each one is only useful because of the one before it.",
          size=8.6, font=F, color=T.TX2_D, va="center")
    save(fig, path)
    return path


# ===========================================================================
# how a pair is built
# ===========================================================================

def pairs(path=OUT + "pairs.png"):
    W_IN = 6.5
    fig, ax = canvas(W_IN, 3.05)

    MONO_ADV = 0.60 * 7.4 * 100.0 / (W_IN * 72.0)     # units per character

    y0 = 39.5
    label(ax, 2.0, y0 + 4.2, "the prompt template", size=7.6, font=FSB,
          color=T.TX2, ha="left")
    tok = [("a", T.TX3, 5.0), ("[subject]", T.CONTENT, 17.5),
           ("[verb]", T.MOTION, 13.5), ("[scene]", T.TX3, 14.5),
           (", high quality, detailed", T.TX3, 33.0)]
    x = 2.0
    for t, col, w in tok:
        fc = (T.CONTENT_WASH if col == T.CONTENT else
              T.MOTION_WASH if col == T.MOTION else T.PANEL_2)
        sqbox(ax, x, y0, w, 3.6, fc=fc, ec=None, z=2)
        label(ax, x + w / 2, y0 + 1.8, t, size=7.4,
              font=FSB if col != T.TX3 else F, color=col, z=4)
        x += w + 0.9

    rows = [
        ("a content pair", T.CONTENT, "vary the subject, hold everything else",
         [("a golden retriever running on a beach", None),
          ("a horse running on a beach", 1)]),
        ("a motion pair", T.MOTION, "vary the verb, hold everything else",
         [("a golden retriever running on a beach", None),
          ("a golden retriever walking on a beach", 3)]),
    ]
    yy = 29.5
    for name, col, sub, lines in rows:
        sqbox(ax, 2.0, yy - 8.6, 1.8, 10.6, fc=col, ec=None, z=3)
        label(ax, 5.4, yy + 0.9, name, size=8.0, font=FSB, color=col, ha="left")
        label(ax, 5.4, yy - 1.5, sub, size=6.9, font=F, color=T.TX3, ha="left")
        for k, (txt, hot) in enumerate(lines):
            ly = yy - 4.4 - k * 3.2
            wx = 5.4
            for wi, w in enumerate(txt.split()):
                is_hot = (hot is not None and wi == hot)
                if is_hot:
                    sqbox(ax, wx - 0.6, ly - 1.5,
                          len(w) * MONO_ADV + 1.2, 3.0,
                          fc=T.CONTENT_WASH if col == T.CONTENT else T.MOTION_WASH,
                          ec=None, z=2)
                label(ax, wx, ly, w, size=7.4, font=FMONO,
                      color=col if is_hot else T.TX2, ha="left", va="center", z=4)
                wx += (len(w) + 1) * MONO_ADV
        yy -= 14.6

    line(ax, [2.0, 98.0], [5.0, 5.0], color=T.RULE, lw=0.8)
    label(ax, 2.0, 2.3,
          "Same seed and the same noise tensor across a pair. Without that, the "
          "difference between two clips" + NL + "is mostly sampling noise and the "
          "probe measures nothing. E2 is what told us this is viable.",
          size=7.2, font=F, color=T.TX2, ha="left", va="center")
    save(fig, path)
    return path


# ===========================================================================
# cross-attention as a lens
# ===========================================================================

def _att_map(rng, sharp, coarse, cx_, cy_, grid=48):
    g = np.linspace(-1, 1, grid)
    X, Y = np.meshgrid(g, g)
    m = np.exp(-(((X - cx_) / sharp) ** 2 + ((Y - cy_) / sharp) ** 2))
    m += 0.45 * np.exp(-(((X + 0.55) / (sharp * 1.5)) ** 2 +
                         ((Y - 0.5) / (sharp * 1.5)) ** 2))
    m += rng.normal(0, 0.10 + 0.22 * sharp, m.shape)
    if coarse > 1:
        k = coarse
        m = m[:grid // k * k, :grid // k * k]
        m = m.reshape(grid // k, k, grid // k, k).mean(axis=(1, 3))
        m = np.repeat(np.repeat(m, k, axis=0), k, axis=1)
    m = np.clip(m, 0, None)
    return m / m.max()


def attention(path=OUT + "attention.png"):
    rng = np.random.default_rng(11)
    fig, ax = canvas(6.5, 3.15)

    nL, nT = 3, 5
    cell, gapx, gapy = 8.0, 3.2, 3.2
    x0, y0 = 15.0, 19.5

    sharps = [0.95, 0.70, 0.50, 0.36, 0.26]      # diffuse -> tight
    coarse = {0: 4, 1: 8, 2: 2}                   # mid block is the blockiest

    for li in range(nL):
        for ti in range(nT):
            x = x0 + ti * (cell + gapx)
            y = y0 + (nL - 1 - li) * (cell + gapy)
            m = _att_map(rng, sharps[ti], coarse[li],
                         0.12 * np.sin(li * 1.7), -0.05 + 0.10 * li)
            ax.imshow(m, cmap=SEQ, extent=[x, x + cell, y, y + cell],
                      vmin=0, vmax=1, zorder=3, interpolation="nearest",
                      aspect="auto")
            sqbox(ax, x, y, cell, cell, fc=None, ec=T.RULE, lw=0.6, z=4)

    for li, nm in enumerate(["down 2", "mid", "up 1"]):
        label(ax, x0 - 1.6, y0 + (nL - 1 - li) * (cell + gapy) + cell / 2, nm,
              size=7.0, font=FMONO, color=T.TX3, ha="right", va="center")
    for ti, nm in enumerate(["t=25", "19", "13", "7", "t=1"]):
        label(ax, x0 + ti * (cell + gapx) + cell / 2,
              y0 + nL * cell + (nL - 1) * gapy + 1.8, nm,
              size=7.0, font=FMONO, color=T.TX3, va="center")
    label(ax, x0 + (nT * cell + (nT - 1) * gapx) / 2, y0 - 2.2,
          "noisy  ->  clean", size=7.0, font=F, color=T.TX3, va="center")
    label(ax, x0 - 9.4, y0 + (nL * cell + (nL - 1) * gapy) / 2, "layer",
          size=7.2, font=FSB, color=T.TX2, rot=90, va="center")
    label(ax, x0, y0 + nL * cell + (nL - 1) * gapy + 4.4,
          "attention for the token  \"retriever\"", size=7.6, font=FSB,
          color=T.TX1, ha="left", va="center")

    # each cell keeps the frame axis
    fy = 7.6
    label(ax, 2.0, fy + 4.8, "and every cell keeps the frame axis",
          size=7.4, font=FSB, color=T.TX2, ha="left")
    n_fr = 10
    fw = (38.0 - 0.5 * (n_fr - 1)) / n_fr
    for i in range(n_fr):
        fx = 2.0 + i * (fw + 0.5)
        m = _att_map(rng, 0.34, 2, -0.55 + 1.1 * i / (n_fr - 1),
                     -0.1 + 0.25 * np.sin(i * 0.9), grid=24)
        ax.imshow(m, cmap=SEQ, extent=[fx, fx + fw, fy, fy + 3.4],
                  vmin=0, vmax=1, zorder=3, interpolation="nearest",
                  aspect="auto")
        sqbox(ax, fx, fy, fw, 3.4, fc=None, ec=T.RULE, lw=0.5, z=4)
    label(ax, 42.0, fy + 1.7,
          "DAAM aggregation with the frame axis kept instead of" + NL +
          "collapsed, so a token's map becomes a small movie.",
          size=7.0, font=F, color=T.TX2, ha="left", va="center")
    label(ax, 2.0, 2.0,
          "Schematic. The real maps arrive with the Stage 2 harness.",
          size=6.9, font=F, color=T.TX3, ha="left", va="center")
    save(fig, path)
    return path


# ===========================================================================
# patch granularities, coarse to fine
# ===========================================================================

def granularity(path=OUT + "granularity.png"):
    fig, ax = canvas(6.4, 2.80)

    items = [
        ("1", "whole tensor", "blunt, probably breaks coherence, but it bounds"
                              " the effect size"),
        ("2", "channel subset", "only the most motion-selective channels"),
        ("3", "attention weights", "swap the temporal map, not the features"),
        ("4", "scaled injection", "blend instead of replace; sweeping alpha"
                                  " gives a dose-response curve"),
    ]
    bh, gap = 8.2, 2.1
    top = 40.0
    gx, gw = 76.0, 20.0

    for i, (num, name, why) in enumerate(items):
        yy = top - i * (bh + gap) - bh
        sqbox(ax, 2.0, yy, 4.4, bh, fc=T.MOTION, ec=None, z=3)
        label(ax, 4.2, yy + bh / 2, num, size=9.0, font=FBK, color="FFFFFF", z=5)
        sqbox(ax, 6.4, yy, 91.6, bh, fc=T.PANEL_2, ec=None, z=2)
        label(ax, 8.2, yy + bh / 2 + 1.6, name, size=8.4, font=FSB,
              color=T.TX1, ha="left", va="center", z=4)
        label(ax, 8.2, yy + bh / 2 - 1.8, why, size=7.1, font=F,
              color=T.TX3, ha="left", va="center", z=4)

        # a glyph of what actually gets swapped
        cy = yy + bh / 2
        if i == 0:
            sqbox(ax, gx, cy - 2.4, gw, 4.8, fc=T.MOTION_EDGE, ec=None, z=4)
        elif i == 1:
            for k in range(6):
                fc = T.MOTION_EDGE if k in (1, 4) else T.PANEL
                sqbox(ax, gx + k * (gw / 6), cy - 2.4, gw / 6 - 0.35, 4.8,
                      fc=fc, ec=None, z=4)
        elif i == 2:
            sqbox(ax, gx, cy - 2.4, gw, 4.8, fc=T.PANEL, ec=None, z=4)
            for k in range(5):
                sqbox(ax, gx + 0.6 + k * 3.9, cy - 1.9 + 0.4 * k, 3.2, 1.2,
                      fc=T.MOTION_EDGE, ec=None, z=5)
        else:
            for k in range(6):
                a = 0.15 + 0.17 * k
                sqbox(ax, gx + k * (gw / 6), cy - 2.4, gw / 6 - 0.35, 4.8,
                      fc=T.MOTION_EDGE, ec=None, z=4, alpha=a)
    label(ax, gx, top + 1.6, "what gets swapped", size=6.9, font=F,
          color=T.TX3, ha="left", va="center")
    label(ax, 2.0, top + 1.6, "coarse  ->  fine", size=6.9, font=F,
          color=T.TX3, ha="left", va="center")
    save(fig, path)
    return path


# ===========================================================================
# the two LoRAs, and the comparison that is the point
# ===========================================================================

def lora(path=OUT + "lora.png"):
    fig, ax = canvas(11.7, 2.85)

    lx = 2.0
    label(ax, lx, 21.4, "two adapters, MotionDirector-style", size=8.0,
          font=FSB, color=T.TX2, ha="left")
    for i, (nm, col, wash, sub) in enumerate([
            ("appearance LoRA", T.CONTENT, T.CONTENT_WASH,
             "on the transformer_blocks attention projections," + NL +
             "trained on one randomly sampled frame per step"),
            ("motion LoRA", T.MOTION, T.MOTION_WASH,
             "on the motion_modules, trained with the" + NL +
             "appearance LoRA attached and frozen")]):
        yy = 11.6 - i * 9.6
        sqbox(ax, lx, yy, 1.8, 8.0, fc=col, ec=None, z=3)
        sqbox(ax, lx + 1.8, yy, 29.0, 8.0, fc=wash, ec=None, z=2)
        label(ax, lx + 3.4, yy + 5.8, nm, size=8.2, font=FSB, color=col, ha="left")
        label(ax, lx + 3.4, yy + 2.6, sub, size=6.9, font=F, color=T.TX2,
              ha="left", va="center")

    line(ax, [34.5, 34.5], [6.6, 22.4], color=T.RULE, lw=0.8)

    rx = 38.0
    label(ax, rx, 21.4, "then train the motion LoRA twice", size=8.0,
          font=FSB, color=T.TX2, ha="left")
    for i, (nm, sub, w, col) in enumerate([
            ("on every temporal projection", "the full-coverage baseline",
             22.0, T.RULE_STRONG),
            ("on the coordinates our map ranked highest",
             "same rank, same data, a fraction of the trainable parameters",
             7.5, T.MOTION)]):
        yy = 14.0 - i * 6.6
        sqbox(ax, rx, yy, 22.0, 4.0, fc=T.PANEL_2, ec=None, z=2)
        sqbox(ax, rx, yy, w, 4.0, fc=col, ec=None, z=3)
        label(ax, rx + 23.6, yy + 2.8, nm, size=7.6, font=FSB,
              color=T.TX1 if i else T.TX2, ha="left", va="center")
        label(ax, rx + 23.6, yy + 0.9, sub, size=6.8, font=F, color=T.TX3,
              ha="left", va="center")

    sqbox(ax, rx, 0.8, 60.0, 4.6, fc=T.MOTION_WASH, ec=None, z=1)
    label(ax, rx + 1.6, 3.1,
          "If the probe-guided one matches the full-coverage one, the map "
          "predicts in weight space and not only",
          size=7.2, font=FSB, color=T.MOTION, ha="left", va="center")
    label(ax, rx + 1.6, 1.6,
          "in activation space. If it does not, that is a real limit on the map "
          "-- and a better result than \"we also trained a LoRA\".",
          size=7.2, font=F, color=T.MOTION, ha="left", va="center")
    save(fig, path)
    return path


# ===========================================================================
# the demonstration grid, and what each cell is scored against
# ===========================================================================

def demo(path=OUT + "demo.png"):
    fig, ax = canvas(11.7, 3.05)

    panels = [
        ("source A", "golden retriever, running", T.CONTENT, True, None),
        ("source B", "horse, walking", T.THIRD, True, None),
        ("output 1", "content of A, motion of B", None, False,
         ("appearance scored against A", "motion scored against B")),
        ("output 2", "content of B, motion of A", None, False,
         ("appearance scored against B", "motion scored against A")),
    ]
    n = len(panels)
    pw, gap = 21.5, 3.2
    x0 = (100 - (n * pw + (n - 1) * gap)) / 2
    py, ph = 8.8, 10.4

    for i, (name, sub, col, given, scored) in enumerate(panels):
        x = x0 + i * (pw + gap)
        sqbox(ax, x, py, pw, ph, fc=T.PANEL_2 if given else T.PAPER,
              ec=T.RULE if given else T.TX1, lw=0.9 if given else 1.15, z=2)
        filmstrip(ax, x + 1.4, py + 2.6, pw - 2.8, ph - 4.4, n=4, gap=0.55,
                  fc=T.PANEL if given else T.PANEL_2, z=3)
        if not given:
            label(ax, x + pw / 2, py + ph / 2 + 0.4, "Stage 6", size=7.2,
                  font=FSB, color=T.TX3, z=6)
        label(ax, x + pw / 2, py + ph + 1.8, name, size=8.4, font=FSB,
              color=col if given else T.MOTION)
        label(ax, x + pw / 2, py + 1.2, sub, size=7.2, font=F, color=T.TX2, z=6)
        if scored:
            for k, sline in enumerate(scored):
                c = T.CONTENT if k == 0 else T.MOTION
                label(ax, x + pw / 2, py - 2.0 - k * 2.2, sline, size=6.7,
                      font=F, color=c, va="center")

    yb = py + ph + 4.6
    line(ax, [x0, x0 + 2 * pw + gap], [yb, yb], color=T.RULE_STRONG, lw=0.9)
    label(ax, x0 + (2 * pw + gap) / 2, yb + 1.4, "given", size=7.2, font=FSB,
          color=T.TX3, va="center")
    x3 = x0 + 2 * (pw + gap)
    line(ax, [x3, x0 + n * pw + (n - 1) * gap], [yb, yb], color=T.MOTION, lw=0.9)
    label(ax, x3 + (2 * pw + gap) / 2, yb + 1.4,
          "produced, and scored in both directions", size=7.2, font=FSB,
          color=T.MOTION, va="center")

    label(ax, 50, 1.4,
          "A method that transfers motion but not content has found a useful "
          "lever, not a disentanglement.",
          size=7.4, font=FSB, color=T.TX2, va="center")
    save(fig, path)
    return path


if __name__ == "__main__":
    import sys
    fns = dict(chain=chain, pairs=pairs, attention=attention,
               granularity=granularity, lora=lora, demo=demo)
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, fn in fns.items():
        if which in ("all", k):
            print(fn())
