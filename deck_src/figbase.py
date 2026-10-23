"""Shared drawing helpers for the deck figures.

Every figure is drawn in a coordinate system that is 100 units wide and
100*h/w units tall, so positions read the same whatever the output size.
Colours and fonts come from theme.py, so figures and slides match exactly.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.path import Path
import matplotlib.patches as mpatches
import numpy as np

import theme as T

DPI = 320

# matplotlib exposes every Lato weight under the single family "Lato",
# so a "font" here is a numeric weight; "mono" switches family.
F = 400
FL = 300
FM = 500
FSB = 600
FBK = 900
FMONO = "mono"


def _face(font):
    if font == "mono":
        return dict(fontfamily="Noto Sans Mono", fontweight=400)
    return dict(fontfamily="Lato", fontweight=font)


def hx(c):
    return "#" + c


def canvas(w_in, h_in, bg=T.PAPER):
    """Return (fig, ax) with a 100-wide user space."""
    fig = plt.figure(figsize=(w_in, h_in), dpi=DPI)
    fig.patch.set_facecolor(hx(bg))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(hx(bg))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h_in / w_in)
    ax.set_axis_off()
    ax.set_aspect("equal")
    return fig, ax


def save(fig, path):
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


# ---------------------------------------------------------------- primitives

def box(ax, x, y, w, h, fc=None, ec=None, lw=1.1, r=0.9, z=2, alpha=1.0, ls="-"):
    p = FancyBboxPatch(
        (x + r, y + r), w - 2 * r, h - 2 * r,
        boxstyle=f"round,pad={r}",
        linewidth=lw, zorder=z, alpha=alpha, linestyle=ls,
        facecolor="none" if fc is None else hx(fc),
        edgecolor="none" if ec is None else hx(ec),
    )
    ax.add_patch(p)
    return p


def sqbox(ax, x, y, w, h, fc=None, ec=None, lw=1.1, z=2, alpha=1.0, ls="-"):
    p = Rectangle((x, y), w, h, linewidth=lw, zorder=z, alpha=alpha, linestyle=ls,
                  facecolor="none" if fc is None else hx(fc),
                  edgecolor="none" if ec is None else hx(ec))
    ax.add_patch(p)
    return p


def label(ax, x, y, s, size=9, color=T.TX1, font=F, ha="center", va="center",
          z=6, rot=0, style="normal", spacing=None):
    kw = {}
    if spacing is not None:
        # matplotlib has no letter-spacing; fake it by spacing the characters
        s = spacing.join(list(s))
    t = ax.text(x, y, s, fontsize=size, color=hx(color),
                ha=ha, va=va, zorder=z, rotation=rot, style=style,
                **_face(font), **kw)
    return t


def arrow(ax, x1, y1, x2, y2, color=T.TX3, lw=1.1, z=3, head=5.0,
          style="-|>", conn="arc3,rad=0", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=head, linewidth=lw, linestyle=ls,
                        color=hx(color), zorder=z, connectionstyle=conn,
                        shrinkA=0, shrinkB=0, joinstyle="miter",
                        capstyle="butt")
    ax.add_patch(a)
    return a


def line(ax, xs, ys, color=T.RULE, lw=1.0, z=1, ls="-", alpha=1.0):
    return ax.plot(xs, ys, color=hx(color), lw=lw, zorder=z, ls=ls,
                   alpha=alpha, solid_capstyle="butt")[0]


def brace_h(ax, x1, x2, y, color=T.TX3, lw=0.9, depth=1.2, z=3):
    """A flat bracket under a span."""
    line(ax, [x1, x1, x2, x2], [y + depth, y, y, y + depth], color=color, lw=lw, z=z)


def tick_label(ax, x, y, s, size=7.5, color=T.TX3, font=F, ha="center", va="top"):
    return label(ax, x, y, s, size=size, color=color, font=font, ha=ha, va=va)


# ---------------------------------------------------------------- filmstrip

def filmstrip(ax, x, y, w, h, n=6, gap=0.6, fc=T.PANEL, ec=None, lw=0.0, z=2):
    """n equal frames across (x, w). Returns list of (fx, fw)."""
    fw = (w - gap * (n - 1)) / n
    out = []
    for i in range(n):
        fx = x + i * (fw + gap)
        sqbox(ax, fx, y, fw, h, fc=fc, ec=ec, lw=lw, z=z)
        out.append((fx, fw))
    return out


def gait(n, kind, phase=0.0):
    """A crude but honest gait trace: vertical displacement per frame."""
    t = np.arange(n) / n
    if kind == "run":
        return 0.5 + 0.44 * np.sin(2 * np.pi * (2.0 * t + phase))
    if kind == "walk":
        return 0.5 + 0.14 * np.sin(2 * np.pi * (1.0 * t + phase + 0.12))
    return np.full(n, 0.5)
