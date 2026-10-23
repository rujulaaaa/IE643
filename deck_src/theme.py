"""Design system for the IE 643 prep deck.

One palette, one type family, an explicit grid.  Everything the build script
draws goes through here so the deck stays internally consistent.

Palette provenance
------------------
The two semantic hues are the categorical slots 1 and 2 of a validated
colour-blind-safe set (blue / orange, a warm-cool pair).  Validated with
scripts/validate_palette.js on both surfaces:

  light  #ffffff : worst all-pairs CVD dE 9.2, normal-vision dE 24.0  -> PASS
  dark   #101820 : worst all-pairs CVD dE 9.4, normal-vision dE 20.9  -> PASS

Blue means *content* and orange means *motion* on every slide of the deck.
Neither hue is ever used decoratively.
"""

from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------


def C(h):
    return RGBColor.from_string(h)


# surfaces
INK = "101820"        # dark full-bleed surface
PAPER = "FFFFFF"      # light content surface
PANEL = "EEF2F5"      # cool grey panel on paper
PANEL_2 = "F6F8FA"    # fainter panel
PANEL_INK = "1A2530"  # panel on ink

# ink on paper
TX1 = "101820"
TX2 = "55636E"
TX3 = "8493A0"
RULE = "DDE4E9"
RULE_STRONG = "C3CDD5"

# ink on the dark surface
TX1_D = "FFFFFF"
TX2_D = "AEBCC7"
TX3_D = "6F7F8C"
RULE_D = "27333E"

# semantic hues  --  blue = content, orange = motion
CONTENT = "2A78D6"
CONTENT_D = "3987E5"
CONTENT_WASH = "DCEAFA"
CONTENT_EDGE = "9EC5F4"

MOTION = "EB6834"
MOTION_D = "F08A5D"
MOTION_WASH = "FBE3D8"
MOTION_EDGE = "F4B39A"

# a third hue, used only for "neither / control"
THIRD = "4A3AA7"
THIRD_WASH = "E2DEF4"

# --------------------------------------------------------------------------
# type
# --------------------------------------------------------------------------

SANS = "Lato"
SANS_L = "Lato Light"
SANS_M = "Lato Medium"
SANS_SB = "Lato Semibold"
SANS_BK = "Lato Black"
MONO = "Noto Sans Mono"

# --------------------------------------------------------------------------
# grid  (13.333 x 7.5 in, 16:9)
# --------------------------------------------------------------------------

SLIDE_W = 13.3333
SLIDE_H = 7.5
MARGIN = 0.82
CONTENT_W = SLIDE_W - 2 * MARGIN          # 11.693
GUTTER = 0.20
NCOL = 12
COLW = (CONTENT_W - (NCOL - 1) * GUTTER) / NCOL   # 0.7911

HEAD_Y = 0.62          # headline top
BODY_Y = 1.86          # first line of content on a standard slide
FOOT_Y = 6.98          # footer baseline
BODY_BOTTOM = 6.62     # nothing below this


def cx(i):
    """Left edge of column i (0-based)."""
    return MARGIN + i * (COLW + GUTTER)


def cw(n):
    """Width spanning n columns."""
    return n * COLW + (n - 1) * GUTTER


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------

from pptx.enum.shapes import MSO_SHAPE


def rect(slide, x, y, w, h, fill=None, line=None, lw=0.75, shape=MSO_SHAPE.RECTANGLE):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    s.shadow.inherit = False
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = C(fill)
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = C(line)
        s.line.width = Pt(lw)
    s.text_frame.text = ""
    return s


def hrule(slide, x, y, w, color=RULE, lw=0.75):
    from pptx.enum.shapes import MSO_CONNECTOR
    ln = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x + w), Inches(y))
    ln.line.color.rgb = C(color)
    ln.line.width = Pt(lw)
    return ln


def vrule(slide, x, y, h, color=RULE, lw=0.75):
    from pptx.enum.shapes import MSO_CONNECTOR
    ln = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(y), Inches(x), Inches(y + h))
    ln.line.color.rgb = C(color)
    ln.line.width = Pt(lw)
    return ln


def tb(slide, x, y, w, h, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, wrap=True):
    """A bare text box: no autofit, no insets, no fill."""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    tf.paragraphs[0].alignment = align
    return box, tf


def para(tf, first=False):
    return tf.paragraphs[0] if first else tf.add_paragraph()


def run(p, text, size=16, font=SANS, color=TX1, bold=False, italic=False,
        spacing=None, caps=False, baseline=None):
    r = p.add_run()
    r.text = text
    f = r.font
    f.name = font
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = C(color)
    if spacing is not None:
        r.font._rPr.set("spc", str(int(spacing * 100)))
    if caps:
        r.font._rPr.set("cap", "all")
    if baseline is not None:
        r.font._rPr.set("baseline", str(int(baseline * 1000)))
    return r


def text(slide, x, y, w, h, s, size=16, font=SANS, color=TX1, bold=False,
         italic=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         line=1.30, space_after=0, spacing=None, caps=False, wrap=True):
    """Single-style text block; newlines become paragraphs."""
    box, tf = tb(slide, x, y, w, h, align=align, anchor=anchor, wrap=wrap)
    for i, chunk in enumerate(s.split("\n")):
        p = para(tf, first=(i == 0))
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(space_after)
        run(p, chunk, size=size, font=font, color=color, bold=bold,
            italic=italic, spacing=spacing, caps=caps)
    return box


def rich(slide, x, y, w, h, parts, size=16, font=SANS, color=TX1,
         align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, line=1.30, space_after=0):
    """parts = list of paragraphs, each a list of (text, **override) tuples."""
    box, tf = tb(slide, x, y, w, h, align=align, anchor=anchor)
    for i, prt in enumerate(parts):
        p = para(tf, first=(i == 0))
        p.alignment = align
        p.line_spacing = line
        p.space_after = Pt(space_after)
        if isinstance(prt, str):
            prt = [(prt, {})]
        for chunk, kw in prt:
            o = dict(size=size, font=font, color=color)
            o.update(kw)
            run(p, chunk, **o)
    return box


# --------------------------------------------------------------------------
# slide chrome
# --------------------------------------------------------------------------

TOTAL_SLIDES = 25


def background(slide, prs, color):
    s = rect(slide, 0, 0, SLIDE_W, SLIDE_H, fill=color)
    slide.shapes._spTree.remove(s._element)
    slide.shapes._spTree.insert(2, s._element)
    return s


def footer(slide, n, section, dark=False):
    """Metropolis-style foot: section name, page, and a progress hairline."""
    c2 = TX3_D if dark else TX3
    r = RULE_D if dark else RULE
    acc = CONTENT_D if dark else CONTENT

    if section:
        text(slide, MARGIN, FOOT_Y, 7.0, 0.22, section, size=9.5,
             font=SANS_SB, color=c2, spacing=0.9, caps=True)
    text(slide, SLIDE_W - MARGIN - 1.2, FOOT_Y, 1.2, 0.22, str(n), size=9.5,
         font=SANS_SB, color=c2, align=PP_ALIGN.RIGHT)

    # progress hairline along the very bottom
    y = SLIDE_H - 0.055
    hrule(slide, 0, y, SLIDE_W, color=r, lw=1.6)
    frac = n / TOTAL_SLIDES
    hrule(slide, 0, y, SLIDE_W * frac, color=acc, lw=1.6)


def headline(slide, s, kicker=None, dark=False, width=None, size=26):
    """Assertion headline. A full sentence, not a topic phrase."""
    c1 = TX1_D if dark else TX1
    c2 = TX3_D if dark else TX3
    w = width or CONTENT_W
    y = HEAD_Y
    if kicker:
        text(slide, MARGIN, y, w, 0.22, kicker, size=10, font=SANS_SB,
             color=c2, spacing=1.0, caps=True)
        y += 0.36
    nlines = s.count("\n") + 1
    text(slide, MARGIN, y, w, 0.5 * nlines + 0.2, s, size=size, font=SANS_SB,
         color=c1, line=1.16)
    return y + (size / 72.0) * 1.16 * nlines
