"""Build IE643_Prep_Presentation.pptx.

Twenty-five slides, against the course instruction sheet in section 9 of the
roadmap.  Layout varies by slide on purpose: full-bleed diagrams, tables,
two-column splits and dark dividers, rather than one template repeated.
"""

import os
from PIL import Image
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

import theme as T
from theme import (cx, cw, text, rich, rect, hrule, vrule, tb, para, run,
                   footer, headline, background, MARGIN, CONTENT_W, SLIDE_W,
                   SLIDE_H, HEAD_Y, FOOT_Y)

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "fig")
OUT = os.path.join(os.path.dirname(HERE), "IE643_Prep_Presentation.pptx")

prs = Presentation()
prs.slide_width = Inches(SLIDE_W)
prs.slide_height = Inches(SLIDE_H)
BLANK = prs.slide_layouts[6]

_n = 0


def newslide(section, dark=False):
    global _n
    _n += 1
    s = prs.slides.add_slide(BLANK)
    background(s, prs, T.INK if dark else T.PAPER)
    footer(s, _n, section, dark=dark)
    return s


def img(slide, name, x, y, w=None, h=None):
    """Place fig/<name>.png; give either w or h and the other is derived."""
    p = os.path.join(FIG, name + ".png")
    im = Image.open(p)
    ar = im.width / im.height
    if w is None:
        w = h * ar
    if h is None:
        h = w / ar
    slide.shapes.add_picture(p, Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def panel(slide, x, y, w, h, fill=T.PANEL_2):
    return rect(slide, x, y, w, h, fill=fill)


def takeaway(slide, s, y=None, dark=False, color=None, width=None):
    """One line at the foot of the slide. Not on every slide."""
    y = y if y is not None else 6.28
    text(slide, MARGIN, y, width or CONTENT_W, 0.30, s, size=13,
         font=T.SANS_SB, color=color or (T.TX2_D if dark else T.TX2))


def lede(slide, s, y=1.60, size=14.5, width=None, color=None, dark=False):
    return text(slide, MARGIN, y, width or CONTENT_W, 0.9, s, size=size,
                color=color or (T.TX2_D if dark else T.TX2), line=1.42)


def rule(slide, y, x=MARGIN, w=None, dark=False, color=None):
    hrule(slide, x, y, w or CONTENT_W,
          color=color or (T.RULE_D if dark else T.RULE), lw=0.9)


# ===========================================================================
# 1  title
# ===========================================================================
s = newslide("", dark=True)
rect(s, 0, 0, 0.09, SLIDE_H, fill=T.CONTENT_D)

text(s, MARGIN + 0.10, 1.28, 8.0, 0.26, "IE 643 course project  ·  prep presentation",
     size=11.5, font=T.SANS_SB, color=T.CONTENT_D, spacing=1.1, caps=True)
text(s, MARGIN + 0.10, 1.86, 10.6, 1.86,
     "Probing the Latent Space\nof Video Diffusion U-Nets",
     size=40, font=T.SANS_L, color=T.TX1_D, line=1.14)
text(s, MARGIN + 0.10, 3.72, 10.0, 0.42,
     "Where does a video model keep what a thing looks like,\n"
     "and where does it keep how the thing moves?",
     size=15, font=T.SANS, color=T.TX2_D, line=1.42)

hrule(s, MARGIN + 0.10, 4.82, 4.4, color=T.RULE_D, lw=1.0)
text(s, MARGIN + 0.10, 5.04, 6.0, 0.30, "Team Claudn't", size=16,
     font=T.SANS_SB, color=T.TX1_D)
text(s, MARGIN + 0.10, 5.42, 7.0, 0.28,
     "Roll numbers — [fill in, one per team member]",
     size=12.5, color=T.TX3_D)
text(s, MARGIN + 0.10, 6.14, 8.0, 0.52,
     "This work is carried out as part of the IE 643 course project.\n"
     "22 September 2026",
     size=11, color=T.TX3_D, line=1.45)

img(s, "title_motif", 7.62, 2.06, w=4.96)

# ===========================================================================
# 2  outline, in three acts
# ===========================================================================
s = newslide("Outline")
headline(s, "Three questions, in order")
lede(s, "Where the split should be, what we have measured so far, and what we "
        "do between now and the assessment.", y=1.26)

acts = [
    ("I", "Where would the split be,\nand who has looked?", T.CONTENT,
     ["Content and motion, and the edit that tests them",
      "How a video diffusion U-Net is put together",
      "Activation patching, and cross-attention as a lens",
      "Where three recent papers already found motion"], "slides 3–8"),
    ("II", "What we have actually\nrun and measured", T.MOTION,
     ["AnimateDiff, and a probing set we generate ourselves",
      "Five pilots: feasibility, seed control, storage, causality"],
     "slides 9–10"),
    ("III", "The five deliverables,\nand the plan behind them", T.TX1,
     ["Harness, selectivity map, patching, the demonstration, a LoRA",
      "Metrics and baselines, the frozen protocol, the roadmap, the risks"],
     "slides 11–22"),
]
y = 2.06
for tag, title, col, bullets, rng in acts:
    h = 0.46 + 0.285 * len(bullets)
    rect(s, MARGIN, y, 0.055, h, fill=col)
    text(s, MARGIN + 0.30, y - 0.04, 0.7, 0.42, tag, size=20, font=T.SANS_BK,
         color=col)
    text(s, MARGIN + 1.15, y - 0.03, 3.5, 0.72, title, size=15,
         font=T.SANS_SB, color=T.TX1, line=1.28)
    for i, b in enumerate(bullets):
        text(s, MARGIN + 5.05, y + i * 0.285, 5.3, 0.26, b, size=12.5,
             color=T.TX2)
    text(s, SLIDE_W - MARGIN - 1.3, y, 1.3, 0.26, rng, size=11,
         font=T.SANS_SB, color=T.TX3, align=PP_ALIGN.RIGHT)
    y += h + 0.34

takeaway(s, "Fifteen minutes. The background is short because the "
            "measurements are the interesting part.", y=6.56)

# ===========================================================================
# 3  the problem
# ===========================================================================
s = newslide("The question")
headline(s, "Two things get settled at once, and we want to move just one")
lede(s, "Content is what is depicted: identity, colour, texture, layout. "
        "Motion is what happens across frames: gait, deformation, the path "
        "traced through the frame. The test is an edit that changes one and "
        "leaves the other alone.", y=1.26, size=14)

img(s, "cm_matrix", MARGIN, 2.28, w=CONTENT_W)
takeaway(s, "If we can say which layers and which denoising steps hold which, "
            "we can make that edit. That is the whole project.", y=6.34)

# ===========================================================================
# 4  the U-Net factorisation   (full-bleed diagram)
# ===========================================================================
s = newslide("Background · architecture")
headline(s, "The architecture already factors the problem — structurally")
text(s, MARGIN, 1.26, CONTENT_W, 0.28,
     "Spatial blocks see one frame at a time. Temporal blocks are the only "
     "path along which information crosses frames.", size=13.5, color=T.TX2)

img(s, "unet", (SLIDE_W - 11.40) / 2, 1.78, w=11.40)

rule(s, 6.18)
text(s, MARGIN, 6.32, 8.6, 0.28,
     "Our contribution is to show the split also holds causally — a "
     "different and much stronger claim than an architectural diagram.",
     size=12.5, font=T.SANS_SB, color=T.TX2)
text(s, MARGIN + 8.7, 6.34, 3.0, 0.24,
     "Ho, NeurIPS 2022  ·  Blattmann, CVPR 2023  ·  Blattmann, 2023",
     size=9, color=T.TX3, align=PP_ALIGN.RIGHT)

# ===========================================================================
# 5  AnimateDiff
# ===========================================================================
s = newslide("Background · the model")
headline(s, "AnimateDiff hands us the split as two separate parameter sets")
lede(s, "It takes a pretrained image model, freezes it, and threads a "
        "separately trained motion module between its blocks. Content and "
        "motion live in different weights by construction — the handle we "
        "grab first.", y=1.26, size=14, width=cw(11))

img(s, "animatediff", MARGIN, 2.08, w=cw(7))

px = cx(7) + 0.22
vrule(s, px - 0.34, 2.12, 3.30, color=T.RULE)
text(s, px, 2.08, cw(5) - 0.2, 0.26, "Sizes, corrected", size=12.5,
     font=T.SANS_SB, color=T.TX1)
rows = [("SD1.5 UNet", "≈ 860M, frozen"),
        ("Motion module", "≈ 417M, trained"),
        ("On disk", "mm_sd_v15_v2, 1.6 GB"),
        ("Peak VRAM", "10.2 GB on a T4 (E1)")]
yy = 2.50
for k, v in rows:
    text(s, px, yy, 1.85, 0.24, k, size=11.5, color=T.TX3)
    text(s, px + 1.95, yy, 2.5, 0.24, v, size=11.5, font=T.SANS_SB, color=T.TX1)
    yy += 0.34
text(s, px, yy + 0.10, cw(5) - 0.2, 0.5,
     "An earlier draft of ours said 1.7B. That was the file size, not a "
     "parameter count.", size=10.5, color=T.TX3, line=1.35)

panel(s, MARGIN, 5.46, CONTENT_W, 0.90, fill=T.MOTION_WASH)
rich(s, MARGIN + 0.26, 5.62, CONTENT_W - 0.52, 0.5, [[
    ("The catch we say before anyone asks. ", dict(font=T.SANS_SB, color=T.MOTION)),
    ("The backbone was frozen, so its layers never saw motion during training. "
     "Some of the separation is an artefact of that recipe — which is why a "
     "jointly-trained model stays on the list.", dict(color=T.TX2))]],
     size=12.5, line=1.38)
text(s, MARGIN, FOOT_Y - 0.28, CONTENT_W, 0.22,
     "Guo et al., ICLR 2024  ·  Zhao et al., ECCV 2024  ·  Wu et al., ICCV 2023",
     size=9.5, color=T.TX3)

# ===========================================================================
# 6  activation patching   (full-bleed diagram)
# ===========================================================================
s = newslide("Background · method")
headline(s, "Patching turns a hunch about a layer into a causal claim")
text(s, MARGIN, 1.26, CONTENT_W, 0.28,
     "Cache what one layer produced at one timestep on clip A, then splice it "
     "into clip B's run at the same coordinate.", size=13.5, color=T.TX2)

img(s, "patching", MARGIN - 0.22, 1.80, w=CONTENT_W + 0.44)

rule(s, 5.28)
text(s, MARGIN, 5.44, cw(8), 0.80,
     "All three of these come from language-model interpretability and the "
     "causal logic carries over unchanged. We are reading Zhang and Nanda "
     "before writing any patching code: their warnings about metric choice, "
     "and about noising versus resampling corruption, describe mistakes we "
     "would otherwise make.", size=12.5, color=T.TX2, line=1.42)
text(s, cx(8) + 0.22, 5.44, cw(4) - 0.22, 0.8,
     "Meng et al., NeurIPS 2022\nWang et al., ICLR 2023\n"
     "Zhang & Nanda, ICLR 2024",
     size=10.5, color=T.TX3, line=1.62)

# ===========================================================================
# 7  cross-attention
# ===========================================================================
s = newslide("Background · method")
headline(s, "Cross-attention shows what a layer is looking at")

img(s, "attention", MARGIN, 1.58, w=cw(7) + 0.45)

px = cx(7) + 0.62
text(s, px, 1.32, cw(5) - 0.3, 0.9,
     "A cross-attention map says which regions, and which frames, a text token "
     "is read into — at every layer and every denoising step.",
     size=13.5, color=T.TX2, line=1.42)
rule(s, 2.40, x=px, w=cw(5) - 0.3)
items = [("Cheap", "No intervention, no second generation. We can look before "
                   "we commit to patching anything."),
         ("Per-token", "The map for “retriever” and the map for "
                       "“running” are different objects, which is "
                       "exactly the distinction we are chasing."),
         ("Kept as a movie", "DAAM collapses the frame axis. We keep it, "
                             "head-averaged and downsampled to 16 × 16.")]
yy = 2.60
for k, v in items:
    text(s, px, yy, cw(5) - 0.3, 0.24, k, size=12.5, font=T.SANS_SB,
         color=T.CONTENT)
    text(s, px, yy + 0.27, cw(5) - 0.3, 0.7, v, size=11.5, color=T.TX2,
         line=1.38)
    yy += 1.06
text(s, px, FOOT_Y - 0.30, cw(5) - 0.3, 0.22,
     "Hertz et al., ICLR 2023  ·  Tang et al., ACL 2023", size=9.5, color=T.TX3)

# ===========================================================================
# 8  where motion has already been found
# ===========================================================================
s = newslide("Background · prior results")
headline(s, "Three recent results tell us where to look first")

img(s, "prior_work", MARGIN, 1.42, w=cw(6) + 0.2)

px = cx(6) + 0.34
findings = [
    ("Layer", "Mid-resolution, not the extremes", T.MOTION,
     "Xiao et al. build a motion feature from U-Net activations and find the "
     "cleanest signal just after the first up-block. Deeper and shallower "
     "blocks both do worse."),
    ("Timestep", "Motion early, appearance late", T.MOTION,
     "Widely assumed, rarely tested. Baherwani et al. measured it and report "
     "two regimes with a reasonably stable boundary."),
    ("Signal", "The attention, not the features", T.TX1,
     "MotionClone finds the dominant components of temporal attention drive "
     "motion; Yatim et al. get a motion descriptor by averaging space-time "
     "features over space."),
]
yy = 1.44
for axis, claim, col, body in findings:
    text(s, px, yy, 1.3, 0.22, axis, size=10, font=T.SANS_SB, color=T.TX3,
         spacing=1.0, caps=True)
    text(s, px, yy + 0.26, cw(6) - 0.34, 0.26, claim, size=13,
         font=T.SANS_SB, color=col)
    text(s, px, yy + 0.56, cw(6) - 0.34, 0.8, body, size=11.5, color=T.TX2,
         line=1.38)
    yy += 1.46

panel(s, MARGIN, 5.92, CONTENT_W, 0.84)
rich(s, MARGIN + 0.26, 6.08, CONTENT_W - 0.52, 0.46, [[
    ("And the older framing. ", dict(font=T.SANS_SB)),
    ("MoCoGAN (2018) split content from motion by hand in the generator's "
     "latent code, and CMD (ICLR 2024) does the diffusion-era version. Both "
     "build the split in. We are asking whether a model that was never "
     "factored this way learns one anyway.", dict(color=T.TX2))]],
     size=12, line=1.38)

# ===========================================================================
# 9  models and the probing set        (table layout)
# ===========================================================================
s = newslide("Setup · models and data")
headline(s, "AnimateDiff is primary, and we generate every clip we probe")

# --- model table
colx = [MARGIN, MARGIN + 2.35, MARGIN + 3.35, MARGIN + 7.25]
colw = [2.25, 0.95, 3.80, CONTENT_W - 7.25 - 0.1]
head = ["Model", "Size", "Why it is on the list", "Why it is not primary"]
yy = 1.30
for i, h in enumerate(head):
    text(s, colx[i], yy, colw[i], 0.22, h, size=9.5, font=T.SANS_SB,
         color=T.TX3, spacing=0.9, caps=True)
yy += 0.30
hrule(s, MARGIN, yy, CONTENT_W, color=T.TX1, lw=1.0)
yy += 0.14

models = [
    ("AnimateDiff", "417M + 860M",
     "Explicit spatial / temporal split; 10.2 GB measured on a T4; mature "
     "diffusers support",
     "The motion module sits on a frozen image model, so the split is partly "
     "given", True),
    ("ModelScope T2V", "1.7B",
     "Spatial and temporal trained jointly, end to end. The harder test case",
     "Thinner tooling for attention extraction than SD-derived pipelines",
     False),
    ("Stable Video Diffusion", "≈ 1.5B",
     "Best output quality of the three, well documented",
     "Image-conditioned only, so matched text-prompt pairs are not possible",
     False),
]
for name, size_, why, whynot, primary in models:
    h = 0.62
    if primary:
        rect(s, MARGIN - 0.14, yy - 0.06, CONTENT_W + 0.28, h + 0.12,
             fill=T.CONTENT_WASH)
    text(s, colx[0], yy, colw[0], 0.24, name, size=12.5, font=T.SANS_SB,
         color=T.CONTENT if primary else T.TX1)
    if primary:
        text(s, colx[0], yy + 0.26, colw[0], 0.2, "primary", size=9.5,
             font=T.SANS_SB, color=T.CONTENT, spacing=0.9, caps=True)
    text(s, colx[1], yy, colw[1], 0.24, size_, size=9.5, font=T.MONO,
         color=T.TX2)
    text(s, colx[2], yy, colw[2], 0.56, why, size=11.5, color=T.TX2, line=1.32)
    text(s, colx[3], yy, colw[3], 0.56, whynot, size=11.5, color=T.TX2,
         line=1.32)
    yy += h + 0.10
    hrule(s, MARGIN, yy - 0.05, CONTENT_W, color=T.RULE, lw=0.75)

# --- probing set + pretraining data, two panels
yy = 3.82
panel(s, MARGIN, yy, cw(7), 2.42)
text(s, MARGIN + 0.26, yy + 0.20, cw(7) - 0.52, 0.24,
     "The probing set, frozen 22 September", size=12.5, font=T.SANS_SB,
     color=T.TX1)
rich(s, MARGIN + 0.26, yy + 0.56, cw(7) - 0.52, 1.4, [
    [("15 subjects × 4 verbs × 2 scenes = 120 clips per seed", dict(font=T.SANS_SB)),
     ("; three seeds (42, 1337, 2024) gives 360.", {})],
    [("Subjects split into two body-plan families — 8 quadrupeds, 7 humans — "
      "so a noun swap does not drag the layout with it.", {})],
    [("104 within-family content pairs · 90 motion pairs · 56 cross-family, "
      "reported separately as a labelled hard subset.", {})],
], size=11, color=T.TX2, line=1.36, space_after=5)

panel(s, cx(7) + 0.22, yy, cw(5) - 0.22, 2.42, fill=T.PANEL)
text(s, cx(7) + 0.48, yy + 0.20, cw(5) - 0.74, 0.24,
     "What these models were pretrained on", size=12.5, font=T.SANS_SB,
     color=T.TX1)
rich(s, cx(7) + 0.48, yy + 0.56, cw(5) - 0.74, 1.4, [
    [("SD1.5 backbone: ", dict(font=T.SANS_SB)),
     ("LAION-5B subsets — web-scraped image–text pairs, with known "
      "demographic and aesthetic bias.", {})],
    [("Motion module: ", dict(font=T.SANS_SB)),
     ("WebVid-10M — watermarked stock footage, biased towards slow camera "
      "motion.", {})],
    [("Both biases show in our own pilots. We are characterising this model, "
      "not video diffusion in general.", dict(color=T.TX3))],
], size=10.5, color=T.TX2, line=1.36, space_after=4)

text(s, MARGIN, FOOT_Y - 0.28, CONTENT_W, 0.22,
     "Vocabulary sampled from MSR-VTT and WebVid-10M captions; verbs from "
     "UCF101 and Kinetics-400 labels. We train on none of these.",
     size=10.5, color=T.TX3)

# ===========================================================================
# 10  the five pilots         (table layout)
# ===========================================================================
s = newslide("Work so far")
headline(s, "Five cheap pilots, and four decisions that came out of them")
text(s, MARGIN, 1.26, CONTENT_W, 0.26,
     "Small and deliberately cheap. The point was to find out whether the "
     "project is tractable, not to produce results.", size=13, color=T.TX2)

cxs = [MARGIN, MARGIN + 0.62, MARGIN + 3.55, MARGIN + 7.00]
cws = [0.55, 2.85, 3.35, CONTENT_W - 7.00]
yy = 1.74
for i, h in enumerate(["", "The question", "What we ran", "What we saw"]):
    if h:
        text(s, cxs[i], yy, cws[i], 0.2, h, size=9.5, font=T.SANS_SB,
             color=T.TX3, spacing=0.9, caps=True)
yy += 0.28
hrule(s, MARGIN, yy, CONTENT_W, color=T.TX1, lw=1.0)
yy += 0.13

pilots = [
    ("E1", "Can we generate video at all on our hardware?",
     "AnimateDiff v1.5 through diffusers. 16 frames, 512×512, 25 DDIM steps, "
     "fp16.",
     [("Runs. ", T.TX1), ("94 s per clip, 10.2 GB VRAM on a T4. ", T.CONTENT),
      ("This is the unit cost behind every budget in the plan.", T.TX2)]),
    ("E2", "Does a fixed seed hold the scene still when one word changes?",
     "Same noise tensor; prompt varied on one axis at a time "
     "(retriever→cat, running→walking).",
     [("Yes, within a body plan. ", T.TX1),
      ("Noun swaps shift appearance 2.6× more than verb swaps; layout SSIM "
       "0.51 for horse against 0.26 for bird.", T.TX2)]),
    ("E3", "Can we get activations out, and what does storage cost?",
     "Forward hooks on the motion_modules and attn2; dumped shapes and sizes.",
     [("Easy to hook: 95 coordinates, +0.6% runtime. ", T.TX1),
      ("A full dump is 88 GB per clip, larger than the disk, so we store "
       "45 MB of summaries.", T.TX2)]),
    ("E4", "Do the temporal modules causally carry motion?",
     "Scaled the motion-module residual towards zero, everything else "
     "identical.",
     [("The clip goes near-static while each frame stays a plausible image. ",
       T.TX1),
      ("The cheapest possible causal demonstration.", T.TX2)]),
    ("E5", "Can we train a LoRA on this hardware, and at what clip length?",
     "Stage 1, this week. One motion LoRA, 100 steps, rank 16, gradient "
     "checkpointing.",
     [("Pending. ", T.MOTION),
      ("E1 set the inference budget before the sweep; E5 does the same for "
       "training.", T.TX2)]),
]
for eid, q, ran, saw in pilots:
    h = 0.72
    text(s, cxs[0], yy + 0.02, cws[0], 0.24, eid, size=13, font=T.SANS_BK,
         color=T.MOTION if eid == "E5" else T.CONTENT)
    text(s, cxs[1], yy, cws[1], 0.7, q, size=11, color=T.TX1, line=1.34)
    text(s, cxs[2], yy, cws[2], 0.7, ran, size=10.5, color=T.TX3, line=1.34)
    rich(s, cxs[3], yy, cws[3], 0.7,
         [[(t, dict(color=c, font=T.SANS_SB if c in (T.TX1, T.CONTENT, T.MOTION)
                    else T.SANS)) for t, c in saw]],
         size=10.8, line=1.34)
    yy += h
    hrule(s, MARGIN, yy - 0.06, CONTENT_W, color=T.RULE, lw=0.75)

panel(s, MARGIN, 5.66, CONTENT_W, 0.86, fill=T.CONTENT_WASH)
rich(s, MARGIN + 0.26, 5.82, CONTENT_W - 0.52, 0.5, [[
    ("Four decisions came out of this. ", dict(font=T.SANS_SB, color=T.CONTENT)),
    ("AnimateDiff is primary (E1, E4) · every probing clip shares a fixed seed "
     "and noise tensor (E2) · the harness stores summaries, not raw tensors "
     "(E3) · content pairs stay inside a body-plan family (E2).",
     dict(color=T.TX2))]], size=12, line=1.38)

# ===========================================================================
# 11  divider: our approach       (dark)
# ===========================================================================
s = newslide("Our approach", dark=True)
text(s, MARGIN, 1.30, 3.0, 0.26, "Part two", size=11.5, font=T.SANS_SB,
     color=T.CONTENT_D, spacing=1.1, caps=True)
text(s, MARGIN, 1.74, 10.0, 0.7, "Five deliverables", size=34,
     font=T.SANS_L, color=T.TX1_D)
img(s, "chain", MARGIN - 0.22, 3.00, w=CONTENT_W + 0.44)
text(s, MARGIN, 5.62, CONTENT_W, 0.5,
     "The trained model is numbered, not an afterthought: done the way "
     "Deliverable 5 describes, it tests Deliverable 2 rather than sitting "
     "beside it.", size=13, color=T.TX2_D, line=1.40)

# ===========================================================================
# 12  deliverable 1 — the harness
# ===========================================================================
s = newslide("Deliverable 1 · harness")
headline(s, "A full dump is larger than the disk, so we store summaries")

steps = [
    ("01", "Hook both paths by name",
     "diffusers builds a UNetMotionModel from a UNet2DConditionModel plus a "
     "MotionAdapter. Temporal: the motion_modules in each down, mid and up "
     "block. Spatial: the transformer_blocks, attn1 and attn2."),
    ("02", "Index everything by one coordinate",
     "A triple of (module path, denoising timestep, clip id). Every later "
     "stage queries by that triple, so it is worth getting right once."),
    ("03", "Sweep in two tiers",
     "A coarse pass over 16 representative coordinates × 25 timesteps × 3 "
     "seeds, then a refined pass at full module resolution inside whichever "
     "block wins."),
    ("04", "Pin down determinism",
     "Each clip records its seed, sampler, step count, resolution, repository "
     "id and commit hash next to its activations."),
]
yy = 1.42
for num, title, body in steps:
    text(s, MARGIN, yy + 0.01, 0.42, 0.24, num, size=11.5, font=T.SANS_BK,
         color=T.CONTENT)
    text(s, MARGIN + 0.52, yy, cw(6) - 0.52, 0.24, title, size=12.5,
         font=T.SANS_SB, color=T.TX1)
    text(s, MARGIN + 0.52, yy + 0.28, cw(6) - 0.52, 0.8, body, size=11,
         color=T.TX2, line=1.36)
    yy += 1.18

px = cx(6) + 0.30
img(s, "storage", px, 1.42, w=cw(6) - 0.30)
rich(s, px, 4.02, cw(6) - 0.30, 0.9, [
    [("What we keep per coordinate. ", dict(font=T.SANS_SB)),
     ("Channel means and variances, the norms of the frame-mean-removed "
      "residual, and cross-attention maps head-averaged and downsampled to a "
      "16 × 16 grid with the frame axis kept — for the subject, verb and "
      "scene tokens only. Full tensors are retained only where we flag a "
      "coordinate as interesting.", dict(color=T.TX2))]],
     size=11.5, line=1.38)
panel(s, px, 5.42, cw(6) - 0.30, 0.86, fill=T.MOTION_WASH)
text(s, px + 0.22, 5.56, cw(6) - 0.74, 0.60,
     "Still open: at 16 × 16 the attention maps are 78% of Kaggle's 20 GB "
     "output cap. An 8 × 8 grid halves the store. Decide before Stage 2 "
     "freezes the format.", size=11, color=T.MOTION, line=1.36)

takeaway(s, "Hooks and generation prototyped in E1 and E3. The production "
            "harness is Stage 2, 28 September to 4 October.", y=6.52)

# ===========================================================================
# 13  deliverable 1 — the probing set
# ===========================================================================
s = newslide("Deliverable 1 · probing set")
headline(s, "Change exactly one word — and hold the noise tensor too")
lede(s, "The probe measures how two clips differ. If anything but the axis "
        "we care about also differs, the measurement is worthless.",
        y=1.26, size=13.5)

img(s, "pairs", MARGIN, 1.72, w=cw(6) + 0.35)
img(s, "e2_ssim", cx(6) + 0.55, 1.90, w=cw(6) - 0.55)

rule(s, 4.62)
cols = [
    ("Pairs are labelled by construction",
     "104 content pairs, 90 motion pairs, 56 cross-family. The anchor design "
     "fixes which pairs exist rather than leaving it to whichever we happened "
     "to use."),
    ("Three seeds, not one",
     "42, 1337, 2024. A pair always lives inside one seed; the spread across "
     "seeds is the error bar on the map."),
    ("Vocabulary borrowed, not invented",
     "Subjects and scenes from MSR-VTT and WebVid-10M captions; verbs from "
     "UCF101 and Kinetics-400 action labels."),
]
for i, (k, v) in enumerate(cols):
    x = MARGIN + i * (CONTENT_W / 3)
    w = CONTENT_W / 3 - 0.40
    text(s, x, 4.80, w, 0.24, k, size=12, font=T.SANS_SB, color=T.TX1)
    text(s, x, 5.10, w, 0.9, v, size=11, color=T.TX2, line=1.38)

takeaway(s, "Without the fixed noise tensor, the difference between two clips "
            "is mostly sampling noise. E2 is what told us this is viable.",
         y=6.30)

# ===========================================================================
# 14  deliverable 2 — three predictions
# ===========================================================================
s = newslide("Deliverable 2 · predictions")
headline(s, "Three predictions, each cheap to falsify")
text(s, MARGIN, 1.26, CONTENT_W, 0.26,
     "Sweeping blind produces a description. Going in with predictions "
     "produces a result.", size=13, color=T.TX2)

hyps = [
    ("H1", "Layer axis", T.MOTION,
     "Motion selectivity peaks in the temporal modules at medium resolution, "
     "around up_blocks[1], rather than at the highest-resolution decoder "
     "layers or at the bottleneck.",
     "Follows Xiao et al. Decided by argmax over blocks with a bootstrap CI; "
     "falsified if that interval reaches an extreme."),
    ("H2", "Timestep axis", T.MOTION,
     "Motion is settled in roughly the first third of the denoising "
     "trajectory and appearance in the last. Content selectivity should run "
     "the opposite way.",
     "Follows Baherwani et al. Decided by a changepoint fit with a bootstrap "
     "CI. A different boundary is a finding, not a bug."),
    ("H3", "The confound", T.CONTENT,
     "Raw activations look content-selective everywhere. A clean motion "
     "signal only appears after subtracting the per-frame mean, which removes "
     "the static component.",
     "The one that matters most. Skip it and we would wrongly conclude motion "
     "is nowhere."),
]
w = (CONTENT_W - 2 * 0.34) / 3
for i, (tag, axis, col, claim, how) in enumerate(hyps):
    x = MARGIN + i * (w + 0.34)
    rect(s, x, 1.70, w, 0.045, fill=col)
    text(s, x, 1.86, w, 0.28, tag, size=17, font=T.SANS_BK, color=col)
    text(s, x + 0.62, 1.92, w - 0.62, 0.24, axis, size=11.5, font=T.SANS_SB,
         color=T.TX3)
    text(s, x, 2.28, w, 1.0, claim, size=12, color=T.TX1, line=1.36)
    hrule(s, x, 3.40, w, color=T.RULE, lw=0.75)
    text(s, x, 3.54, w, 0.8, how, size=10.5, color=T.TX3, line=1.36)

rule(s, 4.42)
text(s, MARGIN, 4.58, cw(6), 0.24, "H3 is no longer only a prediction — E2 "
     "already measured it", size=12.5, font=T.SANS_SB, color=T.CONTENT)
text(s, MARGIN, 4.90, cw(6) - 0.2, 1.4,
     "Our raw pixel motion proxy read higher for noun swaps (0.46) than for "
     "verb swaps (0.39). Appearance swamped the motion measure before we "
     "touched a single activation — which is exactly why the map is computed "
     "on the frame-mean-removed residual, and why the static map is printed "
     "beside it as a control.",
     size=11.5, color=T.TX2, line=1.42)
img(s, "e2_swaps", SLIDE_W - MARGIN - 4.80, 4.20, w=4.80)

# ===========================================================================
# 15  deliverable 2 — the map
# ===========================================================================
s = newslide("Deliverable 2 · the map")
headline(s, "One number per layer and timestep, and its sign is the point")

img(s, "selectivity", MARGIN, 1.36, w=cw(6) + 0.85)

px = cx(6) + 1.05
# the formula, set as real text with real subscripts
box, tf = tb(s, px, 1.38, cw(6) - 1.05, 0.5)
p = tf.paragraphs[0]
run(p, "S(L, t)  =  ", size=19, font=T.SANS_SB, color=T.TX1)
run(p, "d", size=19, font=T.SANS_SB, color=T.CONTENT)
run(p, "c", size=19, font=T.SANS_SB, color=T.CONTENT, baseline=-30)
run(p, " − ", size=19, font=T.SANS_SB, color=T.TX1)
run(p, "d", size=19, font=T.SANS_SB, color=T.MOTION)
run(p, "m", size=19, font=T.SANS_SB, color=T.MOTION, baseline=-30)
box2, tf2 = tb(s, px + 1.44, 1.82, cw(6) - 1.05, 0.5)
p2 = tf2.paragraphs[0]
run(p2, "d", size=19, font=T.SANS_SB, color=T.CONTENT)
run(p2, "c", size=19, font=T.SANS_SB, color=T.CONTENT, baseline=-30)
run(p2, " + ", size=19, font=T.SANS_SB, color=T.TX1)
run(p2, "d", size=19, font=T.SANS_SB, color=T.MOTION)
run(p2, "m", size=19, font=T.SANS_SB, color=T.MOTION, baseline=-30)
hrule(s, px + 1.40, 1.78, 1.62, color=T.TX1, lw=1.1)

rich(s, px, 2.46, cw(6) - 1.05, 0.9, [[
    ("d", dict(font=T.SANS_SB, color=T.CONTENT)),
    ("c", dict(font=T.SANS_SB, color=T.CONTENT, baseline=-30)),
    (" is how far the activations move under a content swap, ",
     dict(color=T.TX2)),
    ("d", dict(font=T.SANS_SB, color=T.MOTION)),
    ("m", dict(font=T.SANS_SB, color=T.MOTION, baseline=-30)),
    (" how far under a motion swap, for a pair that shares a seed.",
     dict(color=T.TX2))]], size=12.5, line=1.40)

items = [
    ("Positive", "the coordinate tracks what is depicted", T.CONTENT),
    ("Negative", "it tracks how things move", T.MOTION),
]
yy = 3.60
for k, v, col in items:
    rect(s, px, yy + 0.05, 0.14, 0.14, fill=col)
    text(s, px + 0.28, yy, 1.0, 0.22, k, size=11.5, font=T.SANS_SB, color=col)
    text(s, px + 1.10, yy, cw(6) - 2.15, 0.22, v, size=11.5, color=T.TX2)
    yy += 0.34

text(s, px, 4.44, cw(6) - 1.05, 0.9,
     "Distance is one minus cosine similarity, fixed before the sweep and not "
     "after seeing it. H3 makes this two maps: the dynamic one is the "
     "headline, the static one sits beside it as the control.",
     size=11.5, color=T.TX2, line=1.40)

panel(s, px, 5.38, cw(6) - 1.05, 1.08, fill=T.PANEL)
text(s, px + 0.22, 5.52, cw(6) - 1.39, 0.9,
     "A second, independent view: a linear probe per coordinate says "
     "what information is present. Patching says what the model actually "
     "uses. Where they disagree is likely the most interesting thing we find.",
     size=10.5, color=T.TX2, line=1.38)

# ===========================================================================
# 16  deliverable 3 — patching framework
# ===========================================================================
s = newslide("Deliverable 3 · patching")
headline(s, "Four granularities, and three controls we run first")

img(s, "granularity", MARGIN, 1.46, w=cw(7) + 0.5)

px = cx(7) + 0.72
text(s, px, 1.48, cw(5) - 0.72, 0.24, "Controls we run first", size=12.5,
     font=T.SANS_SB, color=T.TX1)
ctl = [
    ("Null patch", "Patch a clip with its own activation. In fp16 nothing is "
                   "bit-identical, so the bar is numerical: latent MAE below "
                   "1e-3, PSNR above 45 dB."),
    ("Random coordinate", "Patch somewhere with low selectivity. If that "
                          "works as well as our chosen coordinate, we have "
                          "not localised anything."),
    ("Symmetry", "A into B, then B into A. The two effects should be near "
                 "inverses. Asymmetry means something is leaking."),
]
yy = 1.84
for k, v in ctl:
    rect(s, px, yy + 0.05, 0.14, 0.14, fill=T.MOTION)
    text(s, px + 0.28, yy, cw(5) - 1.00, 0.22, k, size=12, font=T.SANS_SB,
         color=T.TX1)
    text(s, px + 0.28, yy + 0.26, cw(5) - 1.00, 0.9, v, size=11, color=T.TX2,
         line=1.36)
    yy += 1.12

panel(s, MARGIN, 5.40, CONTENT_W, 0.90, fill=T.PANEL_2)
rich(s, MARGIN + 0.26, 5.56, CONTENT_W - 0.52, 0.52, [[
    ("Both runs must start from the same noise tensor, ", dict(font=T.SANS_SB)),
    ("or the comparison means nothing. Corruption is resampling rather than "
     "noising, after Zhang and Nanda. Design is settled; implementation is "
     "Stage 5, 19 to 25 October.", dict(color=T.TX2))]], size=12, line=1.38)

# ===========================================================================
# 17  deliverable 4 — the demonstration
# ===========================================================================
s = newslide("Deliverable 4 · the demonstration")
headline(s, "The demonstration is a 2 × 2, and both diagonals have to work")
text(s, MARGIN, 1.26, CONTENT_W, 0.26,
     "Two edits per test case, each with an unambiguous target, across twelve "
     "test pairs in both directions.", size=13, color=T.TX2)

img(s, "demo", MARGIN - 0.22, 1.72, w=CONTENT_W + 0.44)

rule(s, 5.20)
text(s, MARGIN, 5.36, cw(7), 0.8,
     "The four placeholders above are replaced with real frames in Stage 6, "
     "26 October to 1 November. Until then they are placeholders and we say "
     "so rather than showing something suggestive.",
     size=12, color=T.TX2, line=1.40)
text(s, cx(7) + 0.30, 5.36, cw(5) - 0.3, 0.8,
     "Alongside the grid we report the α sweep — the dose-response curve from "
     "granularity 4 — because a single hard swap hides whether the effect is "
     "graded or a cliff.", size=12, color=T.TX2, line=1.40)

# ===========================================================================
# 18  deliverable 5 — the trained model
# ===========================================================================
s = newslide("Deliverable 5 · the trained model")
headline(s, "Train the motion LoRA twice — and the comparison is the point")

img(s, "lora", MARGIN - 0.22, 1.36, w=CONTENT_W + 0.44)

rule(s, 4.42)
threes = [
    ("A third view, nearly free",
     "‖ΔW‖ per module ranks layers by where training chose to put the motion."),
    ("Three rankings, one question",
     "Patching says used, the linear probe says present, ΔW says learned. We "
     "report the rank correlation between all three."),
    ("Ordering is forced",
     "Nothing starts before the Stage 4 map exists, because the probe-guided "
     "variant depends on it."),
]
for i, (k, v) in enumerate(threes):
    x = MARGIN + i * (CONTENT_W / 3)
    w = CONTENT_W / 3 - 0.40
    text(s, x, 4.60, w, 0.24, k, size=12, font=T.SANS_SB, color=T.TX1)
    text(s, x, 4.90, w, 0.9, v, size=11, color=T.TX2, line=1.38)

panel(s, MARGIN, 5.82, CONTENT_W, 0.84, fill=T.PANEL_2)
rich(s, MARGIN + 0.26, 5.98, CONTENT_W - 0.52, 0.44, [[
    ("Cost and disclosure. ", dict(font=T.SANS_SB)),
    ("Four training runs, about 6 GPU-hours, in Stage 6. The LoRA pair is "
     "fitted on roughly 20 real clips from UCF101 and Kinetics-400 — a "
     "bounded use of real video that touches nothing in Deliverables 1 to 4, "
     "and it is in the report's data section.", dict(color=T.TX2))]],
     size=11.5, line=1.36)

# ===========================================================================
# 19  evaluation
# ===========================================================================
s = newslide("Evaluation")
headline(s, "One headline number, three baselines, one sanity check")

met = [
    ("Content fidelity", T.CONTENT,
     "CLIP similarity to the intended content prompt, plus CLIP or DINO "
     "image-to-image similarity against the reference frames. The "
     "image-to-image version is the honest one."),
    ("Motion fidelity", T.MOTION,
     "Trajectory-based, after Yatim et al.: point tracks from CoTracker, "
     "compared as displacement vectors over time. RAFT flow correlation as a "
     "cheaper second opinion."),
    ("Did we wreck the clip?", T.TX1,
     "Two or three VBench dimensions — subject consistency, motion "
     "smoothness, temporal flickering. Cheap to run, and it answers the "
     "obvious objection before it is raised."),
]
w = (CONTENT_W - 2 * 0.34) / 3
for i, (k, col, v) in enumerate(met):
    x = MARGIN + i * (w + 0.34)
    rect(s, x, 1.40, w, 0.045, fill=col)
    text(s, x, 1.58, w, 0.24, k, size=12.5, font=T.SANS_SB, color=col)
    text(s, x, 1.88, w, 1.0, v, size=11.5, color=T.TX2, line=1.38)

panel(s, MARGIN, 3.04, CONTENT_W, 1.16, fill=T.PANEL)
box, tf = tb(s, MARGIN + 0.30, 3.28, 7.4, 0.4)
p = tf.paragraphs[0]
for txt, col, bl in [("D  =  ½[ s", T.TX1, None), ("c", T.CONTENT, -30),
                     ("(out, B) + s", T.TX1, None), ("m", T.MOTION, -30),
                     ("(out, A) ]  −  ½[ s", T.TX1, None), ("c", T.CONTENT, -30),
                     ("(out, A) + s", T.TX1, None), ("m", T.MOTION, -30),
                     ("(out, B) ]", T.TX1, None)]:
    run(p, txt, size=16, font=T.SANS_SB, color=col, baseline=bl)
text(s, MARGIN + 0.30, 3.76, 7.6, 0.32,
     "for an output meant to take B's content and A's motion",
     size=11, color=T.TX3)
text(s, MARGIN + 7.85, 3.22, CONTENT_W - 8.05, 0.92,
     "Both terms are min–max normalised across the evaluation set first: a "
     "CLIP cosine and a flow correlation are not commensurable. Raw values "
     "are printed alongside.", size=10.5, color=T.TX2, line=1.40)

text(s, MARGIN, 4.52, CONTENT_W, 0.24,
     "Baselines — without these, the number means nothing", size=12.5,
     font=T.SANS_SB, color=T.TX1)
base = [
    ("Prompt-only", "Same seed, new prompt, no patching. If we cannot beat "
                    "this, we have built an expensive way to do nothing. "
                    "Mandatory."),
    ("Training-free", "MotionClone. Public code, comparable in spirit to what "
                      "we are doing. Decided, not “or MOFT”."),
    ("Trained, our own", "The probe-guided LoRA pair. Cheaper than running "
                         "MotionDirector's pipeline and directly comparable."),
]
for i, (k, v) in enumerate(base):
    x = MARGIN + i * (CONTENT_W / 3)
    ww = CONTENT_W / 3 - 0.40
    text(s, x, 4.88, ww, 0.22, k, size=11.5, font=T.SANS_SB, color=T.MOTION)
    text(s, x, 5.16, ww, 0.9, v, size=11, color=T.TX2, line=1.38)

takeaway(s, "Success: beat prompt-only in both directions with a bootstrap CI "
            "excluding zero, and stay within 10% of source on VBench subject "
            "consistency.", y=6.30)

# ===========================================================================
# 20  the frozen protocol
# ===========================================================================
s = newslide("Method · protocol")
headline(s, "Twenty-two decisions, written down before the data arrives")
text(s, MARGIN, 1.34, CONTENT_W, 0.26,
     "A method whose choices are still open when the data lands is a method "
     "that can be fitted to whatever came out.", size=13, color=T.TX2)

left = [
    ("The distance behind the map",
     "One minus cosine similarity — on the frame-mean-removed residual for "
     "motion, on the frame mean for content."),
    ("Two maps, not one",
     "S_dyn is the headline, S_stat is printed beside it. That is what H3 "
     "implies: demonstrate the confound instead of asserting it."),
    ("Pairs, seeds and vocabulary",
     "Anchor design: 104 content pairs, 90 motion pairs, three seeds over the "
     "whole grid, enumerated in a config file."),
]
right = [
    ("The null-patch tolerance",
     "fp16 is never bit-identical, so “identical” needed a number: "
     "latent MAE below 1e-3, PSNR above 45 dB."),
    ("D was not commensurable",
     "It added a CLIP cosine to a flow correlation. Both terms are now "
     "min–max normalised, with raw values printed alongside."),
    ("What would count as success",
     "Beat prompt-only in both directions with a bootstrap CI excluding zero, "
     "within 10% of source on VBench subject consistency."),
]
for ci, (head_, items) in enumerate([("Implicit before, fixed now", left),
                                     ("Thresholds, decided in advance", right)]):
    x = MARGIN + ci * (cw(6) + 0.36)
    ww = cw(6) - 0.18
    text(s, x, 1.86, ww, 0.22, head_, size=10, font=T.SANS_SB, color=T.TX3,
         spacing=1.0, caps=True)
    hrule(s, x, 2.16, ww, color=T.TX1, lw=1.0)
    yy = 2.32
    for i, (k, v) in enumerate(items):
        text(s, x, yy + 0.02, 0.32, 0.22, str(i + 1), size=11.5,
             font=T.SANS_BK, color=T.CONTENT if ci == 0 else T.MOTION)
        text(s, x + 0.42, yy, ww - 0.42, 0.22, k, size=12, font=T.SANS_SB,
             color=T.TX1)
        text(s, x + 0.42, yy + 0.28, ww - 0.42, 0.7, v, size=11, color=T.TX2,
             line=1.36)
        yy += 1.18

panel(s, MARGIN, 6.00, CONTENT_W, 0.76, fill=T.CONTENT_WASH)
text(s, MARGIN + 0.26, 6.16, CONTENT_W - 0.52, 0.5,
     "All twenty-two are listed in §5.7 of the roadmap, dated and "
     "append-only, so a choice made in September is not quietly remade in "
     "October to suit the data.", size=12, font=T.SANS_SB, color=T.CONTENT,
     line=1.36)

# ===========================================================================
# 21  roadmap and budget
# ===========================================================================
s = newslide("Roadmap")
headline(s, "Eight stages, and Stage 2 is the one most likely to slip")

img(s, "budget", MARGIN - 0.22, 1.32, w=CONTENT_W + 0.44)

rule(s, 5.06)
notes = [
    ("Where the slack is",
     "Video inference and hook-based extraction are slower and more fragile "
     "than the image-diffusion equivalent. If Stage 2 overruns, the time "
     "comes out of Stage 5, not Stage 3: a rushed patching implementation is "
     "recoverable, a rushed sweep is not."),
    ("Why the totals carry ±10%",
     "E1 measured 94 s per clip with a within-session SD of 0.1 s, but "
     "throughput varies about 10% between Kaggle sessions — E3's no-hook "
     "baseline on a later session read 85.1 s. Hence 34.7 ± 3.5."),
    ("What the tiering bought",
     "A naive sweep of 9 layers × 7 timesteps × 12 pairs × 2 directions is "
     "1,512 generations, 39 GPU-hours on its own. The two-tier plan exists "
     "because of that arithmetic, not for elegance."),
]
for i, (k, v) in enumerate(notes):
    x = MARGIN + i * (CONTENT_W / 3)
    ww = CONTENT_W / 3 - 0.40
    text(s, x, 5.22, ww, 0.22, k, size=12, font=T.SANS_SB, color=T.TX1)
    text(s, x, 5.50, ww, 1.1, v, size=10.5, color=T.TX2, line=1.38)

# ===========================================================================
# 22  risks
# ===========================================================================
s = newslide("Risks")
headline(s, "Four things that could sink this, and what we do about each")

risks = [
    ("Compute limits the layer × timestep sweep",
     "Start coarse — 16 representative coordinates, then refine only around "
     "what looks promising. E1 measured 94 s per clip and E3 adds only 0.6% "
     "for hooks; the tiered plan totals 34.7 GPU-hours against a 30 GPU-h "
     "weekly quota, where the naive sweep alone would have needed 39."),
    ("AnimateDiff's split is too easy to be a finding",
     "Fair criticism, and we say it up front. The answer is ModelScope T2V, "
     "where spatial and temporal were trained together. The probe-guided LoRA "
     "is the cheaper second answer: it asks whether the map predicts "
     "anything, not whether it echoes the architecture."),
    ("Patched outputs come out incoherent rather than wrong",
     "Run the null patch and the random-coordinate control before believing "
     "anything. Prefer scaled injection over hard replacement, and report the "
     "whole α curve rather than one point on it."),
    ("Everything looks content-selective and we find nothing",
     "This is H3, and it is the failure mode we most expect. Subtract the "
     "per-frame mean before scoring. E2 already showed the confound at pixel "
     "level; reading Xiao et al. saved us a week here."),
]
yy = 1.44
for i, (r, a) in enumerate(risks):
    h = 1.16
    if i % 2 == 0:
        rect(s, MARGIN - 0.14, yy - 0.08, CONTENT_W + 0.28, h, fill=T.PANEL_2)
    text(s, MARGIN, yy, cw(4) - 0.1, 0.62, r, size=12.5, font=T.SANS_SB,
         color=T.TX1, line=1.32)
    text(s, cx(4) + 0.10, yy + 0.02, 0.26, 0.22, "→", size=13,
         font=T.SANS_SB, color=T.MOTION)
    text(s, cx(4) + 0.46, yy, CONTENT_W - cw(4) - 0.56, 0.9, a, size=11.5,
         color=T.TX2, line=1.36)
    yy += h + 0.08

takeaway(s, "Open questions for the TA are listed in the roadmap document, "
            "including what counts as an acceptable interface for this project.",
         y=6.46)

# ===========================================================================
# 23–24  references
# ===========================================================================
REFS_A = [
    "Ho, Jonathan, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad "
    "Norouzi and David J. Fleet. Video Diffusion Models, NeurIPS, 2022. "
    "https://arxiv.org/abs/2204.03458",
    "Blattmann, Andreas, Robin Rombach, Huan Ling, Tim Dockhorn, Seung Wook "
    "Kim, Sanja Fidler and Karsten Kreis. Align your Latents: High-Resolution "
    "Video Synthesis with Latent Diffusion Models, CVPR, 2023. "
    "https://arxiv.org/abs/2304.08818",
    "Blattmann, Andreas, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, "
    "Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, "
    "Adam Letts, Varun Jampani and Robin Rombach. Stable Video Diffusion: "
    "Scaling Latent Video Diffusion Models to Large Datasets, arXiv, 2023. "
    "https://arxiv.org/abs/2311.15127",
    "Guo, Yuwei, Ceyuan Yang, Anyi Rao, Zhengyang Liang, Yaohui Wang, Yu Qiao, "
    "Maneesh Agrawala, Dahua Lin and Bo Dai. AnimateDiff: Animate Your "
    "Personalized Text-to-Image Diffusion Models without Specific Tuning, "
    "ICLR, 2024. https://arxiv.org/abs/2307.04725",
    "Zhao, Rui, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jiawei Liu, "
    "Weijia Wu, Jussi Keppo and Mike Zheng Shou. MotionDirector: Motion "
    "Customization of Text-to-Video Diffusion Models, ECCV, 2024. "
    "https://arxiv.org/abs/2310.08465",
    "Wu, Jay Zhangjie, Yixiao Ge, Xintao Wang, Stan Weixian Lei, Yuchao Gu, "
    "Yufei Shi, Wynne Hsu, Ying Shan, Xiaohu Qie and Mike Zheng Shou. "
    "Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video "
    "Generation, ICCV, 2023. https://arxiv.org/abs/2212.11565",
    "Meng, Kevin, David Bau, Alex Andonian and Yonatan Belinkov. Locating and "
    "Editing Factual Associations in GPT, NeurIPS, 2022. "
    "https://arxiv.org/abs/2202.05262",
    "Wang, Kevin, Alexandre Variengien, Arthur Conmy, Buck Shlegeris and Jacob "
    "Steinhardt. Interpretability in the Wild: a Circuit for Indirect Object "
    "Identification in GPT-2 Small, ICLR, 2023. "
    "https://arxiv.org/abs/2211.00593",
    "Zhang, Fred and Neel Nanda. Towards Best Practices of Activation Patching "
    "in Language Models: Metrics and Methods, ICLR, 2024. "
    "https://arxiv.org/abs/2309.16042",
    "Hertz, Amir, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch and "
    "Daniel Cohen-Or. Prompt-to-Prompt Image Editing with Cross Attention "
    "Control, ICLR, 2023. https://arxiv.org/abs/2208.01626",
    "Tang, Raphael, Akshat Pandey, Zhiying Jiang, Gefei Yang, Karun Kumar, "
    "Jimmy Lin and Ferhan Ture. What the DAAM: Interpreting Stable Diffusion "
    "Using Cross Attention, ACL, 2023. https://arxiv.org/abs/2210.04885",
    "Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, "
    "Shean Wang, Lu Wang and Weizhu Chen. LoRA: Low-Rank Adaptation of Large "
    "Language Models, ICLR, 2022. https://arxiv.org/abs/2106.09685",
]
REFS_B = [
    "Xiao, Zeqi, Yifan Zhou, Shuai Yang and Xingang Pan. Video Diffusion "
    "Models are Training-free Motion Interpreter and Controller, NeurIPS, "
    "2024. https://arxiv.org/abs/2405.14864",
    "Yatim, Danah, Rafail Fridman, Omer Bar-Tal, Yoni Kasten and Tali Dekel. "
    "Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer, "
    "CVPR, 2024. https://arxiv.org/abs/2311.17009",
    "Ling, Pengyang, Jiazi Bu, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Tong Wu, "
    "Huaian Chen, Jiaqi Wang and Yi Jin. MotionClone: Training-Free Motion "
    "Cloning for Controllable Video Generation, ICLR, 2025. "
    "https://arxiv.org/abs/2406.05338",
    "Baherwani, Vatsal, Yixuan Ren and Abhinav Shrivastava. Characterizing "
    "Motion Encoding in Video Diffusion Timesteps, arXiv, 2025. "
    "https://arxiv.org/abs/2512.22175",
    "Kwon, Mingi, Jaeseok Jeong and Youngjung Uh. Diffusion Models already "
    "have a Semantic Latent Space, ICLR, 2023. "
    "https://arxiv.org/abs/2210.10960",
    "Tulyakov, Sergey, Ming-Yu Liu, Xiaodong Yang and Jan Kautz. MoCoGAN: "
    "Decomposing Motion and Content for Video Generation, CVPR, 2018. "
    "https://arxiv.org/abs/1707.04993",
    "Yu, Sihyun, Weili Nie, De-An Huang, Boyi Li, Jinwoo Shin and Anima "
    "Anandkumar. Efficient Video Diffusion Models via Content-Frame "
    "Motion-Latent Decomposition, ICLR, 2024. "
    "https://arxiv.org/abs/2403.14148",
    "Teed, Zachary and Jia Deng. RAFT: Recurrent All-Pairs Field Transforms "
    "for Optical Flow, ECCV, 2020. https://arxiv.org/abs/2003.12039",
    "Karaev, Nikita, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea "
    "Vedaldi and Christian Rupprecht. CoTracker: It is Better to Track "
    "Together, ECCV, 2024. https://arxiv.org/abs/2307.07635",
    "Huang, Ziqi, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, "
    "Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui "
    "Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao and Ziwei Liu. VBench: "
    "Comprehensive Benchmark Suite for Video Generative Models, CVPR, 2024. "
    "https://arxiv.org/abs/2311.17982",
    "Hugging Face. Diffusers documentation, text-to-video pipelines. "
    "https://huggingface.co/docs/diffusers/using-diffusers/text-img2vid. "
    "Accessed on: 22 September 2026.",
    "Guo, Yuwei et al. AnimateDiff model card. "
    "https://huggingface.co/guoyww/animatediff. Accessed on: 22 September 2026.",
]


def refs_slide(title, refs, subtitle):
    sl = newslide("References")
    headline(sl, title, size=24)
    text(sl, MARGIN, 1.20, CONTENT_W, 0.24, subtitle, size=12, color=T.TX3)
    hrule(sl, MARGIN, 1.56, CONTENT_W, color=T.TX1, lw=1.0)
    box, tf = tb(sl, MARGIN, 1.70, CONTENT_W, 5.0)
    for i, r in enumerate(refs):
        p = para(tf, first=(i == 0))
        p.line_spacing = 1.18
        p.space_after = Pt(6.5)
        pPr = p._p.get_or_add_pPr()
        pPr.set("marL", str(int(0.22 * 914400)))
        pPr.set("indent", str(int(-0.22 * 914400)))
        run(p, r, size=9.5, font=T.SANS, color=T.TX2)
    return sl


refs_slide("References · 1 of 2",
           REFS_A,
           "Video diffusion foundations, motion modules, activation patching, "
           "cross-attention, and low-rank adaptation.")
refs_slide("References · 2 of 2",
           REFS_B,
           "Where motion has been located, the older content/motion framing, "
           "evaluation tooling, and model documentation.")

# ===========================================================================
# 25  closing
# ===========================================================================
s = newslide("", dark=True)
rect(s, 0, 0, 0.09, SLIDE_H, fill=T.CONTENT_D)
text(s, MARGIN + 0.10, 2.42, 10.0, 0.9, "Thank you", size=40, font=T.SANS_L,
     color=T.TX1_D)
text(s, MARGIN + 0.10, 3.44, 10.0, 0.34, "Questions and discussion", size=16,
     color=T.TX2_D)
hrule(s, MARGIN + 0.10, 4.30, 4.4, color=T.RULE_D, lw=1.0)
text(s, MARGIN + 0.10, 4.54, 8.0, 0.30, "Team Claudn't", size=14,
     font=T.SANS_SB, color=T.TX1_D)
text(s, MARGIN + 0.10, 4.86, 8.0, 0.28,
     "Roll numbers — [fill in, one per team member]", size=12, color=T.TX3_D)
text(s, MARGIN + 0.10, 5.52, 9.0, 0.28,
     "IE 643 course project  ·  prep presentation  ·  22 September 2026",
     size=11, color=T.TX3_D)

prs.save(OUT)
print("wrote", OUT, "-", len(prs.slides.__iter__.__self__._sldIdLst), "slides")
