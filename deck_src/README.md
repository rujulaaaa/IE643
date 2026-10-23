# Deck source

`IE643_Prep_Presentation.pptx` is generated, not hand-edited. To change it,
edit the source here and rebuild — otherwise the next rebuild overwrites your
edit.

```
cd deck_src
python3 fig_concept.py      # U-Net, AnimateDiff, patching, the 2x2, title motif
python3 fig_data.py         # charts from E1/E2/E3_result.json + the two maps
python3 fig_plan.py         # deliverable chain, pairs, attention, LoRA, demo
python3 build.py            # writes ../IE643_Prep_Presentation.pptx
libreoffice --headless --convert-to pdf --outdir .. ../IE643_Prep_Presentation.pptx
```

Requires `python-pptx`, `matplotlib`, `Pillow`, and the **Lato** and
**Noto Sans Mono** fonts (both installed on this machine). Figures are
rendered at 320 dpi into `fig/` and placed by `build.py`.

## Design system

`theme.py` holds the whole system: palette, type scale, 12-column grid and the
pptx primitives. Two rules matter:

- **Blue means content, orange means motion**, on every slide and in every
  figure. Neither hue is ever decorative. Violet is the second *subject* in the
  2x2 — a cool hue, so orange stays exclusively "motion".
- Colours are the validated colour-blind-safe slots 1/2/7 of the Anthropic
  data-viz palette. Checked on both surfaces (white and `#101820`): worst
  all-pairs CVD dE 9.2 / 9.4, normal-vision dE 24.0 / 20.9, all contrast
  >= 3:1. Re-run the validator if you change a hue.

## Still to fill in

| What | Where | When |
|---|---|---|
| Roll numbers | slides 1 and 25 (`[fill in, ...]`) | before recording |
| E5 outcome | slide 10, the `E5` row in `pilots` | Stage 1, by 27 Sep |
| Measured selectivity map | slide 15 — `fig_data.selectivity_map` draws the *prediction* | Stage 4 |
| Real frames in the 2x2 | slide 17 — `fig_plan.demo` draws placeholders | Stage 6 |

The deck uses the corrected compute budget, **34.7 +/- 3.5 GPU-hours**, not the
36.5 in the roadmap PDF. See `../ROADMAP_CORRECTIONS_E2_E3.md`.
