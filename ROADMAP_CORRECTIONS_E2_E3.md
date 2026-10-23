# Roadmap corrections from E2 and E3 measurements
Generated 22 Sep 2026. Source: `E1_result.json`, `E2_result.json`, `E3_result.json`.
The roadmap exists only as a PDF here, so these are paste-ready replacements.

Measured on Kaggle Tesla T4, diffusers 0.40.0, torch 2.10.0+cu128,
16 frames / 512x512 / 25 DDIM steps / fp16 / CFG 7.5 / seed 42.

---

## 1. §2.2, E3 outcome cell  — WRONG BY 25.9x

**Currently:** "a full fp16 dump of every module at every timestep is 3.4 GB per clip,
which is 1.2 TB across the planned set."

**Replace with:**
> Straightforward to hook: 95 coordinates (21 motion modules, 21 temporal attn1,
> 21 temporal attn2, 16 spatial attn1, 16 spatial attn2), and hooks add only 0.6% to
> generation time. Storage is the real constraint, not compute: a full fp16 dump of
> every module at every timestep is 87.9 GB per clip, which is 31.6 TB across the
> planned set — and larger than a single Kaggle disk (73 GB), so even one clip cannot
> be dumped whole. The harness therefore stores summaries by default, 44.6 MB per
> clip, and full tensors only on request (D-09).

**Likely origin of 3.4:** 87.9 / 25 timesteps = 3.52 GB. The old figure looks like a
*per-timestep* number labelled per-clip.

## 2. §2.3 decisions

**Currently:** "E3 killed the original plan of dumping every activation to disk, at
3.4 GB per clip."

**Replace with:** "E3 killed the original plan of dumping every activation to disk, at
87.9 GB per clip — more than the whole working disk."

## 3. §5.1 What gets stored

**Currently:** "E3 measured a full dump at 3.4 GB per clip."

**Replace with:** "E3 measured a full dump at 87.9 GB per clip, against 44.6 MB for the
summaries: a 2,017x reduction."

**Also add** (the 16x16 grid is now the dominant cost):
> At a 16 x 16 grid the attention maps are 28.9 MB of the 44.6 MB per clip (65%), or
> 15.7 GB across 360 clips — 78% of Kaggle's 20 GB output cap. An 8 x 8 grid halves the
> store to 22.9 MB per clip (8.1 GB total). Decide before Stage 2 freezes the format.

## 4. §7 budget table  — CHANGES THE TOTAL

**Currently:**
| Probing set generation with hooks | 360 clips x 113 s (94 s + ~20% hook overhead) | 11.3 | 2 |

**Replace with:**
| Probing set generation with hooks | 360 clips x 94.6 s (94 s + 0.6% measured hook overhead, E3) | 9.5 | 2 |

**Total: 36.5 -> 34.7 GPU-hours.** Stage 2's single-week load drops from ~11 h to ~9.5 h.

## 5. §7 preamble

**Currently:** "All figures use the 94 s per clip measured in E1."

**Replace with:**
> All figures use the 94 s per clip measured in E1. That figure has a within-session
> standard deviation of 0.1 s but varies by about 10% between Kaggle sessions (E3's
> no-hook baseline on a later session measured 85.1 s), so the totals below carry
> roughly +/-10%: 34.7 +/- 3.5 GPU-hours.

## 6. §9 change log

**Currently:** "E3's storage figure filled in (3.4 GB per clip)."

**Replace with:** "E3's storage figure filled in (87.9 GB per clip, replacing an
erroneous 3.4 GB that was a per-timestep figure)."

Also update the §8 deck checklist line — E3's per-clip storage figure is no longer
outstanding; only E5's outcome is.

---

## Optional: add E2's quantitative results

§2.2's E2 row is currently qualitative. The measured numbers:

| swap | d_c (appearance) | d_m (motion proxy) | layout SSIM |
|---|---|---|---|
| noun (cat/horse/bird) | 0.260 | 0.461 | 0.367 |
| verb (walking/jumping/standing) | 0.099 | 0.390 | 0.400 |

Two findings worth stating:
- Noun swaps shift appearance **2.6x** more than verb swaps — content control works.
- Layout SSIM falls monotonically with body-plan distance: horse 0.508, cat 0.338,
  bird 0.255. This is the quantitative basis for D-05.
- **The pixel-level motion proxy inverted** (0.461 for noun vs 0.390 for verb swaps):
  appearance change swamped the motion measure. That is H3 observed before the sweep
  began, and it is direct evidence for the frame-mean-removed residual in §5.1.
