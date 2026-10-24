# Frozen protocol: the decisions register

Every methodological choice the project depends on, fixed in writing so that a choice
made in September is not quietly remade in October to suit the data.
**Append-only:** a change after 22 September 2026 is a new dated entry with a reason,
never an edit to an earlier row.

Source for D-01 to D-22: roadmap §5.7 (revised 22 September 2026).

| ID | Question | Decision | Implemented in |
|---|---|---|---|
| D-01 | What distance is `d`? | `1 − cos` on the flattened activation. Content channel on the static part `x̄`, motion channel on the frame-mean-removed part `x̃`. L2 as an appendix robustness check. | `summaries.paired_distances` |
| D-02 | One selectivity map or two? | Two. `S_dyn` is the headline; `S_stat` is published beside it as the H3 control. | `summaries.selectivity` |
| D-03 | How are pairs drawn? | Anchor design. 104 within-family content pairs, 90 motion pairs, 56 cross-family content pairs held out as a labelled hard subset. | `probing_set.ProbingSet` |
| D-04 | How many seeds? | Three: 42, 1337, 2024. The full grid regenerated under each. Pairs never cross a seed. Across-seed spread is the error bar. | `configs/probing_set.yaml` |
| D-05 | E2 showed noun swaps drag the layout. | Subjects split into two body-plan families; content pairs within a family by default. Cross-family pairs reported separately, never pooled. | `probing_set.cross_family_pairs` |
| D-06 | What are the 15 subjects, 4 verbs, 2 scenes? | Enumerated and frozen 22 September 2026. | `configs/probing_set.yaml` |
| D-07 | Which timesteps? | Summaries at all 25. Full tensors and all patching limited to `t ∈ {0, 4, 8, 12, 16, 20, 24}`, refined to ±2 around any peak. | `coords.D07_STEPS` |
| D-08 | Which layers? | Coarse pass: one representative module per block, 9 temporal and 7 spatial. Refined pass: all modules inside the winning block. | `coords.coarse_coordinates` |
| D-09 | What exactly is stored? | Channel means and variances, residual norms, attention maps head-averaged at 16×16 with the frame axis kept. Full tensors only for flagged coordinates. | `harness.SummaryRecorder` (see D-24) |
| D-10 | "Most motion-selective channels" by what score? | `S_c(L, t, c)`: the same ratio restricted to channel `c` of `x̃`. Top k% with k ∈ {1, 5, 10, 25}, reported as a curve. | `summaries.paired_distances(per_channel=True)` |
| D-11 | Null patch "must come back identical" to what tolerance? | Mean absolute latent difference < 1e-3 and decoded-clip PSNR > 45 dB. | `patching.null_patch_check` |
| D-12 | Noising or resampling corruption? | Resampling, from a real second clip. | `patching.Patcher` |
| D-13 | Is a patch one timestep or a range? | A contiguous window, swept over {1 step, 5 steps, first third, all 25}. | `patching.window` |
| D-14 | Is `D` commensurable? | Both terms min–max normalised across the evaluation set before entering `D`; raw values printed alongside. | Stage 6 |
| D-15 | How many test pairs for `D`? | Twelve, both directions, six per family, spanning all four verbs. | Stage 6 |
| D-16 | What counts as success? | Pre-registered three-part criterion (roadmap §5.4): patching beats prompt-only with a CI excluding zero, in both directions, with VBench subject consistency within 10% of source. | Stage 6 |
| D-17 | MotionClone or MOFT? | MotionClone. | Stage 6 |
| D-18 | Does the demo generate live? | No. Precomputed cache; live generation behind an explicit button. | Stage 7 |
| D-19 | What is the trained model? | Dual LoRA pair plus a probe-guided motion LoRA (roadmap §5.5). | Stage 6 |
| D-20 | Which checkpoint, exactly? | Resolved base repository id and commit hash written into every result file. | `pipeline.build_pipeline` → `revisions` |
| D-21 | How is the linear probe validated? | Split by scene; PCA fitted on the training split only; chance level plotted. | Stage 3 |
| D-22 | How are H1 and H2 decided? | argmax over blocks with a bootstrap CI for H1; single changepoint with a bootstrap CI for H2. | [hypotheses.md](hypotheses.md) |

## Appended entries

### D-23 · 2026-10-16 · How is H3 decided?
**Decision.** `f_stat ≥ 0.75` (fraction of coarse coordinates × D-07 steps whose `S_stat`
CI lies above 0) **and** `n_flip ≥ 1` (temporal coordinates where the `S_stat` CI is above 0
while the `S_dyn` CI is below 0). Full statement in [hypotheses.md](hypotheses.md).
**Reason.** D-22 fixed test statistics for H1 and H2 only, which left H3, the hypothesis
the roadmap calls the most important, without a falsification criterion.

### D-24 · 2026-10-16 · Summary contents, reconciled with E3's measurement
**Decision.** Per (coordinate, step), on the text-conditional CFG branch:
- channel mean `(C)` and channel variance `(C)` over frames and space;
- the residual norm `‖x̃_f‖` per frame `(F)`;
- for each of the 74 attention coordinates, one head-averaged map `(F, 16, 16)`:
  - `spatial_attn2` (cross-attention): mass on the subject-phrase tokens;
  - `spatial_attn1` (self-attention in a frame): attention *received* by each position;
  - `temporal_attn1/2` (self-attention across frames): cross-frame mass `1 − P[f, f]`
    per position, i.e. how much each frame reads from the other frames.

This is `2C + F` floats per record plus `F·16·16` per attention record, which is exactly the
budget E3 measured: **44.62 MiB per clip**.
**Reason.** D-09 says "‖x̃‖ per channel" and cross-attention maps "for subject/verb/scene
tokens", but E3's storage accounting, which produced the 44.6 MB figure the deck and
roadmap quote, stored one residual norm per frame and one map per attention coordinate.
We implement what was measured and keep the alternatives available as options
(`cross_phrases=("subject", "verb", "scene")` triples the cross-attention share).
Per-channel residual information is still available where D-10 needs it, through the
paired distances (D-26), which do not depend on the summary store.

### D-25 · 2026-10-16 · Which module represents a block in the coarse pass?
**Decision.** Temporal: the last motion module of each of the 9 blocks. Spatial: `attn2`
of the last spatial transformer of each of the 7 blocks that have one.
**Reason.** D-08 fixed the counts (9 + 7) but not the modules. The last module is closest
to the block output; `attn2` is where the prompt enters the spatial path.

### D-26 · 2026-10-16 · Where do the D-01 distances come from?
**Decision.** From paired runs, not from the summary store. The anchor clip is generated
with its activations cached (CPU, text-conditional branch) at the coordinates and steps
being scored; the partner clip is generated from the same noise tensor and scored
against that cache online. Only scalars are kept.
**Reason.** D-01 is a cosine on the *flattened* tensor. The compact summaries (D-24) cannot
reconstruct it, and caching every coordinate at every step for both clips is the 87.9 GB
full dump E3 ruled out. The paired run uses the same cache-then-intervene machinery as
patching, so it costs no extra generations.

### D-27 · 2026-10-16 · Running on the 8 GB laptop GPU
**Decision.** Two memory profiles that change nothing in the computation: `gpu`
(everything resident, VAE slicing; E1's configuration) and `lowvram` (model CPU offload
plus VAE slicing). Default `gpu` wherever it fits.
**Reason.** The roadmap assumed the 8 GB laptop GPU could not run the configuration,
because E1 measured a 10.2 GB peak on a Kaggle T4 (sm_75). Re-measured locally on the
same software stack, an RTX 4060 Laptop GPU (sm_89, which has flash attention) peaks
at 6.4 GB reserved and runs a clip in 66 s, so the pilots reproduce locally.

### D-28 · 2026-10-20 · What is the unit the selectivity ratio is computed on?
**Decision.** A matched unit is the clip that a content pair and a motion pair both end at.
For B = (s, v, scene), s not the family anchor and v ≠ running: content pair (anchor(s), v) → B,
motion pair (s, running) → B. 78 units per seed. The ratio-of-medians map is reported beside it.
**Reason.** S needs one content and one motion distance, which come from different pairs;
this is the only pairing with one word changed each way into the same clip, and it supports
the per-unit bootstrap the roadmap asks for.

### D-29 · 2026-10-20 · Component patches
**Decision.** Besides the four granularities, a patch may transplant only one D-01 component:
`dynamic` (out ← x̄_target + x̃_source) or `static` (out ← x̄_source + x̃_target), per CFG branch.
In the edits the motion direction uses `dynamic`, the content direction `static`.
**Reason.** The motion-module output is the residual stream, so a whole-tensor patch there
moves content and motion together. The x̄/x̃ split that defines the selectivity maps is the
natural way to move one and keep the other, and it turns the Deliverable 2 decomposition
into the Deliverable 3 intervention. Verified exact on the tiny UNet (`tests/test_component.py`).

### D-30 · 2026-10-20 · The twelve test pairs, and a separate dev set
**Decision.** `configs/edit_pairs.yaml`, frozen before any edit ran. Test: 12 pairs, 6 per
family, every verb on both sides, scenes and seeds cycled; A and B differ in subject AND verb.
Dev: 4 further pairs (2 per family), used only to choose the configuration (D-31).
**Reason.** D-15 fixed the count, not the pairs. Choosing the patch location on the pairs it
is then scored on would overfit the twelve.

### D-31 · 2026-10-20 · How Stage 5 picks the configuration for Stage 6
**Decision.** Coarse arms on the dev pairs (`edit_plan.coarse_arms`): component patch on the
motion-module outputs of each of the 9 temporal blocks × windows {first third, all}, and on all
21 motion modules × {one step, five steps, first third, all}, both directions; plus whole-tensor
and attention-injection references. Per direction, the component arm with the largest median
ΔD = D(arm) − D(prompt-only) over the dev pairs whose median subject-consistency ratio to the
source is ≥ 0.9 wins; ties go to fewer patched modules (`score_edits.select`).
**Reason.** D-13 says the winner of the coarse pass is what Deliverable 4 uses, without
saying how the winner is chosen.

### D-32 · 2026-10-20 · Where the probe-guided LoRA goes, and its control
**Decision.** `probe` = motion LoRA on `up_blocks.0` only (the whole H1 95% bootstrap set).
Added `anti` = the same on `down_blocks.2` + `down_blocks.3` (the two lowest M(b)), which has more
parameters than `probe`. Compared on held-out HMDB51 clips (validation denoising loss).
**Reason.** "Top-k coordinates" needed a k; the bootstrap set is the data's own answer. Without
a wrong-place control, "probe ≈ full" could mean any block would do.

### D-33 · 2026-10-20 · LoRA data and recipe as run
**Decision.** HMDB51 (`divm/hmdb51`), classes run/walk/jump/turn → running/walking/jumping/
spinning, filtered on HMDB's own tags (full body, no camera motion, one person, quality not
bad, one clip per film): 19 clips (jump has 4), last clip per class held out (15 train, 4 val).
Caption "a person {verb}". 16 frames × 256², rank 32, alpha 32, AdamW 5e-5, 800 steps, batch 1,
fp16 autocast with fp32 LoRA weights, gradient checkpointing. The appearance LoRA is dropped
(D-19 as amended); the motion LoRA is therefore trained without one attached.
**Reason.** Roadmap §5.5 values where they exist (rank, lr, top of the step range); 16 frames
instead of 8 because AnimateDiff's motion modules are trained at 16; no camera motion because a
LoRA trained on panning shots learns the pan.

### D-34 · 2026-10-24 · Probe controls: control task, non-linear probe, permutation null
**Decision.** Next to the D-21 linear probe, `analyze_sweep.py --probe-controls` runs, per
(coordinate, step, channel) and on the same scene split and PCA:
- a **control task** (Hewitt & Liang 2019): each clip type (subject, verb) gets a label drawn by
  permuting the real labels over types, fixed per seed; **selectivity = acc(real) − acc(control)**;
- an **MLP probe** (one hidden layer of 10, L2 1.0, lbfgs; Hewitt & Liang found small hidden
  layers plus weight decay keep MLP probes selective) beside the linear one;
- a **permutation null** (Ojala & Garriga 2010, test 1): training labels permuted, test labels
  kept, 99 draws, p = (1 + #{null ≥ observed}) / (1 + R) on the seed-averaged accuracy. The
  permutation is restricted and type-level: verbs are shuffled within each subject (family
  labels across subjects), identically in both scenes, so the null keeps subject identity and
  removes only the target. The draws are shared by all cells, so the per-draw maximum over the
  map gives a family-wise-error p (`p_fwer`) besides the per-cell one.
"The layer carries the verb" needs high selectivity and p < 0.05, not accuracy alone.
**Reason.** Every scene-split clip type appears once in each scene, so a probe can score by
recognising the prompt and memorising type → label rather than by reading a verb direction;
the control task keeps that route and removes the verb. A more expressive probe raises both
accuracies, so its gain only counts through selectivity. Hyperparameters are fixed here, before
the run, not tuned on the result.

### D-35 · 2026-10-24 · Experiment 2: pre-registered rules (fixed before any Phase 2 data)
Design: `reports/Content motion probing experiment design.md`; code: `scripts/probe_sweep.py --exp2`,
`scripts/exp2_*.py`. All Phase 2 sweeps use seeds 42 and 1337; seed 2024 is held out for Phase 5.
- **Confirmatory test.** Region = `up_blocks.0.motion_modules.2`, steps 0–2, S_dyn. Per-subject mean
  over its matched units; exact one-sided sign flip over subjects; replicates if p < 0.05. Run in
  each new cell (`exp2_analysis.py`, section `confirm`). Everything else in Phase 2 is exploratory.
- **Maps.** Sign-flip t per (coordinate, step) with max-|t| FWER over the map (S maps), permutation
  within group with max-over-map FWER (RSA, 49 draws). No cell-wise FDR on the 2,375-cell map.
- **Primary cell** (`exp2_quality.py`): highest median VBench subject consistency among cells with
  dynamic share ≥ 0.5 and CLIP-T within 0.01 of the best; **ladder verbs** = the two verbs with the
  highest X-CLIP hit rate on it.
- **Phase 5 candidates** (primary cell): (1) the Experiment 1 region, dynamic component;
  (2) the refined coordinate with the highest RSA unique motion share, 5-step window at its peak,
  dynamic component; (3) `up_blocks.1` motion modules, steps 0–7, dynamic component (the literature's
  site); (4) the coordinate with the highest RSA unique content share, 5-step window at its peak,
  static component. **Pairs:** 25 motion + 25 content pairs drawn with `numpy.default_rng(2024)` from
  seed 2024's probing pairs. Each candidate: A→B (sufficiency), B→A (necessity), C→B (unrelated
  clip C = same family anchor's other-verb clip with a different subject) — scored with the
  normalised contrastive effects of the report (PE, NE), subject-level bootstrap CIs.
- **LoRA target T** (Phase 6): the 3 width-1280 motion modules with the highest mean RSA unique
  motion share over steps 0–14 in the primary cell, rank 32; all-temporal arm at the budget-matched
  rank; null = 19 random width-matched 3-module sets (`lora.matched_random_subsets`, seed 643);
  primary metric MFS (official direction); p = (1 + #{R_i ≥ T}) / 20 on T's first seed.
- **LoRA data** (deviation from the report's "4–8 hand-checked clips"): HMDB51 static-camera
  cartwheels come from 4 films and only 5 are cut-free, so the target set is those 5 (4 train,
  1 validation). Training: debiased loss (β = 1), lr 1e-4, 1,000 steps, 256 px, frame stride ≤ 6.
**Reason.** The report's plan, with each free choice fixed before seeing the data it is applied to.
