# What we did — IE 643 final submission log

Chronological record of the work from the intermediate (prep) submission to the final
submission. Every entry: what was done, files touched, commands/configs, actual results,
and the decisions taken with their reasons. Failed attempts stay in the log.

Project: *Probing the Latent Space of Video Diffusion U-Nets: Disentangling Content vs. Motion*
(team Claudn't). Problem statement: `problem_statement.md`.

---

## 2026-10-20 · Session 1

### 1. Inspection of what already exists

**This folder (`~/IE643_CourseProject_Prep/`)**: prep-stage material.
- `problem_statement.md`: the brief. Four technical deliverables (setup + probing tools;
  content/motion analysis over layers and timesteps; patching/intervention framework;
  editing demonstration) and the final package (trained/finetuned model, code + docs,
  interface, performance metric + qualitative results, stage-wise report).
- `IE643_Roadmap_Submission_Guide.pdf/.html` (revised 22 Sep): the full plan. Probing set
  15 subjects × 4 verbs × 2 scenes × 3 seeds = 360 clips; anchor pair design (104 content,
  90 motion, 56 cross-family pairs per seed); selectivity maps `S_dyn`/`S_stat` from 1−cos
  distances on the static part x̄ and frame-mean-removed residual x̃; hypotheses H1 (layer),
  H2 (timestep), H3 (static confound) with pre-registered tests; patching with null /
  random-coordinate / symmetry controls; disentanglement score D with a pre-registered
  success criterion; dual LoRA + probe-guided LoRA as the trained model; Gradio interface
  over a precomputed cache. Decisions register D-01…D-22.
- `ROADMAP_CORRECTIONS_E2_E3.md`: E3's storage figure is 87.9 GB/clip, not 3.4 GB.
- `E1/E2/E3_*.ipynb` + `E*_result.json`: Kaggle T4 pilot notebooks and numbers of record.
- `deck_src/` + `IE643_Prep_Presentation_*.pdf`: the prep deck, generated from Python.
- `main.pdf`, `updated_instructions.pdf` (dated 24 Sep): **these are not IE 643 documents.**
  Both are for CS 6013 *Efficient AI* (compressing Qwen3.5-4B for maths; GitHub/HF layout
  rules; a strict no-AI rule for that course's report). Not used as instructions here;
  flagged to the user.

**The code repository (`~/MI_DiffusionModels/`)**, last touched 16 Oct, is the real
starting point. Package `vdprobe` (≈2,000 lines) with:
- `coords.py` (95 hookable coordinates: 21 motion modules, 21+21 temporal attention, 16+16
  spatial attention; canonical (branch, F, C, S) layout), `harness.py` (hooks, step counter,
  `SummaryRecorder`), `attention.py` (attention-probability reconstruction for SDPA),
  `summaries.py` (x̄/x̃ split, D-01 distances, S), `patching.py` (`TensorCache`, `Patcher`
  with whole-tensor / channel-subset / α-scaled patches, `PairDistance`, null-patch
  verdict), `pipeline.py` (AnimateDiff loading with mirror fallback + commit hashes, CPU
  fp16 noise policy), `probing_set.py` (frozen vocabulary, pair design), `metrics.py`.
- `docs/` (protocol register extended to D-27, hypotheses with test statistics incl. D-23
  for H3, harness and patching design), `configs/` (frozen generation + vocabulary).
- `results/local/E1–E3`: the pilots re-run on the laptop GPU.
- Status at hand-over: everything up to E3; Stage 2 onward (probing set, sweep, patching on
  the real model, edits, LoRA, interface, report) **not started**.

**Environment check (today)**
- GPU: RTX 4060 Laptop, 8 GB (sm_89). RAM 15 GB. Free disk: ~14 GB on /home, ~17 GB on /.
- `~/MI_DiffusionModels/.venv`: Python 3.10, torch 2.10.0+cu128, diffusers 0.40.0,
  transformers 5.20.0, CUDA available. Missing: `peft`, `gradio`, `scikit-learn`, `opencv`.
- Weights already cached in `HF_HOME=/var/tmp/rujula_hf_cache` (SD1.5 mirror, motion
  adapter v1-5-2, CLIP ViT-B/32).
- `env -u PYTHONPATH .venv/bin/python -m pytest -q` → **18 passed in 3.38 s** (ROS
  `PYTHONPATH` must be unset, as the README says).
- Local E1 record: 66 s/clip, 6.4 GB peak; E3: full `SummaryRecorder` costs +72 %.

### 2. Decisions so far

- **Work continues in `~/MI_DiffusionModels/`** (the existing package, tests and docs) rather
  than a new tree, so nothing valid is rewritten. This log stays in the prep folder next to
  `problem_statement.md`.
- **LoRA data source candidate:** HMDB51 on the HF Hub (`divm/hmdb51`) ships individual mp4s
  of ~300 KB with classes `run`, `walk`, `jump`, `turn`, so ~20 real clips can be fetched
  without the multi-GB UCF101/Kinetics archives. (UCF101 has no plain running class;
  Kinetics needs YouTube scraping.) Pending the user's confirmation.

### 3. Constraints received from the user (2026-10-20)

- **Deadline:** Stage 1 of the intensive assessment is due **25 Oct 2026, 23:59 IST**
  (`IE643 2026 Course Project Intensive Assessment Instructions.pdf`); in-person
  presentations run 26–31 Oct.
- **Scope for now:** only the final-submission technical deliverables (code, trained model,
  interface, metrics, qualitative results). No report, slides or videos yet. (The
  instructions also give a zero for AI-generated report content, so I will not write report
  prose in any case.)
- **Working mode:** after each major chunk of code, report to the user and pause until they
  write "continue".
- **Compute:** laptop RTX 4060 for long or overnight jobs, plus the user's Kaggle account
  (T4) for anything that does not fit in 8 GB.
- Still open: the MotionClone baseline replacement (D-17) and the LoRA scope (D-19).
- **Decided by the user (2026-10-20):**
  - **D-17 changed.** The training-free baseline is our own temporal-attention-map injection
    (MotionClone's core idea, without gradient guidance), implemented as patch granularity 3.
    Reason: official MotionClone needs per-step gradient guidance through the U-Net (memory on
    8 GB, plus a second codebase) and does not fit the 5-day window.
  - **D-19 changed.** Two motion LoRAs on ~20 HMDB51 clips (run / walk / jump / turn): one with
    full temporal coverage, one probe-guided (top selectivity-map layers only), plus a ‖ΔW‖ layer
    ranking. The appearance LoRA is dropped to fit the deadline.

### 4. Chunk 1 plan: lockstep paired sampler for the probing sweep

Implementation choice (does not change the protocol): instead of caching the anchor clip's
activations for all 25 steps (≈7.6 GB of host RAM for the 16 coarse coordinates), all clips of
a group are denoised **in lockstep**, one step at a time. At step t the anchors run first and
their step-t activations are kept, then every partner runs and is scored against them. The
step-t activations are then dropped. Memory holds one step's activations instead of 25, so the
D-01 distances can be computed at full resolution on the flattened tensors, at all 25 steps,
for all 95 coordinates, and each clip is generated exactly once. The loop replicates
`AnimateDiffPipeline.__call__` (diffusers 0.40) line by line: CFG concatenation [neg, pos],
`repeat_interleave` of the prompt embeddings over frames, `scale_model_input`, the CFG
combination, and stateless `DDIMScheduler.step`. Equivalence with `pipe()` is verified before
use.

### 5. Chunk 1 implementation (2026-10-20)

**Packages added** to `~/MI_DiffusionModels/.venv` with
`env -u PYTHONPATH uv pip install --python .venv/bin/python peft scikit-learn opencv-python-headless gradio`:
peft 0.21.2, scikit-learn 1.7.2, gradio 6.30.0, opencv-python-headless. torch stayed at
2.10.0+cu128 and diffusers at 0.40.0.

**New `src/vdprobe/lockstep.py`**
- `run_lockstep(loaded, coords, tracks, noise, after_track, after_step)`: the unrolled
  denoising loop described in §4. `LockstepHarness` reuses the existing `Harness`/`Probe`
  machinery, but the sampler sets the step index instead of a counter incrementing on every
  UNet call, and the probes are swapped per clip.
- Probes:
  - `StepCache`: one clip's text-conditional activations at the current step, fp16, on the CPU.
  - `PairScorer`: D-01 distances of the live clip against reference clips that ran earlier in
    the same step, plus per-channel dynamic distances (D-10) at chosen steps.
  - `Sketcher`: count sketch (dim 512) of the flattened x̃ and x̄ per (coordinate, step), as
    linear-probe features.
- Decision: **count sketch for the linear-probe features.** D-21 asks for PCA + logistic
  regression on the flattened x̃. Storing that is impossible (21M values per step at 64×64),
  and spatial pooling would destroy exactly the spatial motion information being probed. A
  count sketch preserves inner products in expectation (it is a random projection), so PCA on
  the sketch approximates PCA on the full tensor. With ≤180 training clips per fold, PCA can
  keep at most n−1 components anyway, so 512 sketch dimensions lose little. The bucket hash
  uses `zlib.crc32` of "coord|kind": a first draft used Python's `hash()`, which is randomised
  per process and would have given every run different buckets. Caught before any run.

**`summaries.paired_distances` extended** (existing keys unchanged): `l2_static`,
`l2_dynamic` (relative L2 ‖a−b‖/(‖a‖+‖b‖), the D-01 appendix robustness check) and
`static_energy_b`.

**New `scripts/lockstep_check.py`**: verifies, against `pipe(...)`, that a solo lockstep clip
matches, that a clip in a 3-clip group is unaffected by its neighbours, that the probes are
read-only, that a clip scored against an identical copy gives distance ≈0 while a verb swap
does not, and that the row counts are right.

**New `scripts/probe_sweep.py`**: one lockstep group per (seed, scene) = 60 clips, all 104 +
90 + 56 pairs of that scene/seed (pairs split evenly by scene: 52 + 45 + 28 per group),
16 coarse coordinates × 25 steps. Clips are topologically ordered so that every anchor runs
before its partners. An anchor's step cache is freed once its last partner has run. Writes
distances.csv, per_channel.pt (D-07 steps), sketches.pt, latents.pt, GIFs, and manifest.json
(written last, used as the "done" marker so the run can resume). Smoke-test flags:
`--subjects N --verbs N`.

### 6. Paused (2026-10-20)

- The user confirmed the compute split: the laptop runs long jobs, Kaggle runs LoRA training
  (and overflow).
- The first run of `scripts/lockstep_check.py` was **stopped by request** before it finished:
  the user needed to close the laptop. No result was recorded. Its stdout was block-buffered
  (Python launched without `-u`), so no partial output exists either; future runs use
  `python -u`.
- To resume: `cd ~/MI_DiffusionModels && env -u PYTHONPATH .venv/bin/python -u scripts/lockstep_check.py`
  (~12 min). If it passes, run the smoke test
  `... -u scripts/probe_sweep.py --seeds 42 --scenes "on a beach" --subjects 2 --verbs 2`,
  then the full sweep (~7 h).
- Side note to check: the 18 unit tests took 127 s on this run against 3 s before the
  package installs (all still pass). Probably the GPU job starting at the same time, or
  gradio's import cost in the pytest plugins.

### 7. Lockstep verification PASSED (2026-10-20, 02:32–02:50)

Command: `env -u PYTHONPATH .venv/bin/python -u scripts/lockstep_check.py`
Output: `results/local/lockstep_check/lockstep_check.json`, log `results/local/lockstep_check_log.txt`.

| check | result |
|---|---|
| floor: `pipe()` vs `pipe()` | latent MAD **0.00e+00** (fp16 runs are bit-reproducible on this GPU) |
| solo lockstep vs `pipe()` | MAD **0.00e+00**: the unrolled loop is bit-identical to diffusers |
| P inside a [P, Q, R] group vs `pipe()` | MAD **0.00e+00**: no leakage between clips |
| P with StepCache + Sketcher attached, group with two scorers | MAD **0.00e+00**: probes are read-only |
| self pair (P scored against an identical copy) | max d_dynamic 1.8e-07, max d_static 3.0e-07 |
| verb swap (running → walking) | median d_dynamic **0.555**, median d_static **0.083** |
| rows | 400 per pair (16 × 25), 48 per-channel entries, 400 sketches |

The single verb-swap pair is a first glimpse of the H3 split: the verb change moves the
frame-varying residual ~7× more than the static part. One pair only; this is not a result yet.

**Speed investigation.** The lockstep runs looked 2–2.5× slower than the first `pipe()` run
(152 s vs 62 s per clip). A 6-step profile (scratchpad `prof.py`) showed: pipe 5.90 s/step,
lockstep without hooks 4.95, lockstep with 95 hooks 5.53, so the sampler is not slower than
the pipeline. The cause was **two other training processes on the same GPU**
(`.venv-gpu/bin/python src/distill.py ...`, from another project, repeatedly relaunched).
They were left alone. GPU also logged SW power-cap and SW thermal-slowdown time (86 °C).

**Decision (user, 2026-10-20):** run the 360-clip sweep on **Kaggle T4 ×2**, one (seed, scene)
group per GPU in parallel (~5 h, ~5 of the 30 weekly GPU-hours), instead of the shared laptop
GPU (~15 h at the shared speed).

### 8. Kaggle runner (2026-10-20, ~03:10)

**New `scripts/kaggle_run.py`** (modelled on `~/kaggle/src/run_on_kaggle.py` from the user's
other IE643 work): builds a private script kernel that embeds `src/vdprobe`, `configs`,
`scripts` as a base64 tar.gz (48 KiB). The kernel then:
- pip-installs `diffusers==0.40.0` (pinned like every earlier result), `peft`, `scikit-learn`;
- prefetches the SD1.5 / motion-adapter / CLIP weights once, so two jobs don't race on the
  HF cache;
- starts one job per GPU (`CUDA_VISIBLE_DEVICES=i`) and streams both logs into the kernel log
  every 60 s.
Outputs go to `/kaggle/working/out/job<i>` and are downloaded to `results/kaggle/<slug>/`.
Settings: `machine_shape: NvidiaTeslaT4` (Kaggle's T4 accelerator is 2 × T4), internet on (for
HF downloads), private. Split for the sweep: GPU 0 = the three beach groups, GPU 1 = the
three forest groups.

**Smoke test pushed first:** kernel `rujularahate/vdprobe-smoke`, seed 42, one scene per GPU,
`--subjects 2 --verbs 2` (8 clips and 4+2+2... pairs per job). Purpose: catch environment,
parallelism and output-path problems before spending ~5 GPU-hours.

### 9. Kaggle smoke test: two failures, both "no GPU" (2026-10-20, ~03:15–03:30)

- **v1** died at `nvidia-smi -L`: `FileNotFoundError` (the current Kaggle image is Python 3.13
  and has no `nvidia-smi` on PATH). Fixed `kaggle_run.py` to detect GPUs with
  `torch.cuda.device_count()` in a child process and print the device names. Also added
  `filter="data"` to the tar extraction (Python 3.14 deprecation warning).
- **v2**: torch reported `2.11.0+cpu` and 0 CUDA devices, so the session was CPU-only even
  though the server-side metadata says `machine_shape: NvidiaTeslaT4`, `enable_gpu: true`.
- Quota API: GPU `timeUsed 0s`, `hasEverRun: false`, `totalTimeAllowed 21600s` (the CLI table
  shows 30 h; the raw field says 6 h, so the real allowance is unclear), refresh 2026-10-24.
  Diagnosis: the account `rujularahate` has never been cleared for GPU (usually missing phone
  verification); Kaggle silently falls back to CPU. No GPU quota was consumed.
- **User action (2026-10-20):** the user will fix the Kaggle account (phone verification /
  enable the GPU once in the web UI). Then the smoke test is re-pushed and, if it passes, the
  full sweep.
- Note for later: Kaggle's image now ships torch 2.11.0, not the 2.10.0 the pilots used.
  Results will record the version (`env_info`). diffusers stays pinned at 0.40.0.

### 10. Analysis stage written in parallel while Kaggle runs (2026-10-20, ~03:40)

The user verified the Kaggle phone number. Smoke test v3 pushed and confirmed on a GPU: the
quota API shows `hasEverRun: true` and GPU time accumulating. The API still reports
`totalTimeAllowed: 21600 s` (6 h until 24 Oct), so the Kaggle budget may be 6 h, not 30 h.
To be re-checked after the smoke test gives real T4 timings.

**New `src/vdprobe/analysis.py` + `scripts/analyze_sweep.py`** (CPU only):
- loads every sweep group; builds matched selectivity units; S maps for `dyn` (x̃), `stat`
  (x̄), `raw` (no split) and `l2dyn` (L2 robustness); bootstrap 1,000 over units within each
  seed, averaged over seeds; ratio-of-medians kept as a robustness map;
- pre-registered tests exactly as in `docs/hypotheses.md`:
  - H1: argmax over the 9 temporal blocks of M(b) = −mean_t median S_dyn, with a 95%
    bootstrap argmax set that must exclude level-0 and level-3 blocks;
  - H2: two-segment least-squares changepoint on the winning block, τ CI ≤ 12 and early > late;
  - H3: f_stat ≥ 0.75 and n_flip ≥ 1 over the coarse coordinates × D-07 steps;
- static-energy share per coordinate; cross-family hard subset reported separately (D-05);
  Spearman agreement cos-vs-L2, median-ratio vs ratio-of-medians, and seed-vs-seed;
- linear probes (D-21): StandardScaler → PCA(min(256, n_train−1)) fitted on the training
  scene only → LogisticRegression(C=1), train on one scene, test on the other, swap, average,
  per seed then over seeds. Four probes: verb and family on the x̃ and x̄ sketches.
- Outputs `analysis.json` + `maps.npz`.

**Decision D-28 (new, resolves an ambiguity, does not change a decision):** the roadmap says
S is "computed per pair then averaged", but S needs a content distance and a motion distance,
which come from different pairs. Matched unit = the clip both pairs end at. For B = (s, v,
scene) with s ≠ family anchor and v ≠ running, the content pair is (anchor(s), v) → B and the
motion pair is (s, running) → B. That gives 7·3·2 + 6·3·2 = **78 units per seed**. Reason: it
is the only pairing that yields a genuine per-unit ratio (same end clip, one word changed
each way), and it supports the per-unit bootstrap the roadmap requires. The ratio-of-medians
alternative is reported beside it.

**New `tests/test_analysis.py`**: a synthetic sweep on disk with a planted answer (motion only
in `up_blocks.1` at steps < 8, static channel content-selective everywhere, verb signal in one
coordinate's sketch).
- The pipeline recovers it: H1 → up_blocks.1 with a 95% set of exactly {up_blocks.1}; H2 →
  τ = 8; H3 → f_stat = 1.0; the verb probe ≥ 0.79 on the planted coordinate vs < 0.6 elsewhere
  (chance 0.25).
- Bugs caught by the test: `cs.clip` resolved to `DataFrame.clip` (the method) instead of the
  column, fixed to `cs["clip"]`; a pandas `groupby.apply` deprecation replaced.
- The probe threshold in the test was lowered from 0.9 to 0.7: the synthetic signal sits in 1
  of 32 dims and PCA keeps 16. Test calibration, not a code change.
- Full suite: **21 passed**.

### 11. Smoke test passed on Kaggle; full sweep launched; editing framework written (2026-10-20, ~03:45–04:20)

**Smoke test v3 (`rujularahate/vdprobe-smoke`): PASSED.** torch 2.11.0+cu128, 2 × Tesla T4.
Two jobs in parallel (seed 42, one scene each, 8 clips, 12 pairs each): both exit 0, all files
written, weights resolved to `stable-diffusion-v1-5/stable-diffusion-v1-5@451f4fe161` and
adapter `@6167b88ffe` (D-20). Cost **97.4 / 97.7 s per clip** per GPU (3.90 s per clip-step,
incl. hooks, distances and sketches), peak host RSS 8.3 GB per process.

**Kaggle quota is 6 GPU-h/week** (API `totalTimeAllowed 21600 s`; the CLI's "30 h" is wrong).
Smoke tests used 958 s. Refresh 2026-10-24 00:00 UTC (05:30 IST).

**Full sweep launched**: kernel `rujularahate/vdprobe-sweep` v1, 2026-10-20 ~04:05 IST,
job0 = `probe_sweep.py --scenes 'on a beach'` (3 seeds), job1 = `--scenes 'in a forest'`.
Expected 3 × 60 × 97.5 s ≈ 4.9 h per GPU in parallel, ~5.0 GPU-h of the remaining ~5.7.
Plan for the rest: patching/edits on the laptop until the quota refreshes; LoRA training on
Kaggle after the 24 Oct refresh.

**Analysis smoke run**: `scripts/analyze_sweep.py results/kaggle/vdprobe-smoke/out` ran end to
end on real sweep files (4 units only, so the verdicts are meaningless; a pipeline check, not
a result). The cos-vs-L2 Spearman was 0.99.

**New `src/vdprobe/edit.py`** (Deliverables 3–4):
- `run_edit(loaded, coords, source_prompt, target_prompt, noise, EditSpec)`: source + target in
  lockstep from one noise tensor. The source's raw activations are cached at step t
  (`TensorCache`, cleared each step) and patched into the target (`Patcher`, granularities
  1/2/4). An empty spec = the prompt-only baseline, with the source still running so every arm
  costs the same.
- Granularity 3 = **temporal attention-weight injection** (the D-17 replacement):
  `InjectingProcessor` reimplements `AttnProcessor2_0` with explicit softmax probabilities.
  The source records P_s per temporal attention module and step; the target uses
  `lerp(P_t, P_s, α) @ V_t`. Outside the active modules/steps it defers to the stock processor.
  `attention_injection` installs and restores the processors. `run_lockstep` gained a
  `before_track` callback for this.
- `channel_mask(S_c, top_frac)` for granularity 2 (D-10; most negative S_c = most
  motion-selective).
- Directions: "motion" = target (new subject, A's verb) + A's motion-selective values;
  "content" = target (A's subject, new verb) + A's content-selective values.

**New `tests/test_edit.py`** (tiny CPU UNet), 4 tests:
- recording reproduces the stock attention output (atol 1e-5) and the processors are restored;
- self-injection ≈ null (atol 1e-3: the recorded probabilities are stored fp16);
- injecting A's weights into B changes B's output; α = 0 is a no-op;
- inactive steps run the stock path; `channel_mask` picks the right channels.
Debugging along the way:
- the "injection changes B" check first failed: the tiny UNet's motion-module `proj_out` path
  needed live weights (fixture `live_unet`);
- then a real test bug: `b_plain` was computed inside the injection context, where the
  installed processors were still bound to the live controller in "inject" mode, so the
  "plain" output was secretly injected. Fixed by computing `b_plain` with stock processors
  outside the context.

**Suite: 25 passed.**

### 12. Edit metrics (2026-10-20, ~04:30)

Added to `src/vdprobe/metrics.py`:
- `flow_fields` (OpenCV Farneback, 128 px; pyr 0.5, 3 levels, win 15, 3 iterations,
  poly_n 5, σ 1.2);
- `motion_fidelity` s_m = mean over frame pairs of the cosine of the flattened flow fields;
- `motion_magnitude` (is anything moving);
- `content_fidelity` s_c = mean per-frame CLIP image cosine;
- `subject_consistency` (VBench-style proxy: mean CLIP cosine of each frame to the first and to
  the previous frame);
- `disentanglement` D (§5.4) with s_c and s_m min-max normalised over the evaluation pool (D-14).

**Decision (substitution, dated 2026-10-20):** motion fidelity by dense optical flow instead of
CoTracker tracks (+RAFT); content and subject consistency by CLIP instead of DINO/VBench's
own models. Reason: no new model downloads or dependencies in the last five days (RAFT needs
torchvision; CoTracker and DINO need new checkpoints). Flow keeps the property that matters:
it measures how pixels move, not what they are.

**New `tests/test_metrics.py`** on synthetic moving squares: same motion in a different colour
→ s_m 0.98; perpendicular → 0.00; opposite → −0.49 (not −1, since the square ends up on
different pixels; test threshold set at −0.3); static → zero magnitude. D reaches exactly ±1 at
the ends of the pool. **Suite: 27 passed.**

### 13. Correction: Kaggle quota is 30 h/week (2026-10-20, 16:15)

Earlier entries (§9, §11) said the quota was 6 h, based on the API field
`totalTimeAllowed: 21600 s`. **That was wrong.** The user's Kaggle quota page shows "24h 32m
available of 30h" (5 h 28 min used: smoke tests + the full sweep). The API field is not the
weekly cap; the CLI's 30 h figure was right. So ~24.5 GPU-hours remain this week, enough to
run patching/edit experiments and LoRA training on Kaggle as well as the laptop.

The session also restarted between 04:30 and 16:15: the background download watcher was
stopped with it. The sweep kernel `vdprobe-sweep` had finished (`COMPLETE`); its outputs are
being downloaded with `scripts/kaggle_run.py --slug vdprobe-sweep --download-only`.

### 14. Probing sweep complete; first real results (2026-10-20, 16:15–17:10)

**Sweep** (`rujularahate/vdprobe-sweep`, Kaggle 2 × T4, torch 2.11.0+cu128, diffusers 0.40.0):
6 groups × 60 clips = **360 clips**, 6 × 125 = **750 pairs** (312 content, 270 motion, 168
cross-family), 16 coarse coordinates × 25 steps → 300,000 distance rows. Beach groups took
96.5–96.7 min each, forest groups 92.5–92.6 min; peak host RSS 8.3 GB per process (21 anchors
fit easily). Total kernel time ≈ 5.2 h.
- Download: the first `kaggle kernels output` died with `ConnectionResetError(104)` at
  366 MB. Re-running **without `--force`** resumed and completed (1,006 MB). All 6 groups have
  7 files + 60 GIFs each.

**Analysis** (`scripts/analyze_sweep.py results/kaggle/vdprobe-sweep/out --out results/analysis`,
1,000 bootstraps, probes took 836 s on the CPU; log `results/analysis_log.txt`):
- Matched units: 78 per seed, exactly as designed (D-28).
- **H1 (layer) — FALSIFIED.** Motion selectivity M(b) is highest at **up_blocks.0** (8×8, level 3,
  the first decoder block after the bottleneck), M = +0.005. Every other temporal block is
  negative (content-leaning): up1 −0.017, up2 −0.019, mid −0.026, down0 −0.052, down2 −0.075.
  The bootstrap argmax lands on up_blocks.0 in **1000/1000** resamples, so the 95% set is
  {up_blocks.0}. That is an extreme-resolution block, so H1's prediction (peak at up_blocks.1)
  fails, but the peak is sharp and sits *adjacent* to the predicted block.
- **H2 (timestep) — SURVIVES.** On up0.mm2, m(t) = 0.096, 0.050, 0.007, then ≈0 or negative;
  changepoint **τ = 2, 95% CI [1, 3]**; early mean 0.073 vs late −0.001. Motion selectivity lives
  in the **first 2–3 of 25 steps** only. Across all coordinates, the median S_dyn falls from
  0.35 at step 0 to ~0.01 by step 9: after the first third, content and motion swaps move the
  activations about equally.
- **H3 (static confound) — FALSIFIED as stated.** f_stat = 0.65 (< 0.75) and n_flip = 0. Removing
  the frame mean does not turn any temporal coordinate motion-selective. What is true:
  - S_dyn < S_stat at almost every coordinate, so the split moves every coordinate towards
    motion (e.g. down2 cross-attention S_stat 0.43 → S_dyn 0.17; up2 cross-attention
    0.26 → 0.08), but the shift never crosses zero.
  - Static energy share F‖x̄‖²/‖x‖²: 0.39–0.70 in the temporal modules, 0.82–0.9996 in spatial
    cross-attention. The static part dominates the spatial path but not the temporal path.
  - S_stat has 5 motion-significant cells (up0.mm2 steps 0–2, mid.mm0 steps 0–1), S_dyn only 1
    (up0.mm2 step 0). The cleaner motion signal at the very start is in x̄, the opposite of H3.
  - S_raw has 19 motion-significant cells, mostly up1 spatial cross-attention at steps 18–24.
    A late, unexpected effect; noted, not interpreted yet.
- **Robustness:**
  - cos vs L2 Spearman 0.987, so the D-01 choice does not drive the map;
  - median-of-ratios vs ratio-of-medians 0.930;
  - seed agreement of the S_dyn map 0.58 / 0.69 / 0.73 (42–1337, 42–2024, 1337–2024): moderate.
- **Linear probes (scene-split, chance verb 0.25 / family 0.50):**
  - verb on x̃: mean 0.44, best 1.00 (mid cross-attention, step 0); by coordinate the highest is
    mid cross-attention 0.74, then down0 cross-attention 0.57, up0.mm2 0.50; most temporal
    modules 0.35–0.40;
  - verb on x̄: mean 0.56;
  - family on x̃ 0.72 and on x̄ 0.82, best at down0.mm1.
  - Caveat: the step-0 maxima on cross-attention are expected, because at step 0 the spatial
    cross-attention output reads the prompt embedding almost directly, so "verb is decodable"
    there means "the verb token is in the prompt", not "motion is represented". The probe
    profile over steps, not the max, is the informative part.

**What this means for Stage 5 (my reading, still a hypothesis until patching tests it):**
content dominates nearly everywhere; the only motion-leaning coordinates are the temporal
modules at up_blocks.0 and mid_block in the first 1–3 steps. The patching sweep should
therefore target temporal modules and temporal attention in the early window (D-13 "first
third" and "five steps at the peak") and include up0/mid, plus a random low-|S| coordinate as
the control.

## 2026-10-20 · Session 2

### 15. Code moved into the submission folder (user request)

The repository now lives at **`~/IE643_CourseProject_Prep/code/`** (src, scripts, tests,
configs, docs, notebooks, results, README, pyproject), so the submission is one folder.
`mv` on the same disk, so no copy was made. The 7.6 GB virtualenv stays at
`~/MI_DiffusionModels/.venv` (kept out of the submission folder); its editable-install `.pth`
now points to `code/src`. Run everything as
`cd ~/IE643_CourseProject_Prep/code && env -u PYTHONPATH ~/MI_DiffusionModels/.venv/bin/python -u ...`.
Check after the move: `vdprobe` imports from `code/src`; **27 passed**. pyproject now also lists
pandas, scikit-learn, opencv-python-headless, peft and gradio (installed in §5, not
declared until now). `results/kaggle/` (1.1 GB of raw sweep output) sits inside `code/` and
will be excluded or summarised in the final zip.

### 16. Stage 5 framework: component patches, multi-arm runner, controls on the real model (2026-10-20, evening)

**New code**
- `coords.frame_mean_raw`: x̄ broadcast back to each of the three raw layouts (spatial attn
  (B·F,S,C), temporal attn (B·S,F,C), motion module (B·F,C,H,W)).
- `PatchSpec.component` ∈ {all, dynamic, static} (**D-29**): transplant only the source's x̃
  (keep the target's x̄) or only its x̄. This turns the D-01 split that defines the selectivity
  maps into the intervention. Reason: the motion-module output is the residual stream, so a
  whole-tensor patch there moves content and motion together.
- `edit.run_arms`: one source clip + N target arms in one lockstep group, all reading one
  per-step source cache (N+1 generations instead of 2N). One `AttnController` whose spec is
  swapped per track. Supports a negative step offset (mis-indexed control keeps 2 steps).
  `run_edit` is now a wrapper around it. `cache_device="cpu"` option for 8 GB GPUs (the first
  controls run hit CUDA OOM with the cache on the GPU).
- `tests/test_component.py` (12 tests): `frame_mean_raw` equals the canonical split on all
  layouts; a dynamic patch leaves the target's x̄ and takes the source's x̃ (and vice versa);
  self-patch is a no-op.
- `configs/edit_pairs.yaml` (**D-30**): 12 test pairs + 4 disjoint dev pairs, frozen before
  any edit ran. `src/vdprobe/edit_plan.py`: the coarse arm set (51 arms per pair) and the
  selected arm set for Stage 6. `tests/test_edit_plan.py` checks the design rules.
- `scripts/patch_controls.py`, `scripts/edit_sweep.py` (resumable, sharded),
  `scripts/score_edits.py` (decode → CLIP + Farneback flow → s_c, s_m, D, consistency,
  flicker; D-16 statistic with paired bootstrap; **D-31** selection rule),
  `scripts/run_seq.py` (chain commands on one Kaggle GPU), `metrics.temporal_flicker`.
- Protocol register: D-28 (was only in this log) to D-33 appended to `docs/protocol.md`.

**Controls on the real model, run 1** (laptop, `results/local/patch_controls_v1/`; 65 s/clip):

| control | result |
|---|---|
| null whole-tensor patch (16 coords × 25 steps) | MAD 0, PSNR ∞: pass |
| null attention injection (42 modules × 25 steps) | MAD 0: pass |
| mis-indexed null patch | MAD 0.103, PSNR 25.1 dB: fails D-11, as it must |
| null x̃ / x̄ component patch | MAD 9.7e-3 / 5.0e-3: **FAIL** |
| unpatched B inside group 1 vs source B of group 2 | MAD 0 |
| unpatched A (group 2) vs source A of group 1 | MAD 6.0e-3: **FAIL** |
| symmetry f(A→B) vs f(B→A), up0 motion modules, first third | 0.172 vs 0.083: ratio 2.07, fails my 2× criterion |

Two bugs found by these controls:
1. The component patch computed `x̄_out + x_src − x̄_src`. That is an identity for a self-patch
   but not in fp16: rare 1-ulp rounding flips, amplified over 25 steps to MAD 1e-2. Rewritten
   as `out + (δ − mean_F δ)` (or `out + mean_F δ`) with δ = src − out, exactly 0 for a self-patch.
2. The attention recorder returned its own explicit-softmax output instead of the stock SDPA
   output, so recording perturbed the source clip (6e-3). Recording now returns the stock
   output and only stores a copy of the probabilities.
Both tests made bit-exact (`torch.equal`). **40 passed.** Controls re-run.

**LoRA preparation**
- `scripts/fetch_lora_data.py` (**D-33**): 19 HMDB51 clips (run 5, walk 5, jump 4, turn 5) at
  240p / 30 fps, 21–168 frames, 2.0 MB total, in `code/data/hmdb51_lora/` with sha256 manifest.
  Jump has only 4 because 2 of its 5 tag-clean clips come from the same film.
- `src/vdprobe/lora.py`: three variants (**D-32**). `full` = all temporal attention projections;
  `probe` = up_blocks.0 only (the H1 95% bootstrap set); `anti` = down_blocks.2+3 (lowest
  M(b)), a wrong-place control with more parameters than `probe`. First draft named "up0 +
  mid, the blocks with M ≥ −0.03": wrong, up1–up3 are also above −0.03; corrected before
  any training.
- `scripts/train_motion_lora.py`: 15 train / 4 held-out clips, validation loss at 8 fixed
  timesteps × 2 fixed noises; errors out if no gradient reaches the LoRA (checkpointing guard).
- `scripts/layer_rankings.py`: selectivity / presence (probe) / use (patching) / learning
  (relative ‖ΔW‖) per temporal block, pairwise Spearman.

**Controls, run 2** (`results/local/patch_controls_v2/`): x̃/x̄ null patches MAD 0 (pass), unpatched A
== other group's source (MAD 0). Remaining: null attention injection MAD 2.3e-3, PSNR 53.6 dB.
That fails the MAD bound and passes PSNR. Cause: the inject path replaced SDPA with an explicit
softmax on fp16-stored probabilities at 42 modules × 25 steps. Fix: injection now returns
`SDPA_out + α·W_out((P_src − P_live) V)` (mathematically equal to `lerp(P, P_src, α) V`),
probabilities stored in fp32 (≈0.9 GB per step), so a self-injection adds exactly zero.

**Controls, run 3 = of record** (`results/local/patch_controls/patch_controls.json`, 2026-10-20):

| control | result |
|---|---|
| null whole tensor / null x̃ / null x̄ / null attention injection | all MAD 0, PSNR ∞: **pass** |
| mis-indexed null patch | MAD 0.103, PSNR 25.1 dB: fails D-11, as it must |
| unpatched arm vs the same clip as another group's source | MAD 0 |
| symmetry, up_blocks.0 motion modules, first third | f(A→B) 0.171, f(B→A) 0.083: **FAIL** (ratio 2.06 vs the 2× criterion written before the run) |

The symmetry failure stands as a result. Same sign both ways, but A (golden retriever running)
→ B (horse walking) transfers twice as much as the reverse. One possible reading: the running
clip carries more motion to move. Not tested; one pair only. Cost: 65 s per clip on the laptop.

`run_arms` also gained `source_trajectory` and `src/vdprobe/inversion.py` was added (DDIM inversion
of an uploaded video onto the sampler's 25 timesteps). This is for the interface's "upload a
video as the motion source" input; not yet run on the GPU.

**Smoke tests (laptop, 2026-10-20 night)**
- `edit_sweep.py --split dev --pairs 1 --only <6 arms>` then `score_edits.py --select`: runs end
  to end, 65 s per clip with the CPU cache (`results/local/edit_smoke/`). One pair, so these are
  not results. Motion direction: x̃ patch on up_blocks.0 over the first third raises s_m(out, A)
  0.31 → 0.48 with s_c(out, B) unchanged (0.92), ΔD +0.22; attention injection (all 42 modules,
  first third) ΔD +0.30. Content direction: x̄ patch on all 21 motion modules at all 25 steps gives
  s_c(out, A) 0.99 but also s_m(out, A) 0.92 (it rebuilds A, motion included), ΔD −0.22. Choosing
  a block and window is exactly what the coarse dev sweep is for.
- `train_motion_lora.py --variant probe --steps 30`: 1.97 M trainable parameters, 2.42 s/step on
  the laptop, held-out validation loss 0.0996 → 0.0979 after 30 steps (gradients do reach the
  adapter). `results/local/lora_smoke/`.
- Added: per-track LoRA arms in `run_arms` (`Arm.lora`, adapters enabled only for that track),
  `--lora` flag in `edit_sweep.py`, LoRA arms in `selected_arms`, `tests/test_lora.py`.
  **42 passed.**

## 17. Stage 5 runs launched (2026-10-20 night, schedule slot "21 Oct")

- Kaggle kernel `rujularahate/vdprobe-dev` (T4 x2): `edit_sweep.py --split dev --arms coarse`,
  shards 0/2 and 1/2 (2 dev pairs each, source + 51 arms per pair). Outputs download to
  `code/results/kaggle/vdprobe-dev/`.
- First push was rejected (400 from SaveKernel): the script was 2.6 MB because the payload carried
  `data/` (the HMDB51 clips). `kaggle_run.py` now ships `data/` only with `--data`; the kernel is 89 KiB.
- Laptop, in parallel: `train_motion_lora.py --variant {full, probe, anti}` (800 steps each) into
  `code/results/local/motion_lora/<variant>/`, log in `code/results/local/motion_lora_log.txt`.
- The first LoRA chain OOMed on the full variant at training step 1 (7.26 GB in use on the 8 GB
  laptop GPU). Cause: diffusers' `enable_gradient_checkpointing()` checkpoints only the resnets in
  UNetMotionModel; the 21 motion modules (where the LoRA sits) keep all activations, and the full
  variant backpropagates through the down blocks too. `train_motion_lora.py` now wraps every
  `AnimateDiffTransformer3D` in non-reentrant checkpointing (same gradients). Full variant peak 5.0 GB.
  All three variants restarted with the same recipe; the OOM log is kept as `motion_lora_oom_log.txt`.

**Motion LoRAs trained (laptop, 2026-10-20 23:00 to 23:58)**, 800 steps, rank 32, lr 5e-5, seed 0,
`code/results/local/motion_lora/<variant>/`:

| variant | blocks | trainable params | val loss step 0 → 100 → 800 | peak GB | time |
|---|---|---|---|---|---|
| full | all 21 motion modules | 9.67 M | 0.0996 → 0.0658 → **0.0638** | 5.0 | 22 min |
| probe | up_blocks.0 | 1.97 M | 0.0996 → 0.0779 → **0.0652** | 4.6 | 17 min |
| anti | down_blocks.2 + 3 | 2.62 M | 0.0996 → 0.0693 → **0.0646** | 4.7 | 18 min |

The anti control ends below the probe-guided variant on held-out loss (0.0646 vs 0.0652), with
33% more parameters. Most of the drop happens in the first 100 steps for every variant, which looks
like adapting to HMDB51's look (grain, low resolution) rather than motion; validation loss is
not a motion-specific measure. The motion test is the Stage 6 edit run (prompt + LoRA arms).
No overfitting by step 800 (validation still falling slowly for all three).

**Dev sweep finished (Kaggle `vdprobe-dev`, 2026-10-21 02:15 IST)**: queued ~1 h, then ran ~2.6 h
wall on T4 x2. All 4 dev pairs done, 52 tracks each (source + 51 arms), exit codes [0, 0], no
tracebacks. 86.7 to 93.1 s per clip, peak 7.52 GB allocated (GPU cache), 5.19 GPU-h in total.
Outputs (290 MB: latents, 104 GIFs per pair, manifests): `code/results/kaggle/vdprobe-dev/out/job{0,1}/<pair>/`.
Not yet scored.

## 2026-10-21 · Session 3

### 18. Dev sweep scored, Stage 5 selection, test sweep launched (17:05–17:20)

**Scoring** (`scripts/score_edits.py results/kaggle/vdprobe-dev/out --select`, laptop GPU for the
VAE decode + CLIP; log `results/kaggle/vdprobe-dev/score_log.txt`): 4 dev pairs × 52 clips.
Outputs `scores.csv`, `scores_D.csv`, `summary.csv`, `selection.json` in `results/kaggle/vdprobe-dev/out/`.

**D-31 selection (rule fixed before scoring):**

| direction | selected arm | median ΔD (dev) | frac ΔD > 0 | 95% CI of D-16 stat |
|---|---|---|---|---|
| motion | x̃ patch, `up_blocks.3` motion modules, first third | **+0.080** | 3/4 | [−0.039, 0.121] |
| content | x̄ patch, `down_blocks.3` motion modules, first third | **+0.015** | 2/4 | [−0.014, 0.407] |

Copied to `code/configs/edit_selection.json` (shipped with the code; the test run reads it).

Observations (dev only, 4 pairs, so none of this is a result yet):
- The selected motion block is **not** the probing map's peak (`up_blocks.0`, ΔD +0.030). The
  editing optimum sits at the highest-resolution decoder block. Presence (probe map) and use
  (patching) disagree; this is what `layer_rankings.py` will quantify.
- The best motion arm overall is not a component patch: **attention injection, all 42 temporal
  attention modules, all steps, ΔD +0.151** (sm_A 0.82, sc_B 0.81). D-31 restricts the choice to
  component arms, so it is not selected; the test set runs attention injection at the selected
  window (first third) as the D-17 baseline.
- Content direction is weak everywhere (best ΔD +0.015). Patching x̄ over all modules/steps rebuilds
  A whole (sc_A 0.994 but also sm_A 0.942, ΔD −0.040): x̄ in the motion modules carries motion too,
  consistent with H3's failure.
- Whole-step windows hurt the motion direction (allmm.all ΔD −0.156): patching late steps copies
  A's appearance along with its motion.

**LoRAs to Kaggle:** private dataset `rujularahate/vdprobe-loras` (full/probe/anti .safetensors +
train logs, 28 MB). `scripts/kaggle_run.py --dataset <slug>` attaches it and links it to
`input/<slug>/` in the repo (searches `/kaggle/input`, since the mount path differs between images).

**Test sweep launched:** kernel `rujularahate/vdprobe-test` v1, 17:19 IST, T4 ×2, shards 0/2 and
1/2 of the 12 test pairs; per pair source + 17 arms (3 references; per direction the selected
patch, α 0.25/0.5/0.75, whole tensor; attention injection over the first third; one prompt-only
arm per LoRA). Expected ≈ 6 pairs × 18 clips × 90 s ≈ 2.7 h wall.

**Test sweep v1 failed** after ~3 min, before any generation (17:22). The dataset mounted fine
(`/kaggle/input/datasets/rujularahate/vdprobe-loras`), but `load_motion_lora` died in peft:
`ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions
above 0.16.0 are supported`. The Kaggle image ships torchao 0.10 and the kernel installed the newest
peft. Fix in `kaggle_run.py`: pin `peft==0.21.2` (the version the LoRAs were trained with locally)
and `pip uninstall torchao` (unused). Log kept in `results/kaggle/vdprobe-test_v1_failed/`.
Re-pushed as v2 at ~17:23.

### 19. Stage 4 figures (17:40–18:00, CPU, while the test sweep runs)

New `code/scripts/sweep_figures.py` reads `results/analysis/{analysis.json, maps.npz}` and writes
five PNGs to `code/results/figures/`: `sweep_selectivity_maps` (S_dyn and S_stat, 16 coordinates ×
25 steps, cells whose 95% CI excludes 0 marked), `sweep_h1_blocks` (M(b) with CIs),
`sweep_h2_timestep` (m(t) on up0.mm2 with τ and its CI), `sweep_static_energy`, `sweep_probes`
(verb probe over steps, x̃ vs x̄, four coordinates). Palette as in the deck (blue content, orange
motion, validated for the prep deck; `node` is not installed here, so it was not re-validated).
The diverging map uses unequal arms (−0.1 to 0.4) so the small motion values are visible;
the figure states this.

Things the figures make visible that the numbers in §14 did not:
- After the motion peak at steps 0–2, m(t) on up0.mm2 **dips below zero at steps 3–4** (−0.024,
  −0.030) before settling near 0. Not covered by the H2 test (two-segment fit); noted.
- S_stat is content-selective at down2 cross-attention at **every** step (the strongest row), while
  S_dyn fades by step ~12 everywhere.
- Verb probe on mid.sp.a20 (cross-attention): x̄ stays at 1.0 for all steps, x̃ falls to ~0.58, so
  the verb is decodable from the spatial path's frame mean throughout: prompt leakage, as §14 warned.
  On the temporal modules (mid.mm0, up0.mm2) x̄ also beats x̃ at every step.

### 20. Interface (checkpoint 6) built and checked on the GPU (18:55–19:35)

**New code**
- `code/src/vdprobe/demo.py`: the engine. `EditRequest` (A/B subjects and verbs, scene, seed,
  direction, method, block/window/α for a custom patch); `build_arms` (source A, optional B_ref, the
  prompt-only baseline, the edit) reusing `edit_plan`'s block/window helpers and the frozen
  selection (`configs/edit_selection.json`); `score_edit` (s_c, s_m, consistency, D, as in
  `score_edits.py`); `Engine` (loads AnimateDiff + CLIP + the three LoRAs on first use, runs
  `run_arms` with the CPU-side cache, saves GIFs + scores + request under `results/interface/`);
  `Engine.upload_edit` (real video → `inversion.invert` → motion-direction edit); readers for the
  cached sweeps. Methods: selected component patch, custom patch (any block/window/α), whole-tensor
  patch, attention injection, prompt + each LoRA. Free-text subjects/verbs/scenes are allowed.
- **D for a single live edit** has no pool of its own, so it is min-max normalised with the pool of
  the scored test run (dev until then); the app says which pool is in use.
- `code/scripts/app.py`: Gradio 6.30 front end, four tabs (Live edit, Upload a video, Results
  browser over the Kaggle sweeps with per-arm scores and the run summary, Where motion lives with
  the per-coordinate S_dyn/S_stat curves and the H1–H3 verdicts). `--no-gpu` for browsing only,
  `--share` for a public link. Queue concurrency 1 (one GPU).
- `code/tests/test_demo.py` (4 tests): arms match the selection in both directions; other methods
  build the right specs and reject invalid combinations; D on a synthetic perfect edit; shot split.
  A `df.clip` bug again (DataFrame.clip), fixed to `df["clip"]`. **46 passed.**

**GPU check, laptop** (`results/local/interface_check_log.txt`, outputs `results/interface_check/`):
- Model + CLIP + 3 LoRAs load in 20 s. Live edit, test pair 1 (golden retriever running → horse),
  motion direction, selected arm, with B: 4 clips in **270 s**, peak 6.37 GB.
  base vs edit: sm_A 0.377 → **0.596**, sc_B 0.959 → 0.936, D 0.285 → 0.286 (dev pool). Visually the
  edited horse gallops with the dog's splash pattern; the prompt-only horse trots.
- Upload, first try (`run_20060723sfjffbartsinger…`): inversion reconstructs the clip
  (sm 0.993, sc 0.949 vs the upload), but the edit was full of streak artefacts. Cause found: the
  clip has a **hard scene cut** (frame diff 66 vs median 5) and the 16-frame window spanned it.
- **11 of the 19 HMDB51 LoRA clips contain a cut** (mean grey-level jump > max(4× median, 20)).
  `train_motion_lora.py` draws random windows, so some training windows span cuts. Same data for all
  three variants, so the comparison is fair, but part of what the LoRAs fit is cuts. Not changed;
  flagged to the user (a cut-free retrain fits the remaining Kaggle budget).
- `inversion.read_video` now takes frames from the longest shot (`shot_bounds`).
- The server starts, serves the page, and serves GIFs as `image/gif` (checked over HTTP).

### 21. Paused by the user (2026-10-21, ~20:10)

- **Upload path, second fix:** with a cut-free clip the edit still failed, and so did the
  prompt-only arm: clips started from the DDIM-inverted noise z_{t_0} are blotchy and grid-patterned
  (inverted latents are not Gaussian; CFG 7.5 on them fails). Fix: `Track.start` (lockstep.py) lets
  the source start at z_{t_0} while the arms start from the seed's Gaussian noise; `run_arms` uses it
  when given both `noise` and `source_trajectory`; `Engine.upload_edit` and the upload tab take a seed.
  Default paths unchanged; **46 passed**. Re-check on `jump_Faith_Rewarded…` was running at pause
  time, log `code/results/local/interface_upload_check2_log.txt`, outputs `code/results/interface_check/`.
  Not yet looked at.
- **Test sweep `vdprobe-test` v2: COMPLETE.** First download died on an SSL error; the re-run
  completed: all 12 pair directories have a manifest, 305 MB in `code/results/kaggle/vdprobe-test/`.
  Kernel log and exit codes not yet checked; **not yet scored**.
- To resume: check the kernel log, then
  `scripts/score_edits.py results/kaggle/vdprobe-test/out` (laptop GPU, ~15 min), then layer
  rankings; look at the upload re-check; user decision pending on the spare Kaggle hours (≈16 h until
  the 24 Oct 05:30 IST reset): proposed cut-free LoRA retrain × 3 seeds + re-run LoRA arms +
  attention injection over all steps on the test pairs.

## 2026-10-22 · Session 4

### 22. Test sweep scored; layer rankings; upload failure diagnosed (19:10–20:00)

**Test kernel `vdprobe-test` v2:** exit codes [0, 0], 12/12 pairs × 18 clips, kernel time 10,017 s.

**Scoring** (`scripts/score_edits.py results/kaggle/vdprobe-test/out`, laptop GPU, ~15 min, log
`results/kaggle/vdprobe-test/score_log.txt`; `summary.csv`, `scores*.csv`, `score_summary.json`).
12 held-out test pairs, ΔD = paired gain over the prompt-only arm, CI = 95 % paired bootstrap of
the D-16 statistic (10,000 resamples):

| direction | arm | median D | median ΔD | frac ΔD > 0 | 95 % CI | consistency |
|---|---|---|---|---|---|---|
| motion | prompt only | 0.149 | 0 | — | — | 1.003 |
| motion | **selected x̃ patch** (up_blocks.3 mm, first third) | 0.195 | **+0.041** | 8/12 | [−0.027, 0.143] | 0.997 |
| motion | x̃ patch α 0.75 / 0.5 / 0.25 | 0.212 / 0.236 / 0.212 | +0.043 / +0.038 / +0.038 | 9, 9, 10 /12 | all include 0 | ≥ 0.997 |
| motion | whole-tensor patch | 0.074 | −0.036 | 5/12 | [−0.109, 0.047] | 0.988 |
| motion | **attention injection** (42 modules, first third; D-17 baseline) | 0.269 | **+0.105** | 10/12 | **[0.001, 0.204]** | 1.003 |
| motion | prompt + LoRA probe / full / anti | 0.179 / 0.096 / 0.061 | +0.023 / −0.021 / −0.020 | 7, 5, 5 /12 | all include 0 | ≤ 1.030 |
| content | prompt only | 0.090 | 0 | — | — | 1.009 |
| content | **selected x̄ patch** (down_blocks.3 mm, first third) | 0.198 | **+0.068** | 10/12 | [−0.010, 0.158] | 0.999 |
| content | x̄ patch α 0.75 / 0.5 / 0.25 | 0.189 / 0.199 / 0.112 | +0.070 / +0.029 / +0.006 | 10, 9, 8 /12 | α 0.5: [0.001, 0.152] | ≥ 0.989 |
| content | whole-tensor patch | 0.133 | −0.026 | 4/12 | [−0.080, 0.106] | 0.998 |

- **D-16 (pre-registered) is not met.** The selected component patches beat the prompt-only arm on
  the median in both directions, with the consistency guard holding, but neither CI excludes zero.
  The criterion required both. Recorded as a failure.
- The training-free baseline (temporal attention injection) is the only motion arm whose CI
  excludes zero, and only just (lower bound 0.001).
- Motion direction raw scores: the selected patch raises s_m(out, A) 0.233 → 0.673 and lowers
  s_c(out, B) 0.850 → 0.809. Most of the motion is transferred, at a small cost in content.
  The whole-tensor patch copies A almost entirely (s_m 0.823 but s_c(out, A) 0.912), so ΔD < 0.
  This is the content/motion entanglement the component split was meant to remove.
- LoRA arms (prompt + LoRA, no patch): the probe-guided LoRA has the highest ΔD of the three
  (+0.023 vs −0.021 / −0.020), but every CI includes zero. The LoRAs learn generic HMDB51
  run/walk/jump/turn, not A's motion, so they are compared with each other, not with the patches.

**Layer rankings** (`scripts/layer_rankings.py --analysis results/analysis/analysis.json --dev
results/kaggle/vdprobe-dev/out --lora results/local/motion_lora --out results/rankings`, CPU; log
`results/rankings_log.txt`):

| block | selectivity M(b) | presence (verb probe x̃) | use (dev ΔD, all / first third) | learning (rel. ‖ΔW‖, full LoRA) |
|---|---|---|---|---|
| down0 | −0.052 | 0.385 | −0.067 / +0.009 | 0.0070 |
| down1 | −0.045 | 0.347 | +0.036 / −0.004 | 0.0088 |
| down2 | −0.075 | 0.398 | +0.069 / +0.066 | 0.0131 |
| down3 | −0.077 | 0.396 | +0.028 / +0.031 | 0.0150 |
| mid | −0.026 | 0.483 | −0.003 / +0.009 | 0.0179 |
| up0 | **+0.005** | **0.498** | +0.029 / +0.030 | **0.0184** |
| up1 | −0.017 | 0.363 | −0.092 / +0.035 | 0.0122 |
| up2 | −0.019 | 0.363 | −0.084 / +0.079 | 0.0096 |
| up3 | −0.028 | 0.355 | −0.115 / **+0.080** | 0.0064 |

- Spearman (n = 9): presence ~ learning **+0.83 (p 0.005)**; selectivity ~ learning +0.23 (p 0.55);
  selectivity ~ use (first third) +0.07; use ~ learning −0.27. Every other pair |ρ| ≤ 0.48, p ≥ 0.19.
- So the LoRA puts its largest relative update at up_blocks.0, the block where both the selectivity
  map and the verb probe peak, and its block profile follows the probe closely. Patching
  usefulness does not follow either: it peaks at the outer decoder blocks (up2, up3).
- Caveat on "learning": relative ‖ΔW‖ is larger for the low-resolution blocks in general (down3,
  mid, up0 are the 8×8 bottleneck). Part of the agreement may be resolution, not motion. Not
  separated here.
- Dev "use" is from 4 pairs only.

**Upload path: why the edit came out as noise.** Look at the 21 Oct re-check (§21,
`results/interface_check/20261021-194658_upload__selected`): inversion reconstructs the video
(sm 0.996), the prompt-only tiger is fine, the selected x̃ patch gives pure colour noise, and
attention injection (first third) gives a near-static tiger (motion magnitude 0.23 vs 0.48 for
the video). Diagnostic (scratchpad `upload_diag.py`, norms logged inside `Patcher`):
- generated pair (A and B from the same noise): at up3 motion modules, ‖x̃_src − x̃_out‖ is
  20–4,000 against ‖out‖ ≈ 8–21k. The patch is a nudge, because A and B already share most of
  their frame-varying structure through the shared noise.
- upload (source on the inverted trajectory, target from Gaussian noise): the same difference is
  4,700–12,400 at step 0, and the target's output norm then grows every step
  (up3.mm2 19k → 54k by step 8). The final latent std is 26.0 against 0.70 for a generated edit.
  The target runs away.
- Cause: the source's frame-varying residual is spatially aligned with the source layout, not
  the target's. Transplanting it into a target with an unrelated layout, at the
  highest-resolution block for 8 steps, compounds. This is a property of the method, worth
  stating in the report: the component patch works on a pair that shares its initial noise and
  is not a layout-free motion transplant.
- Fix candidates being tested: target arms start from (A) the inverted noise z_{t0}, (B) a
  unit-variance blend of z_{t0} and Gaussian noise, (C) Gaussian noise with attention injection
  at all 25 steps. Script: scratchpad `upload_fix.py`, log `code/results/local/upload_fix_log.txt`.

### 23. Upload path fixed: layout-aligned start + attention injection at all steps (20:00–20:05)

**Comparison** (scratchpad `upload_fix.py`, log `code/results/local/upload_fix_log.txt`; video
`jump_Faith_Rewarded…` (baseball pitcher), "a man jumping" → "a tiger jumping", seed 42; sm_video =
flow cosine against the upload, 1.0 for the upload itself):

| new clips start from | prompt only | selected x̃ patch | attn, first third | attn, all 25 steps |
|---|---|---|---|---|
| A: inverted noise z_t0 | 0.118 | 0.519 (man/tiger hybrid, colour artefacts) | 0.309 | **0.668** |
| B: unit-variance blend of z_t0 and seed noise | 0.322 | 0.025 (runs away, noise) | 0.395 | **0.656** |
| C: seed noise (old behaviour) | 0.085 | 0.087 (runs away, noise) | −0.053 | 0.366 (near-static tiger) |

CLIP match to the target prompt is 0.33–0.36 for every non-broken clip (0.23 for the broken ones),
so the tiger content survives the transfer. Visually, A and B with attention at all steps give a
tiger in the pitcher's pose that follows the pitch, with purple/green grain. B has a cleaner
background than A.

**Decision:** upload default = **attention injection at all 25 steps, start "blend"**. Reasons:
- highest motion fidelity of the stable options;
- the blend keeps half of the start Gaussian, so it is less tied to one clip's inversion
  artefacts than z_t0;
- the component patch is stable only from z_t0, and even there it gives hybrids.
The patch stays selectable, with the start choice exposed and a warning in the tab text.

**Code:**
- `demo.METHODS["attn_all"]` (attention injection over the whole trajectory; also offered in the
  live tab, where it was the best motion arm of the dev sweep, ΔD +0.151);
- `demo.upload_start(z_inv, gauss, start)`;
- `Engine.upload_edit(..., method="attn_all", start="blend")`, with the start recorded in
  request.json;
- the app's upload tab gets a start radio, the new default and new explanatory text;
- tests for `attn_all` and `upload_start`. **47 passed.**

**Check through the real engine on a second clip** (`run_20060723sfjffbartsinger…`, a woman
walking towards the camera; "a man running" → "a robot running"; log
`code/results/local/interface_upload_check3_log.txt`; outputs
`code/results/interface_check/20261022-195627_upload__attn_all/`; 249 s): sm_video base 0.123 →
edit **0.427**, CLIP-target 0.288 → 0.274. Only partly successful: the approach-the-camera motion
is transferred, but onto the scene (a pillar grows in the frame) rather than onto the robot, which
stays small and nearly static. The temporal attention carries *where* things move in the frame,
not *which object* moves. A limitation to state, not to hide.

### 24. Paused: GPU handed back to the user (2026-10-22, ~20:15)

Nothing was running at the pause: the laptop GPU was idle (no compute process; the last job,
`upload_check3.py`, had finished and written its outputs) and every Kaggle kernel (`vdprobe-sweep`,
`-dev`, `-test`) is COMPLETE. No partial state to recover.

To resume:
1. Point the interface's D pool at the test run (`demo.SCORED_RUNS` / `reference_pool`), then a
   click-through of all four tabs (`scripts/app.py`, ~10 min of GPU).
2. README + usage docs + submission packaging (no GPU).
3. Pending user decision: Kaggle run (~4 h wall on T4 ×2) = cut-free LoRA retrain, 3 variants ×
   3 seeds + LoRA arms and attention injection over all steps on the 12 test pairs.

## 2026-10-24 · Session 5

### 25. New machine; Kaggle account switched (afternoon)

- Work moved to `~/IE643` on a machine with **no GPU**. CPU venv `code/.venv` (torch 2.10.0+cpu,
  diffusers 0.40.0, torchvision 0.25.0+cpu): **47 passed** before any change.
- Kaggle CLI 2.2.5 in `~/kaggle/.venv` (Python 3.12; 1.7.x cannot read the new `KGAT_` access
  tokens), token in `~/.kaggle/access_token`. Account: `shivanshisme`.
  `scripts/kaggle_run.py` takes `--user` (or `$KAGGLE_USER`) instead of the hard-coded account.
- The sweep outputs of §14–22 (`results/kaggle/vdprobe-{sweep,dev,test}`, incl. `sketches.pt`) are
  **not on this machine** and are private to the old account: the D-34 probes below cannot run on
  real data until they are copied over (or the sweep is re-run, ≈5 h of quota).
- Demo kernel `shivanshisme/vdprobe-demo-clips` (T4): retriever running / walking / horse running
  from seed 42, 79–92 s per clip. Outputs in the session scratchpad (not in the repo).

### 26. Probe controls, D-34 (CPU)

- `analysis.probe_scores`: Hewitt–Liang control task (type = (subject, verb) within a seed, labels
  permuted over types), MLP probe (hidden 10, L2 1.0, lbfgs, 500 iterations), type-level restricted
  permutation null (verbs within subject; family across subjects), draws shared by all cells, so a
  max-over-map p (`p_fwer`) is reported beside the per-cell p. `probe_accuracy` now calls it
  (bit-identical, tested). `analyze_sweep.py --probe-controls [--n-perm 99]`.
- First version used hidden 64 and free label shuffles; changed before any real run after the
  methods review (small hidden + weight decay are what kept MLP probes selective in Hewitt & Liang;
  a free shuffle also destroys subject structure, so it is not the right null).
- Speed: sklearn fits at n = 60 ran ~30x slower with multi-threaded BLAS (MLP 14 s per fit);
  `threadpool_limits(1)` per cell + joblib over cells.
- Tests: planted verb code is selective (control ≈ chance) and beats every null draw; a pure
  prompt-identity code scores high on real *and* control labels, selectivity ≈ 0. **55 passed**.

### 27. Cross-attention visualisation; model variants (Kaggle `vdprobe-xattn`, T4 ×2)

- `harness.CrossAttentionMaps` (DAAM aggregation per frame: head-averaged mass on a phrase's
  tokens, bicubic upsampling, summed over the 16 spatial attn2 layers and 25 steps; τ = 0.4 masks)
  and `scripts/cross_attention_maps.py` (overlay GIFs, layer × step-window grids, masks.npz).
- `configs/variants/`: `epicrealism_mm2`, `sd15_mm3`, `epicrealism_mm3` (only base model / adapter
  changed). v1-5-3 has no mid-block motion module.
- Run on "a golden retriever running on a beach", seed 42, all four variants, 72–75 s per clip:
  subject-mask IoU across frames 0.64 (sd15+v2, the frozen model), 0.72 (epiCRealism+v2), 0.56
  (sd15+v3), 0.82 (epiCRealism+v3). Not a stability score: IoU also falls when the subject travels
  (the sd15+v3 dog runs across the frame). By eye, the frozen model's dog morphs and both
  epiCRealism dogs keep their identity; a proper identity-consistency metric is still to be run.
  Outputs: `code/results/kaggle/vdprobe-xattn/out/`, contact sheet `variants_contact_sheet.png`.
- Seen in the grids, not yet measured: the verb token's attention sits on the dog's body from
  step 3 on, and the mid block's map concentrates on the lower half (legs, splash).

### 28. Experiment 2 groundwork (CPU)

- `metrics.raft_flow` (torchvision RAFT-large), `remove_camera_motion` (median background flow),
  `masked_motion_fidelity` (flow cosine inside the subject masks). RAFT checked on a synthetic
  2-pixel translation (opt-in test, `VDPROBE_DOWNLOAD_TESTS=1`).
- `analysis.debiased_cka` (unbiased HSIC), `rsa_commonality` + `rsa_permutation` (multiple-regression
  RSA with unique/shared variance and permutation p).
- Design research notes: `research_notes/Content motion probing experiment design/`.

### 29. Experiment 2 launched ("run everything", 2026-10-24 evening)

- **Code** (all tested on CPU, **74 passed**; GPU paths smoke-tested on Kaggle):
  - `probe_sweep.py --exp2` (+ `--set ladder`), `configs/ladder.yaml` + `ladder.py`;
  - `exp2_quality.py` (VBench subject consistency / dynamic degree, CLIP-T, X-CLIP verbs);
  - `exp2_analysis.py`, `exp2_grid.py`, `exp2_ladder.py`, `exp2_select.py` (CPU analyses);
  - `exp2_patching.py` (A→B, B→A, C→B), `exp2_erasure.py` (LEACE vs random), `erasure.py`;
  - `train_motion_lora.py` (module-set arms, MotionDirector debiased loss, shot-aware windows,
    `--max-stride`), `eval_motion_lora.py` + `motion_eval.py` (MFS), `fetch_lora_data.py` (any class);
  - runbook `code/docs/experiment2.md`; rules fixed before any data in protocol D-35.
- **Bugs caught before they cost GPU time:** a `global` declared after use in the training script
  (SyntaxError; pyflakes does not catch it — every script is now also compile-checked); the LEACE
  eraser's affine term shifted x̄ (fixed by re-centring over frames); zero-variance layers at step 0
  (text-blind ResNets before any step: identical across clips) divided by zero in the ANOVA.
- **Smoke test** (`vdprobe-exp2-smoke`, sd15_mm2 + epicrealism_mm3, 4 clips each): 117 / 114
  coordinates, 43 / 42 scored, 108 s per clip (+12 % over the plain sweep), peak RSS 8.3–8.5 GB.
  ~13 MB of outputs per clip. RAFT on the frozen model's clip: 19 px/frame mean flow (its frames
  change by 26 grey levels per frame: morphing + splashes), vs 0.5 px on epiCRealism+v3 — the
  better model barely moves the dog, which the dynamic-degree gate will have to weigh.
- **Data:** cartwheel clips, HMDB51 static camera, ≤3 per film: 8 candidates, 3 have cuts,
  **5 kept** (private Kaggle dataset `shivanshisme/vdprobe-cartwheel`).
- **Kaggle limit:** at most **2 GPU batch sessions** at once ("Maximum batch GPU session count of
  2 reached"); kernels are queued by a polling script.
- **Running at the end of this entry:** Phase 2 `vdprobe-exp2-sd15-mm2` and `-epicrealism-mm2`
  (T4 ×2 each, ~4 h); queued: `-sd15-mm3`, `-epicrealism-mm3`, then `vdprobe-lora-pilot`
  (T on up_blocks.0 r32 vs all-temporal r7, 1,000 steps each, then evaluation).

### 30. Interface finished for the demo (2026-10-24, night)

- `demo.py`: D pool falls back (test → dev → `results/local/edit_smoke` → `configs/d_pool.json`) and
  edits still run without one (s_c / s_m only); results browser also lists the local scored run;
  `Engine(variant=...)` for the Experiment 2 model cells (LoRAs that do not fit a cell are skipped
  and reported); `Engine.cross_attention` (live DAAM maps); `cached_xattn`; `link_assets`.
- `app.py`: fifth tab **Cross-attention** (cached 2x2 maps + contact sheet, live maps, the
  Experiment 2 replication table once analyses exist); `--variant`, `--assets`.
- Checked in a real browser (Playwright, `--no-gpu`): all CPU tabs render, no console errors.
- GPU paths: `scripts/interface_check.py` (both edit directions, LoRA, upload, live cross-attention,
  app build) queued on Kaggle as `vdprobe-interface-check`, ahead of the remaining Phase 2 cells.
- Live demo without a local GPU: Kaggle notebook `shivanshisme/ie643-interface`
  (`code/notebooks/run_interface_kaggle.ipynb`) + private datasets `vdprobe-code`,
  `vdprobe-app-assets`; prints a public gradio.live URL. Its CPU-only run found both datasets.
- Still missing on this machine: the dev/test sweep outputs, so the browser shows only the local
  smoke pair and D uses the 1-pair smoke pool until they are copied in.

### 31. Interface redesign with the apple-design skill (2026-10-25, early morning)

Skill: `github.com/dickwu/apple-design-skill` (HIG reviewer; read in an isolated clone, nothing executed).
For a web app it applies the principles and foundations only. Read: accessibility, layout, typography, color,
dark-mode, writing, loading, feedback, progress-indicators, charting-data, branding.
- **Findings (measured):** subdued text #BBBBC2 on white 1.91:1 (needs 4.5:1); white on the orange
  primary button 2.8:1 (bold needs 3:1); orange meant both "button" and "motion" (color › one colour, one
  meaning); flickering clips looped as GIFs with no control (accessibility › Cognitive); 4–5 min edits with no
  progress (progress indicators › determinate); blue/orange lines told apart by colour alone; jargon labels
  (`sc_A`, `motion.dynamic.up_blocks.0.ft`); unreadable 16-row attention grid and contact sheet.
- **Changes:** tokens with contrast checked in light and dark (text ≥ 5.4:1, data graphics ≥ 3.3:1); neutral ink
  controls, colour reserved for the two channels (■ content blue, ▲ motion orange, always with the shape);
  clips as MP4 with play/pause (`demo.playable`, ffmpeg); determinate progress per denoising step
  (`demo.StepProgress` → `gr.Progress`); plain-language names (`arm_label`, `pretty_scores`); chart with a
  headline, line styles, markers and direct labels; scores formulas moved into "How the scores work";
  wide attention grid (rows = step windows); readable contact sheet; selected tab in ink (slate-500 was 3.6:1
  on dark). Checked in the browser, light and dark. **74 passed.**
- Kaggle: `vdprobe-code` dataset re-versioned; GPU check rebuilt with progress ticks; cross-attention re-run
  queued for the new grids.

## 2026-10-25 · Session 6 (Stage 6b)

### 32. Stage 6b: attention-level editing operators from the literature (D-37), 11:45–13:00

**Why.** The editing demonstration has to work. Stage 6's verdict stands as recorded (D-16 not met for
the component patches; only temporal attention injection's CI excluded 0). Literature check:
- MotionClone (ICLR 2025) and MOFT (NeurIPS 2024) work through *guidance* on temporal attention or on
  the frame-mean-removed feature (our x̃), at up_blocks.1, not by copying activations.
- UniEdit injects temporal attention Q/K at all layers and steps (motion), and replaces spatial
  self-attention values from a reconstruction branch (content).
- MasaCtrl (ICCV 2023): the target's spatial self-attention queries the source's K/V from step 4/50 in
  decoder layers 10–15. Appearance transfers; layout and motion stay the target's own (its queries).
Our x̄ patch failed because it overwrites the residual stream, which carries motion too (H3).
Attention-level operators keep the target's own queries.

**Code.**
- `edit.KVSpec`, `edit.KVInjectingProcessor`: spatial attn1 K/V injection, framewise or frame-0,
  optionally extended.
- `AttnSpec.source` / `KVSpec.source`: any earlier unedited track (e.g. `B_ref`) can be a source, so
  "dual" arms take appearance from one clip and motion from another.
- `run_arms` now runs one controller per operator, with recordings keyed by (track, module).
- `edit_plan.literature_specs/literature_arms/literature_test_arms`; `edit_sweep.py --arms lit [--lit-sel]`;
  `score_edits.py --select-lit` (D-37 rule) and `--out` (for `run_seq.py`).
- `demo` methods `lit_single` / `lit_dual` (read `configs/lit_selection.json`, papers' defaults until it
  exists); labels in `arm_label`.
- Three **probe-placed** arms (the other session's suggestion, user approved): temporal attention at
  up_blocks.0 + mid at steps 0–2 (H1 + H2) and at all steps; K/V at down_blocks.2.
- Decision number: D-36 was already taken by the Experiment 2 corrections, so this is **D-37**.
**Tests.** `tests/test_kv.py`, 7 tests:
- recording is read-only, and self-injection of K/V is bit-exact;
- the injected source is the named track (end to end through `run_arms` on the tiny UNet with a fake
  pipeline: A's prompt + A's own K/V == A; B's prompt + B_ref's temporal attention == B_ref);
- source-order validation; the arm builders; the demo methods.
Full suite: **81 passed, 1 skipped** (before the demo test was added).

**Run.** Kaggle `shivanshisme/vdprobe-lit-dev` (T4 ×2), pushed ~12:55 IST: `edit_sweep.py --split dev
--arms lit` sharded 0/2, 1/2 (source + 3 refs + 14 arms per pair), then `score_edits.py` per shard.
Scores are merged locally before the D-37 selection.

### 33. Experiment 2 steps 2-8: analysis fixes and restarts (2026-10-25, 09:00-13:20)

**Analysis corrections (D-36, made before any selection).**
- Flow decoding: the ridge strength was chosen by leave-one-clip-out CV, which leaks through the ± residual
  pair of each prompt (the left-out clip's partner predicts it exactly). Replaced by grouped 4-fold inner CV
  over subjects with an SVD ridge path (`exp2._cv_r2`, `_ridge_path`), alphas 10^-1…10^6, and the gain floored
  in `exp2_analysis.py` (G = max(R²_layer, 0) − max(R²_latent, 0)). Test:
  `test_paired_residuals_do_not_leak_into_the_ridge_choice`. Before: median R² −0.18/−0.39, gains up to 1.34.
- RSA: the CLIP text-embedding RDM added as the third model RDM, as the design report specifies.
- `exp2_verdict.py` reads patching CSVs with `keep_default_na=False` ("null" is a scene name, not NA).

**Runs.** Per-cell analyses (local CPU, ~50 min each, `--n-jobs 4` because 8 workers exceeded 7 GB RAM):
| cell | Exp 1 site replicates? (one-sided p) | best decoding gain over latent |
|---|---|---|
| sd15_mm2 | no (0.189) | 0.138, mid_block.resnets.0, step 19 |
| epicrealism_mm2 | no (0.063) | 0.253, down_blocks.3.motion_modules.0…attn2, step 12 |
| sd15_mm3 | no (0.167) | (rerun after the shutdown) |

**Interruptions.**
- 12:28 the machine was powered off (clean shutdown) and came back at 13:08; the sd15_mm3 analysis was lost
  minutes before finishing and `/tmp` (the queue/wait helpers and logs) was wiped.
- The helpers now live in the repo (`scripts/kaggle_queue.sh`, `scripts/kaggle_wait.sh`; the wait helper
  touches `.download_ok`), and `exp2_orchestrate.sh` logs to `results/exp2/logs/` (survives a reboot).
  Restarted 13:13; it resumes from the first missing output.
- Disk: 5.3 GB free before the Kaggle step; freed the uv cache and Chrome cache (7.2 GB free), so the
  orchestrator's prune of `latents.pt` / `sketches_refined.pt` does not trigger.

**Shared Kaggle account.** The Stage 6b session's kernels (`vdprobe-lit-dev`, `vdprobe-switch-a`) occupy both
GPU slots; the Experiment 2 kernels wait in the queue helper (retry every 5 min) and share the weekly quota.
Full test suite after the other session's `edit.run_arms` changes: 87 passed, 1 skipped.

### 33. Stage 6b dev results and the D-37 selection (13:25–13:45)

**Dev kernel** `vdprobe-lit-dev`: exit 0 on both shards. Per pair: source + 17 arms, ~1,520 s (84.5 s per clip),
then scoring ~160 s per shard. The two shards' raw `scores.csv` were merged into one pool
(`results/kaggle/vdprobe-lit-dev/merged/`) before D and the selection. `score_edits.select_lit` first looked for
`arms.json` with `rglob`, which does not follow the merged directory's symlinks; it now looks one level down first.

**Dev, 4 pairs, merged pool** (median ΔD over prompt-only; CIs from 4 pairs are wide):

| arm | family | median ΔD | frac > 0 | sm_A | sc_B | sm_B | sc_A |
|---|---|---|---|---|---|---|---|
| motion.attn.all.all | single | **+0.152** | 3/4 | 0.823 | 0.806 | 0.392 | 0.777 |
| motion.dual.up23.from2 | dual | +0.147 | 4/4 | 0.660 | 0.922 | 0.475 | 0.761 |
| motion.dual.dec.from2 | dual | +0.095 | 3/4 | | | | |
| motion.attn.up1.all (MotionClone block) | single | +0.024 | 2/4 | | | | |
| motion.attn.up0.all / probe.early / probe.all | single | −0.002 / −0.005 / −0.020 | | | | | |
| content.dual.all | dual | **+0.121** | 3/4 | 0.546 | 0.774 | 0.601 | 0.930 |
| content.kv.probe.from2 (down_blocks.2, probe-placed) | single | **+0.056** | 3/4 | 0.272 | 0.738 | 0.321 | 0.864 |
| content.kv.up23.from2.first | single | +0.019 | 3/4 | | | | |
| content.kv.up23.from2 / from5 / dec.from2 (MasaCtrl defaults) | single | −0.007 / −0.003 / −0.032 | | | | | |
| content.dual.ft | dual | −0.014 | | | | | |

**D-37 selection** (`configs/lit_selection.json`):
- motion: single `motion.attn.all.all`, dual `motion.dual.up23.from2`;
- content: single `content.kv.probe.from2`, dual `content.dual.all`.

Observations (dev only, not results):
- Content: the MasaCtrl placements (up_blocks.2–3 / all decoder, framewise) raise sc_A but also raise sm_A
  (0.35 → 0.63–0.86). Framewise K/V carries A's per-frame layout, so motion comes along with appearance:
  the same entanglement as the x̄ patch. The frame-0 variant halves that leak. The probe-placed down_blocks.2 K/V
  leaks no motion (sm_A 0.27, same as prompt-only) and is the only positive single-source content arm.
- Motion: the probe-placed temporal attention arms (up0 + mid, steps 0–2 or all) do nothing. The probe's
  *location* of motion selectivity is not where motion can be *injected*, which repeats Stage 5's presence-vs-use
  gap. Injection must cover all temporal attention to transfer motion.
- **Looked at the clips** (contact sheets `results/figures/stage6b_dev_*.png`):
  - motion.attn.all.all works on the woman→soldier and fox→German shepherd pairs: B's subject follows A's path
    and composition.
  - On child→elderly man it reproduces A's child: the content leaks.
  - On the bear pair it gives a ghosted, semi-transparent retriever.
- **Caveat on the dual family:** dual arms inject from B_ref, the clip they are then scored against (sc_B for
  motion, sm_B for content). Part of their D is therefore by construction. They answer a different question:
  compositing two clips, not editing from a prompt.
  - On the bear pair B_ref is itself broken (sand texture, no dog), and motion.dual copies that broken look.
  - So dual arms are reported separately from single-source arms and are not compared with the prompt-only
    baseline as if they were the same kind of edit.

### 34. Stage 6b test results (16:07–16:30)

**Kernel** `vdprobe-lit-test`: exit 0 on both shards (6,117 s and 6,674 s for 6 pairs each, 11 tracks per pair),
scoring 280 s and 310 s. The shards were merged into one pool (`results/kaggle/vdprobe-lit-test/merged/`, copied to `out/`).

| arm | family | median ΔD | frac > 0 | 95% CI | D-16 | sm_A | sc_B | sm_B | sc_A |
|---|---|---|---|---|---|---|---|---|---|
| motion.base | — | 0 | — | — | — | 0.234 | 0.850 | 0.172 | 0.769 |
| motion.attn.all.all | single | +0.097 | 10/12 | [−0.019, 0.185] | ✗ | 0.761 | 0.811 | 0.094 | 0.830 |
| motion.dual.up23.from2 | dual | **+0.173** | 12/12 | **[0.111, 0.266]** | ✓ | 0.570 | 0.891 | 0.218 | 0.762 |
| motion.attn.all.ft (Stage 6 baseline) | — | +0.096 | 9/12 | [−0.030, 0.213] | ✗ | 0.641 | 0.828 | 0.098 | 0.793 |
| motion.sel (Stage 6 x̃ patch) | — | +0.028 | 8/12 | [−0.057, 0.123] | ✗ | 0.673 | 0.809 | 0.119 | 0.829 |
| content.base | — | 0 | — | — | — | 0.118 | 0.776 | 0.259 | 0.830 |
| content.kv.probe.from2 | single | +0.022 | 8/12 | [−0.027, 0.151] | ✗ | 0.279 | 0.783 | 0.204 | 0.867 |
| content.dual.all | dual | **+0.145** | 12/12 | **[0.102, 0.279]** | ✓ | 0.251 | 0.765 | 0.562 | 0.898 |
| content.sel (Stage 6 x̄ patch) | — | +0.092 | 11/12 | [0.004, 0.191] | (✓ in this pool) | 0.313 | 0.760 | 0.182 | 0.894 |

Reading:
- **Single-source attention edits do not pass on test.**
  - Temporal attention over all steps transfers A's motion strongly (sm_A 0.23 → 0.76), but A's appearance comes
    along (sc_A 0.77 → 0.83, sc_B 0.85 → 0.81), and the metric charges for that.
  - The probe-placed K/V content arm adds little on test (+0.022; it was +0.056 on dev).
- **Dual-source edits pass, 12/12 in both directions.**
  - Motion: A's motion (sm_A 0.23 → 0.57) with B's look (sc_B 0.85 → 0.89), without copying B's motion
    (sm_B 0.17 → 0.22).
  - Content: A's look (sc_A 0.83 → 0.90) with B's motion (sm_B 0.26 → 0.56).
  - They need clip B and read it directly, so they answer "swap motion and appearance between two clips". They do
    not answer "edit from a prompt".
- The same Stage 6 clips score differently in this pool (content.sel now excludes 0). D is min-max normalised
  over each run's pool, so cross-pool comparisons of CIs are fragile. The Stage 6 verdict stands as recorded.
- **Looked at the clips** (`results/figures/stage6b_test/<pair>.png`, rows = arms with their D, columns = frames
  0/5/10/15):
  - athlete→soldier: motion.attn.all.all and the dual arm give a soldier walking away along the athlete's path.
    It is the clearest single-source success on test.
  - deer→bear: the bear holds the deer's standing pose.
  - soldier→dancer: content.dual.all scores D = 0.39, but the clip is sand and a boot. A high per-pair D does not
    guarantee a good edit, so demo clips are chosen by eye.

<<<<<<< HEAD
### 34. Quality gate, primary cell (D-39), local steps done (2026-10-25, 15:11-19:20)

- **Quality gate** (local CPU, 2 h 15 min, 960 clips). No cell met both D-35 criteria:
  - The epiCRealism cells are nearly static (dynamic share 0.021 / 0.004; median top-5 % flow 1.9 / 1.1 px).
  - The SD1.5 cells move (0.83 / 0.93; 25 / 17 px) but miss the CLIP-T tolerance by 0.001–0.002.
  - The code's unwritten fallback picked epicrealism_mm3 (highest subject consistency). I stopped the
    orchestrator after selection, before any kernel was built, and nothing was pushed.
- **D-39** (decided by me at the user's request): drop the CLIP-T tolerance and keep dynamic share ≥ 0.5.
  The primary cell is **sd15_mm2**; the ladder verbs are walking and running.
  - `exp2_quality.decide` implements it; `--from-csv` re-derives the decision from `quality.csv`.
  - The superseded selection is kept in `results/exp2/selection_superseded_epicrealism_mm3/`, and the
    original gate output in `quality_D35_fallback.json`.
- **Candidates** on sd15_mm2 (D-35 rules):
  - exp1_region (up0.mm2, steps 0–2);
  - rsa_motion_peak (up0.mm2…attn2, steps 22–24);
  - up1_motion_modules (steps 0–7);
  - rsa_content_peak (mid attn1, steps 0–3, static).
  - LoRA target: up0.mm2, up0.mm1, up1.mm2.
- **Fixes:**
  - `exp2_grid.py` failed unpacking the interaction contrast; fixed.
  - The channels section was recomputed in fp64 (D-36 item 3). Medians barely moved: sd15_mm2 0.2971 → 0.2970.
- **Kaggle account:** the user asked that nothing more run on shivanshisme; another account will be provided.
  - The orchestrator now needs `KAGGLE_USER=<account>`. Without it, it stops after the local steps.
  - `kaggle_wait.sh` takes the account from the same variable; no account name is hardcoded.
  - The CLI can take a second account's token through `KAGGLE_API_TOKEN` without touching `~/.kaggle/access_token`.
- The machine rebooted again at 19:00; nothing was lost (all steps resume from their outputs).
=======
---

> **Merged log (2026-10-25).** The two sessions below were logged on the `main` branch, in parallel
> with Sessions 5–6 above on `final-submission`; their section numbers (§25–28) repeat those above and are
> kept so references to them stay valid.

## 2026-10-24 · Session 5

### 25. Environment rebuilt; cut-free LoRA retrain launched on Kaggle (18:55–19:25)

- `~/MI_DiffusionModels/` (and with it the venv) no longer exists. Rebuilt the venv at
  **`code/.venv`** (gitignored) with uv, same pins: Python 3.10, torch 2.10.0+cu128, diffusers 0.40.0,
  transformers 5.20.0, peft 0.21.2, scikit-learn 1.7.2, gradio 6.30.0; `vdprobe` installed editable.
  Packages hardlinked from the uv cache, but `/home` still dropped from 4.2 to 1.3 GB free. Run
  everything as `cd ~/IE643_CourseProject_Prep/code && env -u PYTHONPATH .venv/bin/python -u ...`.
  The HF cache at `/var/tmp/rujula_hf_cache` is intact. Suite: 47 passed.
- **User decision (2026-10-24):** run the cut-free LoRA retrain on Kaggle (quota reset this morning).
- **D-34** (protocol register): retrain full / probe / anti with windows inside one shot, seeds 0, 1, 2;
  on the 12 test pairs one prompt-only arm per LoRA plus attention injection over all 25 steps,
  scored in one pool with the Stage 6 test run.
- Code:
  - `train_motion_lora.py --cut-free` (`usable_shots`, `pick_shot`; with one shot no random draw, so
    the old recipe's stream is unchanged), `--data-dir` (finds a flat copy by file name); the seed
    and the flag go into the LoRA metadata;
  - `edit_plan.lora_arms`, `edit_plan.adapter_name` (`lora_<variant>_s<seed>` for the new files,
    `lora_<variant>` for the old ones);
  - `edit_sweep.py --arms lora --lora-dir --lora-wait N` (waits for N `train_log.json`, aborts on a
    `*.FAILED` marker or after 3 h); `run_seq.py` writes `<name>.FAILED` when a command fails;
  - `score_edits.py --with RUN` (pools another run's `scores.csv`; duplicates keep the earlier run's row).
  - Tests: `test_lora_windows.py` (3), one more in `test_edit_plan.py`. **51 passed.**
- Laptop smoke test (`--variant probe --cut-free --seed 1 --steps 3`): all 19 clips get a clean
  window (shortest: BLACK_HAWK_DOWN, 14 frames, last frame repeated); base val loss 0.0982 (0.0996
  with the old windows); metadata carries seed and cut_free.
- Private dataset `rujularahate/vdprobe-hmdb51`: the 19 clips flat + manifest (the payload limit
  rejected `data/` inside the kernel on 20 Oct).
- **Kernel `rujularahate/vdprobe-lora` v1 pushed 19:20 IST**, RUNNING at once. GPU 0: full s0–s2,
  probe s0, then sweep shard 0/2; GPU 1: probe s1–s2, anti s0–s2, then sweep shard 1/2. Each
  sweep waits for all 9 LoRAs. Per pair: source, B_ref, motion.base, motion.attn_all, 9 LoRA arms
  = 13 clips. Expected ≈ 2 h training + 2 h sweep. The CLI returns no log while a kernel runs.

### 26. Interface: D pool, click-through, docs, packaging (19:25– )

- D pool: `demo.SCORED_RUNS` already lists `test` first and the test run is scored, so live edits
  are normalised with the test pool. No change needed.
- Click-through through the running app with `gradio_client` (same handlers as the buttons):
  results browser on all 12 test + 4 dev pairs (gallery and score table agree for every pair,
  summary 16 / 50 rows), maps tab on all 16 coordinates: no errors.
- `README.md` rewritten for the whole project (deliverables, results, setup, interface, reproduce,
  data, layout, limitations); the pilot write-up moved unchanged to `docs/pilots.md`. The LoRA
  section has a placeholder for the D-34 numbers.
- `.gitignore`: `results/kaggle/` records, scores and logs are now tracked (58 MB, mostly the six
  9 MB `distances.csv` of the probing sweep, the input of the Stage 4 analysis); GIFs and `.pt`
  tensors (~1.8 GB) and `results/interface/` stay out.
- GPU handlers through the app (log `code/results/local/interface_clickthrough_log.txt`):
  - live edit, content direction, selected x̄ patch (cat walking → dog jumping, forest, seed 7):
    4 clips in 429 s; ΔD over the prompt-only arm +0.014 (D 0.096 → 0.110);
  - live edit, motion direction, prompt + LoRA probe (test pair 1): 422 s; D 0.269 → 0.244;
  - content direction + attention injection: rejected with the intended message (gr.Error);
  - upload, attention injection at all steps, blend start (OldSchool spin, "a man" → "a bear"):
    279 s; sm_video 0.17 (prompt only) → 0.66 (edit); CLIP-target 0.26 → 0.30.
  - The first upload call failed in my test client (Gradio 6 wants `handle_file(path)`, not
    `{"video": ...}`); not an app bug.
  - Live edits ran at ~105 s per clip vs 68 s on 22 Oct with nothing else on the GPU (probably
    clocks or temperature); `SEC_PER_CLIP` and the app text now give 70–105 s.

## 2026-10-25 · Session 6

### 27. Cut-free LoRA run done and scored (15:00–16:30)

- Kernel `rujularahate/vdprobe-lora` v1: COMPLETE, exit codes [0, 0], no tracebacks or `.FAILED`
  markers. 9 LoRAs trained (full 1,787–1,797 s, probe 1,312–1,450 s, anti 1,376–1,378 s per T4), then
  the edit sweeps (job0 7,023 s, job1 6,138 s), 12 pairs × 13 clips.
- Downloaded without the `.safetensors` (disk: 1.8 GB free on /home):
  `kaggle kernels output ... --file-pattern '^(?!.*\.safetensors$).*$'` → `code/results/kaggle/vdprobe-lora/`
  (225 MB). **The nine cut-free LoRA weights are still only on Kaggle**; to ship them, download with
  `--file-pattern '.*\.safetensors$'` (each ~4–20 MB).
- Regeneration check: source, B_ref and motion.base latents equal the test run's on all 12 pairs
  (MAD 0), so pooling is valid.
- Scoring: `score_edits.py results/kaggle/vdprobe-lora/out --with results/kaggle/vdprobe-test/out`
  (laptop, ~12 min, log `results/kaggle/vdprobe-lora/score_log.txt`).

| arm (motion direction) | median ΔD | ΔD > 0 | 95 % CI |
|---|---|---|---|
| attention injection, all 25 steps | **+0.120** | 11/12 | **[0.020, 0.203]** |
| LoRA probe s0 / s1 / s2 (cut-free) | +0.014 / +0.015 / +0.009 | 7, 7, 6 /12 | all include 0 |
| LoRA full s0 / s1 / s2 | +0.004 / +0.008 / −0.013 | 6, 6, 5 /12 | all include 0 |
| LoRA anti s0 / s1 / s2 | −0.011 / −0.031 / −0.001 | 6, 5, 6 /12 | all include 0 |

- Held-out val loss at step 800 (untrained 0.0982): full 0.0627 ×3, anti 0.0635/0.0636/0.0635,
  probe 0.0641 ×3. Same ordering as the first recipe; seed spread < 1e-4.
- Reading: probe is the only placement above zero in every seed, but the gain is small and comes
  from s_m(out, B) dropping (0.17 → 0.06–0.08), not s_m(out, A) rising (0.23 → 0.14–0.17).
- Pool shift: the test-run arms move slightly in the pooled scoring (selected x̃ +0.041 → +0.045;
  attn first third +0.105 → +0.108, CI [0.006, 0.202]; content x̄ α 0.5 CI now [−0.004, 0.137]).
  The test run's own numbers stay the numbers of record.
- README: D-34 section filled in; "4 body-plan families" corrected to 2 (the config has Q and H).

### 28. Report drafted in the course template (15:15– )

- Template fetched from the Drive folder linked in the instructions
  (`IE643_Project_Report.tex` + `demo.bib`); kept as `report/template_original.tex` and
  `report/template_reference.pdf`.
- `report/IE643_Project_Report.tex` → `.pdf` (19 pages): template preamble and header unchanged,
  sections in the template's order, before/after-prep-review split in §4, references written in the
  format the instructions give (thebibliography). `report/make_figures.py` makes the frame strips
  (test pair 1, both directions) and the LoRA validation curves; the other figures are copied from
  `code/results/figures/` and `deck_src/fig/unet.png`.
- TinyTeX lacked `footmisc` (needed by ragged2e); installed from the TeX Live 2025 frozen repository.
- Open: roll numbers in the header; the team must read and rewrite the prose in its own words
  before submission (the instructions give zero for AI-generated report content).
>>>>>>> origin/final-submission
