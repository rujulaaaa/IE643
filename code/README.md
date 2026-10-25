# Probing the Latent Space of Video Diffusion U-Nets

**Disentangling content vs. motion in AnimateDiff** · IE 643 course project · Team Claudn't

A video diffusion U-Net alternates spatial layers, which work inside one frame and settle
*what the scene looks like*, with temporal layers, which work across frames and settle
*how it moves*. This project asks where that division actually sits: which layers,
which denoising steps, and which part of an activation. It then tests the answer with an
edit that needs it: keep a golden retriever's running gait while turning it into a horse,
and the reverse.

Model: AnimateDiff = Stable Diffusion 1.5 + `guoyww/animatediff-motion-adapter-v1-5-2`,
16 frames, 512×512, 25 DDIM steps, CFG 7.5, fp16 (frozen in `configs/generation.yaml`).

| Deliverable | What was built | Code |
|---|---|---|
| 1 · Setup and probing tools | Hooks on all 95 spatial/temporal coordinates; a lockstep sampler that denoises a whole group of clips step by step, bit-identical to `AnimateDiffPipeline` | `harness.py`, `lockstep.py`, `coords.py` |
| 2 · Content/motion analysis | 360 clips, 750 one-word-swap pairs, selectivity maps over 16 coordinates × 25 steps, three pre-registered hypotheses, linear probes | `probe_sweep.py`, `analysis.py` |
| 3 · Patching framework | Whole-tensor, channel, α-scaled and **component** patches (frame-varying part x̃ or frame mean x̄ only), temporal attention injection, null / mis-indexed / symmetry controls | `patching.py`, `edit.py` |
| 4 · Editing demonstration | Dev pairs choose the patch site; 12 held-out test pairs score it against a prompt-only baseline | `edit_sweep.py`, `score_edits.py` |
| Trained model | Motion LoRAs on HMDB51: full, probe-guided (placed by the Deliverable 2 map), wrong-place control | `lora.py`, `train_motion_lora.py` |
| Interface | Gradio app: live edits, upload your own video, browse every scored clip, the selectivity maps | `scripts/app.py`, `demo.py` |

Every decision that shaped an experiment is dated and justified in the register
[docs/protocol.md](docs/protocol.md) (D-01 … D-34). The hypotheses and their test statistics,
written before the sweep, are in [docs/hypotheses.md](docs/hypotheses.md). The pilot phase
(E1–E3, before the prep presentation) is in [docs/pilots.md](docs/pilots.md).

## Results

### Where content and motion live (Deliverable 2)

The selectivity S of a coordinate compares how far it moves under a verb swap with how far
it moves under a subject swap (1 − cosine, same noise tensor, one word changed).
S < 0 means motion-selective. S_dyn is measured on the frame-mean-removed residual x̃, S_stat on
the frame mean x̄. 78 matched units per seed, 1,000 bootstrap resamples.

| Hypothesis (pre-registered) | Verdict | What the data say |
|---|---|---|
| H1 · motion peaks in a mid-resolution decoder block (`up_blocks.1`) | falsified | The peak is `up_blocks.0` (8×8, right after the bottleneck), argmax in 1000/1000 resamples. Every other temporal block leans content. |
| H2 · motion is decided early in denoising | survives | Changepoint τ = 2, 95% CI [1, 3]. Motion selectivity lives in the first 2–3 of 25 steps. |
| H3 · removing the frame mean exposes hidden motion selectivity | falsified | f_stat = 0.65 (< 0.75), no coordinate flips sign. The split shifts every coordinate towards motion but never past zero. |

Robustness: cosine vs relative-L2 maps agree (Spearman 0.987); seed-to-seed agreement of the
S_dyn map is moderate (0.58–0.73). Verb probes trained on one scene and tested on the other
reach 0.74 on mid-block cross-attention (chance 0.25), but much of that is the verb token
leaking through the prompt; on the temporal modules the frame mean x̄ decodes the verb better
than x̃ at every step. Figures: `results/figures/sweep_*.png`.

### Patching controls on the real model (Deliverable 3)

| Control | Result |
|---|---|
| Null patch: whole tensor, x̃ only, x̄ only, attention injection | MAD 0, bit-identical |
| Mis-indexed null patch (wrong step) | MAD 0.103, PSNR 25.1 dB: caught, as it must be |
| Unpatched arm vs the same clip in another group | MAD 0 (no leakage between clips) |
| Symmetry f(A→B) vs f(B→A), `up_blocks.0`, first third | 0.171 vs 0.083: fails the 2× criterion set before the run |

### Edits on the 12 held-out test pairs (Deliverable 4)

Metrics: s_c = mean per-frame CLIP image cosine (content), s_m = cosine of Farneback optical-flow
fields (motion). D = ½[ŝc(out, B) + ŝm(out, A)] − ½[ŝc(out, A) + ŝm(out, B)] in the motion
direction (reversed for content), with ŝ min-max normalised over the run. ΔD = gain over the
prompt-only arm of the same pair; CI = 95% paired bootstrap of the median difference.

| Direction | Arm | Median ΔD | ΔD > 0 | 95% CI |
|---|---|---|---|---|
| keep A's motion | x̃ patch, `up_blocks.3` motion modules, steps 0–8 (selected on dev) | +0.041 | 8/12 | [−0.027, 0.143] |
| keep A's motion | whole-tensor patch, same place | −0.036 | 5/12 | [−0.109, 0.047] |
| keep A's motion | temporal attention injection, steps 0–8 (training-free baseline) | **+0.105** | 10/12 | **[0.001, 0.204]** |
| keep A's content | x̄ patch, `down_blocks.3` motion modules, steps 0–8 (selected on dev) | +0.068 | 10/12 | [−0.010, 0.158] |
| keep A's content | x̄ patch at α = 0.5 | +0.029 | 9/12 | [0.001, 0.152] |
| keep A's content | whole-tensor patch, same place | −0.026 | 4/12 | [−0.080, 0.106] |

The pre-registered success criterion (both selected patches with a CI above zero) is **not
met**. The component patches beat the prompt-only edit on the median in both directions, and
both beat the whole-tensor patch, which copies clip A wholesale (motion direction:
s_m(out, A) 0.82 but s_c(out, A) 0.91). In the motion direction the selected patch raises
s_m(out, A) from 0.23 to 0.67 at a small content cost (s_c(out, B) 0.85 → 0.81). Full table:
`results/kaggle/vdprobe-test/out/summary.csv`.

Where motion is *represented* and where patching *uses* it differ: the probing map peaks at
`up_blocks.0`, the best patch site is the outermost decoder block `up_blocks.3`.

### Motion LoRAs (trained model)

Three LoRAs with the same rank (32), data, and steps (800), differing only in placement:
`full` (all 21 motion modules, 9.67 M parameters), `probe` (`up_blocks.0` only, the block the
selectivity map picked, 1.97 M), `anti` (`down_blocks.2`+`3`, the two least motion-selective
blocks, 2.62 M, a wrong-place control).

| Variant | Held-out val loss, step 0 → 800 | Test pairs, prompt + LoRA, median ΔD |
|---|---|---|
| full | 0.0996 → 0.0638 | −0.021 |
| probe | 0.0996 → 0.0652 | +0.023 |
| anti | 0.0996 → 0.0646 | −0.020 |

All three ΔD CIs include zero. Across the 9 temporal blocks, where the full LoRA puts its
largest relative weight change follows the verb-probe accuracy (Spearman +0.83, p = 0.005),
not the patching usefulness (−0.27).

**Cut-free retrain, three seeds (D-34).** The same recipe with every training window inside
one shot (11 of the 19 clips contain a hard cut), seeds 0, 1, 2, on Kaggle
(`results/kaggle/vdprobe-lora/`). Edits scored in one pool with the test run
(`score_edits.py results/kaggle/vdprobe-lora/out --with results/kaggle/vdprobe-test/out`); the
regenerated source, B and prompt-only clips are bit-identical to the test run's on all 12 pairs.

| Variant | Held-out val loss at 800, seeds 0 / 1 / 2 | Median ΔD, seeds 0 / 1 / 2 |
|---|---|---|
| full | 0.0627 / 0.0627 / 0.0627 | +0.004 / +0.008 / −0.013 |
| probe | 0.0641 / 0.0641 / 0.0641 | +0.014 / +0.015 / +0.009 |
| anti | 0.0635 / 0.0636 / 0.0635 | −0.011 / −0.031 / −0.001 |

(Untrained: 0.0982 on the cut-free validation windows.) The held-out loss ranks full < anti < probe
in every seed; on the edit, probe is the only variant above zero in every seed, but every CI
includes zero, and its gain comes from moving less like B (s_m(out, B) 0.17 → 0.06–0.08) rather
than more like A (s_m(out, A) 0.23 → 0.14–0.17).

Same run: **attention injection over all 25 steps** on the test pairs: median ΔD **+0.120**,
11/12 pairs, 95% CI **[0.020, 0.203]**, s_m(out, A) 0.76 at s_c(out, B) 0.81. The pooled
normalisation shifts the other arms slightly (selected x̃ patch +0.045 [−0.019, 0.148]; attention
injection, first third +0.108 [0.006, 0.202]); the table above keeps the test run's own numbers.

### Upload path (interface)

A real video is DDIM-inverted and used as the motion source. Attention injection at all 25
steps from a blend of the inverted and the seed noise transfers the motion (flow cosine to
the upload 0.12 → 0.67 on a baseball pitch turned into a tiger). The component patch does not
survive this setting: the video's frame-varying activations do not line up with a new clip's
layout and the latent runs away. The patch is a tool for pairs that share their initial noise,
not a layout-free motion transplant. Attention injection carries *where* things move in the
frame, not *which object* moves.

## Setup

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -e ".[dev]"
```

Tested with Python 3.10, torch 2.10.0+cu128, diffusers 0.40.0, transformers 5.20.0,
peft 0.21.2, gradio 6.30.0 (Kaggle: torch 2.11.0, same diffusers and peft). Weights (~4.6 GB:
SD1.5 fp16, the motion adapter, CLIP ViT-B/32) download on first use into the Hugging Face
cache; the resolved repository and commit hash are written into every result file (D-20).

A CUDA GPU is needed for generation. An 8 GB RTX 4060 Laptop runs everything here
(65 s per clip); a Kaggle T4 needs 10.2 GB and ~90 s per clip.

If a ROS environment is sourced, its `PYTHONPATH` leaks packages into the venv: prefix
commands with `env -u PYTHONPATH`.

## Interface

```bash
python scripts/app.py                 # http://127.0.0.1:7860, loads the models on the first edit
python scripts/app.py --preload       # load at start (~20 s)
python scripts/app.py --no-gpu        # results browser and maps only
python scripts/app.py --share         # public Gradio link
```

- **Live edit.** Pick clip A (subject, verb), clip B, a scene and a seed, a direction (keep A's
  motion or keep A's content) and a method: the selected component patch, a custom patch
  (any block, window, α), the whole-tensor patch, attention injection (first third or all
  steps), or prompt + one of the LoRAs. Out come A, B, the prompt-only edit and the patched edit,
  with s_c, s_m and D (normalised with the test run's pool). Subjects and verbs can be typed
  freely. 4 clips ≈ 4.5 min on the laptop.
- **Upload a video.** Any mp4/avi/mov; the longest shot is cut to 16 frames, inverted, and its
  motion moved onto the subject in the target prompt. Reports flow cosine to the upload and
  the CLIP match to the target prompt.
- **Results browser.** Every clip of the dev and test sweeps with its scores, and the run
  summary. Needs `results/kaggle/` (see *Results on disk*).
- **Where motion lives.** S_dyn and S_stat over the 25 steps for each of the 16 coordinates,
  the H1–H3 verdicts and the sweep figures.

Each live edit is saved under `results/interface/<time>_<pair>/` (GIFs, `scores.csv`,
`request.json`).

## Reproduce

Commands run from the repository root. Long jobs ran on Kaggle (2 × T4) through
`scripts/kaggle_run.py`, which ships `src/`, `configs/` and `scripts/` in a private kernel,
runs one job per GPU and downloads the outputs to `results/kaggle/<slug>/`.

```bash
python -m pytest                                   # 51 tests, CPU, tiny random UNet, ~10 s

# Stage 1 · the lockstep sampler equals the diffusers pipeline (laptop, ~12 min)
python -u scripts/lockstep_check.py

# Stage 2-4 · probing sweep: 6 groups of 60 clips (Kaggle, ~5.2 h on 2 × T4)
python scripts/kaggle_run.py --slug vdprobe-sweep \
    --job "scripts/probe_sweep.py --scenes 'on a beach'" --job "scripts/probe_sweep.py --scenes 'in a forest'"
python scripts/analyze_sweep.py results/kaggle/vdprobe-sweep/out --out results/analysis    # CPU, ~15 min
python scripts/sweep_figures.py

# Stage 5 · patching controls (laptop), then the coarse dev sweep (Kaggle, ~2.6 h) and the selection
python -u scripts/patch_controls.py
python scripts/kaggle_run.py --slug vdprobe-dev \
    --job "scripts/edit_sweep.py --split dev --arms coarse --shard 0/2" \
    --job "scripts/edit_sweep.py --split dev --arms coarse --shard 1/2"
python scripts/score_edits.py results/kaggle/vdprobe-dev/out --select      # -> configs/edit_selection.json

# Trained model · HMDB51 clips, then the three LoRAs (laptop, ~20 min each)
python scripts/fetch_lora_data.py
python -u scripts/train_motion_lora.py --variant full      # also: probe, anti; --cut-free --seed N (D-34)

# Stage 6 · test pairs with the selected arms and the LoRAs (Kaggle, ~2.8 h), then scoring
#   (dataset vdprobe-loras = results/local/motion_lora/<v>/motion_lora.safetensors saved as <v>.safetensors)
python scripts/kaggle_run.py --slug vdprobe-test --dataset vdprobe-loras \
    --job "scripts/edit_sweep.py --split test --arms selected --sel configs/edit_selection.json \
           --lora input/vdprobe-loras/full.safetensors --lora input/vdprobe-loras/probe.safetensors \
           --lora input/vdprobe-loras/anti.safetensors --shard 0/2" --job "... --shard 1/2"
python scripts/score_edits.py results/kaggle/vdprobe-test/out
python scripts/layer_rankings.py --analysis results/analysis/analysis.json \
    --dev results/kaggle/vdprobe-dev/out --lora results/local/motion_lora --out results/rankings
```

Every script writes a JSON record with the software versions, model revisions, timings and
peak memory next to its outputs.

## Data

- **Generated probing set** (no download): 15 subjects in 2 body-plan families × 4 verbs
  (running, walking, jumping, spinning) × 2 scenes × 3 seeds, prompt
  "a {subject} {verb} {scene}, high quality, detailed" (`configs/probing_set.yaml`). Pairs change
  one word with the noise tensor fixed; content pairs stay inside a family (D-05).
- **Edit pairs**: 12 test + 4 dev pairs, A and B differing in subject and verb, frozen in
  `configs/edit_pairs.yaml` before any edit ran.
- **LoRA training clips**: 19 HMDB51 clips (run 5, walk 5, jump 4, turn 5) from the Hugging Face
  mirror `divm/hmdb51` (revision pinned), selected by a deterministic rule on HMDB's own tags
  (full body, no camera motion, one person, one clip per film); 15 train, 4 validation.
  `scripts/fetch_lora_data.py` downloads them and checks sha256 against
  `data/hmdb51_lora/manifest.json` (2.0 MB, included). 11 of the 19 contain a hard cut, which
  the `--cut-free` recipe avoids.

## Results on disk

| Path | Contents | In git |
|---|---|---|
| `results/analysis/`, `results/figures/`, `results/rankings/` | Stage 4 maps, verdicts, figures; layer rankings | yes |
| `results/local/` | pilots, lockstep check, patch controls, LoRA weights and logs, interface checks | yes |
| `results/kaggle/<run>/` | sweep outputs: per-pair records, scores, summaries, logs | records and scores only |
| `results/kaggle/<run>/**/gifs/`, `*.pt` | every generated clip (GIF) and latent / activation tensors, ~1.8 GB | no |

## Layout

```
configs/      generation protocol, probing vocabulary, edit pairs, Stage 5 selection
data/         HMDB51 LoRA clips + manifest
docs/         decisions register, hypotheses, harness and patching design, pilot phase
src/vdprobe/
  coords.py        the 95 coordinates, blocks, layouts, frame-mean split per layout
  harness.py       hooks, step counter, summary recorder
  lockstep.py      group-wise denoising loop (bit-identical to the pipeline), step caches, pair scorer
  summaries.py     x̄ / x̃ split, D-01 distances, selectivity S
  analysis.py      selectivity maps, bootstrap, H1-H3 tests, linear probes
  patching.py      TensorCache, Patcher (whole / channel / α / component), controls
  edit.py          source + N edited arms in one lockstep group, attention injection
  edit_plan.py     edit pairs and the arm sets of Stages 5-6
  metrics.py       CLIP content fidelity, optical-flow motion fidelity, D, consistency, flicker
  lora.py          motion LoRA placement (full / probe / anti), save and load
  inversion.py     DDIM inversion of an uploaded video, shot detection
  demo.py          the interface engine
  pipeline.py, probing_set.py, config.py, runio.py, store.py, attention.py
scripts/      one script per stage (see Reproduce), app.py, kaggle_run.py, run_seq.py
tests/        CPU tests on a tiny random UNet: hooks, patches, controls, analysis, metrics, demo
notebooks/kaggle_pilots/   the original E1-E3 notebooks
```

## Known limitations

- The pre-registered edit criterion (D-16) is not met; only attention injection has a CI that
  excludes zero (just, over the first third; clearly, over all 25 steps).
- D is min-max normalised within a run, so D values are comparable inside a run, not across runs.
- Motion fidelity is optical-flow cosine and content fidelity is CLIP, not CoTracker / DINO
  (no new model downloads in the final week; noted in the log).
- The symmetry control fails (A→B transfers twice as much as B→A on the one pair tested).
- The LoRAs are trained on 15 clips; they learn the generic action, not clip A's motion, so they
  are compared with each other rather than with the patches.
