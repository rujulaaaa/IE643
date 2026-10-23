# The activation-extraction harness

`vdprobe.harness.Harness` hooks every spatial and temporal module of AnimateDiff's
U-Net, counts the 25 denoising steps, and hands each activation to a list of
probes that either record it or intervene on it.

## The 95 coordinates

`coords.enumerate_coordinates(unet)` walks `UNetMotionModel` (SD1.5 backbone +
`mm_sd_v15_v2` motion adapter) and classifies modules by path:

| kind | what it is | count | output layout |
|---|---|---:|---|
| `temporal_module` | a whole motion module (`AnimateDiffTransformer3D`), output = input + temporal update | 21 | `(B·F, C, H, W)` |
| `temporal_attn1` | first self-attention across frames inside a motion module | 21 | `(B·H·W, F, C)` |
| `temporal_attn2` | second self-attention across frames | 21 | `(B·H·W, F, C)` |
| `spatial_attn1` | SD1.5 self-attention within a frame | 16 | `(B·F, H·W, C)` |
| `spatial_attn2` | SD1.5 cross-attention to the text | 16 | `(B·F, H·W, C)` |
| | | **95** | |

Motion modules sit in every block (8 down, 1 mid, 12 up); spatial transformers only
in the blocks inherited from SD1.5's cross-attention blocks (6 down, 1 mid, 9 up).
Every coordinate also carries its block and resolution level (latent side ÷ 2^level:
64, 32, 16, 8 at 512×512).

`B = 2` because classifier-free guidance batches the unconditional and
text-conditional branches (in that order). `coords.canonical()` maps all three
layouts to one `(B, F, C, S)` form; the tests check that a given (frame, position)
lands in the same cell whichever layout it came from.

## Indexing

A measurement is addressed by **(module path, step index, clip id)**. A forward
pre-hook on the UNet increments the step counter; the UNet is called exactly once
per step (E3 confirmed 25 calls for 25 steps), and the run fails loudly if not.
The scheduler's actual timestep values are recorded alongside, in the manifest.

## What is stored, and why it is 2,017× smaller

A full fp16 dump of every coordinate at every step, as the tensors flow through
the network (both CFG branches), is computed exactly from the architecture:

| | per coordinate per step | × 25 steps |
|---|---|---|
| 16 spatial transformers, 2 attention outputs each | `2 · 16 · S · C` fp16 values | 34.42 GiB |
| 21 motion modules, module output + 2 attention outputs | `2 · 16 · S · C` fp16 values | 53.47 GiB |
| **full dump** | | **87.89 GiB per clip** |

That is larger than a whole Kaggle working disk (73 GB) for a *single* clip, and
about 31,600 GiB for the 360-clip probing set. So the harness stores summaries
(`SummaryRecorder`, D-09 / D-24), computed on the text-conditional branch:

| per (coordinate, step) | floats | for all 95 × 25 |
|---|---|---|
| channel mean, channel variance over frames and space | `2C` | |
| residual norm `‖x̃_f‖` per frame (how far each frame departs from the static component) | `F = 16` | |
| channel statistics, fp32 | `2C + F` | **15.71 MiB** |
| one head-averaged attention map per attention coordinate, `(F, 16, 16)`, fp32 | 4,096 | **28.91 MiB** |
| **summary store** | | **44.62 MiB per clip** |

87.89 GiB / 44.62 MiB = **2,017×**. Over 360 clips the store is 15.7 GiB, inside
Kaggle's 20 GB output cap; the attention grid is the dominant term and is a single
flag (`--grid 8` halves the store). `tests/test_protocol.py` recomputes all of these
numbers from the layer shapes and checks them against the Kaggle E3 record.

**What the ratio compares, stated precisely.** The 87.89 GiB baseline is the dump of
what actually flows through the network, both CFG branches; the summaries cover the
text-conditional branch, which is the one prompt pairs differ in. Against a dump of
the conditional branch alone (43.95 GiB) the reduction is 1,008×.

The attention maps, one `(F, 16, 16)` movie per attention coordinate and step, keep
the frame axis (DAAM extended to video):

| kind | map |
|---|---|
| `spatial_attn2` | attention mass on the subject-phrase tokens (`cross_phrases` adds verb/scene) |
| `spatial_attn1` | attention *received* by each position, averaged over queries |
| `temporal_attn1/2` | cross-frame mass `1 − P[f, f]`: how much frame f reads from the other frames |

SDPA never materialises attention probabilities, so `vdprobe.attention` recomputes
`q` and `k` from the module's own inputs with SDPA's scaling. The tests check that
`probs @ v` through `to_out` reproduces each attention module's real output.

Full tensors are kept only on request (`patching.TensorCache`), for flagged
coordinates at the D-07 steps.

## On-disk format

One directory per clip, `store/<clip id>/`:

- `summary.safetensors` with keys `<module path>::mean|var|resnorm|attn`, each with a
  leading step axis of 25;
- `manifest.json` with the prompt, seed, token positions, the 25 scheduler timesteps,
  the generation config, the resolved model repositories and commit hashes (D-20),
  and the coordinate table.

```python
from vdprobe.store import load_clip, query
tensors, manifest = load_clip("results/local/E3/store/s42__golden-retriever__running__on-a-beach")
query(tensors, "up_blocks.1.motion_modules.2", "resnorm", step=4)   # (16,) one value per frame
```

## Measured

| | Kaggle T4 (record) | RTX 4060 Laptop (local) |
|---|---|---|
| coordinates hooked | 95 | 95 |
| UNet calls per clip | 25 | 25 |
| full dump | 87.89 GiB | 87.89 GiB |
| summary store | 44.62 MiB (formula) | 44.62 MiB formula, 44.66 MiB on disk |
| reduction | 2,017× | 2,015× on disk |
| overhead, metadata hooks only | +0.6 % | −0.9 % (noise) |
| overhead, full `SummaryRecorder` | not measured | +72 % (110 s vs 64 s per clip) |

The recorder's overhead is not free. The likely cause is recomputing attention
probabilities for 74 maps per step; it has not been profiled. Levers if it matters:
subsample queries for the 64×64 `spatial_attn1` maps, or record statistics only
(`attention_maps=False`) and attention maps on a subset of clips.
Sources: `results/kaggle_t4/E3_result.json`, `results/local/E3/E3_result.json`.
