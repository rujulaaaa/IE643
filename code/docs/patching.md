# The causal probing framework: activation patching via paired runs

Selectivity (H1–H3) says where content and motion information *is*. Patching says
where the model *uses* it. The two can disagree, and that disagreement is the most
interesting thing the project could find.

## Paired runs

Every causal measurement is a pair of generations that share the seed, the fp16
noise tensor drawn on the CPU, the sampler, the 25 DDIM steps and every prompt token
but one:

```
source run   clip A  "a golden retriever running on a beach"   ──► TensorCache
                                                                      │ A's activations at
                                                                      │ (coordinates × steps)
target run   clip B  "a golden retriever walking on a beach"   ◄──────┘
                     a probe reads the cache at the same (L, t) while B is generated
```

The shared noise tensor is not optional. E2 showed that with it, changing one word
leaves the composition largely in place; without it, the difference between two clips
is mostly sampling noise and the comparison measures nothing.

The same machinery serves two purposes:

| target-run probe | what it does | output |
|---|---|---|
| `Patcher` | replaces B's activation at (L, t) with A's | a patched clip |
| `PairDistance` | scores B's activation against A's with the D-01 distances | scalars only (D-26) |

So the paired run is both the intervention for Deliverable 3 and the measurement
behind the selectivity maps of Deliverable 2, and computing the maps costs no
generations beyond the clips themselves.

## Where a patch lands: coordinates × the 25 denoising steps

A patch is addressed by the same triple as every stored summary: (module path, step
index, clip id). Module paths come from `coords.enumerate_coordinates` (95 of them,
see [harness.md](harness.md)); step indices run 0–24 and are counted by a pre-hook on
the UNet, which is called exactly once per step (the two CFG branches are batched).

The time axis is a contiguous **window** (D-13), swept over:

| window | steps | `patching.window(...)` |
|---|---|---|
| one step | `t` | `window("one", center=t)` |
| five steps around a peak | `t−2 … t+2` | `window("five", center=t)` |
| first third | `0 … 8` | `window("first_third")` |
| whole trajectory | `0 … 24` | `window("all")` |

Full tensors are cached only at the D-07 steps `{0, 4, 8, 12, 16, 20, 24}` unless a
peak calls for refinement: caching all 16 coarse coordinates at all 25 steps would be
~15 GiB of host memory per source clip.

## Granularity

Following roadmap §5.3, coarse to fine:

| # | granularity | `PatchSpec` | status |
|---|---|---|---|
| 1 | whole tensor | `alpha=1` | implemented, tested |
| 2 | channel subset (top-k% by the per-channel score `S_c`, D-10) | `channels={coord: mask}` | implemented, tested |
| 3 | temporal attention weights only (after MotionClone) | needs an attention-processor swap | Stage 5 |
| 4 | scaled injection, `out ← out + α·(cached − out)`, α ∈ {0, .25, .5, .75, 1} | `alpha=α` | implemented, tested |

Granularities 2 and 4 compose (`torch.lerp` with weight `α · mask`, exact at both ends).
`branches="cond"` restricts the patch to the text-conditional CFG branch.

The corruption is **resampling** from a real second clip, not Gaussian noising (D-12),
because Zhang & Nanda (ICLR 2024) show the two answer different questions.

## Controls, in the order they must pass

1. **Null patch (D-11).** Patch a clip with its own cached activations. The output
   must come back unchanged: final-latent MAD < 1e-3 and decoded PSNR > 45 dB
   (`patching.null_patch_verdict`). fp16 kernels are not bit-reproducible, so the
   run-to-run floor of two unhooked generations is reported beside it.
2. **Mis-indexed null patch (our addition).** The same patch read one D-07 step late
   must **fail** D-11. A check that cannot fail cannot catch an indexing bug.
3. **Random coordinate** (Stage 5). Patch a coordinate with low |S|. If it transfers as
   much as the chosen one, nothing has been localised.
4. **Symmetry** (Stage 5). A into B and B into A should be near inverses; asymmetry means
   something is leaking.

**Status.** The framework is designed and its core is implemented and unit-tested:
`tests/test_patching.py` runs controls 1 and 2, plus α = 0 / α = 1,
conditional-branch-only and channel-subset patches, on a small randomly initialised
`UNetMotionModel` on the CPU in fp32, where "unchanged" can be checked bit for bit.
Running the controls and the patching sweeps on the real model is Stage 5
(19–25 October); this phase stops at E3.

## Cost

One paired run is two generations: ~136 s on the RTX 4060 Laptop GPU, ~188 s on a
Kaggle T4. The source run can be shared by every patch that uses the same source clip,
so a sweep over k (coordinate, window, α) settings for one pair costs k + 1
generations, not 2k. This is what makes the tiered plan in roadmap §7 (189 coarse +
360 refined runs) affordable.

## API

```python
from vdprobe.patching import TensorCache, Patcher, PairDistance, PatchSpec, run_with, window

cache = TensorCache(coords=["up_blocks.1.motion_modules.2"], steps=window("first_third"))
run_with(loaded, all_coords, [cache], prompt_a, noise)                       # source run

spec = PatchSpec(coords=("up_blocks.1.motion_modules.2",), steps=window("first_third"), alpha=0.5)
patched = run_with(loaded, all_coords, [Patcher(cache, spec)], prompt_b, noise)  # target run
```
