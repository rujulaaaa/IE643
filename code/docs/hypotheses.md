# Three falsifiable hypotheses

Sweeping every layer and timestep blind produces a description. Going in with
predictions produces a result. Each hypothesis below comes from the background
reading (roadmap group 5), names the evidence that would refute it, and fixes its
test statistic **before** the sweep runs (D-22, D-23). The analysis code may not
pick a different statistic after seeing the data; changes go into
[protocol.md](protocol.md) with a date and a reason.

They cover the three places content and motion could be localised:

| | Axis | Question |
|---|---|---|
| **H1** | layer | *Which* modules carry motion? |
| **H2** | timestep | *When* in the 25-step denoising trajectory is motion settled? |
| **H3** | representation | *Which part* of an activation carries it: the static component or the frame-varying residual? |

## The measurement all three share

For a coordinate (module `L`, denoising step `t`) and one clip, take the activation
in canonical form `x ∈ R^{F×C×S}` (16 frames, channels, spatial positions; text-conditional
CFG branch). Split it:

```
x̄ = mean_F(x)        static component, broadcast over frames
x̃ = x − x̄            frame-mean-removed residual
```

For a pair of clips (A, B) that share a seed and noise tensor and differ in one
prompt word, the distance is `d = 1 − cos` on the flattened tensor (D-01): on `x̄`
for the static channel, on `x̃` for the dynamic channel. With `d_c` measured across
a content pair (subject swapped) and `d_m` across a motion pair (verb swapped):

```
S_dyn(L, t)  = (d_c[x̃] − d_m[x̃]) / (d_c[x̃] + d_m[x̃])      headline map
S_stat(L, t) = (d_c[x̄] − d_m[x̄]) / (d_c[x̄] + d_m[x̄])      H3 control
```

`S > 0`: the coordinate tracks what is depicted. `S < 0`: it tracks how things move.
Ratios are computed per pair, then the median is taken over pairs with a 95%
bootstrap CI (1,000 resamples) within each seed, then averaged across the three seeds.

The distances need both clips' full activations at once, which the 44.6 MB summary
store deliberately does not keep. They are therefore computed **online during paired
runs**: the anchor clip's activations are cached in a source run and the partner clip
is scored against that cache as it is generated
(`vdprobe.patching.PairDistance`, see [patching.md](patching.md)).

---

## H1 · Layer axis: motion peaks at medium resolution

**Claim.** Motion selectivity peaks in the temporal modules at medium resolution,
around `up_blocks[1]` (16×16 latent), not at the highest-resolution decoder layers
and not at the bottleneck.

**Why we expect it.** Xiao et al. (NeurIPS 2024) build a motion feature from U-Net
activations and find the cleanest signal just after the first up-block; shallower
and deeper blocks both do worse.

**Test statistic (D-22).** For each block `b`, `M(b) = −mean_t median_pairs S_dyn(L_b, t)`
over its representative temporal coordinate (larger = more motion-selective).
Bootstrap over pairs (1,000 resamples), recording `argmax_b M(b)` each time. The
95% set is the smallest set of blocks holding 95% of the bootstrap argmax mass.

**Survives if** the 95% set excludes every resolution-level-0 block
(`down_blocks.0`, `up_blocks.3`, 64×64) **and** every level-3 block
(`down_blocks.3`, `mid_block`, `up_blocks.0`, 8×8).

**Falsified if** the peak sits at an extreme, or the profile is flat (the 95% set
spans both extremes).

## H2 · Timestep axis: motion early, appearance late

**Claim.** Motion is settled in roughly the first third of the 25-step trajectory and
appearance in the last; content selectivity runs the opposite way.

**Why we expect it.** Widely assumed, rarely tested. Baherwani et al. (arXiv 2025)
measured it and report two regimes with a reasonably stable boundary.

**Test statistic (D-22).** On the motion-selectivity profile `m(t) = −median_pairs S_dyn(L*, t)`
of the winning block `L*` from H1, fit a single changepoint `τ ∈ {1..24}` by least
squares on a two-segment step function. Bootstrap over pairs for a 95% CI on `τ`.

**Survives if** the CI on `τ` lies in the first half of the trajectory (`τ ≤ 12`) and
`mean m(t<τ) > mean m(t≥τ)`.

**Falsified if** the changepoint falls in the second half or the early segment is not
the more motion-selective one. A boundary somewhere other than Baherwani et al.'s is
a finding about AnimateDiff, not a bug.

## H3 · Representation axis: the static-content confound

**Claim.** Raw activations, *even in temporal layers*, are dominated by the static
component. A clean motion signal appears only in the frame-mean-removed residual `x̃`.

**Why we expect it.** Xiao et al. only obtain a usable motion signal after subtracting
the per-frame mean across the frame axis.

**What it predicts.**
1. `S_stat > 0` (content-selective) at most coordinates, temporal ones included.
2. At some temporal coordinates, `S_dyn` flips sign relative to `S_stat`: content
   selective on `x̄`, motion selective on `x̃`.
3. The static share of activation energy, `F·‖x̄‖² / ‖x‖²`, is large in every block.

**Test statistic (D-23, added 2026-10-16; the roadmap fixed tests for H1 and H2 only).**
Over the coarse coordinate set and the D-07 timesteps:
`f_stat` = fraction of (L, t) with the 95% CI of `S_stat` entirely above 0;
`n_flip` = number of temporal coordinates (L, t) with the CI of `S_stat` above 0 **and**
the CI of `S_dyn` below 0.

**Survives if** `f_stat ≥ 0.75` and `n_flip ≥ 1`.

**Falsified if** raw/static activations already separate motion (`f_stat < 0.75`), or the
residual never recovers a motion-selective coordinate (`n_flip = 0`). The second outcome
would mean motion is not linearly visible in single-coordinate activations at all, which
is itself worth reporting.

**Why it matters most.** Skip the split and every layer reads as content-selective; we
would wrongly conclude motion is nowhere. That is the mistake we would have made in
week one of the sweep.

### Early evidence for H3 (pixel level, E2)

The fixed-noise pilot E2 already shows the confound in *pixel space*. Its motion proxy
(cosine distance between frame-to-frame difference maps) should move under verb swaps
and stay put under noun swaps. It did the opposite:

| swap (vs. "a golden retriever running on a beach") | d_c appearance (CLIP) | d_m motion proxy |
|---|---|---|
| noun: cat / horse / bird (motion held fixed) | **0.260** | **0.461** |
| verb: walking / jumping / standing (subject held fixed) | **0.099** | **0.390** |

Content control works (noun swaps move CLIP appearance 2.6× more than verb swaps), but
the motion proxy is *larger* for noun swaps, whose motion did not change, and
`standing` (no locomotion at all) scores 0.450, indistinguishable from the noun swaps.
Changing the animal relocates the frame differences, so a pixel motion measure mostly
measures what is depicted. Numbers: `results/kaggle_t4/E2_result.json` (record) and `results/local/E2/` (local rerun).

---

## How the hypotheses get tested causally

H1–H3 are stated over correlational selectivity. The patching framework
([patching.md](patching.md)) turns a high-|S| coordinate into a causal claim: cache the
activation at (L, t-window) in clip A's run, splice it into clip B's run from the same
noise tensor, and ask whether B now moves like A (motion transferred) while still
looking like B. A coordinate that scores high on `S_dyn` but transfers nothing under
patching is the "present but not used" disagreement the project is looking for.
