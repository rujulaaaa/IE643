# Report notes

Notes, numbers and sources for the team to write the report from. Bullet points, not report text: the
course scores AI-written report content as zero, so the prose in the report is ours. Check each arXiv
ID before citing it.

## Temporal identity drift (limitation)

**What we see.**
- `results/kaggle/vdprobe-xattn/out/variants_contact_sheet.png` shows "a golden retriever running on a beach",
  frames 0/5/10/15, for the four model cells.
  - SD1.5 + v1-5-2: fur, face and pose change from frame to frame; frame 10 grows a blob near the paw.
  - SD1.5 + v1-5-3: the head's shape and size change, and the framing jumps.
  - epiCRealism + v1-5-3: the dog is nearly identical across frames, but it barely moves (trade-off
    between consistency and motion).
- Quantitative, from the Phase 1 quality gate over 960 clips (`results/exp2/quality/quality.json`).
  Columns: VBench subject consistency (DINO, median); dynamic share (fraction of clips VBench's RAFT
  dynamic degree counts as moving); median top-5 % flow magnitude in px; CLIP-T (median).

  | cell | subject consistency | dynamic share | top-5 % flow (px) | CLIP-T |
  |---|---|---|---|---|
  | sd15_mm2 | 0.865 | 0.833 | 24.8 | 0.3317 |
  | sd15_mm3 | 0.858 | 0.929 | 16.8 | 0.3332 |
  | epicrealism_mm2 | 0.961 | 0.021 | 1.9 | 0.3442 |
  | epicrealism_mm3 | 0.970 | 0.004 | 1.1 | 0.3436 |

  - The trade-off in numbers: the cells that keep the dog's identity (epiCRealism) are the ones that
    hardly move. The SD1.5 cells move but drift.
  - High subject consistency on near-static video is a known property of this metric: VBench reports
    subject consistency next to dynamic degree for this reason.
  - The primary cell (sd15_mm2) is chosen per D-39.

**Why AnimateDiff drifts (mechanism).**
- AnimateDiff = a frozen SD1.5 image U-Net plus inserted motion modules (Guo et al., ICLR 2024,
  arXiv:2307.04725). The spatial layers (ResNets, spatial self-/cross-attention) process each frame
  independently; frames are coupled only through the motion modules.
- Motion-module attention is 1-D over frames at each spatial location (spatial positions are folded
  into the batch). A location attends only to the same location in other frames. A moving subject occupies
  different locations over time, so no layer attends to "the dog" across frames.
- The SD1.5 VAE decodes each frame separately, so there is extra high-frequency flicker (fur texture).
- The motion module was trained on WebVid-10M (Bain et al., ICCV 2021, arXiv:2104.00650): low
  resolution and watermarked stock video, with the image model frozen.

**How recent video models address it (context, not used in this project).**
- Full 3-D (spatio-temporal) attention in a Diffusion Transformer: every space-time patch attends
  to every other, so a subject can be matched across frames wherever it moves.
  - Sora technical report (OpenAI 2024, "Video generation models as world simulators"): "spacetime patches".
  - CogVideoX (Yang et al., arXiv:2408.06072): argues separate spatial + temporal attention makes it hard to
    keep large-motion objects consistent, and uses 3-D full attention.
  - HunyuanVideo (Kong et al., arXiv:2412.03603); Wan 2.1 (arXiv:2503.20314); LTX-Video (arXiv:2501.00103).
- 3-D causal VAE: compresses time and space jointly (e.g. 4× temporal, 8×8 spatial) and decodes the clip
  as a whole, which removes per-frame decode flicker (CogVideoX, HunyuanVideo, Wan-VAE).
- Video-native training at scale: joint image + video training on large re-captioned video datasets;
  no frozen image backbone.
- Whole-clip generation in one pass: Lumiere's space-time U-Net (Bar-Tal et al., arXiv:2401.12945),
  instead of keyframes plus temporal super-resolution.
- Why we still use AnimateDiff: the project asks for a video diffusion U-Net, and the DiT models are 2–14 B
  parameters and do not fit the free T4 budget.

**Fixes that stay within AnimateDiff (not applied in our runs unless stated).**
- FreeInit (Wu et al., ECCV 2024, arXiv:2312.07537): training-free; re-samples the noise keeping its
  low-frequency band. Available in diffusers as `pipe.enable_free_init()`; about 3× sampling time.
- Cross-frame / first-frame attention: spatial self-attention reads frame 0's keys and values
  (Tune-A-Video, arXiv:2212.11565; Text2Video-Zero, arXiv:2303.13439). Our Stage 6b K/V injection with
  `frames="first"` (`edit.KVSpec`, D-37) is this operator.
- Image-to-video conditioning on a fixed first frame: SparseCtrl (arXiv:2311.16933), ConsistI2V (arXiv:2402.04324).

**Consequence for our analysis.**
- We split activations into x̄ (frame mean, "content") and x̃ (residual over frames, "motion"). Identity
  drift varies over frames, so it loads on x̃: part of what x̃ measures is appearance flicker, not motion.
- Guards already in the design:
  - the primary cell is chosen by subject consistency (D-35 quality gate);
  - patching effects are compared against the unrelated-clip control (D-35);
  - flow decoding targets RAFT motion inside subject masks, with camera motion removed.
- An observation to check, not tested: in the per-cell ANOVA, the seed's share of x̃ variance is large
  (mean 0.26 on sd15_mm2, 0.21 on epicrealism_mm2; `results/exp2/<cell>/exp2_analysis.json`, `anova.seed`).
  That fits seed-driven appearance variation, but other explanations aren't ruled out.
- Possible follow-up (not run): correlate per-clip DINO subject consistency with the x̃ energy at the
  selected candidates; or re-run a small set with FreeInit and compare.
