# Interpretability of (Video) Diffusion U-Nets: Cross-Attention, Layer Roles, Motion Features, AnimateDiff Variants, Optical Flow

Scope: supports a course project that probes content vs. motion in AnimateDiff (SD1.5 + `guoyww/animatediff-motion-adapter-v1-5-2`, diffusers, 16 frames, 512x512, 25 DDIM steps, CFG 7.5). The project found motion selectivity only in the `up_blocks.0` motion module (8x8) during the first 2-3 denoising steps. Researched 2026-10-24. Every citation below was checked against arXiv, CVF, ACL Anthology, ICLR, ECVA, Hugging Face, GitHub or torchvision pages unless marked otherwise.

Resolution convention, which several items below depend on: for a 512x512 SD1.5 input the latent is 64x64. In diffusers, `down_blocks.0-3` run at 64/32/16/8, `mid_block` at 8, and `up_blocks.0-3` at 8/16/32/64; each of `up_blocks.0-2` ends with an upsampler. This is standard SD1.5 structure and was not re-fetched here; confirm it by printing `pipe.unet`.

---

## Q1. Cross-attention visualisation: DAAM, Prompt-to-Prompt, and video extensions

### Takeaway
DAAM upsamples every cross-attention probability map bicubically to image size, sums it over heads, all down and up layers and all timesteps, and thresholds the sum at τ·max (τ = 0.3 to 0.5). The paper's ablation shows that every timestep and every layer adds to segmentation quality. Prompt-to-Prompt established that cross-attention maps set the spatial layout of each token. FateZero and Video-P2P carry this to video: cross-attention maps are kept per frame and thresholded into blending masks. That gives the project a ready-made subject mask for object-only flow.

### Cited Findings
- **Citation verified:** Tang, Liu, Pandey, Jiang, Yang, Kumar, Stenetorp, Lin, Ture, "What the DAAM: Interpreting Stable Diffusion Using Cross Attention", *Proc. ACL 2023 (Vol. 1: Long Papers)*, pp. 5644-5659; arXiv:2210.04885. Affiliations: Comcast Applied AI, UCL, U. Waterloo. — [ACL Anthology PDF](https://aclanthology.org/2023.acl-long.310.pdf); [arXiv](https://arxiv.org/abs/2210.04885)
- **DAAM attention definition (Eq. 5):** F_t^{(i)↓}(ĥ, X) = softmax((W_q ĥ)(W_k X)^T / √d), with shape ⌈w/c^i⌉ × ⌈h/c^i⌉ × l_H × l_W (l_H heads, l_W tokens). The same form applies to up-block maps F_t^{(i)↑}. Scores are "already normalized in [0, 1]". — [ACL PDF](https://aclanthology.org/2023.acl-long.310.pdf)
- **DAAM aggregation (Eq. 6):** "we upscale all intermediate attention score arrays to the original image size using bicubic interpolation, then sum them over the heads, layers, and time steps": D_k^R[x,y] := Σ_{i,j,ℓ} ( F̃^{(i)↓}_{t_j,k,ℓ}[x,y] + F̃^{(i)↑}_{t_j,k,ℓ}[x,y] ). Here k is the word, i the layer (down and up), j the timestep and ℓ the head. — [ACL PDF](https://aclanthology.org/2023.acl-long.310.pdf)
- **DAAM normalisation and thresholding (Eq. 7):** no extra normalisation is applied, because "summing normalized values preserves linear scale". The binary map is D_k^{Iτ}[x,y] := 𝟙( D_k^R[x,y] ≥ τ · max_{i,j} D_k^R[i,j] ). τ ∈ {0.3, 0.4, 0.5} were evaluated (DAAM-0.3/0.4/0.5). — [ACL PDF](https://aclanthology.org/2023.acl-long.310.pdf)
- **DAAM ablation (App. A.1):** restricting to the first or last n timesteps, or to a subset of layer resolutions, lowers segmentation quality. The authors conclude that "all time steps and layers contribute positively to segmentation quality". Setup: SD 2.0 base, 512x512, 30 steps, CFG 7.5, DPM solver. — [ACL PDF](https://aclanthology.org/2023.acl-long.310.pdf)
- **DAAM head-dependent study:** some heads' (words') maps "inappropriately subsume their dependents'" (feature entanglement). — [ACL PDF](https://aclanthology.org/2023.acl-long.310.pdf)
- **Citation verified:** Hertz, Mokady, Tenenbaum, Aberman, Pritch, Cohen-Or, "Prompt-to-Prompt Image Editing with Cross-Attention Control", ICLR 2023 (oral, notable top 25%); arXiv:2208.01626. — [ICLR 2023 poster page](https://iclr.cc/virtual/2023/poster/11646); [arXiv](https://arxiv.org/abs/2208.01626)
- **What the P2P maps control:** "cross-attention layers are the key to controlling the relation between the spatial layout of the image and each word in the prompt". Edits inject the source image's attention maps, which supports word swap (localised edit), adding a phrase (global edit) and attention re-weighting. — [arXiv abstract](https://arxiv.org/abs/2208.01626)
- **Video extension, FateZero:** Qi et al., "FateZero: Fusing Attentions for Zero-shot Text-based Video Editing", ICCV 2023, pp. 15932-15942; arXiv:2303.09535. Attention maps are stored during DDIM inversion because they "retain both structural and motion information". Self-attention is blended with "a binary mask from thresholding the cross-attention" of the edited words. Self-attention is reformed into spatial-temporal attention. — [CVF ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html); [arXiv](https://arxiv.org/abs/2303.09535)
- **Video extension, Video-P2P:** Liu et al., "Video-P2P: Video Editing with Cross-attention Control", arXiv:2303.04761. It is listed as CVPR 2024 by MLAnthology and Adobe Research. It adapts P2P cross-attention control (word swap, prompt refinement, re-weighting) to video using decoupled guidance with separate unconditional embeddings for the source and target branches. — [arXiv](https://arxiv.org/abs/2303.04761); [MLAnthology CVPR 2024](https://mlanthology.org/cvpr/2024/liu2024cvpr-videop2p)
- **Attention analysis in video diffusion:** Liu, Wang, Su, Ten, Huang, Guo, Jia, "Understanding Attention Mechanism in Video Diffusion Models", arXiv:2504.12027 (Apr 2025; no peer-reviewed venue listed). It perturbs spatial and temporal attention, finds attention-map entropy is linked to quality ("high-entropy attention maps are often key elements linked to superior video quality"), and links low-entropy maps to intra-frame structure. — [arXiv](https://arxiv.org/abs/2504.12027)
- **Head-level motion work on video DiTs (not U-Nets):** "Controlling Motion Transfer in Diffusion Transformers via Attention Heads" (arXiv:2607.11081) reports that temporal heads show cross-frame diagonal patterns and align with optical flow. "IMAP / I'm a Map! Interpretable Motion-Attentive Maps" (arXiv:2603.02919) localises motion concepts with motion-specific heads in video DiTs. Both were seen only in search snippets and not opened. — [arXiv 2607.11081](https://arxiv.org/html/2607.11081v1); [arXiv 2603.02919](https://arxiv.org/html/2603.02919v1)

### Inferences
- **Per-frame DAAM for AnimateDiff.** In diffusers' AnimateDiff, the spatial transformer blocks see frames folded into the batch axis, so each cross-attention call yields attention probabilities of shape (2·B·F·heads, HW, 77) under CFG. The recipe:
  1. Keep only the conditional half.
  2. Reshape to (F, heads, H, W, 77).
  3. Pick token k, bicubically upsample to 512x512, and sum over heads, layers and steps, as in Eq. 6.

  The result is a 16-frame heat-map video D_k^R[f, x, y]. Hook it with a custom `AttnProcessor` that stores `attention_probs` for `attn2` modules only (`attn1` is spatial self-attention, and the motion modules' attention is temporal).
- **A temporal-attention analogue.** Motion-module attention has shape (B·H·W·heads, F, F): one F×F frame-to-frame matrix per spatial location. Summary maps per location:
  - attention entropy;
  - off-diagonal mass;
  - the top-1 index, which is what MotionClone uses (Q3).

  These can be upsampled and aggregated in the DAAM style to produce a "where does temporal attention mix frames" map.
- **Layer-resolved DAAM.** Because DAAM sums over everything, keep per-layer and per-step maps rather than only the sum. That lets the project test whether cross-attention localisation of the subject at a given layer or step correlates with the motion selectivity seen in `up_blocks.0`.
- **CFG matters for the maps.** With CFG 7.5, the unconditional-branch maps attend to an empty prompt and should be discarded.

### Gaps
- No published "video DAAM" paper was found that defines the aggregation over frames. The per-frame extension above is my inference.
- No OpenReview URL was confirmed for P2P; the ICLR virtual page was used instead.
- The Video-P2P venue is CVPR 2024 per MLAnthology and Adobe, but the CVF page was not opened directly.

---

## Q2. Layer roles in the SD U-Net: P+, Plug-and-Play features, DIFT

### Takeaway
The evidence is consistent:
- **Coarse layers (8x8, 16x16)** carry structure, shape and semantics.
- **Fine layers (32x32, 64x64)** carry colour, appearance and high-frequency detail.
- **The first two decoder blocks** (`up_blocks.0` and `up_blocks.1`) carry the most semantic and layout information; PnP injects at `up_blocks[1].resnets[1]` and DIFT uses `up_ft_index=1` for semantics.

This fits the project's finding that `up_blocks.0` is the most "structural" decoder block. The literature puts the semantic and layout sweet spot one block later, at 16x16.

### Cited Findings
- **Citation verified:** Voynov, Chu, Cohen-Or, Aberman, "P+: Extended Textual Conditioning in Text-to-Image Generation", arXiv:2303.09522 (v3 15 Jul 2023; Google Research and Tel Aviv University). No peer-reviewed venue is listed on arXiv. — [arXiv](https://arxiv.org/abs/2303.09522)
- **P+ layer count:** P+ assigns a separate token embedding to each of 16 cross-attention layers in SD. Resolution 8 has 1 layer. Resolutions 16, 32 and 64 each have 2 layers in the down path and 3 in the up path. — [arXiv HTML v3](https://arxiv.org/html/2303.09522v3)
- **P+ layer roles:** "the coarse layers primarily affect the structure of the image" and "the fine layers determine the color appearance of the image". In attention analysis, coarse layers (8, 16) attend more to the object token and fine layers (32, 64) to the appearance token. Colour "requires that we condition the full set of layers". For style, "every layer has some contribution". Caveat: "the disentanglement among the layers of U-net is not perfect". — [arXiv HTML v3](https://arxiv.org/html/2303.09522v3)
- **Citation verified:** Tumanyan, Geyer, Bagon, Dekel, "Plug-and-Play Diffusion Features for Text-Driven Image-to-Image Translation", CVPR 2023, pp. 1921-1930; arXiv:2211.12572. — [CVF CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Tumanyan_Plug-and-Play_Diffusion_Features_for_Text-Driven_Image-to-Image_Translation_CVPR_2023_paper.html)
- **PnP findings on decoder features:**
  - The shallowest decoder layer is "mostly dominated by foreground-background separation".
  - Decoder layer 4 encodes "localized semantic information" shared across objects.
  - Later layers "capture more high-frequency information".
  - Spatial features are injected from decoder layer 4 for the first τ_f = 40 of 50 steps (0.8). Self-attention is injected for τ_A = 25 of 50 steps (0.5).
  - Uses SD v1-4.

  — [arXiv HTML](https://arxiv.org/html/2211.12572)
- **PnP code, exact diffusers modules:** `conv_module = model.unet.up_blocks[1].resnets[1]` (feature injection). The self-attention injection map is `res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}`, so decoder attention layers in `up_blocks.1-3` are used and `up_blocks.0` is skipped. — [pnp-diffusers/pnp_utils.py](https://github.com/MichalGeyer/pnp-diffusers/blob/main/pnp_utils.py)
- **Citation verified:** Tang, Jia, Wang, Phoo, Hariharan, "Emergent Correspondence from Image Diffusion" (DIFT), NeurIPS 2023; arXiv:2306.03881. — [arXiv](https://arxiv.org/abs/2306.03881)
- **DIFT defaults:** "`t=261` by default for semantic correspondence", "`up_ft_index=1` by default", `ensemble_size=8`. For HPatches (geometric/homography), DIFT_sd uses `t=0, up_ft_index=2`. For DAVIS video label propagation it uses `t=51, up_ft_index=2`. In short, semantics come from a coarser up block at moderate noise, and geometry from a finer block at low noise. — [DIFT GitHub README](https://github.com/Tsingularity/dift)

### Inferences
- **Mapping to diffusers names (my inference from the indexing):**
  - PnP's "decoder layer 4" is `up_blocks.1.resnets.1`, which runs at 16x16 for 512 input.
  - DIFT `up_ft_index=1` is the output of `up_blocks.1`.
  - DIFT `up_ft_index=2` is the output of `up_blocks.2`.
- **The project's `up_blocks.0` result in context.** `up_blocks.0` at 8x8 is the coarsest decoder block. P+ classes it as "structure", and PnP says its predecessor-level features are dominated by foreground/background separation. Motion selectivity there is plausibly coarse layout or position change, not fine motion.
- **Refined layer sweep.** Include `up_blocks.1` (all 3 resnets, 3 attentions and 3 motion modules) explicitly, at sub-module granularity. PnP, DIFT, MOFT and MotionClone all point to the 16x16 decoder level (see Q3).
- **Timestep reading.** P+ and DAAM imply that cross-attention layout is fixed early. That fits motion and layout being decided in the first denoising steps (see Baherwani et al. in Q3).

### Gaps
- The P+ venue could not be verified; arXiv lists none.
- PnP's paper text says self-attention injection goes into "decoder layers" (WebFetch summary said "all"), while the code injects into `up_blocks.1-3` only. The code was taken as authoritative.

---

## Q3. Video diffusion motion features: MOFT, MotionClone, DMT/SMM, Baherwani et al., probing studies

### Takeaway
All three training-free motion-feature methods locate motion at the 16x16 decoder level (`up_blocks.1`) in early or mid noise:
- **MOFT:** "upper block 1", frame-mean removal, then PCA-selected top 4% of channels.
- **MotionClone:** temporal attention in `up_block.1`, a top-1 sparse mask, extracted at t = 400.
- **DMT:** the 2nd decoder block of ZeroScope, SMM over the first 20 of 50 steps.

Baherwani et al. (arXiv Dec 2025) give quantitative support for an early motion-dominant regime. The boundary sits at τ ≈ 700 of 1000 for ModelScope and Latte, roughly the first 30% of the trajectory.

### Cited Findings
- **Citation verified:** Xiao, Zhou, Yang, Pan, "Video Diffusion Models are Training-free Motion Interpreter and Controller", NeurIPS 2024; arXiv:2405.14864. — [arXiv](https://arxiv.org/abs/2405.14864); [NeurIPS proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/8b2fc235787852ead92da2268cd9e90c-Paper-Conference.pdf)
- **MOFT content removal (Eq. 1):** X^norm = X − (1/F) Σ_{i=1}^{F} X_i, i.e. subtract the mean over frames. **Motion feature (Eq. 2):** ℳ = X[j] − (1/F) Σ_i X_{i,[j]} for motion channels j ∈ 𝒞. — [arXiv HTML v3](https://arxiv.org/html/2405.14864v3)
- **MOFT channel selection:**
  - PCA on features from videos with known panning motion; channels are ranked by their projection weights on the first principal component P₁ ∈ ℝ^{D×1}, since "only a few channels significantly contribute".
  - The default is the top 4% of channels.
  - Ablation (faithfulness / naturalness): 100%: 67.5/0.671; 10%: 82.7/0.692; 5%: 83.2/0.694; 4%: 84.0/0.693; 1%: 76.3/0.677.

  — [arXiv HTML v3](https://arxiv.org/html/2405.14864v3)
- **MOFT block and timestep:**
  - "we select the features after upper block 1"; medium-resolution features align best. Per-layer evidence is qualitative only (Fig. 4e-h).
  - The reference MOFT is taken at the start of the inversion stage.
  - Guidance runs on 25-step DDIM, 512x512, 16 frames, with step thresholds t₁ = 19, t₂ = 18, t₃ = 5 (of 25).
  - Base models: AnimateDiff (version not stated) and SparseCtrl; it also generalises to ModelScope, ZeroScope, SVD and Open-Sora.

  — [arXiv HTML v3](https://arxiv.org/html/2405.14864v3)
- **Citation verified:** Ling, Bu, Zhang, Dong, Zang, Wu, Chen, Wang, Jin, "MotionClone: Training-Free Motion Cloning for Controllable Video Generation", ICLR 2025; arXiv:2406.05338. — [ICLR 2025 proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2406.05338)
- **MotionClone representation:**
  - Mask (Eq. 5): M^{tα} = 1 where the reference temporal-attention value is among the top-k along the temporal axis; k = 1 in experiments.
  - Representation: L^{tα} = M^{tα} ⊙ A_ref^{tα}.
  - Guidance (Eq. 6): g = ‖L^{tα} − M^{tα} ⊙ A_gen^t‖²₂.
  - It is extracted in a single denoising step at t_α = 400 (no inversion).
  - Ablations: t_α = 800 "loses substantial motion information"; 200, 400 and 600 all work.

  — [arXiv HTML v6](https://arxiv.org/html/2406.05338v6)
- **MotionClone layers and steps:**
  - Guidance is applied to the temporal-attention layers in "up_block.1". The ablation (Fig. 9) finds that up_block.1 gives the best motion control while keeping visual quality.
  - Camera motion: 100 steps, guidance on 50. Object motion: 300 steps, guidance on the first 180.
  - s = 7.5, λ = 2000.
  - Base model: AnimateDiff (T2V) and SparseCtrl (I2V and sketch).

  — [arXiv HTML v6](https://arxiv.org/html/2406.05338v6)
- **Citation verified:** Yatim, Fridman, Bar-Tal, Kasten, Dekel, "Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer", CVPR 2024, pp. 8466-8476; arXiv:2311.17009. — [CVF CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Yatim_Space-Time_Diffusion_Features_for_Zero-Shot_Text-Driven_Motion_Transfer_CVPR_2024_paper.html)
- **DMT SMM feature:** SMM[f(x_t)] = (1/(M·N)) Σ_i Σ_j f(x_t)_{i,j}. The spatial mean of the F×M×N×D space-time feature gives one D-vector per frame.
- **DMT loss:** Δ_{(i,j)} = φ_i − φ_j and L = Σ_i Σ_j ‖Δ_{(i,j)} − Δ̃_{(i,j)}‖²₂ (pairwise frame differences of SMM, which removes per-video appearance offsets).
- **DMT settings:**
  - Features come from the 2nd decoder block of ZeroScope, taken right after the upsampling layer ("semantic DIFT features").
  - Guidance runs during the first 20 of 50 steps.
  - Optimiser: Adam, lr 0.01, 30 steps per denoising step.
  - 24 frames at 576x320.

  — [arXiv HTML](https://arxiv.org/html/2311.17009)
- **Baherwani paper, exact identity verified:** Vatsal Baherwani, Yixuan Ren, Abhinav Shrivastava (University of Maryland), "Characterizing Motion Encoding in Video Diffusion Timesteps", arXiv:2512.22175, submitted 18 Dec 2025. It is an arXiv preprint (10 pages, 4 figures), and no venue appears on the arXiv page. One aggregator claimed an IJCNN 2025 listing, which I could not verify and which is implausible for a Dec 2025 preprint. — [arXiv](https://arxiv.org/abs/2512.22175)
- **Baherwani method and results:**
  - Method: inject a new (appearance-editing) condition over chosen timestep ranges and measure the trade-off between appearance editing and motion preservation.
  - Motion preservation is the cosine similarity of normalised Lucas-Kanade flow against the original. Appearance editing is the change in CLIP score. Data: the 76 TGVE videos.
  - Models: ModelScope (U-Net), Latte (DiT with decoupled spatio-temporal attention), CogVideoX (unified attention).
  - Result: "an early, motion-dominant regime and a later, appearance-dominant regime", with boundary τ = 700 (of 1000) for ModelScope and Latte, and τ = 950 for CogVideoX.
  - Latte ablation: query and key parameters did not encode motion; tuning value and output projections sufficed.

  — [arXiv HTML](https://arxiv.org/html/2512.22175)
- **Related follow-up (same group):** "Timestep-Constrained One-Shot Video Motion Customization", CVPR 2026 Workshops (P13N). Seen in search results only, not opened. — [CVF CVPRW 2026 PDF](https://openaccess.thecvf.com/content/CVPR2026W/P13N/papers/Baherwani_Timestep-Constrained_One-Shot_Video_Motion_Customization_CVPRW_2026_paper.pdf)

### Inferences
- **Reconciling with the project's result.**
  - The project's frame-mean removal is MOFT Eq. 1. MOFT also needs channel selection (top ~4% by PCA on known-motion videos). Without it, the motion signal is diluted by the roughly 96% content-dominated channels, which the ablation shows: 100% channels scores 67.5 vs 84.0 at 4%. This likely explains why mean removal alone "did not reveal motion selectivity".
  - Fix: build a small set of clips with known motion direction, either from controlled prompts ("camera pans left/right") or from optical flow labels. Run PCA on the frame-mean-removed features per hooked module, then probe only the top-k% channels.
- **Timestep window.** "First 2-3 of 25 steps" with DDIM linspace corresponds to roughly t ≈ 960-880. That is earlier than MotionClone's t_α = 400 and Baherwani's boundary near t = 700 (≈ first 30% ≈ first 7-8 of 25 steps).
  - Probe the whole of steps 1-10, not only 1-3.
  - Evaluate on the conditional branch separately; CFG mixing may blur things.
- **Add temporal-attention maps as a probe target.** Following MotionClone, record the F×F attention probabilities in `up_blocks.1` motion modules (and in all others for the refined sweep), not only the module outputs. Top-1 sparsified attention is a published motion representation that is free of appearance by construction.
- **Add the SMM-difference feature as a third motion descriptor.** Take the spatial mean per frame, then pairwise frame differences. It is cheap and removes static appearance. It also gives a natural "motion vs content" test: SMM-diff should be content-invariant if motion is held fixed.

### Gaps
- MOFT does not name the exact module (resnet vs motion module) or the resolution of "upper block 1". Whether it means diffusers `up_blocks.1` (16x16 at 512 input) is my inference.
- MotionClone does not state the resolution of "up_block.1". In the original AnimateDiff and diffusers indexing it is 16x16 at 512 input (inference).
- I found no paper that probes individual temporal attention heads inside AnimateDiff motion modules. Head-level studies found target video DiTs (CogVideoX, Wan).

---

## Q4. Model variants usable with the same diffusers code

### Takeaway
- **Lowest-risk upgrades:** keep `guoyww/animatediff-motion-adapter-v1-5-2` and swap the base to `SG161222/Realistic_Vision_V5.1_noVAE` (the diffusers docs default) or `emilianJR/epiCRealism`. Then try `guoyww/animatediff-motion-adapter-v1-5-3`.
- **v1-5-3 caveat:** v1-5-3 has **no mid-block motion module** (`use_motion_mid_block: false`), unlike v1-5-2 (`true`). Probing code that hooks `mid_block.motion_modules` must handle its absence.
- **Domain adapter:** the v3 Domain-Adapter LoRA has no official diffusers-format route.
- **Other options:** AnimateLCM (`wangfuyun/AnimateLCM`) is a distilled alternative. FreeInit (`pipe.enable_free_init`) reduces flicker at about 2-5× cost.

### Cited Findings
- **Citation verified:** Guo, Yang, Rao, Liang, Wang, Qiao, Agrawala, Lin, Dai, "AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning", ICLR 2024 (Spotlight); arXiv:2307.04725. — [GitHub README](https://github.com/guoyww/AnimateDiff); [arXiv](https://arxiv.org/abs/2307.04725)
- **AnimateDiff architecture:**
  - The motion module is a "temporal Transformer" with "several self-attention blocks along the temporal axis, with sinusoidal position encoding", zero-initialised output projection and a residual connection (identity at init).
  - The domain adapter is LoRA in the base model's self- and cross-attention, scaled by α (1 in training, adjustable at inference, 0 removes it).
  - MotionLoRA is LoRA in the motion module's self-attention.
  - Built on SD v1.5; trained on WebVid-10M.

  — [arXiv HTML](https://arxiv.org/html/2307.04725)
- **Versions per the README:**
  - v1 (2023.07): `mm_sd_v14.ckpt` and `mm_sd_v15.ckpt`, 417M parameters.
  - v2 (2023.09): `mm_sd_v15_v2.ckpt`, 453M parameters, "trained upon larger resolution and batch size", with 8 camera MotionLoRAs.
  - v3 (2023.12): `v3_sd15_mm.ckpt` plus Domain Adapter `v3_adapter_sd_v15.ckpt` (97.4 MB) plus SparseCtrl RGB and scribble encoders. "Image model finetuning through Domain Adapter LoRA for more flexibility at inference"; lowering the adapter scale removes training-video artefacts such as watermarks; SparseCtrl needs a full Domain Adapter.
  - SDXL beta: `mm_sdxl_v10_beta.ckpt`.
  - Example community models: ToonYou, Realistic Vision V2.0, FilmVelvia, MajicMix, RcnzCartoon, Lyriel, Tusun.

  — [GitHub README](https://github.com/guoyww/AnimateDiff); [search summary of README](https://github.com/guoyww/AnimateDiff/blob/main/README.md)
- **v1-5-2 config:** `use_motion_mid_block: true`, `motion_layers_per_block: 2`, `motion_mid_block_layers_per_block: 1`, `motion_num_attention_heads: 8`, `motion_max_seq_length: 32`, `block_out_channels: [320, 640, 1280, 1280]`. — [HF config v1-5-2](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/raw/main/config.json)
- **v1-5-3 config:** `use_motion_mid_block: false`; otherwise the same `motion_layers_per_block: 2`, 8 heads, `max_seq_length 32` and block channels. The model card shows `MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3")` with `SG161222/Realistic_Vision_V5.1_noVAE`, `EulerAncestralDiscreteScheduler(beta_schedule="linear")`, 16 frames, CFG 7.5, 25 steps. — [HF config v1-5-3](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3/raw/main/config.json); [HF model card](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-3)
- **diffusers motion-module internals:** each motion module is an `AnimateDiffTransformer3D` with `positional_embeddings="sinusoidal"`. Up blocks are `UpBlockMotion` / `CrossAttnUpBlockMotion`, and the mid block is `UNetMidBlockCrossAttnMotion`; each creates one motion module per layer. — [diffusers unet_motion_model.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_motion_model.py)
- **Repo ids in the diffusers docs:**
  - Adapters: `guoyww/animatediff-motion-adapter-v1-5-2`, `guoyww/animatediff-motion-adapter-v1-5-3`, `guoyww/animatediff-motion-adapter-sdxl-beta`, `wangfuyun/AnimateLCM`.
  - Bases: `SG161222/Realistic_Vision_V5.1_noVAE`, `emilianJR/epiCRealism`, `frankjoshua/toonyou_beta6`.
  - VAE: `stabilityai/sd-vae-ft-mse`.
  - SparseCtrl: `guoyww/animatediff-sparsectrl-scribble` / `-rgb`, plus `guoyww/animatediff-motion-lora-v1-5-3`.
  - Motion LoRAs: `guoyww/animatediff-motion-lora-zoom-out`, `-tilt-up`, etc. These "work with" v1-5-2.

  — [diffusers AnimateDiff docs](https://huggingface.co/docs/diffusers/api/pipelines/animatediff)
- **Recommended scheduler in the docs:** `DDIMScheduler.from_pretrained(model_id, subfolder="scheduler", clip_sample=False, timestep_spacing="linspace", beta_schedule="linear", steps_offset=1)`. Checkpoints are "sensitive to the beta schedule", so `linear` is recommended. Motion adapters target SD1.4/1.5 fine-tunes. Original `.ckpt` files load via `MotionAdapter.from_single_file` (diffusers ≥ 0.30). — [diffusers AnimateDiff docs](https://huggingface.co/docs/diffusers/api/pipelines/animatediff)
- **Base-model repos verified to exist:**
  - `SG161222/Realistic_Vision_V6.0_B1_noVAE`: diffusers format, beta; the card recommends the `sd-vae-ft-mse` VAE and the DPM++ SDE Karras sampler. — [HF](https://huggingface.co/SG161222/Realistic_Vision_V6.0_B1_noVAE)
  - `emilianJR/epiCRealism`: diffusers format, from Civitai model 25694. — [HF](https://huggingface.co/emilianJR/epiCRealism)
- **v3 Domain Adapter in diffusers:** the docs never mention the v3 domain-adapter LoRA, and I found no official diffusers loading example. A community copy of `v3_sd15_mm.ckpt` in diffusers format exists (`vladmandic/animatediff-v3`, third party). — [diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/animatediff); [vladmandic/animatediff-v3](https://huggingface.co/vladmandic/animatediff-v3/commit/f78580ac6834048dcb418daefd445a51b8bfcda5)
- **Citation verified:** Wang, Huang, Bian, Shi, Sun, Song, Liu, Li, "AnimateLCM: Computation-Efficient Personalized Style Video Generation without Personalized Video Data", SIGGRAPH Asia 2024 Technical Communications (short paper); arXiv:2402.00769. It uses decoupled learning (style vs acceleration, image vs motion acceleration). — [arXiv](https://arxiv.org/abs/2402.00769)
- **AnimateLCM diffusers usage:** `MotionAdapter.from_pretrained("wangfuyun/AnimateLCM")`, `LCMScheduler(beta_schedule="linear")`, `pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="sd15_lora_beta.safetensors")`, CFG 1.5, 6 steps, with `emilianJR/epiCRealism`. — [diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/animatediff)
- **Citation verified:** Wu, Si, Jiang, Huang, Liu, "FreeInit: Bridging Initialization Gap in Video Diffusion Models", ECCV 2024 (DOI 10.1007/978-3-031-72646-0_22); arXiv:2312.07537. — [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/423_ECCV_2024_paper.php)
- **FreeInit method:**
  1. Sample and run DDIM to get z₀.
  2. Diffuse z₀ to T with the same ε.
  3. Keep the low spatio-temporal frequencies of the diffused latent and take the high frequencies from fresh noise, using a 3D Gaussian LPF with D₀ = 0.25 (Butterworth also offered).
  4. Resample, and repeat.

  — [arXiv HTML v2](https://arxiv.org/html/2312.07537v2)
- **FreeInit cost:**
  - Each iteration is a full sampling pass. The paper's quantitative results use 4 extra iterations; in practice 2-3, since gains saturate and the first iteration gives the most.
  - One extra iteration (2n steps) beats vanilla 5n steps. A coarse-to-fine schedule T_i = ⌊(T/N)(i+1)⌋ cuts cost.
  - DINO consistency on AnimateDiff: 85.24 → 92.01 (UCF-101) and 83.24 → 91.86 (MSR-VTT).
  - diffusers: `pipe.enable_free_init(method="butterworth", use_fast_sampling=True)`.

  — [arXiv HTML v2](https://arxiv.org/html/2312.07537v2); [diffusers docs](https://huggingface.co/docs/diffusers/api/pipelines/animatediff)

### Inferences
- **The likely cause of morphing.** The current setup probably uses the plain SD1.5 base (`runwayml`/`stable-diffusion-v1-5`). The AnimateDiff docs and README always pair motion modules with fine-tuned community bases. Also check the scheduler: the docs require `beta_schedule="linear"`, `clip_sample=False`, `timestep_spacing="linspace"`, `steps_offset=1`. A mismatched schedule is a known cause of flicker and morphing.
- **Suggested experimental grid (same code):**
  1. Base ∈ {SD1.5, Realistic_Vision_V5.1_noVAE, epiCRealism} × adapter v1-5-2.
  2. Best base × {v1-5-2, v1-5-3}.
  3. Optionally AnimateLCM as a distilled model with very different timestep dynamics.
  4. Optionally FreeInit as a consistency control.
- **Probing code changes for v1-5-3:**
  - Build hook lists dynamically from `unet.named_modules()`; v1-5-3 has no `mid_block.motion_modules`.
  - Report per-block counts, not absolute layer indices, so the two adapters stay comparable.
- **Probing caveat for FreeInit and AnimateLCM.** FreeInit changes the initial noise distribution, so it confounds per-step probing comparisons. Probe the final FreeInit iteration only, and say so. AnimateLCM's 4-6 steps map to very different timesteps, so compare by t (out of 1000), not by step index.

### Gaps
- **Motion modules per up block in diffusers.** I could not verify the count (`motion_layers_per_block` vs +1) because the diffusers source fetch was truncated. Confirm empirically: `len(pipe.unet.up_blocks[i].motion_modules)`.
- **Domain-adapter filename discrepancy.** The README lists `v3_adapter_sd_v15.ckpt`, but I could not open the HF `guoyww/animatediff` file list to check the exact name there. Whether `pipe.load_lora_weights(...)` accepts that `.ckpt` is unverified. Using the v3 motion module without the domain adapter is the documented diffusers path (the v1-5-3 model card example has no adapter).
- The AnimateDiff training resolution, frame count and stride are in the supplementary material, which was not retrieved.
- The diffusers version string "0.40" in the brief was not checked against PyPI. The docs fetched are from `main`, current as of Oct 2026.
- Realistic Vision's SD1.5 base is inferred from its HF collection title, not stated in the card. The epiCRealism card does not state its base either; both are known in practice as SD1.5 fine-tunes, which the diffusers docs imply by using them with SD1.5 motion adapters.

---

## Q5. Optical flow for probing: RAFT vs Farneback, camera vs object motion, masks, VBench metrics

### Takeaway
Use torchvision `raft_large(weights=Raft_Large_Weights.DEFAULT)` (C_T_SKHT_V2; 5.26M parameters; inputs in [-1, 1], H and W divisible by 8, minimum 128 px; last of 12 iterative predictions) on 512x512 frames. Derive per-clip motion labels:
- global (camera) flow, from the median flow or a homography fit;
- residual (object) flow inside the subject mask from per-frame DAAM.

Report VBench-style Dynamic Degree, Motion Smoothness and Temporal Flickering so the model variants can be compared on standard terms.

### Cited Findings
- **RAFT citation:** Teed & Deng, "RAFT: Recurrent All-Pairs Field Transforms for Optical Flow", ECCV 2020 (Best Paper Award per secondary sources; DOI 10.1007/978-3-030-58536-5_24); arXiv:2003.12039. — [arXiv](https://arxiv.org/abs/2003.12039v2); [MLAnthology](https://mlanthology.org/eccv/2020/teed2020eccv-raft/); [Synced on the award](https://syncedreview.com/2020/08/24/eccv-2020-best-paper-award-goes-to-princeton-team/)
- **torchvision builders and weights:**
  - Builders: `torchvision.models.optical_flow.raft_large` and `raft_small`.
  - `Raft_Large_Weights`: `C_T_V1`, `C_T_V2`, `C_T_SKHT_V1`, `C_T_SKHT_V2`, `C_T_SKHT_K_V1`, `C_T_SKHT_K_V2`. `DEFAULT` = `C_T_SKHT_V2`.
  - 5,257,536 parameters, 211.01 GFLOPS, 20.1 MB.
  - Sintel-test EPE (C_T_SKHT_V2): 1.819 clean / 3.067 final.
  - Transforms rescale images to [-1, 1]; minimum input 128 px.

  — [torchvision raft_large](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.optical_flow.raft_large.html); [torchvision RAFT index](https://docs.pytorch.org/vision/main/models/raft.html)
- **torchvision usage:**
  - Resize so "dimensions are divisible by 8" (512 already is), with `antialias=False`.
  - The output is a list of flows, one per iteration, each (N, 2, H, W) in pixel units. 12 iterations; use the last ("most accurate").
  - `raft_small` is "smaller and faster ... sacrificing a bit of accuracy".
  - `torchvision.utils.flow_to_image` renders flow as colour.

  — [torchvision optical flow tutorial](https://docs.pytorch.org/vision/main/auto_examples/others/plot_optical_flow.html)
- **Citation verified:** Huang, He, Yu, Zhang, Si, Jiang, Zhang, Wu, Jin, Chanpaisit, Wang, Chen, Wang, Lin, Qiao, Liu, "VBench: Comprehensive Benchmark Suite for Video Generative Models", CVPR 2024, pp. 21807-21818; arXiv:2311.17982. Follow-ups: VBench++ (arXiv:2411.13503) and VBench-2.0. — [CVF CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Huang_VBench_Comprehensive_Benchmark_Suite_for_Video_Generative_Models_CVPR_2024_paper.html)
- **VBench Dynamic Degree:**
  - RAFT flow between consecutive frames; per pair, take the mean of the top 5% flow magnitudes (`cut_index = int(h*w*0.05)`).
  - A pair counts as moving if this exceeds `thres = 6.0*(scale/256.0)`, where scale is the shorter side. A video is dynamic if at least `count_num = round(4*(count/16.0))` pairs move.
  - Frames are subsampled to about 8 fps (`interval = max(1, round(fps/8))`). RAFT large, 20 iterations.
  - The model score is the fraction of dynamic videos.

  — [VBench dynamic_degree.py](https://github.com/Vchitect/VBench/blob/master/vbench/dynamic_degree.py); [VBench arXiv HTML](https://arxiv.org/html/2311.17982)
- **VBench Motion Smoothness:** drop odd frames, reconstruct them with the AMT frame-interpolation model, compute MAE against the originals, and normalise as (255 − MAE)/255. — [VBench arXiv HTML](https://arxiv.org/html/2311.17982)
- **VBench Temporal Flickering:** MAE between consecutive frames on static videos only (flow used to filter), S = (255 − MAE)/255. — [VBench arXiv HTML](https://arxiv.org/html/2311.17982)
- **VBench Subject Consistency:** (1/(T−1)) Σ_t ½(⟨d₁,d_t⟩ + ⟨d_{t−1},d_t⟩) with DINO features. Background Consistency uses the same formula with CLIP features. These directly quantify the "morph" problem. — [VBench arXiv HTML](https://arxiv.org/html/2311.17982)
- **A flow metric in a probing study:** Baherwani et al. measure motion preservation as the cosine similarity of normalised Lucas-Kanade flow vectors against a reference. That is a lightweight alternative to RAFT for comparing motion between two clips. — [arXiv HTML 2512.22175](https://arxiv.org/html/2512.22175)
- **Cross-attention as a subject mask:** FateZero thresholds cross-attention into a binary mask to localise edits. DAAM Eq. 7 gives the threshold rule (τ·max, τ ≈ 0.3-0.5). — [CVF ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/QI_FateZero_Fusing_Attentions_for_Zero-shot_Text-based_Video_Editing_ICCV_2023_paper.html); [DAAM ACL PDF](https://aclanthology.org/2023.acl-long.310.pdf)

### Inferences
These are proposed methods and engineering judgement, not sourced facts.

- **RAFT vs Farneback.**
  - RAFT is far more accurate on smooth, low-texture generated content, and its cost is small next to video diffusion sampling. Use `raft_large`, 15 frame pairs per clip, fp32 on GPU.
  - OpenCV `cv2.calcOpticalFlowFarneback` is CPU-only and less accurate on large motions. It is acceptable only as a sanity check.
  - I did not fetch a source comparing the two; treat this as an untested judgement.
- **Camera vs object motion separation (per frame pair):**
  - **Option (a), median flow:** global = median(u), median(v) over background pixels; object flow = flow − global.
  - **Option (b), homography:** sample a grid of points p, set p' = p + flow(p), and fit `cv2.findHomography(p, p', cv2.RANSAC, 3.0)` on background points (outside the subject mask). Warp-predicted flow = H(p) − p. Residual = flow − predicted. A homography captures pan, zoom and rotation for distant or planar scenes; an affine fit (`cv2.estimateAffinePartial2D`) is more stable for small motions.
  - **Summary statistics per clip:**
    - camera magnitude ‖global‖;
    - object magnitude, the mean ‖residual‖ inside the mask;
    - dominant direction, the angle of the mean residual;
    - a "morph index", the residual magnitude in the background. Flicker and morphing show up as large, incoherent residual flow off the subject.
- **Subject mask from cross-attention.**
  1. For the subject token k, compute per-frame DAAM D_k^R[f] (Q1 inference), aggregated over all layers and steps.
  2. Threshold at τ = 0.4 (DAAM-0.4 was a strong setting in the paper).
  3. Downsample to the flow resolution.

  Using maps from late steps or fine layers sharpens boundaries. Check them against SAM or a simple saliency mask on a few clips.
- **Integrating flow into probing (motion labels).**
  1. Derive per-clip, and optionally per-region, motion targets from RAFT: camera direction class, object speed, and the object displacement field pooled to 8x8 and 16x16 so it matches `up_blocks.0` and `up_blocks.1` resolution.
  2. Regress the frame-mean-removed activations at each hooked module and step onto these targets (ridge or linear probe).
  3. Compare against a content target (prompt identity) to get a content-vs-motion selectivity index per layer and timestep.

  Pooling flow to the feature resolution lets the probe be spatial (per-location motion vectors) rather than only clip-level. MOFT shows the motion signal is spatially aligned with actual motion.
- **Using VBench metrics for model comparison.** Report Dynamic Degree, Motion Smoothness, Subject Consistency and Temporal Flickering for each base/adapter variant. Higher subject consistency with a non-trivial dynamic degree separates a variant that genuinely moves from one that merely flickers.

### Gaps
- No primary source was fetched for Farneback accuracy or speed, or for a RAFT-small vs RAFT-large timing benchmark on GPU. Timing should be measured locally.
- The VBench paper text does not give the Dynamic Degree threshold; the numbers above come from the released code (`dynamic_degree.py`), which may change between versions.
- No published paper was found that combines cross-attention-derived subject masks with optical flow to measure object-only motion in AnimateDiff. The combination above is proposed, not established.
