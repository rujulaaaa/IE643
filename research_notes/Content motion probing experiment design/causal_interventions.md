# Causal-intervention methods for localising what a layer computes, and how they transfer to AnimateDiff U-Nets

Scope: activation patching best practice, causal mediation analysis, causal tracing in LLMs and text-to-image (T2I) diffusion, attribution patching (AtP, EAP, AtP*), concept erasure (INLP, amnesic probing, LEACE), interchange interventions and DAS, and rules for combining probes with interventions. Each section maps the method onto the study setup. That setup is AnimateDiff on SD1.5: 16 frames, 25 DDIM steps, CFG 7.5, 95 hooked modules (21 motion modules, 42 temporal attention, 32 spatial attention), paired clips A/B that share initial noise and differ by one prompt word, splice modes (whole / channel subset / alpha mix / frame-mean "static" / frame-mean-removed "dynamic"), CLIP for content and optical-flow cosine for motion, about 90 s per clip on a Kaggle T4.

Notation used throughout: h^{m,t}_X is the output of module m at DDIM step t in run X. "Run B with do(h^{m,t} ← h^{m,t}_A)" means generating B's clip with that activation overwritten by A's.

Verification status: every paper's metadata below was checked against arXiv, ACL Anthology, or the NeurIPS/ICLR/ICML proceedings pages (URLs inline). Equations were copied from the papers' own text (PDFs were converted to text and read directly), except where marked "(inference)". Unverified items are listed under Gaps.

---

## 1. Activation-patching best practice (Zhang & Nanda, ICLR 2024; Heimersheim & Nanda, 2024): denoising vs noising, corruption, metrics, sufficiency vs necessity, sliding windows

### Takeaway
Patching clean→corrupt ("denoising") tests **sufficiency**, and patching corrupt→clean ("noising") tests **necessity**. Neither result implies the other, and AND-like or OR-like (redundant) circuits make the two directions disagree in predictable ways. The papers recommend in-distribution counterfactual corruptions (symmetric token replacement, which the A/B one-word swap already is) over Gaussian noise. They recommend contrastive, continuous, roughly linear metrics (logit difference) over probabilities, and single-layer patching before sliding windows, because windows inflate localisation.

### Cited Findings
**Metadata**
- Fred Zhang & Neel Nanda, "Towards Best Practices of Activation Patching in Language Models: Metrics and Methods," arXiv:2309.16042 (v1 27 Sep 2023, v2 17 Jan 2024). The arXiv comments field says ICLR 2024. — [arXiv](https://arxiv.org/abs/2309.16042)
- Stefan Heimersheim & Neel Nanda, "How to use and interpret activation patching," arXiv:2404.15255 (23 Apr 2024). No venue is listed beyond arXiv. — [arXiv](https://arxiv.org/abs/2404.15255)

**Definitions (Zhang & Nanda)**
- The setup uses three runs: a clean run on X_clean (activations cached), a corrupted run on X_corrupt, and a patched run on X_corrupt with one component restored from the clean cache. Their example pairs "The Eiffel Tower is in" with "The Colosseum is in". — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)
- **Gaussian noising (GN)** adds ε ~ N(0, ν) to the key-token embeddings, with ν equal to 3 times the standard deviation of the token embeddings in the dataset. **Symmetric token replacement (STR)** swaps key tokens for semantically related tokens of the same length, so the corrupted prompts are "identically distributed as a fresh draw of a clean prompt." — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)
- Metrics, written explicitly:
  - Probability effect: P_pt(r) − P_*(r).
  - Logit difference: LD(r, r′) = Logit(r) − Logit(r′), with effect LD_pt − LD_*. Normalised (following Wang et al. 2023): (LD_pt − LD_*) / (LD_cl − LD_*), which typically lies in [0, 1].
  - KL effect: D_KL(P_cl ‖ P_*) − D_KL(P_cl ‖ P_pt).

  Here "pt" is the patched run, "*" the corrupted run, and "cl" the clean run. — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)

**Recommendations (Zhang & Nanda)**
- "We recommend STR whenever possible." GN may be used "when token alignment or lack of analogous tokens makes STR unsuitable." — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)
- Avoid probability. Probability is bounded below by 0, so negative components are hard to detect: in one setting the corrupted probability was about 0.03, which caps how negative an effect can look. Logit difference is preferred for localisation, and KL "can be a reasonable metric for circuit discovery." — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)
- Sliding-window patching (restoring several adjacent layers jointly) gave "at least 20% more peak effect" than the sum of single-layer effects. The authors warn this "may" inflate localisation plots and recommend single-layer patching first, with windows only "when individual layers seem to induce small effects." — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)
- Try different tokens to corrupt, because the choice can change localisation results. — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)
- Zhang & Nanda do not analyse noising vs denoising. They study clean→corrupt patching and leave the comparison of directions to future work. — [Zhang & Nanda HTML](https://arxiv.org/html/2309.16042v2)

**Direction, sufficiency, and necessity (Heimersheim & Nanda)**
- **Denoising** caches activations from the clean prompt and patches them into the corrupted run. It shows "whether the patched activations were sufficient to restore the model behaviour."
- **Noising** caches activations from the corrupted prompt and patches them into the clean run. It shows "whether the patched activations were necessary to maintain the model behaviour."
- The two directions are "not just symmetric mirrors of each other."

— [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)

**AND/OR circuits (Heimersheim & Nanda)**
- For C = A AND B, denoising A or B alone has no effect, so denoising finds only C, while noising finds A, B, and C.
- For C = A OR B, the pattern reverses: denoising finds all components, and noising finds only C.

— [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)

**Metrics (Heimersheim & Nanda)**
- They "personally trust logit difference (or equivalently logprob difference) the most."
- Probability is non-linear and saturates.
- Discrete metrics (accuracy, rank) miss sub-threshold effects and suit confirmatory work.
- Compute several metrics, including at least one continuous metric that is roughly linear in logits.

— [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)

**Choosing the corruption (Heimersheim & Nanda)**
- Each corruption traces only the properties it changes.
- Exploratory work should use narrow variations. Narrow distributions raise denoising's typically low sensitivity and lower noising's typically high sensitivity.
- Confirmatory work should use a wide distribution, noise the non-circuit components, and check that the behaviour persists.

— [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)

**Pitfalls (Heimersheim & Nanda)**
- A positive result shows only that the component handles the clean-vs-corrupt difference.
- Results apply only to the tested prompt distribution.
- Patching shows sufficiency, not minimality.
- Backup behaviour (the "Hydra effect") can make a main component look less important. The compensation is lossy (one cited result recovers about 0.7× of the effect).
- Negative components can make recovery exceed 100%, which the authors call "an unsolved problem."

— [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)

**Workflow (Heimersheim & Nanda)**
- Start coarse (for example, the residual stream), then increase granularity, then use path patching to separate mediation (A→B→C) from B reading A and also acting on C directly.
- Attribution patching can speed this up.
- Prefer patching over ablation, because it is "more targeted and controlled."

— [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)

### Inferences (mapping to AnimateDiff)
- **The A/B one-word swap is already STR.** Both clips are in-distribution, so the design follows Zhang & Nanda's recommendation and avoids Basu et al.'s Gaussian-noise corruption (Section 3). Keep it.
- **Rename the directions to avoid the clash with diffusion "denoising."**
  - "A→B patch (sufficiency)": run B and splice in A's activation at (m, t). The question is whether B's output acquires A's property.
  - "B→A patch (necessity)": run A and splice in B's activation. The question is whether A's property breaks.

  Run both directions. In a U-Net, skip connections and the parallel spatial and temporal paths create OR-like redundancy. Under OR-redundancy, a single-module B→A patch can show a small necessity effect even when that module is sufficient on its own.
- **Use contrastive, normalised metrics, the analogue of the normalised logit difference.**
  - Content: D_c(x) = CLIP(x, p_A) − CLIP(x, p_B). Contrasting the two prompts, or only the swapped word, controls for generic quality changes. The normalised effect is PE_c = [D_c(x_patch) − D_c(x_B)] / [D_c(x_A) − D_c(x_B)].
  - Motion: D_m(x) = cos(F(x), F(x_A)) − cos(F(x), F(x_B)), where F is the flattened flow field. PE_m has the same normalised form.

  Both are 0 for no transfer and 1 for full transfer, and both can go below 0 or above 1 (negative components). Avoid raw "CLIP to target prompt" alone. It behaves like the probability metric: it is bounded, non-contrastive, and saturating.
- **Single (m, t) cells first, step windows only when single cells are small.** Report window results explicitly as joint effects (Zhang & Nanda's "at least 20% more" warning).
- **Exploratory vs confirmatory.** Use a narrow one-word swap (verb ↔ verb with the same noun) for exploration. Use many pairs and seeds, including noun swaps, for confirmation.
- **Compute (inference).** 95 modules × 25 steps = 2,375 cells. At 90 s each, that is about 59 h per pair, per direction, per splice mode, which is infeasible on Kaggle. A coarse-to-fine scheme is needed:
  1. Group modules by U-Net level and type (for example, down/mid/up level × {motion module, spatial self-attention, cross-attention}).
  2. Use step windows (for example, 5 windows of 5 steps).
  3. Refine only the groups whose |PE| clears a threshold.
- **Trajectory caching (inference; DDIM with η=0 is deterministic given the shared noise).** A patched run is identical to the unpatched base run up to the first patched step. Cache the base latent z_t at the start of each window and resume from there. For 5 equal windows starting at steps 0, 5, 10, 15, and 20, the mean fraction of UNet calls needed is (1 + 0.8 + 0.6 + 0.4 + 0.2)/5 = 0.6, a saving of about 40%.

### Gaps
- Heimersheim & Nanda is arXiv-only. I found no peer-reviewed venue.
- Neither paper addresses iterative samplers, where a patch at step t also changes every later step's input. For diffusion, "patch only at step t" (later steps run freely) and "patch at every step" are different interventions, and neither source says which to prefer.

---

## 2. Causal mediation analysis (Vig et al., NeurIPS 2020): total, direct, and indirect effects

### Takeaway
Vig et al. define a unit-level total effect of an input intervention as a proportional change in a response y. They split it into a natural direct effect (intervene on the input, hold the mediator at its original value) and a natural indirect effect (keep the input, set the mediator to its post-intervention value). Empirically, the total effect approximately equals NDE + NIE, and effects concentrate in few components.

### Cited Findings
**Metadata**
- The NeurIPS 2020 version is "Investigating Gender Bias in Language Models Using Causal Mediation Analysis," by Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber, in Advances in Neural Information Processing Systems 33. — [NeurIPS proceedings](https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html)
- The expanded arXiv version has a different title: "Causal Mediation Analysis for Interpreting Neural NLP: The Case of Gender Bias," arXiv:2004.12265. It adds authors Simas Sakenis and Jason Huang. — [arXiv](https://arxiv.org/abs/2004.12265)

**Definitions** (equations below are from the arXiv expanded version, Sec. 3.2)
- The two do-operations are **set-gender** (replace "nurse" with "man") and **null** (leave the input unchanged). y_x(u) is the value y takes in unit u under do(x = x), and y is a bias ratio such as p(he|u)/p(she|u). — [arXiv 2004.12265](https://arxiv.org/abs/2004.12265)
- Total effect:
  - TE(set-gender, null; y, u) = [y_set-gender(u) − y_null(u)] / y_null(u) = y_set-gender(u)/y_null(u) − 1.
  - Averaged: TE(set-gender, null; y) = E_u[ y_set-gender(u)/y_null(u) − 1 ].

  The difference is made proportional "to control for the high variance of y across examples." — [arXiv 2004.12265](https://arxiv.org/abs/2004.12265)
- Natural direct effect, with the mediator z held at its null value: NDE(set-gender, null; y) = E_u[ y_{set-gender, z_null(u)}(u) / y_null(u) − 1 ]. — [arXiv 2004.12265](https://arxiv.org/abs/2004.12265)
- Natural indirect effect, with the input left as is and z set to its value under set-gender: NIE(set-gender, null; y) = E_u[ y_{null, z_set-gender(u)}(u) / y_null(u) − 1 ]. — [arXiv 2004.12265](https://arxiv.org/abs/2004.12265)
- A mediator z "might be a particular neuron, a full layer, an attention head, or a certain attention weight." The definitions follow Pearl (2001). — [arXiv 2004.12265](https://arxiv.org/abs/2004.12265)
- Findings: the effect is **sparse** (concentrated in few components) and **synergistic** (some components amplify each other). It is also **decomposable**: TE ≈ NDE + NIE, "a surprising result given the non-linear nature of the model." — [arXiv 2004.12265](https://arxiv.org/abs/2004.12265)

**How later work reuses these terms**
- ROME's causal tracing uses: TE = P[o] − P_*[o] (clean minus corrupted), and IE = P_{*, clean h_i^(l)}[o] − P_*[o] (corrupted input with one state restored), averaged as ATE and AIE over 1,000 statements. — [Meng et al. arXiv](https://arxiv.org/abs/2202.05262)
- AtP* writes I(n) = I_direct(n) + I_indirect(n), and uses this decomposition to explain why attribution patching can fail when the two terms nearly cancel. — [Kramár et al.](https://arxiv.org/abs/2403.00745)

### Inferences (mapping to AnimateDiff)
- **NIE is an A→B patch.** Hold the prompt at B (the "null" input) and set module m at step t to its value under prompt A. Then NIE_m = E[D(x_{B, h^m←h^m_A})] − E[D(x_B)], using the contrastive D from Section 1. The AnimateDiff splice is exactly this, so a splice result is an NIE.
- **NDE is a B→A patch on the prompt side.** Feed prompt A but freeze module m at B's value. NIE/TE then estimates the fraction of the prompt's effect that flows through m.
- **Additivity check.** If NDE + NIE ≈ TE for the top modules (Vig's decomposability result), interactions are weak and single-module rankings can be trusted. A large gap signals synergy or redundancy, so joint patching is needed.
- **Proportional metrics need care.** CLIP and flow-cosine differences can be near zero or negative, so Vig's proportional form (dividing by y_null) is unstable here. Use the normalised-difference form from Section 1 instead.

### Gaps
- I did not verify that the NeurIPS camera-ready uses equations identical to the arXiv expanded version. The NeurIPS abstract page did not show equations.

---

## 3. Causal tracing (ROME) and its application to T2I diffusion (Basu et al., ICLR 2024 and ICML 2024)

### Takeaway
ROME-style causal tracing corrupts the subject embedding with 3σ Gaussian noise and restores one clean state, measuring the AIE. In GPT, facts localise to mid-layer MLPs. Applied to SD v1.4, the same procedure (CLIP-Score as the metric, restoring a layer at every timestep) finds UNet knowledge **distributed**, with different attributes using different component sets. In the CLIP text encoder it is **localised** to the first self-attention layer at the last subject token.

The ICML 2024 follow-up shows causal tracing breaks down for SD-XL and DeepFloyd. It replaces tracing with LocoGen, which feeds a counterfactual prompt embedding to a window of m adjacent cross-attention layers and measures the **direct** effect on the image. That localises style, objects, and facts to a few layers (for example, SD-v1-5 style at cross-attention layer 8 with m=2).

### Cited Findings
**ROME (Meng et al.)**
- Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov, "Locating and Editing Factual Associations in GPT," NeurIPS 2022 (36th Conference on Neural Information Processing Systems). — [arXiv](https://arxiv.org/abs/2202.05262)
- The corruption is noise ~ N(0; ν) with ν = 3σ_t, "three times larger than the observed standard deviation" of embeddings, added to the subject tokens. — [arXiv](https://arxiv.org/abs/2202.05262)
- The corrupted-with-restoration run forces one clean state h_î^(l̂) at token î and layer l̂, with "future computations execut[ing] without further intervention." TE and IE are as given in Section 2. — [arXiv](https://arxiv.org/abs/2202.05262)
- For GPT-2 XL over 1,000 statements, individual states reach AIE = 8.7% at layer 15. MLP at the last subject token reaches AIE 6.6%, versus 1.6% for attention there. The paper also plots AIE for "a run of 10 MLP lookups" and "a run of 10 Attn modules", which is a 10-layer sliding window. — [arXiv](https://arxiv.org/abs/2202.05262)
- Appendix comparisons with uniform ±3σ noise give measurable but smaller effects than 3σ Gaussian noise, so the noise choice matters. — [arXiv](https://arxiv.org/abs/2202.05262)
- **Caveat on localisation vs editing.** Hase, Bansal, Kim & Ghandeharioun (NeurIPS 2023) find that Causal Tracing localisation "does not indicate which MLP layer is best to edit," and that edit-layer choice predicts edit success much better. — [NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)

**Basu et al., ICLR 2024**
- Samyadeep Basu, Nanxuan Zhao, Vlad Morariu, Soheil Feizi, Varun Manjunatha, "Localizing and Editing Knowledge in Text-to-Image Generative Models," ICLR 2024 (poster). — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/4bfcebedf7a2967c410b64670f27f904-Abstract-Conference.html), [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)

*Method*
- Three model configurations: clean ε_θ (with CFG), corrupted ε_θ^corr (the word embedding of the attribute's subject, e.g. "Van Gogh", corrupted with Gaussian noise), and restored ε_θ^restored, in which "one of its layers is restored from the clean model at each time-step." The restore operation is ε_θ^restored[a_i] : ε_θ^corr[a_i] = ε_θ[a_i]. — [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)
- CFG is then applied: ε̂^restored(z_t, c, t) = ε^restored(z_t, c, t) + α(ε^restored(z_t, c, t) − ε^restored(z_t, t)), ∀t ∈ [T, 1], followed by DDIM sampling and VAE decoding. — [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)
- Metric: AIE = |CLIPScore(x_0^restored, c) − CLIPScore(x_0^corr, c)|. Because x_corr is shared across layers, they use CLIPScore(x_0^restored, c) directly. — [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)
- The causal/non-causal threshold comes from a small human study (10 validation prompts per attribute): the lowest average CLIP-Score among human-confirmed causal states is used as the cutoff. — [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)
- The model is Stable Diffusion v1.4. The UNet has "70 unique layers" across down, mid, and up blocks. Attributes traced are objects, style, color, and action, with viewpoint and count also captioned. — [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)

*UNet findings*
- Knowledge is distributed, and the sets of causal components "are often distinct for different visual attributes."
- Causal states are denser in the up-block than in the down or mid blocks.
- For style, the first self-attention layer is causal but is not causal for other attributes.
- For action, the mid-block cross-attention is causal but is not causal for other attributes.
- Figure 2 labels include down-1-resnet-0/1, down-0-resnet-0, mid-block-resnet-1, and mid-block-cross-attn.
- "To our surprise ... the cross-attention layers are not causally important states and a significant amount of knowledge is in fact stored in components such as the ResNet blocks and the self-attention blocks."
- When the whole text embedding is noised, some causal states can still restore the output, "highlighting that the conditional information can be completely bypassed if certain causal states are active" (Appendix M).

— [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)

*Text-encoder finding*
- A single causal state: self-attn-0 at the last subject token. This motivates DiffQuickFix, a closed-form edit that runs in under a second, about 1000× faster than fine-tuning. — [arXiv 2310.13730](https://arxiv.org/abs/2310.13730)

**Basu et al., ICML 2024**
- Samyadeep Basu, Keivan Rezaei, Priyatham Kattakinda, Ryan Rossi, Cherry Zhao, Vlad Morariu, Varun Manjunatha, Soheil Feizi, "On Mechanistic Knowledge Localization in Text-to-Image Generative Models," ICML 2024. — [arXiv 2405.01008](https://arxiv.org/abs/2405.01008), [ICML poster page](https://icml.cc/virtual/2024/poster/33449)

*Why causal tracing fails*
- Causal tracing works for SD v1-5 and v2-1 but "fails in pinpointing localized knowledge" for SD-XL and DeepFloyd.
- In SD-XL only about 5% of layers are causal ("extremely sparse"). DeepFloyd has "no strong causal states in the UNet."
- For SD-XL and DeepFloyd, no single causal state exists in the text encoder either.

— [arXiv 2405.01008](https://arxiv.org/abs/2405.01008)

*LocoGen*
- Causal tracing measures indirect effects. LocoGen "measures the direct effect of intermediate layers."
- A subset C′ of the cross-attention layers {C_l}_{l=1}^M receives a counterfactual prompt embedding c′ (e.g. "...in the style of painting"), while all other cross-attention layers receive the original c. The model output is then ε̂(z_t, c, c′, t) = ε_θ(z_t, c, c′, t) + α(ε_θ(z_t, c, c′, t) − ε_θ(z_t, t)).
- C′ is restricted to m *adjacent* layers. All M − m + 1 windows are scanned and the window with the minimum average CLIP-Score to the original prompt is chosen (maximum CLIP-Score to the target prompt for facts). The minimum m that meets a CLIP-Score threshold is selected.

— [arXiv 2405.01008](https://arxiv.org/abs/2405.01008)

*Layers found*
- SD-v1-5: style at layer 8 (m=2); objects and facts at layer 6 (m=2).
- SD-v2-1: style and objects at layer 8, facts at layer 7 (m=3).
- OpenJourney: same as SD-v1-5.
- SD-XL (70 cross-attention layers): style and facts at layer 45, objects at layer 15.
- DeepFloyd: control points depend on the prompt rather than the attribute.

— [arXiv 2405.01008](https://arxiv.org/abs/2405.01008)

*Validation*
- A human study of 132 image pairs with five evaluators gave a "92.58% verification rate." — [arXiv 2405.01008](https://arxiv.org/abs/2405.01008)

### Inferences (mapping to AnimateDiff)
- **AnimateDiff's base is SD1.5, so both papers' SD1.x results are directly relevant priors.**
  - Content attributes are spread across ResNet and self-attention blocks, so causal tracing with a single-module restore may show weak per-module effects.
  - Cross-attention is a narrow, controllable entry point (LocoGen layers 6 and 8).
  - Prediction for "content vs motion": a noun swap should localise to a few cross-attention layers under LocoGen-style intervention, whereas a verb swap may route through motion modules.
- **A LocoGen-style test needs no activation caching.** It is just a different encoder_hidden_states per cross-attention layer, so it is cheap. Scan windows of m adjacent cross-attention layers (SD1.5 has 16), feeding c_B to the window and c_A elsewhere. Note that AnimateDiff motion modules do not take text conditioning, so prompt-embedding windows test only spatial cross-attention.
- **Basu's AIE metric is non-contrastive** (CLIPScore to the original caption only), with a human-calibrated threshold. The contrastive D_c from Section 1 is preferable. Copying their threshold procedure (a small human-rated validation set, threshold set at the lowest causal score) is cheap and worth adopting for the motion metric too.
- **Basu restores at every timestep.** The study's step-window splicing adds a time dimension Basu does not resolve. That is a genuine extension, and a likely place where content (early steps) and motion separate. This is a hypothesis, not a cited finding.
- **The Appendix M bypass result warns that redundancy is real in U-Nets.** That supports running both patch directions (Section 1 AND/OR logic).

### Gaps
- Neither Basu paper studies video or temporal (motion) modules. I found no published causal-tracing study of AnimateDiff motion modules (the search was not exhaustive).
- The exact noise scale Basu et al. use for the U-Net corruption was not extracted. ROME's 3σ is likely but unconfirmed.
- Basu ICML 2024 has an internal inconsistency on SD-XL. The Fig. 5 caption says style m=3 and facts m=5 at layer 45, while the body text says style and facts at layer 45 with m=3 and objects at layer 15 with m=5. Treat SD-XL m values as uncertain.

---

## 4. Attribution patching (Nanda 2023), Edge Attribution Patching (Syed, Rager & Conmy 2023), AtP* (Kramár et al. 2024): formula, failure modes, feasibility on a T4

### Takeaway
Attribution patching replaces each patch with a first-order Taylor estimate, (Δactivation)·∇metric, for every component at once, from two forward passes and one backward pass. It works well for "small" activations (head or module outputs) and badly for "big" ones (the residual stream) and for saturated nonlinearities. AtP* fixes attention-saturation false negatives (the QK fix) and direct/indirect cancellation (GradDrop), and adds a subsampling diagnostic that bounds missed effects.

For AnimateDiff, a single-step backward through one UNet call is probably feasible on 16 GB with fp16 and gradient checkpointing, but this is unverified and must be measured. Backpropagating CLIP or optical-flow through 25 steps plus the VAE is not practical, so a per-step differentiable proxy metric is required.

### Cited Findings
**Metadata**
- Neel Nanda, "Attribution Patching: Activation Patching At Industrial Scale." The site shows "Feb 4"; the LessWrong linkpost is dated 16 March 2023. — [neelnanda.io](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching), [LessWrong](https://www.lesswrong.com/posts/gtLLBhzQTG6nKTeCZ/attribution-patching-activation-patching-at-industrial-scale)
- Aaquib Syed, Can Rager, Arthur Conmy, "Attribution Patching Outperforms Automated Circuit Discovery," NeurIPS 2023 ATTRIB Workshop. — [arXiv 2310.10348](https://arxiv.org/abs/2310.10348)
- János Kramár, Tom Lieberum, Rohin Shah, Neel Nanda (Google DeepMind), "AtP*: An efficient and scalable method for localizing LLM behaviour to components," arXiv:2403.00745 (1 Mar 2024). The arXiv page lists no venue. — [arXiv](https://arxiv.org/abs/2403.00745)

**Formula**
- Nanda's per-activation estimate is `((clean_act - corrupted_act) * corrupted_grad_act).sum()`: the gradient is taken in the corrupted run and the clean activation is patched in (the denoising direction). "Every single patch can be computed after caching activations from two forward passes and one backwards pass." — [Nanda](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- Syed et al., Eq. 2: L(x_clean | do(E = e_corr)) ≈ L(x_clean) + (e_corr − e_clean)^⊤ ∂L(x_clean | do(E = e_clean))/∂e_clean. The second term is the attribution score Δ_e L. They rank edges by |Δ_e L| averaged over (clean, corrupt) pairs and keep the top k (EAP). — [Syed et al.](https://arxiv.org/abs/2310.10348)
- AtP*'s target quantity (noising direction): I(n; x_clean, x_noise) = L(M(x_clean | do(n ← n(x_noise)))) − L(M(x_clean)), with c(n) = |E_D[I(n; ·)]|. — [Kramár et al.](https://arxiv.org/abs/2403.00745)

**Failure modes**
- Nanda lists five nonlinearities that break the approximation: attention softmax, MLP activations, value-weighted attention, LayerNorm, and the final softmax.
  - Saturation: the case where an attention weight "starts near zero and ends up near one post patch" is handled poorly.
  - LayerNorm: gives near-zero derivatives for patches parallel to the residual stream.
  - The approximation is "decent" for "small" activations like head outputs and "bad" for "big" ones like the residual stream. It "works badly for residual streams and MLP0, well for everything else."

  — [Nanda](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- Syed et al.: attribution patching fails when the metric's gradient is zero. KL divergence is 0 and minimal on the unablated run, so all its gradients vanish there. They therefore use task metrics instead. — [Syed et al.](https://arxiv.org/abs/2310.10348)
- Syed et al. report that EAP has higher AUC than ACDC averaged over tasks (IOI, Docstring, Greater-Than), although ACDC with KL beats EAP on one task. — [Syed et al.](https://arxiv.org/abs/2310.10348)
- AtP* identifies two classes of false negatives:
  1. **Attention saturation.** The pre-activation on x_clean sits in a flat region of the softmax but the one on x_noise does not. The fix is the "QK fix," which recomputes attention probabilities exactly for query and key patches, at less cost than two forward passes.
  2. **Cancellation.** I = I_direct + I_indirect nearly cancel, so small errors leave |Î| orders of magnitude below |I|. The fix is GradDrop, which costs L extra backward passes.

  — [Kramár et al.](https://arxiv.org/abs/2403.00745)
- AtP* diagnostic: use AtP to filter out high-impact nodes, then patch random subsets of the remainder. This gives an upper confidence bound on the effect size of any missed false negative. — [Kramár et al.](https://arxiv.org/abs/2403.00745)

**Diffusion precedent**
- I found no published application of attribution patching to diffusion U-Nets. — [search result: Nanda post is the method source](https://neelnanda.io/attribution-patching)
- AnimateDiff's README gives only an inference figure: "~13GB VRAM" for SDXL-beta at 1024×1024 with 16 frames. It gives nothing for SD1.5 training or backward-pass memory. — [AnimateDiff GitHub](https://github.com/guoyww/AnimateDiff)

### Inferences (mapping to AnimateDiff; none of this is published, all of it needs testing)
- **Screening formula for module m at step t (denoising direction, on run B's trajectory).**
  - Score: Î(m, t) = Σ (h_A^{m,t} − h_B^{m,t}) ⊙ ∂L_t/∂h^{m,t} |_{run B}.
  - Per-step proxy: L_t = ⟨ε̂_t − ε̂_{B,t}, ε̂_{A,t} − ε̂_{B,t}⟩ / ‖ε̂_{A,t} − ε̂_{B,t}‖². This is the projection of the guided noise prediction onto the A−B direction (a normalised, logit-difference-like metric).
  - It is linear and non-saturating, and its gradient is not zero at the base point. That avoids the KL zero-gradient trap Syed et al. describe; an MSE-to-A metric would have a non-zero gradient too, but it is quadratic.
- **Myopia.** L_t measures only the immediate effect on that step's noise prediction. It ignores how the change propagates through later DDIM steps. Use it only to rank and screen, then confirm the top cells with real patches scored by CLIP and flow.
- **Content and motion proxies.** Split the A−B direction into its frame-mean ("static") and frame-mean-removed ("dynamic") parts in ε-space, giving L_t^static and L_t^dynamic as candidate content and motion proxies. Validate them by rank correlation against true PE_c and PE_m on a subset of patched cells, in the spirit of AtP*'s diagnostics.
- **Free decompositions.** The score is a dot product, so it splits exactly into per-channel contributions and into static plus dynamic parts: Î = Σ_c Î_c = Î_static + Î_dyn, because (Δh) decomposes orthogonally along the frame axis. One backward pass therefore ranks channel subsets and static/dynamic splices for every module.
- **Expected failure points.** These are the softmaxes in temporal attention over 16 frames and spatial self-attention over 4,096 tokens (64×64), and GroupNorm, the analogue of LayerNorm. Patch scores on attention outputs and motion-module outputs, which are "small" activations, rather than on the ResNet residual stream. If temporal-attention patterns change a lot between A and B (check max |Δ attention prob|, as AtP* does), expect false negatives. The MotionClone-style map-copy baseline is the exact intervention to check those cells against.
- **Memory (unverified estimate).** One forward plus backward through a single UNet+motion-module call covers 16 frames, or 32 with CFG, at a 64×64 latent.
  - To fit a T4's 16 GB: fp16 weights; `requires_grad_(False)` on parameters, with gradients only on hooked outputs (no parameter-gradient buffers); diffusers' `unet.enable_gradient_checkpointing()`; SDPA memory-efficient attention; and, if needed, backpropagating only through the conditional branch.
  - Measure with `torch.cuda.max_memory_allocated()` before committing.
  - Do not reduce frames to save memory: that changes motion-module behaviour.
  - The 8 GB laptop is probably not enough for the backward pass. Use it for analysis, probes, and LEACE fitting on cached, pooled activations.
- **Screening cost.** 25 single-step backward passes on run B per pair (plus the forward passes for A and B) rank all 2,375 (m, t) cells. That is roughly the cost of 2–3 normal clips, compared with about 2,375 clips for brute-force patching.
- **Activation storage.** One module output at 64×64×16×320 in fp16 is 4096·16·320·2 B ≈ 42 MB. Caching all 95 modules × 25 steps per clip is tens of GB, so compute A−B differences on the fly or cache only the screened cells.

### Gaps
- No measured backward-pass memory for AnimateDiff SD1.5 at 512×512×16 on a T4 was found. The figure must be measured.
- No source validates per-step ε-space proxies against end-of-trajectory CLIP or flow metrics for diffusion.
- Syed et al. may also have appeared at a later venue (possibly BlackboxNLP 2024). Not verified; cite as the NeurIPS 2023 ATTRIB Workshop paper.

---

## 5. Concept erasure as a causal test (INLP, amnesic probing, LEACE): formulas, and erasing "verb" information during generation

### Takeaway
INLP iteratively projects representations onto the nullspace of trained linear probes. Amnesic probing uses that projection as a counterfactual: it removes property P at a layer, measures the behavioural change, and compares against a random-direction control of the same rank. LEACE gives the closed-form, least-squares-optimal affine eraser, r(x) = x − W⁺P_{WΣ_XZ}W(x − E[X]), which provably makes the concept linearly unpredictable while removing minimal rank. Concept scrubbing applies it layer by layer.

For AnimateDiff: fit LEACE for a verb label on a module's (dynamic-component) activations, apply it at chosen steps during sampling, and measure PE_m and PE_c against a random same-rank erasure control.

### Cited Findings
**Metadata**
- Shauli Ravfogel, Yanai Elazar, Hila Gonen, Michael Twiton, Yoav Goldberg, "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection," ACL 2020, pp. 7237–7256. — [ACL Anthology](https://aclanthology.org/2020.acl-main.647/)
- Yanai Elazar, Shauli Ravfogel, Alon Jacovi, Yoav Goldberg, "Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals," TACL vol. 9 (2021), pp. 160–175. — [ACL Anthology](https://aclanthology.org/2021.tacl-1.10/)
- Nora Belrose, David Schneider-Joseph, Shauli Ravfogel, Ryan Cotterell, Edward Raff, Stella Biderman, "LEACE: Perfect linear concept erasure in closed form," NeurIPS 2023 (per the paper header). — [arXiv 2306.03819](https://arxiv.org/abs/2306.03819)

**INLP**
- Repeatedly train a linear classifier W_i to predict Z from X, then project X onto its nullspace N(W_i) = {x | W_i x = 0}. The final projection is P = P_N(W_m) P_N(W_{m−1}) ⋯ P_N(W_0). — [INLP](https://aclanthology.org/2020.acl-main.647/)

**Amnesic probing**
- The method uses INLP to remove a property from a layer and measures the change in the model's own task (LM accuracy and D_KL). — [Amnesic probing](https://aclanthology.org/2021.tacl-1.10/)
- *Control over information (Rand).* The control removes the same number of directions as INLP, but randomly: "If the Rand impact on performance is lower than the impact of amnesic probing for some property, we conclude that we removed important directions for the main task. Otherwise, when the Rand control has a similar or higher impact, we conclude that there is no evidence for property usage by the main task." — [Amnesic probing](https://aclanthology.org/2021.tacl-1.10/)
- *Control over selectivity.* The removed features may only correlate with the property. To test this, the gold property information is provided back and the subsequent layers are fine-tuned, to see whether performance is restored. — [Amnesic probing](https://aclanthology.org/2021.tacl-1.10/)
- Headline finding: "probe accuracy does not correlate with task importance," so high probing performance does not entail that the model uses the information. — [Amnesic probing](https://aclanthology.org/2021.tacl-1.10/)

**LEACE**
- The guarding conditions are equivalent: "X linearly guards Z" (no linear classifier beats the constant predictor) holds iff the class-conditional means equal the unconditional mean E[X], iff the cross-covariance Σ_XZ = 0 (Thms 3.x). — [LEACE](https://arxiv.org/abs/2306.03819)
- Thm 4.1: an affine r(x) = Px + b guards Z iff ker(P) ⊇ colsp(Σ_XZ), since Cov(PX + b, Z) = PΣ_XZ. — [LEACE](https://arxiv.org/abs/2306.03819)
- Thm 4.2 (centred case): argmin_P E‖PX − X‖²_M subject to Cov(PX, Z) = 0 is solved by P* = I − W⁺ P_{WΣ_XZ} W, where:
  - W = (Σ_XX^{1/2})⁺ is the whitening transform;
  - P_{WΣ_XZ} = (WΣ_XZ)(WΣ_XZ)⁺ is the orthogonal projector onto colsp(WΣ_XZ);
  - the solution is optimal for both the Euclidean (M = I) and the Mahalanobis (M = Σ_XX⁺) norms.

  — [LEACE](https://arxiv.org/abs/2306.03819)
- Thm 4.3 adds the bias b* = E[X] − P*E[X], which gives the LEACE formula **r_LEACE(x) = x − W⁺ P_{WΣ_XZ} W (x − E[X])**. Geometrically: whiten, orthogonally project out colsp(WΣ_XZ), then unwhiten. — [LEACE](https://arxiv.org/abs/2306.03819)
- Rank efficiency: in a replication of amnesic-probing experiments, INLP needed 20 iterations, still left accuracy above majority, and reduced the embedding rank by 360. LEACE reduced it by 17. — [LEACE](https://arxiv.org/abs/2306.03819)
- Concept scrubbing (Algorithm 1) fits LEACE sequentially from the first layer onward, with each layer's eraser fit on the already-scrubbed upstream activations. — [LEACE](https://arxiv.org/abs/2306.03819)
- Limitations: guardedness is linear only, and models might "circumvent" erasure by encoding attributes nonlinearly. — [LEACE](https://arxiv.org/abs/2306.03819)

### Inferences (procedure for erasing "verb" in AnimateDiff; not published)
1. **Data.** Generate N clips over K verbs × several nouns × seeds, crossing nouns and verbs so that verb is not confounded with content. Z is the one-hot verb label, so the rank removed is at most K−1 after centring.
2. **Features.** For module m and step window τ, use samples x ∈ R^C (C = channels, 320–1280). Samples can be per (frame, h, w) position, or pooled.
   - Fit separate erasers on the **dynamic component** (h − mean_frames h) and on the **static component** (mean_frames h). This directly tests whether verb information lives in the temporal-variation part.
   - Statistics are timestep-dependent, so fit per step window.
   - Positions within a clip are highly correlated. Treat the clip, not the position, as the unit for train/test splits.
3. **Intervene during sampling.** Apply r_LEACE to h^{m,t} via a forward hook at the chosen steps, on the conditional CFG branch (or on both branches: report which).
4. **Measure.**
   - Behaviour: the motion change PE_m relative to the unerased clip (or flow similarity to verb-matched references), and the content change PE_c, which should stay unchanged if the erasure is selective.
   - Leakage: a held-out linear probe on the erased activations, which should be at chance, plus a nonlinear MLP probe to detect nonlinear residual information.
5. **Controls (from amnesic probing).**
   - Rand: erase a random subspace of the same rank, matched in whitened space so it is comparable to LEACE.
   - Selectivity: erase an unrelated label, such as noun identity.
   - "Verb used" is concluded only if the motion drop clearly exceeds the Rand control. Repeat across seeds and report a confidence interval.
6. **Concept scrubbing.** Erasing at one module may be undone by redundancy, given U-Net skip connections and parallel motion modules. Scrubbing every module at or below a depth, in sequence, tests whether the property is recoverable downstream.

**Cost.** Fitting needs only means and covariances, which is cheap on the laptop. Applying the eraser is one matrix multiply per hook, so generation cost stays about 90 s per clip.

### Gaps
- No published application of LEACE or INLP inside a diffusion or video U-Net was found.
- How Σ_XX should be estimated with spatially correlated samples (effective sample size) is not addressed in the sources.
- The rank of verb information might exceed K−1 in a nonlinear sense. LEACE cannot detect this.

---

## 6. Interchange interventions and Distributed Alignment Search (DAS): can a subspace-level intervention isolate a motion subspace?

### Takeaway
DAS learns an orthogonal rotation R so that swapping a low-dimensional rotated subspace from a source run into a base run reproduces a hypothesised high-level causal variable's counterfactual behaviour. Success is scored by interchange intervention accuracy (IIA). In principle it can find a "motion subspace" inside a layer. But subspace patching can create illusions by activating dormant pathways, so learned subspaces need extra checks.

### Cited Findings
- Atticus Geiger, Zhengxuan Wu, Christopher Potts, Thomas Icard, Noah D. Goodman, "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations," Proceedings of the 3rd Conference on Causal Learning and Reasoning (CLeaR 2024), PMLR vol. 236. — [arXiv 2303.02536](https://arxiv.org/abs/2303.02536)
- DAS "find[s] the best alignment via gradient descent rather than conducting a brute-force search" and uses distributed interchange interventions on non-basis-aligned subspaces. The IIA metric gives "a graded notion" of abstraction. — [DAS](https://arxiv.org/abs/2303.02536)
- Definition 3 (distributed interchange intervention):
  - Let R: N → Y be invertible, with Y = ⊕_{j=0}^k Y_j an orthogonal decomposition.
  - Rotate the representation, replace the component in subspace Y_j with the corresponding component from source input s_j, and rotate back.
  - In DAS, R is an orthogonal matrix found by gradient descent with an interchange-intervention training objective.

  — [DAS](https://arxiv.org/abs/2303.02536)
- Illusion warning: Aleksandar Makelov, Georg Lange, (Atticus Geiger,) Neel Nanda, "Is This the Subspace You Are Looking for? An Interpretability Illusion for Subspace Activation Patching," ICLR 2024. A subspace patch can have the intended end-to-end effect "by activating a dormant parallel pathway" that is causally disconnected from the output in normal operation. The authors also show a success case on IOI. — [arXiv 2311.17030](https://arxiv.org/abs/2311.17030), [ML Anthology ICLR 2024](https://mlanthology.org/iclr/2024/makelov2024iclr-subspace)

### Inferences (mapping to AnimateDiff)
- **For a k-dimensional subspace with orthonormal rows R_k (k×C)**, the interchange is h′ = h_B + R_kᵀ(R_k h_A − R_k h_B). This follows from Definition 3 with a single subspace.
- **The existing static/dynamic splice is already a fixed (not learned) distributed interchange intervention along the frame axis.** The frame-mean is the projection onto the constant ("DC") vector in R^16. A natural generalisation is a temporal-DCT basis: patch low versus high temporal frequencies. A channel subset is a basis-aligned intervention, and DAS generalises it to rotated channel subspaces.
- **DAS objective for motion.**
  - A and B differ only in the verb, so the counterfactual "B's content with A's verb" is just A's run. Train R_k to maximise the projection of ε̂ onto A's noise prediction (the L_t proxy from Section 4) after patching only the subspace.
  - Constraint: on noun-swap pairs, the same subspace patch must *not* transfer content, measured by a held-out PE_c near 0.
  - Small k (1–8) plus held-out prompts and seeds is the test. A motion subspace that transfers verbs but not nouns is evidence of a dedicated subspace.
- **Cost.** DAS needs backprop through a UNet call per training step, with the same memory issue as Section 4. Cheaper first steps:
  1. A difference-of-means direction, mean over pairs of (h_A − h_B) projected onto the dynamic component, used as a 1-D subspace.
  2. The LEACE subspace W⁺colsp(WΣ_XZ) from Section 5.

  Patch along these directions and compare against random directions of the same dimension.
- **Makelov check.** Test whether the found subspace's component varies naturally between A and B, i.e. whether the projection actually differs between runs. If the subspace barely varies naturally but the patch has a large effect, suspect a dormant pathway.

### Gaps
- I did not verify Boundless DAS (Wu et al., "Interpretability at Scale," NeurIPS 2023) or any DAS application to diffusion or vision generative models.
- Author lists for Makelov et al. differ between sources: arXiv and MATS list three authors, while the ICLR proceedings reportedly add Atticus Geiger. Check the proceedings before citing.

---

## 7. Combining probes (correlational) with patching and erasure (causal): when to say a component "encodes" vs "uses" a property

### Takeaway
The sources agree that probing accuracy shows only that information is *decodable* ("encodes"). It does not show that the model *uses* it: amnesic probing found probe accuracy uncorrelated with task importance.

"Uses" requires an intervention with a matched control:
- the amnesic rule (effect beyond a random same-rank erasure);
- necessity via noising/B→A patches;
- sufficiency via denoising/A→B patches.

Read the results with the AND/OR logic and with the caveat that localisation does not imply the best edit site. No source gives a single formal decision rule. The rule below is assembled from these pieces.

### Cited Findings
- Yonatan Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances," *Computational Linguistics* 48(1):207–219 (2022). It reviews the methodological limits of probes, and the fact that probes are correlational is the motivation for causal "advances" such as amnesic probing. — [ACL Anthology](https://aclanthology.org/2022.cl-1.7/)
- Amnesic probing's explicit decision rule: erasure impact > Rand control → "we removed important directions for the main task"; similar or lower → "no evidence for property usage." Their conclusion is that high probing performance does not entail usage. — [Amnesic probing](https://aclanthology.org/2021.tacl-1.10/)
- Denoising establishes sufficiency and noising establishes necessity. Neither implies the other, and AND/OR structure determines which components each direction reveals. — [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)
- A positive patching result shows only that the component carries the clean–corrupt difference within the tested distribution. Backup components and negative components distort single-component readings. — [Heimersheim & Nanda](https://arxiv.org/html/2404.15255)
- Mediation decomposition: NIE through mediator z relative to TE gives the share of the effect flowing through z. Vig et al. found TE ≈ NDE + NIE. — [Vig et al. arXiv](https://arxiv.org/abs/2004.12265)
- Causal-tracing localisation did not predict the best layer to edit (Hase et al.). "Where the information is causally present" and "where an intervention works best" are separate claims. — [Hase et al., NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/hash/3927bbdcf0e8d1fa8aa23c26f358a281-Abstract-Conference.html)
- In T2I U-Nets, indirect-effect tracing (Basu, ICLR 2024) and direct-effect intervention (LocoGen, ICML 2024) gave different pictures: distributed versus few cross-attention layers. Which effect is measured changes the localisation conclusion. — [Basu ICLR](https://arxiv.org/abs/2310.13730), [Basu ICML](https://arxiv.org/abs/2405.01008)
- Subspace interventions can succeed via dormant pathways. — [Makelov et al.](https://arxiv.org/abs/2311.17030)

### Inferences (a proposed decision rule for content vs motion; synthesised by me, not taken from any one paper)
For module m (and step window τ), with property P ∈ {content, motion}:
1. **Encodes P (correlational).** A linear probe on h^{m,τ} decodes P (noun for content, verb for motion) on held-out clips, well above a control. Controls: a probe trained on shuffled labels, or a probe on the same module in a random-init or unconditioned run.
2. **Sufficient for P (A→B patch).** The normalised effect PE_P(A→B) has a confidence interval excluding 0 across ≥ 10–20 pairs and seeds, and exceeds a matched control: patching the same module from an *unrelated* clip C, which controls for disruption.
3. **Necessary for P.** Either PE_P(B→A) is reliably negative, or LEACE erasure of P at m drops the P-metric more than Rand erasure of the same rank (the amnesic rule).
4. **Selectivity / dissociation.** Module m is a *motion* locus if its motion effect is large and its content effect is near 0, i.e. a high dissociation index DI_m = (|PE_m| − |PE_c|)/(|PE_m| + |PE_c|). It is a *content* locus if the reverse holds. Strongest is a **double dissociation**: module X transfers motion but not content, and module Y transfers content but not motion, under the same pairs and metric.
5. **Labels.**
   - "Stores/encodes but does not use P": 1 holds, but 2 and 3 fail beyond controls. This is the amnesic-probing pattern.
   - "Uses P": 2 or 3 holds. Say "necessary," "sufficient," or both, explicitly.
   - "Redundant use" (OR-like): 2 holds, 3 fails, and joint B→A patching of the module group succeeds.
   - "Required but not sufficient" (AND-like): 3 holds and 2 fails.
6. **Time.** Report each label per step window. "Content at early steps, motion at mid steps" is a valid and likely outcome.

**Budgeted pipeline for Kaggle T4s**
- (a) Attribution-patching screen: about 25 single-step backward passes per pair to rank all (m, t) cells and static/dynamic splits.
- (b) Coarse real patching: module groups × 5 step windows × 2 directions, with trajectory caching (about 2 h per pair per direction at 90 s per clip).
- (c) Refine the top 5–10 cells to single modules and steps, plus static/dynamic splices.
- (d) LEACE and Rand erasure on the 2–3 strongest candidates.
- (e) Optional subspace (difference-of-means, then DAS) test on the strongest motion candidate.
- Run every causal claim on ≥ 10 pairs, report bootstrap confidence intervals, and validate the CLIP and flow thresholds on a small human-rated set (Basu's procedure).

### Gaps
- None of the cited sources gives a formal, quantitative decision rule combining probes and interventions. The rule above is a synthesis and should be presented as such.
- I did not verify Hewitt & Liang's "control tasks" / selectivity (EMNLP 2019) or Ravichander et al.'s "Probing the probing paradigm" in this session. Cite them only after checking.
- The Kaggle weekly GPU quota (commonly about 30 h per week) was not verified here, and it constrains how many pairs are feasible.
