# Fine-tuning methods that decouple appearance from motion in video diffusion models (for an AnimateDiff content/motion probing project)

Scope note: primary sources were read as full text. arXiv PDFs were downloaded and grepped, and training scripts were read from GitHub. Hugging Face repo ids were checked against the HF API on 2026-10-24. "Verified" below means the item was confirmed against the paper PDF, the repo/API, or a proceedings index. Anything else is flagged.

---

## 1. MotionDirector (Zhao et al., ECCV 2024): dual-path LoRA, appearance-debiased temporal loss, data, steps, hardware, evaluation

### Takeaway
MotionDirector trains two LoRA sets. **Spatial LoRAs** go on spatial self-attention and FFN layers and are trained on single random frames. **Temporal LoRAs** go on temporal self-attention and FFN layers and are trained on the whole clip with the spatial LoRAs loaded. The temporal objective is plain MSE plus an **appearance-debiased MSE**, which subtracts a random "anchor" frame's noise from every frame before comparing. The paper reports 400 steps for one video and 1000 for a few videos, rank 32, lr 1e-4, 16 frames at 384², about 14 GB VRAM, and 8–20 min on an A5000. An AnimateDiff (SD1.5) port exists and is the most direct fit for this project.

### Cited Findings
- **Citation (verified).** Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, David Junhao Zhang, Jia-Wei Liu, Weijia Wu, Jussi Keppo, Mike Zheng Shou. "MotionDirector: Motion Customization of Text-to-Video Diffusion Models." ECCV 2024 (Oral, per the repo description), arXiv:2310.08465. — [arXiv](https://arxiv.org/abs/2310.08465); [ECCV PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07327.pdf); [repo showlab/MotionDirector, Apache-2.0](https://github.com/showlab/MotionDirector). The arXiv PDF lists the first authors as Rui Zhao, Yuchao Gu and Jay Zhangjie Wu. The remaining names above come from my memory of the paper and were not re-read from the PDF, so treat them as **partially verified**.
- **Spatial path.** LoRAs go into spatial self-attention and feed-forward layers. They deliberately leave out spatial cross-attention "to maintain the learned strong and diverse textual alignment ability". Each step trains on **one randomly sampled frame**:
  `L_spatial = E_{z0,y,ε,t,i~U(0,F)} || ε − ε_θ(z_{t,i}, t, τ_θ(y)) ||²₂` (Eq. 4). — [arXiv 2310.08465 §3.2](https://arxiv.org/abs/2310.08465)
- **Temporal path.** LoRAs go into temporal-transformer self-attention and FFN layers. The spatial LoRAs, with weights shared from the spatial path, are also loaded "to force the trainable temporal LoRAs to ignore the appearance of the training data". — [arXiv 2310.08465 §3.2](https://arxiv.org/abs/2310.08465)
- **Appearance-debiased temporal loss (Eqs. 5–7).**
  `φ(ε_i) = sqrt(β² + 1) · ε_i − β · ε_anchor`
  `L_ad-temp = E_{z0,y,ε,t} || φ(ε) − φ(ε_θ(z_t, t, τ_θ(y))) ||²₂`
  `L_temporal = L_org-temp + L_ad-temp`, where L_org-temp is the standard full-clip MSE.
  The paper sets β = 1 and picks the anchor as a random frame of the same clip. The motivation is that, in latent space, "the internal connectivity structure between latent codes is more influenced by motion, while the distance between sets of latent codes is primarily affected by the difference in appearance." — [arXiv 2310.08465 §3.2, Fig. 4](https://arxiv.org/abs/2310.08465)
- **Code matches the equation.** `alpha=(beta**2+1)**0.5; ran_idx=randint(F); model_pred_decent = alpha*model_pred - beta*model_pred[:,:,ran_idx]; target_decent = alpha*target - beta*target[:,:,ran_idx]; loss_temporal = mse + mse(decent)`. — [MotionDirector_train.py](https://github.com/showlab/MotionDirector/blob/main/MotionDirector_train.py)
- **Inference-time strength.** `W = W0 + γΔW`, with LoRA scale γ adjustable separately for the spatial (appearance) and temporal (motion) sets. — [arXiv 2310.08465](https://arxiv.org/abs/2310.08465)
- **Hyperparameters (App. A.2).**
  - Adam (0.9, 0.999, eps 1e-8), lr 1e-4, weight decay 5e-4, LoRA dropout 0.1, **rank 32**.
  - Both temporal-loss coefficients are 1.
  - **16 frames, 384×384, 8 fps**.
  - **400 steps (single video) / 1000 steps (multiple videos)**, fp16.
  - **A5000: ~8 min (single) / ~20 min (multi)**.
  - Inference: 30 DDPM steps, CFG 12.
  — [arXiv 2310.08465 App. A.2](https://arxiv.org/abs/2310.08465)
- **Parameters and memory.** On ZeroScope (>1.8B parameters), each spatial LoRA set adds about 9M parameters and each temporal set about 12M. Training needs **14 GB VRAM**. — [arXiv 2310.08465 §4.3](https://arxiv.org/abs/2310.08465)
- **README numbers differ from the paper.** The README says multi-video training "usually takes 300~500 steps, about 9~16 minutes using one A5000", single-video training "50~150 steps, about 1.5~4.5 minutes", "required VRAM … around 14GB". The function default in the training script is `lora_rank=16`, versus 32 in the paper. — [showlab/MotionDirector README](https://github.com/showlab/MotionDirector); [train script](https://github.com/showlab/MotionDirector/blob/main/MotionDirector_train.py)
- **Data and benchmarks.**
  - Multi-video: an adapted UCF Sports Action set with **95 videos across 12 motions (4–20 videos each)** and 72 test prompts with appearance changes.
  - Single-video: **LOVEU-TGVE-2023 (76 videos × 4 editing prompts + 3 extra prompts with large appearance changes)**.
  - Base models: ModelScope T2V and ZeroScope.
  — [arXiv 2310.08465 §4, App. A.3](https://arxiv.org/abs/2310.08465)
- **Metrics.**
  - "Appearance diversity" = mean CLIP score between the new prompts and all frames.
  - "Temporal consistency" = mean inter-frame CLIP similarity.
  - Mean PickScore.
  - MTurk pairwise human preference on appearance diversity, temporal consistency and **motion fidelity** (5 raters per video).
  - There is **no automatic motion-fidelity metric**.
  — [arXiv 2310.08465 §4.1](https://arxiv.org/abs/2310.08465)
- **AnimateDiff port.** "ExponentialML/AnimateDiff-MotionDirector" (Apache-2.0):
  - Implements MotionDirector for AnimateDiff on SD1.5. Only the **v3 motion module** has been tested.
  - "Roughly 10-15 minutes to converge on roughly 14GB of VRAM" at the "preferred" setting.
  - Rule of thumb: "300 steps per video … with AdamW", and single videos at 500–700 steps.
  - Recommends 3–5 similar videos for multi-video training.
  — [README](https://github.com/ExponentialML/AnimateDiff-MotionDirector)
- **Port code details.** Temporal LoRA target `TemporalTransformerBlock`. Function default `lora_rank=8`. The spatial path is randomly masked with probability 0.2 (`mask_spatial_lora = random.uniform(0,1) < 0.2`). It uses the same β=1 debiased loss. — [AnimateDiff-MotionDirector train.py](https://github.com/ExponentialML/AnimateDiff-MotionDirector/blob/main/train.py)

### Inferences
- The project's current motion LoRAs (all-temporal vs probe-selected vs opposite) use plain MSE, so their deltas likely absorb appearance as well as motion. Adding the debiased term (about 4 lines of code, verified above) is the cheapest change that makes "temporal LoRA ⇒ motion" a cleaner claim. It is also a natural 2×2 factor: {plain MSE, debiased MSE} × {layer subset}.
- The spatial-path LoRA, trained on single frames, gives the "change appearance, keep motion" demo for free. The reverse demo, "keep appearance, change motion", comes from swapping the temporal LoRA while keeping the spatial LoRA or the prompt.
- **Budget fit.** 14 GB at 384² puts full MotionDirector at the edge of a 16 GB T4 and over 8 GB. The project already trains at 256²×16 frames on its hardware, so MotionDirector-style training at 256² with fp16, gradient checkpointing and 8-bit Adam should fit both GPUs. This is an inference, not measured. The T4 is roughly 2–4× slower than an A5000 (my estimate, unverified), which suggests about 20–80 min per run. At ~30 GPU-h/week that allows several dozen runs, enough for multiple seeds per condition.

### Gaps
- The full author list was only partly confirmed from the PDF text. The ECCV page numbers were not retrieved.
- The paper's Table 1 numbers could not be reliably attributed from the text extraction, so no numeric baselines are quoted.
- There is no reported VRAM figure at 256² or on a T4.

---

## 2. VMC: Video Motion Customization (Jeong et al., CVPR 2024): motion distillation loss on consecutive-frame noise residuals; which layers are tuned

### Takeaway
VMC fine-tunes **only the temporal-attention Q/K/V projections of the keyframe UNet** of the cascaded Show-1 model. This is full-weight tuning of those matrices, not LoRA. Its loss aligns *frame-to-frame residuals* of the predicted noise with those of the true noise, using **cosine distance**. Training uses an "appearance-invariant" prompt that strips background descriptions. It runs for 400 steps at lr 1e-4, takes 15 GB VRAM and under 5 min on an RTX 6000. At inference VMC starts from **DDIM-inverted latents** (75 steps), so it is partly an editing method.

### Cited Findings
- **Citation (verified).** Hyeonho Jeong, Geon Yeong Park, Jong Chul Ye. "VMC: Video Motion Customization using Temporal Attention Adaption for Text-to-Video Diffusion Models." CVPR 2024, pp. 9212–9221. arXiv:2312.00845. — [mlanthology](https://mlanthology.org/cvpr/2024/jeong2024cvpr-vmc/); [arXiv](https://arxiv.org/abs/2312.00845); [repo HyeonHo99/Video-Motion-Customization, Apache-2.0](https://github.com/HyeonHo99/Video-Motion-Customization)
- **Motion vectors.** Frame residual `δv_t^n := v_t^{n+c} − v_t^n`, with stride c = 1. Similarly `δε_t^n = ε_t^{n+c} − ε_t^n` and `δε_{θ,t}^n := ε_θ^{n+1}(v_t^{1:N}, t) − ε_θ^n(v_t^{1:N}, t)`. Since `δv_t^n = sqrt(ᾱ_t) δv_0^n + sqrt(1−ᾱ_t) δε_t^n`, aligning the denoised motion vectors (via Tweedie) is equivalent to aligning noise residuals:
  `ℓ_align(δv_0^n, δv̂_0^n(t)) = ((1−ᾱ_t)/ᾱ_t) · || δε_t^n − δε_{θ,t}^n ||²` (Eq. 16). — [arXiv 2312.00845 §3.1](https://arxiv.org/abs/2312.00845)
- **Final objective.** The authors switch to cosine distance:
  `min_θ E_{t,n,ε_t^n,ε_t^{n+c}} [ ℓ_cos(δε_t^n, δε_{θ,t}^n) ]`, with `ℓ_cos(x,y) = 1 − ⟨x,y⟩/(||x|| ||y||)` (Eq. 17). They say cosine "may further improve the distillation". — [arXiv 2312.00845 §3.1](https://arxiv.org/abs/2312.00845)
- **Layers.** "We exclusively fine-tune the temporal attention layers θ_TA ⊂ θ". Concretely, the appendix says "we finetune Query, Key, Value projection matrices W^Q, W^K, W^V of temporal attention layers of the keyframe UNet". The interpolation and super-resolution modules are frozen. — [arXiv 2312.00845 §3.1, App. D](https://arxiv.org/abs/2312.00845)
- **Appearance-invariant prompts.** Example: "A bird is flying above a lake in the forest" becomes "A bird is flying", to suppress background and appearance learning. — [arXiv 2312.00845 §3.2](https://arxiv.org/abs/2312.00845)
- **Training setup.**
  - AdamW, weight decay 0.01, lr 1e-4, **400 steps**, mixed precision.
  - Keyframes are 8 frames at low resolution (64×40 latent keyframes are shown).
  - 15 GB VRAM, under 5 min, single NVIDIA RTX 6000.
  - Inference (29 frames at 576×320) needs 18 GB and about 12 min.
  - DDIM inversion uses 75 steps.
  — [arXiv 2312.00845 §3, §4.1, App. D](https://arxiv.org/abs/2312.00845)
- **Data and metrics.**
  - 24 videos: 13 from DAVIS, 10 from WebVid, 1 from LAMP.
  - CLIP text alignment (prompt vs frames), CLIP frame consistency (mean pairwise cosine of frame features), and a user study.
  - Baselines are depth-conditioned methods (VideoComposer, Gen-1, and others).
  — [arXiv 2312.00845 §4.2](https://arxiv.org/abs/2312.00845)
- **Layer ablation.** Fine-tuning (a) temporal attention vs (b) spatial self+cross attention: "both frameworks exhibit proficient motion learning capabilities", with temporal giving "smoother frame transitions". This comparison is qualitative only. — [arXiv 2312.00845 §4.3](https://arxiv.org/abs/2312.00845)

### Inferences
- The VMC loss is model-agnostic for ε-prediction models such as AnimateDiff/SD1.5. It needs only the per-frame noise predictions the training loop already computes, so it can replace or augment MSE in the project's LoRA runs at no extra memory cost.
- VMC's ablation suggests a caution: *spatial* layers can also learn motion. A probe-selected temporal subset therefore needs a random-subset control, not just an "opposite set" control, to support a localisation claim.
- VMC's inversion-initialised inference makes it hard to compare with free generation. If the VMC loss is adopted, generate from pure noise so motion fidelity reflects what the weights learned.

### Gaps
- The VRAM figure is for Show-1's 64×40 keyframes. No figure was found for AnimateDiff at 256².
- The CLIP and user-study numbers in Table 1 were not extracted cleanly.

---

## 3. DreamVideo, Customize-A-Video, Motion Inversion, MotionCrafter, Spectral Motion Alignment, and 2025–2026 follow-ups

### Takeaway
Every method follows the same pattern: put motion-specific parameters only in or near temporal layers, and block appearance from flowing into them. The blocking mechanism differs:
- **DreamVideo**: conditions the motion adapter on a CLIP image embedding of a frame.
- **Customize-A-Video**: pre-trains "appearance absorbers" (spatial LoRA and/or a textual-inversion token).
- **Motion Inversion**: learns embeddings with no spatial dimension, or takes their frame differences.
- **MotionCrafter**: adds a KL term against a frozen copy of the model.
- **SMA**: aligns the frequency content of motion vectors.

Most relevant 2024–2026 follow-ups:
- **CustomTTT (AAAI 2025)**: on AnimateDiff, picks *specific* temporal layers by a B-LoRA-style injection analysis.
- **"Separate Motion from Appearance" (2025)**: a probing study leads it to apply LoRA to the temporal Key projection only, and only in early denoising steps.
- **DeT (ICCV 2025)**: a benchmark and hybrid motion-fidelity metric.
- **CMC (arXiv, Oct 2026)**: changes the objective to reduce leakage.

### Cited Findings
**DreamVideo**
- **Citation (verified).** Yujie Wei, Shiwei Zhang, Zhiwu Qing, Hangjie Yuan, Zhiheng Liu, Yu Liu, Yingya Zhang, Jingren Zhou, Hongming Shan. "DreamVideo: Composing Your Dream Videos with Customized Subject and Motion." CVPR 2024, pp. 6537–6549. arXiv:2312.04433. — [arXiv](https://arxiv.org/abs/2312.04433); [mlanthology](https://mlanthology.org/cvpr/2024/wei2024cvpr-dreamvideo/); code inside [ali-vilab/VGen](https://github.com/ali-vilab/VGen) (no SPDX licence detected by the GitHub API).
- **Subject learning.** Two steps. First, textual inversion of "S*" (~3000 iterations, lr 1e-4). Second, an **identity adapter** (~800 iterations, lr 1e-5). The adapter is a bottleneck: `h' = h + σ(h W_down) W_up`, GELU activation, W_up zero-initialised, hidden dimension half of the input. It is placed in **cross-attention layers**. — [arXiv 2312.04433 §3.2, App. A.1](https://arxiv.org/abs/2312.04433)
- **Motion adapter.** Placed in **all layers of the temporal transformer**. It adds appearance guidance: a CLIP image embedding `e` of a random training frame, broadcast across frames. `ĥ^e_t = ĥ_t + broadcast(e W_cond)`; `ĥ'_t = ĥ_t + σ(ĥ^e_t W_down) W_up` (Eqs. 3–4). Training is ~1000 iterations at lr 1e-5, with the standard ε-MSE loss. — [arXiv 2312.04433 §3.2, §4.1](https://arxiv.org/abs/2312.04433)
- **Layer-placement analysis.** Uses weight change rate `Δ_l = ||θ'_l − θ_l||₂ / ||θ_l||₂` after full fine-tuning. When learning a subject, **cross-attention** changes most. When learning motion, "all parameters contribute close to importance". This analysis decides the adapter placement. — [arXiv 2312.04433 §3.3](https://arxiv.org/abs/2312.04433)
- **Setup.**
  - Base model: ModelScopeT2V, further fine-tuned on about 30k internal text-video pairs.
  - 32 frames, 256×256, 8 fps.
  - One A100: ~15 min (single video) or ~30 min (multiple videos) per motion; ~12 min for the identity adapter.
  - Data: 20 subjects and 30 motion patterns from the Internet, UCF101, UCF Sports and DAVIS; 42 prompts.
  - Metrics: CLIP-T, CLIP-I, DINO-I, and temporal consistency.
  - Baselines include fine-tuning AnimateDiff's motion module.
  — [arXiv 2312.04433 §4.1, App. A](https://arxiv.org/abs/2312.04433)

**Customize-A-Video**
- **Citation (verified).** Yixuan Ren, Yang Zhou, Jimei Yang, Jing Shi, Difan Liu, Feng Liu, Mingi Kwon, Abhinav Shrivastava. "Customize-A-Video: One-Shot Motion Customization of Text-to-Video Diffusion Models." ECCV 2024. arXiv:2402.14780. — [arXiv](https://arxiv.org/abs/2402.14780); [mlanthology](https://mlanthology.org/eccv/2024/ren2024eccv-customizeavideo)
- **Method.** **Temporal LoRA (T-LoRA)** on temporal cross-frame attention only. Stage 1 trains "appearance absorbers" on unordered frames, with all temporal layers bypassed:
  - S-LoRA on spatial self- and cross-attention: `L = E||ε − ε_{θ0+ΔθS}(x_t^f, t, τ(y_S))||`
  - and/or a textual-inversion token: `L = E||ε − ε_{θ0}(x_t^f, t, τ_{v0+Δv}(y_S))||`
  Stage 2 freezes the absorbers and trains T-LoRA on the full clip with the full caption: `L = E||ε − ε_{θ'+ΔθT}(x_t^{1..F}, t, τ_{v'}(y))||`. Inference loads **only the T-LoRA**. — [arXiv 2402.14780 §3](https://arxiv.org/abs/2402.14780)
- **Hyperparameters.**
  - T-LoRA rank 4 (α = 1); S-LoRA rank 1 (α = 0.5).
  - lr 5e-4 for T-LoRA and 5e-5 for the absorbers.
  - ModelScope base; 2 s at 8 fps, 256×256.
  - One RTX A5000 in half precision: T-LoRA ~7 min, S-LoRA ~0.5 min, textual inversion ~1 min.
  - MotionDirector takes 8 min on the same device.
  — [arXiv 2402.14780 App. B](https://arxiv.org/abs/2402.14780)
- **Ablation and parameter-count note.** S-LoRA alone or ST-LoRA (spatial+temporal) "significantly impairs the motion modeling", and models with spatial LoRAs "memorize the video by its spatial layout". This evidence is qualitative (Fig. 4). The authors also note that MotionDirector adds LoRA to the temporal FFNs as well, which gives "approximately twice the quantity of parameters" at the same rank. They therefore compare at the same rank *and* at comparable parameter count. — [arXiv 2402.14780 §4.2, App. B](https://arxiv.org/abs/2402.14780)
- **Data and metrics.**
  - Videos from LOVEU-TGVE-2023, WebVid-10M and DAVIS; quantitative results on 53 LOVEU videos.
  - CLIPScore text alignment.
  - Temporal consistency as LPIPS between consecutive frames.
  - Diversity as LPIPS between samples from different seeds.
  - A user study.
  — [arXiv 2402.14780 §4](https://arxiv.org/abs/2402.14780)

**Motion Inversion (motion embeddings)**
- **Citation.** Luozhou Wang, Ziyang Mai, Guibao Shen, Yixun Liang, Xin Tao, Pengfei Wan, Di Zhang, Yijun Li, Ying-Cong Chen. "Motion Inversion for Video Customization." arXiv:2403.20193. **SIGGRAPH 2025** per Adobe Research's listing; not checked against the ACM DL. — [arXiv](https://arxiv.org/abs/2403.20193); [Adobe Research](https://research.adobe.com/publication/motion-inversion-for-video-customization); repo [EnVision-Research/MotionInversion](https://github.com/EnVision-Research/MotionInversion) (formerly wileewang/MotionInversion; the GitHub API redirects).
- **Embeddings.** Each temporal attention module i gets a Motion Query-Key embedding `m^QK_i ∈ R^{1×N×C}`, with no spatial dimensions, and a Motion Value embedding `m^V_i ∈ R^{(H×W)×N×C}`:
  `TA_i(F) = softmax( W_q(F+m^QK_i) (W_k(F+m^QK_i))ᵀ / sqrt(d_k) ) · W_v(F+m^V_i)` (Eq. 3).
  The embeddings are zero-initialised and trained with the standard loss `M* = argmin_M E ||ε_t − ε_θ(x_t, t, M)||²` (Eq. 4) while the model stays frozen. The method also supports VMC's and MotionDirector's losses, and the latter "can boost our performance". — [arXiv 2403.20193 §3](https://arxiv.org/abs/2403.20193)
- **Inference debiasing.** Differencing: `m̃^V[:,j] = m^V[:,j] − m^V[:,j−1]` for j > 1 (Eq. 5). — [arXiv 2403.20193 §3](https://arxiv.org/abs/2403.20193)
- **Setup and results.**
  - Main experiments use ZeroScope. The README lists ZeroScope, ModelScope and VideoCrafter2 code, and says AnimateDiff was used in the paper. Training is "under 10 minutes".
  - Evaluation: CLIP text similarity; the **Motion Fidelity Score of Yatim et al.** (tracklets); CLIP temporal consistency; FID against frames of 89 DAVIS videos; a user study with 121 participants.
  - 66 video-prompt pairs from 22 videos.
  - The method beats MotionDirector on motion fidelity and trails VMC slightly on temporal consistency.
  — [arXiv 2403.20193 §4](https://arxiv.org/abs/2403.20193); [repo README](https://github.com/EnVision-Research/MotionInversion)

**MotionCrafter**
- **Citation.** Yuxin Zhang, Fan Tang, Nisha Huang, Haibin Huang, Chongyang Ma, Weiming Dong, Changsheng Xu. "MotionCrafter: One-Shot Motion Customization of Diffusion Models." arXiv:2312.05288 (v2, Jan 2024). **No peer-reviewed venue was found.** — [arXiv](https://arxiv.org/abs/2312.05288); [repo zyxElsa/MotionCrafter, Apache-2.0](https://github.com/zyxElsa/MotionCrafter)
- **Method.**
  - Parallel spatial-temporal fine-tuning: a spatial loss on one random frame, and a temporal loss on all frames that tunes temporal modules in the down, mid and up blocks.
  - A **dual-branch** setup with a frozen copy of the U-Net produces "normalization videos".
  - Appearance normalisation loss: `L_appearance = D_KL(q_θ(z_t|x_t) || p(ẑ_t))`.
  - Total: `L_motion = L_temporal + β L_appearance` (Eqs. 3–4).
  - Base model: ZeroScope.
  - Metrics: CLIP diversity, consistency and motion fidelity, plus a user study.
  — [arXiv 2312.05288 §3–4](https://arxiv.org/abs/2312.05288)

**Spectral Motion Alignment (SMA)**
- **Citation (verified).** Geon Yeong Park, Hyeonho Jeong, Sang Wan Lee, Jong Chul Ye. "Spectral Motion Alignment for Video Motion Transfer Using Diffusion Models." AAAI 2025, 39(6):6398–6405. arXiv:2403.15249. — [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/32685); [arXiv](https://arxiv.org/abs/2403.15249); [repo geonyeong-park/Spectral-Motion-Alignment](https://github.com/geonyeong-park/Spectral-Motion-Alignment)
- **Loss.** SMA is an add-on regulariser for any motion-distillation objective:
  `min_θ E[ ℓ_align(δv_0^n, δv̂_0^n(t)) + λ_g ℓ_global(δv_0, δv̂_0(t)) + λ_l ℓ_local(δv_0^n, δv̂_0^n(t)) ]` (Eq. 18).
  - `ℓ_global` = L1 distance between **1-D discrete wavelet** coefficients of the motion vectors along time.
  - `ℓ_local` = low-frequency-weighted L1 on the **2-D Fourier** amplitude and phase of each frame residual.
  - Shown with VMC (Show-1, ZeroScope), MotionDirector, VideoComposer-style methods and DMT.
  - Data: 30 text-video pairs from DAVIS and WebVid.
  — [arXiv 2403.15249 §3–4](https://arxiv.org/abs/2403.15249)

**2024–2026 follow-ups (decoupled appearance/motion, AnimateDiff-relevant)**
- **CustomTTT.** Xiuli Bi, Jian Lu, Bo Liu, Xiaodong Cun, Yong Zhang, Weisheng Li, Bin Xiao. "CustomTTT: Motion and Appearance Customized Video Generation via Test-Time Training." AAAI 2025 (per arXiv comments).
  - **Base model: AnimateDiff.**
  - Layer selection: swap text embeddings at individual cross-attention layers, following B-LoRA. The paper chooses spatial layers 2 and 6 for appearance and **temporal layers 2 and 5 for motion**. Per the fetched summary, the text also calls layers 2 and 4 most critical for motion, which looks like an internal inconsistency.
  - Training: rank 32; lr 1e-5 (spatial) / 5e-5 (temporal); 500 steps each; Lion optimiser; 16 frames at 256²; one A6000.
  - Test-time training: 30 steps at lr 1e-6.
  - Ablation: LoRAs at other layers (spatial 0,1 / temporal 1,7) "failed to customize".
  - Comparison: a "Full LoRA" baseline with 28.26M parameters vs 12.12M for the selected layers, which scores lower on most metrics.
  - Motion data: 18 sets from UCF101, UCF Sports and DAVIS.
  — [arXiv 2412.15646](https://arxiv.org/abs/2412.15646); [HTML v2](https://arxiv.org/html/2412.15646v2). Index-to-block mapping not given; details obtained via a summarising fetch, so **verify against the PDF before quoting numbers**.
- **Separate Motion from Appearance.** Huijie Liu, Jingyun Wang, Shuai Ma, Jie Hu, Xiaoming Wei, Guoliang Kang. "Separate Motion from Appearance: Customizing Motion via Customizing Text-to-Video Diffusion Models." arXiv:2501.16714 (v2 Sep 2025; no venue listed).
  - A probing study (Table 1, motion-classifier-scored) leads them to **adapt only the Key projection** of temporal attention with LoRA ("temporal attention purification").
  - "Appearance highway": skip connections start from the spatial-attention output, scaled by β = 1.1, applied at inference.
  - "Phased LoRA integration": the adapted model is used only for early denoising steps (t > τ).
  - Base model: ZeroScope.
  - Metrics: CLIP text alignment, temporal consistency, aesthetic score, ViCLIP, VBench average, and a human study.
  - Small gains over MotionDirector, e.g. one-shot VBench average 76.33 vs 75.95.
  — [arXiv 2501.16714](https://arxiv.org/abs/2501.16714); [HTML v2](https://arxiv.org/html/2501.16714v2). Details obtained via a summarising fetch.
- **DeT.** Qingyu Shi et al. "Decouple and Track: Benchmarking and Improving Video Diffusion Transformers for Motion Transfer." ICCV 2025. Targets DiTs (full 3D attention): a temporal smoothing kernel plus dense-trajectory supervision. Introduces **MTBench** and a hybrid global+local motion-fidelity metric. — [arXiv 2503.17350](https://arxiv.org/abs/2503.17350); [mlanthology](https://mlanthology.org/iccv/2025/shi2025iccv-decouple)
- **CMC.** Youngyoon Choi, Kihyun Kim, Jeongwoo Shin, Joonseok Lee. "Diverse Motion Customization via Control-based Dynamic Optimization." arXiv:2610.07911 (6–7 Oct 2026, preprint). It attributes content leakage to "direct regression" onto the reference. It proposes stochastic-optimal-control training with a **timestep-adaptive motion cost targeting early generative stages** and claims 2.5× faster training. — [arXiv](https://arxiv.org/abs/2610.07911)
- **Other relevant work.**
  - MotionClone (ICLR 2025): training-free; uses the principal components of temporal-attention maps, visualised from AnimateDiff's `up_blocks.1`. — [arXiv 2406.05338](https://arxiv.org/abs/2406.05338); [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bc82dbfbfa43232be85b8d9838f49c3e-Abstract-Conference.html)
  - VideoMage (CVPR 2025, multi-subject + motion): [CVF PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_VideoMage_Multi-Subject_and_Motion_Customization_of_Text-to-Video_Diffusion_Models_CVPR_2025_paper.pdf)
  - Follow-Your-Motion (2025, DiT, spatial-temporal decoupled LoRA): [arXiv 2506.05207](https://arxiv.org/abs/2506.05207)
  - For VideoMage and Follow-Your-Motion, only titles and abstract-level claims were seen.
- **AnimateDiff MotionLoRA (the baseline the project is extending).** LoRA on the motion module's self-attention layers. "As few as 20 ∼ 50 reference videos, 2,000 training iterations (around 1 ∼ 2 hours) … about 30M storage". With **N = 5 videos, quality degrades** and MotionLoRA "relies on capturing texture information from the reference videos". — [Guo et al., arXiv 2307.04725 §4.3, §5.3](https://arxiv.org/abs/2307.04725) (ICLR 2024 per my knowledge; the venue was not re-checked here).

### Inferences
- **Fit for an 8 GB / T4 budget, best to worst:**
  1. MotionDirector-style loss on AnimateDiff. Code exists and is verified.
  2. VMC or SMA losses as drop-in alternatives. Pure loss changes, no extra memory.
  3. Customize-A-Video absorbers. Cheap, but no verified code repo was found.
  4. Motion Inversion embeddings. Tiny and fast, but embeddings are not LoRA, so they do not fit a matched-rank layer comparison. The H×W-sized value embeddings make the parameter count resolution-dependent.
  5. DreamVideo adapters. Portable, but the training code targets ModelScope and adds a CLIP image encoder.
  6. MotionCrafter. A second frozen U-Net roughly doubles activation memory, so it is a poor fit for 8 GB.
- Three independent works localise motion customisation to a *subset* of temporal parameters and report equal or better results than tuning everything: CustomTTT (specific temporal layers), Separate-Motion (Key projection only) and Customize-A-Video (temporal attention, not FFN). This supports the project's hypothesis framing. None of them includes a parameter-matched **random-subset** control, so that control would be a real contribution.
- The project's probing found motion selectivity in the first 2–3 of 25 steps. That matches the "early steps" emphasis in Separate-Motion's phased integration and CMC's early-stage motion cost. A second probe-derived factor is cheap to test: restrict LoRA training timesteps to the high-noise range, or apply the LoRA only for early sampling steps.

### Gaps
- No public code was found for Customize-A-Video: `yorkucvil/Customize-A-Video` returned 404. The project page customize-a-video.github.io was not checked.
- MotionCrafter training hyperparameters (steps, lr, GPU) were not found in the text.
- CustomTTT and Separate-Motion numbers come from summarised HTML fetches, not line-by-line PDF reads.
- No 2026 peer-reviewed paper on AnimateDiff-specific layer-restricted motion LoRA was found.

---

## 4. LoRA basics, parameter count for matched-budget comparisons, and work comparing fine-tuning of different layer subsets

### Takeaway
A rank-r LoRA on a d_out×d_in matrix adds r(d_in + d_out) parameters. In AnimateDiff v1-5-2 under diffusers, each motion module has two temporal self-attentions, each with four C×C projections. LoRA on q/k/v/out therefore costs **16·r·C per motion module**. All 21 motion modules together cost **302,080·r** (≈9.67M at r = 32). **up_blocks.0 alone is 61,440·r** (≈1.97M at r = 32, 20.3% of the total).

Diffusion "which layers" evidence:
- **B-LoRA**: 2 of 11 SDXL transformer blocks, found by prompt-injection probing, separate content from style.
- **CustomTTT**: temporal layers 2 and 5 in AnimateDiff.
- **DreamVideo**: cross-attention dominates the weight change for appearance.
- **Separate-Motion**: Key-only.

### Cited Findings
- **LoRA (verified).** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. "LoRA: Low-Rank Adaptation of Large Language Models." ICLR 2022. arXiv:2106.09685.
  - `h = W0 x + ΔW x = W0 x + B A x`, with B ∈ R^{d×r}, A ∈ R^{r×k}, B zero-initialised, output scaled by α/r.
  - Parameter count when LoRA is applied to L̂ square d_model matrices: `|Θ| = 2 × L̂_LoRA × d_model × r`.
  - On GPT-3, adapting W_q and W_v together at rank 4 beat spending the budget on one matrix at a higher rank.
  — [arXiv 2106.09685 §4, §7.1, App.](https://arxiv.org/abs/2106.09685) The author list beyond Hu, Shen, Wallis and Allen-Zhu is from memory; the first four were read from the PDF.
- **AnimateDiff v1-5-2 motion adapter config (verified).**
  - `block_out_channels [320, 640, 1280, 1280]`
  - `motion_layers_per_block 2`
  - `use_motion_mid_block true`, `motion_mid_block_layers_per_block 1`
  - `motion_num_attention_heads 8`, `motion_attention_bias false`, `motion_cross_attention_dim null`
  — [guoyww/animatediff-motion-adapter-v1-5-2 config.json](https://huggingface.co/guoyww/animatediff-motion-adapter-v1-5-2/blob/main/config.json)
- **diffusers module structure (verified).** `AnimateDiffTransformer3D` builds `BasicTransformerBlock`s with `double_self_attention=True` by default, so attn1 and attn2 are both temporal self-attention. Down blocks have `motion_layers_per_block` motion modules and up blocks have one more. — [diffusers unet_motion_model.py](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unets/unet_motion_model.py)
- **B-LoRA (verified).** Yarden Frenkel, Yael Vinker, Ariel Shamir, Daniel Cohen-Or. "Implicit Style-Content Separation using B-LoRA." ECCV 2024. arXiv:2403.14572.
  - SDXL's attention layers are grouped into **11 transformer blocks**.
  - Injecting a different prompt into a single block shows that blocks 2 and 4 dominate content.
  - Joint LoRA on **ΔW⁴ (content) and ΔW⁵ (style)** separates the two.
  - Rank 64, 1000 steps, about 10 min per image on one A100.
  - Training two blocks rather than all also reduces overfitting.
  — [arXiv 2403.14572](https://arxiv.org/abs/2403.14572); [mlanthology](https://mlanthology.org/eccv/2024/frenkel2024eccv-implicit)
- **DreamVideo weight-change-rate analysis** (`Δ_l = ||θ'_l − θ_l||/||θ_l||`): cross-attention dominates for subjects, while all temporal parameter types contribute about equally for motion. — [arXiv 2312.04433 §3.3](https://arxiv.org/abs/2312.04433)
- **CustomTTT.** Selected temporal layers (2, 5) succeed, while other placements (1, 7) "failed to customize". The selected set uses 12.12M parameters vs 28.26M for Full LoRA and scores better on most metrics. — [arXiv 2412.15646](https://arxiv.org/abs/2412.15646)
- **Separate Motion from Appearance.** A probing table over Q/K/V adaptation leads to Key-only temporal LoRA. — [arXiv 2501.16714](https://arxiv.org/abs/2501.16714)
- **Customize-A-Video.** Explicitly flags that equal rank does not mean equal parameters across LoRA placements, and compares at both. — [arXiv 2402.14780 App. B](https://arxiv.org/abs/2402.14780)
- **General LoRA layer-selection literature.** Mostly LLM-focused 2026 preprints (similarity-metric and gradient-guided layer selection). The search did not surface a broad head-to-head study of U-Net blocks for video-diffusion fine-tuning. — [arXiv 2602.05988](https://arxiv.org/abs/2602.05988); [arXiv 2604.15351](https://arxiv.org/abs/2604.15351). Seen via search snippets only, not verified.

### Inferences
**(a) LoRA parameter counts for AnimateDiff v1-5-2.** This is my arithmetic from the verified config and module structure: r(C+C) per projection × 4 projections × 2 attentions = **16rC per motion module**. FFN, proj_in and proj_out are excluded.

| Block | Motion modules × channels | ΣC | Parameters at r = 32 |
|---|---|---|---|
| down0 | 2 × 320 | 640 | 0.33M |
| down1 | 2 × 640 | 1,280 | 0.66M |
| down2 | 2 × 1280 | 2,560 | 1.31M |
| down3 | 2 × 1280 | 2,560 | 1.31M |
| mid | 1 × 1280 | 1,280 | 0.66M |
| **up0** | **3 × 1280** | **3,840** | **1.97M** |
| up1 | 3 × 1280 | 3,840 | 1.97M |
| up2 | 3 × 640 | 1,920 | 0.98M |
| up3 | 3 × 320 | 960 | 0.49M |
| **Total** | 21 modules | **18,880** | **9.67M** |

**(b) Matched-budget designs.** At equal rank, an up0-only LoRA has 20.3% of the full-temporal parameters. Two ways to match:
- **Fixed budget of about 1.97M.** Full-temporal at r ≈ 6.5 (r = 6 gives 1.81M, r = 7 gives 2.11M) vs up0 at r = 32.
- **Fixed budget of about 9.67M.** up0 at r ≈ 157 (C = 1280, so feasible) vs all temporal layers at r = 32.

The first is cheaper and avoids very high ranks.

**(c) Exactly matched random controls.** There are 11 motion modules with C = 1280: down2 ×2, down3 ×2, mid ×1, up0 ×3, up1 ×3. Any 3 of these give exactly the same parameter count and width as up0. Excluding up0's 3 modules, C(8,3) = **56 random, exactly matched subsets** remain. up1 is the natural "adjacent block" control. Block-level matched sets also exist: down1+down2, mid+down2, down1+down3. Sample 3–5 random subsets × 2–3 seeds to get a null distribution, and report where the probe-selected subset falls (permutation-style p-value). This is the "better than random layer subsets" test the project wants.

**(d) Asymmetric controls.** The project's probing found selectivity in up0 while patching usefulness peaked in the outer decoder blocks (up2/up3). So "selectivity-selected" (up0) and "patching-selected" (up2+up3, ΣC = 2,880; match rank via r × 3840/2880) can be tested as two hypotheses against the same random null.

**(e) Projection granularity.** Separate-Motion's Key-only finding suggests a cheap extra arm: q/k-only vs v/out-only LoRA within up0. This tests *what* in up0 carries motion.

### Gaps
- I found no paper that runs a parameter-matched, random-layer-subset control for motion LoRA in video diffusion. This appears to be an open niche, though absence of evidence from a limited search is not proof.
- The exact SDXL module names of B-LoRA's blocks 4 and 5 were not confirmed from the paper text. The official code reportedly uses `up_blocks.0.attentions.0/1`, but that is unverified here.

---

## 5. Evaluating motion customisation: metrics and public action recognisers (Hugging Face repo ids)

### Takeaway
The standard automatic set is:
- **CLIP text-frame alignment** (appearance/edit fidelity)
- **CLIP inter-frame consistency**
- **Motion Fidelity Score** (Yatim et al.): Chamfer-style max normalised cross-correlation of CoTracker tracklets between the reference and the generated video
- optionally **VBench** dimensions (motion_smoothness, dynamic_degree, human_action) and PickScore/ImageReward
- human 2AFC

For checking "is the intended action present", Kinetics-400 classifiers are a poor fit for run/walk/jump/turn: K400 has no plain "walk" or "run" class. HMDB51 has all four classes, and X-CLIP HMDB few-shot checkpoints and zero-shot X-CLIP exist on Hugging Face.

### Cited Findings
- **Motion Fidelity Score.** Danah Yatim, Rafail Fridman, Omer Bar-Tal, Yoni Kasten, Tali Dekel. "Space-Time Diffusion Features for Zero-Shot Text-Driven Motion Transfer." arXiv:2311.17009. CVPR 2024 per my knowledge; **venue not re-verified here**.
  - Tracklets T = {τ_1..τ_n} (input) and T̃ = {τ̃_1..τ̃_m} (output) come from **CoTracker** (Karaev et al.).
  - `MFS = (1/m) Σ_{τ̃∈T̃} max_{τ∈T} corr(τ,τ̃) + (1/n) Σ_{τ∈T} max_{τ̃∈T̃} corr(τ,τ̃)`
  - `corr(τ,τ̃) = (1/F) Σ_k (v_k^x ṽ_k^x + v_k^y ṽ_k^y) / ( sqrt((v_k^x)²+(v_k^y)²) · sqrt((ṽ_k^x)²+(ṽ_k^y)²) )`, where v_k is the frame-k displacement.
  - The authors designed it because flow-field comparison is "unsuitable" when structure changes.
  - Evaluated on 54 video-edit pairs from 21 DAVIS videos, with 2AFC on 7000 judgments.
  — [arXiv 2311.17009 §5](https://arxiv.org/abs/2311.17009); [repo](https://github.com/diffusion-motion-transfer/diffusion-motion-transfer)
- **Adoption of MFS.** Motion Inversion uses it as its main motion metric. — [arXiv 2403.20193 §4](https://arxiv.org/abs/2403.20193)
- **Hybrid metric.** DeT adds a global+local hybrid motion-fidelity metric with MTBench. — [arXiv 2503.17350](https://arxiv.org/abs/2503.17350)
- **VBench.** Ziqi Huang et al., CVPR 2024 (Highlight, per repo). arXiv:2311.17982. 16 dimensions: subject_consistency, background_consistency, temporal_flickering, motion_smoothness, dynamic_degree, aesthetic_quality, imaging_quality, object_class, multiple_objects, **human_action**, color, spatial_relationship, scene, temporal_style, appearance_style, overall_consistency.
  - human_action uses **UMT** on 100 Kinetics-400 actions.
  - dynamic_degree uses **RAFT** optical flow.
  - `--mode=custom_input` evaluates your own videos on subject/background consistency, motion_smoothness, dynamic_degree, aesthetic_quality and imaging_quality.
  — [arXiv 2311.17982](https://arxiv.org/abs/2311.17982); [Vchitect/VBench, Apache-2.0](https://github.com/Vchitect/VBench)
- **Other metric conventions in this literature.**
  - CLIP-T / CLIP-I / DINO-I (DreamVideo).
  - LPIPS consecutive-frame consistency and LPIPS diversity (Customize-A-Video).
  - PickScore (MotionDirector).
  - ViCLIP, aesthetic score and VBench average (Separate-Motion).
  - A motion classifier used as a probing score (Separate-Motion).
  — [DreamVideo](https://arxiv.org/abs/2312.04433); [Customize-A-Video](https://arxiv.org/abs/2402.14780); [MotionDirector](https://arxiv.org/abs/2310.08465); [Separate-Motion](https://arxiv.org/abs/2501.16714)
- **Kinetics-400 label check (verified).** The id2label of `MCG-NJU/videomae-base-finetuned-kinetics` has 400 classes. Labels containing run/walk/jump/turn are only: bungee jumping, cartwheeling, high jump, jogging, jumping into pool, jumpstyle dancing, long jump, riding or walking with horse, running on treadmill, ski jumping, somersaulting, spinning poi, triple jump, walking the dog. **No plain "walk", "run" or "turn".** — [config.json](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/blob/main/config.json)
- **Hugging Face repo ids** (all verified to exist via the HF API on 2026-10-24; pipeline tag `video-classification` unless noted):

| Repo id | Architecture | Licence | Notes |
|---|---|---|---|
| `MCG-NJU/videomae-base-finetuned-kinetics` | VideoMAEForVideoClassification | CC-BY-NC-4.0 | 16 frames |
| `MCG-NJU/videomae-small-finetuned-kinetics` | VideoMAE | CC-BY-NC-4.0 | |
| `MCG-NJU/videomae-large-finetuned-kinetics` | VideoMAE | CC-BY-NC-4.0 | |
| `MCG-NJU/videomae-huge-finetuned-kinetics` | VideoMAE | — | listed in search; not individually checked |
| `MCG-NJU/videomae-base-finetuned-ssv2` | VideoMAE | CC-BY-NC-4.0 | Something-Something v2 (motion-centric) |
| `facebook/timesformer-base-finetuned-k400` | TimeSformer | CC-BY-NC-4.0 | |
| `facebook/timesformer-base-finetuned-ssv2` | TimeSformer | CC-BY-NC-4.0 | |
| `google/vivit-b-16x2-kinetics400` | ViViT | MIT | |
| `facebook/vjepa2-vitl-fpc16-256-ssv2` | VJEPA2ForVideoClassification | MIT | |
| `OpenGVLab/VideoMAEv2-Base` | VideoMAEv2 | CC-BY-NC-4.0 | |
| `microsoft/xclip-base-patch32` | X-CLIP | MIT | |
| `microsoft/xclip-base-patch16-zero-shot` | X-CLIP | MIT | **zero-shot with arbitrary text labels such as "a person walking"** |
| `microsoft/xclip-base-patch16-kinetics-600` | X-CLIP | — | |
| `microsoft/xclip-base-patch16-hmdb-{2,4,8,16}-shot` | X-CLIP | MIT | 8-shot card self-reports **62.8% top-1 on HMDB-51** (`verified: False`) |
| `nateraw/videomae-base-finetuned-ucf101` | VideoMAE | MIT | 101 labels, 16 frames, 224 px; accuracy not stated on card |
| `arielb30/videomae-base-finetuned-hmdb51_dataset` | VideoMAE | — | community; low downloads; unvetted |
| `dronefreak/mc3-18-hmdb51-kinetics` | MC3-18 | — | community; low downloads; unvetted |

  — [HF API / model pages](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics), e.g. [X-CLIP zero-shot](https://huggingface.co/microsoft/xclip-base-patch16-zero-shot), [X-CLIP HMDB 8-shot](https://huggingface.co/microsoft/xclip-base-patch16-hmdb-8-shot), [nateraw UCF101](https://huggingface.co/nateraw/videomae-base-finetuned-ucf101)
- **HMDB51 classes.** The class list includes **run, walk, jump, turn**, alongside cartwheel, somersault, handstand, climb stairs, and others. — [divm/hmdb51 dataset card](https://huggingface.co/datasets/divm/hmdb51)

### Inferences
- For a *specific* customised motion (one-shot or few-shot), MFS against the reference clip(s) is the most defensible primary metric. CoTracker runs fine on a T4 for 16-frame clips (an inference, not measured).
- Pair MFS with CLIP-T on prompts with new subjects, which measures appearance leakage and editability, and with CLIP-I against reference frames, where *lower* is better and indicates less appearance copying. This yields the motion-vs-leakage trade-off plot that all papers above use.
- For *class-level* actions (run/walk/jump/turn), use:
  - `microsoft/xclip-base-patch16-zero-shot` with the four HMDB class names as text prompts, or
  - the HMDB few-shot X-CLIP checkpoints, which were trained on HMDB-51 label space and so share it.

  Avoid K400-only classifiers for these four classes. Validate the chosen classifier on the real HMDB training clips and some base-model AnimateDiff generations before trusting it. Base-model accuracy is the floor, and the LoRA must beat it.
- **Why the current LoRAs showed no effect.** The base AnimateDiff already produces run/walk/jump, so a classifier has little headroom. The AnimateDiff paper itself notes MotionLoRA with N = 5 videos degrades into texture copying. Two fixes:
  - a motion that is **out-of-distribution for the base model**, such as a specific camera trajectory, a specific choreography, cartwheels or somersaults; or
  - a **one-shot reference-trajectory** target scored by MFS rather than a class label.

  MotionDirector explicitly motivates customisation for motions "that lie in the long-tailed distribution that can not be synthesized well by pre-trained models". — [arXiv 2310.08465](https://arxiv.org/abs/2310.08465)

### Gaps
- Accuracies for HMDB/UCF community checkpoints are unverified, and X-CLIP's HMDB score is self-reported on the card.
- No published accuracy was found for any classifier on *AnimateDiff-generated* clips.
- VBench's human_action uses UMT weights distributed through the VBench toolkit; the exact HF repo id was not checked.

---

## 6. Datasets for few-shot motion customisation with clean, cut-free clips (licences and sizes)

### Takeaway
Benchmarks in this literature reuse:
- **DAVIS** (short, single-shot, object-centric clips; VMC, DMT, Motion Inversion, SMA)
- **LOVEU-TGVE-2023** (76 clips with captions and editing prompts; MotionDirector, Customize-A-Video, Separate-Motion)
- **UCF Sports Action** (curated to 95 clips × 12 motions in MotionDirector)
- **WebVid** clips
- **UCF101 / HMDB51** subsets (DreamVideo, CustomTTT)

For a T4 budget, DAVIS-2017 480p and UCF101/HMDB51 subsets are small and easy to get. Licences are mostly non-commercial or research-only, which is fine for a course project.

### Cited Findings
- **MotionDirector.** Multi-video uses UCF Sports Action (original 150 videos / 10 motions, curated to 95 videos / 12 motions with 4–20 clips each, plus inconsistent and low-resolution clips removed). Single-video uses LOVEU-TGVE-2023 (76 videos × 4 prompts). — [arXiv 2310.08465 App. A.3](https://arxiv.org/abs/2310.08465); LOVEU track page cited by the paper: [sites.google.com/view/loveucvpr23/track4](https://sites.google.com/view/loveucvpr23/track4) (not fetched).
- **VMC.** 24 videos: 13 DAVIS, 10 WebVid, 1 LAMP. — [arXiv 2312.00845 §4.2](https://arxiv.org/abs/2312.00845)
- **DMT.** Driving videos from DAVIS; 21 videos / 54 edit pairs. — [arXiv 2311.17009](https://arxiv.org/abs/2311.17009)
- **Motion Inversion.** DAVIS, WebVid and online clips; 22 videos / 66 pairs; FID reference from 89 DAVIS videos. — [arXiv 2403.20193](https://arxiv.org/abs/2403.20193)
- **SMA.** 30 text-video pairs from DAVIS and WebVid-10M. — [arXiv 2403.15249](https://arxiv.org/abs/2403.15249)
- **DreamVideo.** 30 motion patterns from the Internet, UCF101, UCF Sports and DAVIS. **CustomTTT.** 18 motion sets from UCF101, UCF Sports and DAVIS. — [arXiv 2312.04433](https://arxiv.org/abs/2312.04433); [arXiv 2412.15646](https://arxiv.org/abs/2412.15646)
- **DAVIS 2017.** Official downloads: semi-supervised TrainVal, Test-Dev and Test-Challenge, at 480p and full resolution. The official page states no licence and only asks for citation. An HF mirror `emirkisa/DAVIS-2017-480p-mp4` (verified) is tagged **CC-BY-NC-4.0**, has under 1K items, and is in mp4 form. — [davischallenge.org](https://davischallenge.org/davis2017/code.html); [HF mirror](https://huggingface.co/datasets/emirkisa/DAVIS-2017-480p-mp4)
- **HMDB51.**
  - Official Serre Lab release on HF, `Serrelab/hmdb51`, **CC-BY-4.0**.
  - **6,849 clips / 51 classes**; `hmdb51_org.rar` about **2 GB**; a **stabilised** version `hmdb51_sta.rar`; 3 official splits.
  - Another card (`divm/hmdb51`) says **6,766 clips**. The clip count conflicts between cards; 6,766 is the commonly used usable count.
  — [Serrelab/hmdb51](https://huggingface.co/datasets/Serrelab/hmdb51); [divm/hmdb51](https://huggingface.co/datasets/divm/hmdb51)
- **UCF101.** HF mirrors exist: `flwrlabs/ucf101`, `quchenyuan/UCF101-ZIP`, `MichiganNLP/ucf-101`, and the small `sayakpaul/ucf101-subset` (Apache-2.0 tag, contains `UCF101_subset.tar.gz`). The official CRCV page could not be fetched (TLS certificate error), so size and licence are unverified here. — [sayakpaul/ucf101-subset](https://huggingface.co/datasets/sayakpaul/ucf101-subset)
- **MotionDirector repo assets.** The repo distributes trained LoRA weights for its demos (download section in the README). No separate curated benchmark-video download was found in the README. — [showlab/MotionDirector README](https://github.com/showlab/MotionDirector)

### Inferences
- HMDB51 clips are movie and web footage, often with cuts, camera shake and low resolution. That likely contributed to the project's LoRAs learning "generic actions" rather than a crisp motion. The stabilised HMDB release, and filtering by HMDB's meta-labels for camera motion and quality, would help. The meta-label claim is from my knowledge of the original paper and was not re-verified.
- **One-shot arm.** 3–5 DAVIS clips with a distinctive motion (e.g. breakdance-flare, parkour, dance-twirl, car-turn). These are short, cut-free and single-shot, and MFS was designed and validated on DAVIS.
- **Few-shot arm.** 4–8 hand-picked, cut-free clips of one long-tail action (e.g. HMDB "cartwheel" or "somersault", or UCF101 "JumpingJack"), checked by eye for single shots.
- Both arms fit a T4 easily: dozens of 16-frame clips at 256².
- **Licence caveat.** DAVIS mirrors are tagged CC-BY-NC, and the VideoMAE/TimeSformer weights are CC-BY-NC. That is acceptable for coursework, not for commercial use.

### Gaps
- LOVEU-TGVE-2023 download size, licence and current availability were not verified (the Google Sites page was not fetched).
- DAVIS-2017 480p TrainVal size was not stated on the official page.
- UCF101 official size and licence were not verified (fetch failed).
- MTBench (DeT) licence and size were not checked.
