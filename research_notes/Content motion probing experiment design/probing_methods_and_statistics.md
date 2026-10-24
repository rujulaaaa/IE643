# Probing, representational-analysis and statistics methods for a content-vs-motion layer study of AnimateDiff

Scope: methods for (a) non-linear probes with control tasks, (b) rigorous layer comparison, (c) deciding which module/timestep "stores" content vs motion, for a factorial clip design (15 subjects x 4 verbs x 2 scenes x 3 seeds = 360 clips; 120 clips per seed; ~60 training clips per cross-scene fold), 95 hooked modules x 25 denoising steps, 512-d count sketches.

Notation used throughout: n = number of clips in an analysis; k = sketch dimension; K = number of classes; a "cell" = one (subject, verb, scene, seed) clip. "Selectivity" is used in two senses below. **Hewitt-selectivity** is task accuracy minus control accuracy. **Study-S** is the study's own S = (d_c - d_m)/(d_c + d_m). Keep them apart in the write-up.

Verification method: arXiv metadata was pulled from the arXiv API. Journal metadata (authors, venue, volume, pages) was pulled from the Crossref API by DOI. Key equations were read from the full text of Hewitt & Liang, Voita & Titov, Pimentel et al., Kornblith et al., Weinberger et al. (all on arXiv) and Kobak et al. (eLife HTML). Items not verified this way are marked **[UNVERIFIED]**.

---

## Q1. Control tasks and selectivity for probes (Hewitt & Liang 2019); building a control task for a factorial design; probe-complexity recommendations

### Takeaway
A control task gives each *type* (word type in NLP) a random label. The probe can only solve it by memorising, and Hewitt-selectivity = task accuracy - control accuracy. A bijective relabelling of the very factor being probed (for example a random permutation of subject IDs for a subject probe) is **not** a control task. It is equally hard, so its accuracy is uninformative. In the factorial design the control label must be a random function of an identity that is shared across the train/test split but carries no task structure. The (subject, verb, seed) triple is the natural "type". Hewitt & Liang found that small hidden layers and weight decay make MLP probes selective, while dropout and early stopping do not. With about 60 training clips, use linear or very narrow MLP probes with strong L2.

### Cited Findings
- **Definition of a control task.** Control tasks have "structure: The output for a word token is a deterministic function of the word type" and "randomness: The output for each word type is sampled independently at random." A control behaviour C(v) is sampled for every type v in V, and the task is f_control(x_{1:T}) = f(C(x_1), C(x_2), ..., C(x_T)) (Eq. 1). The control task shares the linguistic task's output space Y. — [Hewitt & Liang, EMNLP 2019, arXiv:1909.03368](https://arxiv.org/abs/1909.03368)
- **Choice of label distribution.** For POS, labels are sampled "from the empirical token distribution of part-of-speech tagging, so the marginal probability of each label is similar"; the authors say "the exact distribution from which we sample isn't crucial". — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Ceiling on control accuracy.** "the ceiling on performance is the fraction of tokens in the evaluation set whose types occur in the training set (plus chance accuracy on all other tokens.)" The control mapping "must be memorized independently for each word type". — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Selectivity.** "Selectivity is defined as the difference between linguistic task accuracy and control task accuracy." A good probe "should be what we call selective, achieving high linguistic task accuracy and low control task accuracy." In symbols: Sel = Acc_task - Acc_control (percentage points). — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Default-hyperparameter result (ELMo layer 1, POS).** The linear probe reached 97.2 accuracy with 26.0 selectivity. The MLP reached 97.3 accuracy with 4.5 selectivity. On dependency edges, the bilinear probe reached 89.0 accuracy with 16.7 selectivity, MLP-1 reached 92.3 accuracy with -0.7 selectivity, and MLP-2 had 1.3 selectivity. "Bilinear and linear probes themselves show a significant capacity for memorization." — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Complexity-control sweep.** Each setting was varied from a default of rank/hidden size 1000, no dropout and no weight decay:
  - rank/hidden size l in {2, 4, 10, 45} for POS and {5, 10, 50, 100} for dependency;
  - dropout p in {0.2, 0.4, 0.6, 0.8} on the inputs and on each hidden layer;
  - training-set size at about 100%, 10% and 1%;
  - weight decay in {0.01, 0.1, 1.0, 10.0};
  - early stopping at a maximum of {50k, 25k, 12.5k, 6k, 3k, 1.5k} steps.

  Of the six methods, "all worked except dropout and early stopping, though never for a broad range of hyperparameters." "Constraining the hidden state dimensionality of MLPs is an effective way to encourage selectivity at little cost to linguistic task accuracy." Dropout 0.4 *reduced* MLP-2 selectivity (4.2 -> 3.4). Constraining the number of training examples was effective for POS. — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Hand-picked selective settings.** POS probes "designed with control tasks all use rank-10 weight matrices (10-dimensional hidden state)". For dependency: bilinear with weight decay 0.01, MLP-1 with weight decay 0.1, MLP-2 with a rank-50 weight. — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Layer comparison changes once selectivity is considered.** On POS, the linear probe on ELMo2 has selectivity 31.4 versus 26.0 on ELMo1, while its accuracy is 96.6 on ELMo2 versus 97.2 on ELMo1. The authors' "alternative hypothesis [is] that probes use word identity as a feature to predict part-of-speech, and that feature is less easily available in ELMo2 than ELMo1." — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Probes as craftspeople.** The paper suggests probes "be thought of as craftspeople; their performance depends not only on the materials they're given, but also on their" expressivity. — [Hewitt & Liang 2019](https://arxiv.org/abs/1909.03368)
- **Survey framing.** Belinkov's squib reviews probing's shortcomings, including baselines/controls, probe complexity, and the gap between correlation and causation (whether the model *uses* decodable information). Metadata: Computational Linguistics 48(1):207-219, 2022. — [Belinkov 2022, Computational Linguistics](https://doi.org/10.1162/coli_a_00422)
- **Causal follow-up.** Amnesic probing tests whether a property is *used* by removing it from the representation and measuring the change in behaviour; it argues that decodability alone does not show use. TACL 9:160-175, 2021. — [Elazar, Ravfogel, Jacovi & Goldberg 2021](https://doi.org/10.1162/tacl_a_00359)

### Inferences
- **Mapping to the factorial design.** The NLP "word type" corresponds to an identity that recurs across the train/test split. In the cross-scene split, every (subject, verb, seed) triple appears once in each scene, so that triple is the type. Recommended control tasks:
  1. **Memorisation control (Hewitt-style).** y_ctl = C(subject, verb, seed), with C drawn i.i.d. from the task's empirical label marginal and the same C used for both scenes. The 4 classes (verb probe) or 15 classes (subject probe) are balanced. The ceiling is 100% because every test type was seen in training. Chance is 1/K. Control accuracy measures how well the probe can learn an arbitrary mapping of clip identities that transfers across scenes. That is exactly the capacity that inflates task accuracy.
  2. **Seed/noise-identity control.** y_ctl = C(seed, subject), or C of anything carried by the fixed per-seed noise. This tests whether a probe can exploit the shared initial noise. Clips that share a seed share their noise, so early-step activations may be dominated by it.
  3. **Cross-factor control (a different question from Hewitt's).** For the verb probe, use y_ctl = C(subject), a random 4-way function of subject. High accuracy here means the layer makes subject identity linearly separable through a 4-class bottleneck. That is a confound check, not a capacity check.
  4. **Not a control.** A random bijection of the probed factor's own labels: identical task, identical accuracy.

  Draw R >= 20 random C's and report the mean and SD of control accuracy, because with only 45-180 types a single draw is noisy.
- **Bias in Hewitt-selectivity at small n.** Task and control accuracies are each evaluated on about 60 test clips. Binomial SE is about sqrt(p(1-p)/60), roughly 0.065 at p = 0.5. Hewitt-selectivity therefore has an SE of roughly 0.09 if the two are independent. Differences between layers of less than about 0.15-0.2 should not be interpreted without the permutation/bootstrap machinery in Q3/Q6.
- **Probe recipe for about 60 training clips (15 subjects means about 4 examples per class).**
  - Fit StandardScaler and PCA inside each training fold only, to avoid leakage. With n_train = 60, PCA rank is at most 59.
  - Use L2 logistic regression, with C chosen by nested CV (leave-one-subject-out or leave-one-seed-out within the training scene).
  - For the non-linear probe, use a one-hidden-layer MLP with h in {4, 8, 16} (following the rank-10 finding) and weight decay alpha in {0.1, 1, 10}. No dropout reliance.
  - Report task accuracy, control accuracy and Hewitt-selectivity for both probe families at every layer x timestep. Interpret "MLP > linear" only where MLP *selectivity* is also high.
- **Content/motion confound specific to diffusion.** Every hooked module downstream of a cross-attention layer receives the prompt tokens directly. Decoding "verb" may therefore reflect text-token identity rather than generated motion. The study-specific control is the cross-attention/text pathway itself. For example, probe the same module at t = T (pure noise plus text) or with the motion module disabled; this is an inference, not from a cited source.
- **Analogy to the ELMo1/ELMo2 result.** A layer with higher raw verb accuracy may simply expose "which prompt" more easily. Rank layers by selectivity, or by gain over a control (Q2), not by accuracy.

### Gaps
- Hewitt & Liang give no guidance for continuous-valued or factorial vision datasets, or for n < 100. All of the design mapping above is inference.
- No published study was found that applies control tasks to diffusion-model activations. None was searched for exhaustively; this is a scope limit.
- ACL Anthology ID D19-1275 for Hewitt & Liang is from memory **[UNVERIFIED]**. The arXiv record (comment "EMNLP 2019") is verified.

---

## Q2. MDL / online-code probing (Voita & Titov 2020) and information-theoretic probing (Pimentel et al. 2020); is MDL usable at n of about 120?

### Takeaway
Online-code MDL replaces accuracy with the number of bits needed to transmit the labels when the probe is trained incrementally on growing prefixes. Codelength rewards representations from which the label is learnable with few examples. Online-code MDL is cheap with logistic regression and usable at n of about 120 *as a relative measure* between layers. It needs coarse block boundaries (Voita's 0.1% first block is less than one clip here) and averaging over many data orderings. Pimentel et al.'s mutual-information view implies that "how much information about the verb" is uninformative in a text-conditioned generator. The label is a deterministic function of the conditioning prompt, so the data-processing inequality caps it and every layer could in principle carry it. That argues for ease-of-extraction measures such as MDL and selectivity, not information content.

### Cited Findings
- **Data codelength under a model:** L_p(y_{1:n} | x_{1:n}) = - sum_{i=1}^n log2 p(y_i | x_i) (Eq. 1), which equals cross-entropy in bits. **Uniform baseline:** L_unif = n log2 K. — [Voita & Titov, EMNLP 2020, arXiv:2003.12298](https://arxiv.org/abs/2003.12298)
- **Information bound.** "the gain in codelength over the trivial codelength H(y) is H(y) - E_q[L(y|x)] <= H(y) - H(y|x) = I(y; x)." Total codelength includes model cost, so "a good representation is the one which also yields simple models predicting y from x." — [Voita & Titov 2020](https://arxiv.org/abs/2003.12298)
- **Online (prequential) code.** Choose timesteps 1 = t_0 < t_1 < ... < t_S = n. Send the first block with a uniform code. For each later block, train p_{theta_i} on examples 1..t_i and encode block t_i+1..t_{i+1} with it:

  L_online(y_{1:n} | x_{1:n}) = t_1 log2 K - sum_{i=1}^{S-1} log2 p_{theta_i}( y_{t_i+1 : t_{i+1}} | x_{t_i+1 : t_{i+1}} )

  The equation was read from the text. — [Voita & Titov 2020](https://arxiv.org/abs/2003.12298)
- **Compression** = L_unif / L_online = n log2 K / L_online. — [Voita & Titov 2020](https://arxiv.org/abs/2003.12298)
- **Block sizes.** "In all experiments in this paper, the timesteps correspond to 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100 percent of" the dataset. — [Voita & Titov 2020](https://arxiv.org/abs/2003.12298)
- **Probe used.** MLP-2 of Hewitt & Liang with default hyperparameters (h = 1000, no dropout), with appendix results for several h. Codelength is reported as "stable across settings", unlike accuracy. — [Voita & Titov 2020](https://arxiv.org/abs/2003.12298)
- **Control tasks under MDL.** "codelengths for control tasks are substantially larger than for the linguistic task." For control tasks "model size is several times larger"; codelength "characterizes how hard it is to achieve this accuracy". Results were also reported over 5 random seeds. — [Voita & Titov 2020](https://arxiv.org/abs/2003.12298)
- **Variational code.** This alternative uses Bayesian compression (Louizos et al. 2017) and needs a Bayesian network implementation. The authors note "the online code is easier to implement." — [Voita & Titov 2020](https://arxiv.org/abs/2003.12298)
- **Pimentel et al.: probing as MI estimation.** The paper argues against the "received wisdom that one should always select simple" probes, and introduces control functions c(.) of the representation. The gain is G(T, R, c) = I(T; R) - I(T; c(R)) = H(T | c(R)) - H(T | R) >= 0, and Proposition 1 shows G = I(T; R | c(R)). G is estimated by variational bounds, using probe cross-entropy as an upper bound on conditional entropy. A further result (Theorem 1): "BERT(S) cannot provide more information about T than the sentence S itself." Empirically, "BERT only encodes at most 12% more information than a simple baseline (a type-level representation)" on the dependency-label task they analysed. — [Pimentel, Valvoda, Hall Maudslay, Zmigrod, Williams & Cotterell, ACL 2020, arXiv:2004.03061](https://arxiv.org/abs/2004.03061)

### Inferences
- **Online code at n of about 120 (one seed) or 360 (all seeds), with logistic regression.**
  - Use geometric block boundaries with a first block of at least K examples, stratified so every class appears. For a 4-way verb probe, t = {8, 16, 32, 64, 120}. For a 15-way subject probe, t = {15, 30, 60, 120}.
  - The first block always costs t_1 log2 K bits regardless of layer (about 16 bits for verb and about 59 bits for subject). Compare layers on L_online - t_1 log2 K, or report compression with the same blocks everywhere.
  - Average over at least 20 random stratified orderings. Report mean and SD of codelength, and of task-minus-control codelength.
  - Cost: each logistic-regression fit on at most 120 x 512 takes milliseconds. 95 modules x 25 steps x 20 orderings x 4-5 blocks x (1 task + about 20 controls) is roughly 5M fits, so subsample controls (for example 5) or run on a coarse timestep grid first. All of this runs on CPU.
- **The cross-scene requirement does not fit the online code.** Prequential coding uses random orderings, not a scene split. Order the data so that each block is drawn from one scene before the other (all of scene A, then scene B). Then the later-block codelength measures transfer. This is a design suggestion, not from the cited papers.
- **Pimentel's argument applied here.** The verb/subject label is a deterministic function of the prompt, and every U-Net module is a function of (noise, prompt). By the data-processing inequality I(T; R) <= H(T), and with an expressive enough probe, any layer that sees the text embedding could approach H(T). Using the prompt's text embedding as the control function would drive G toward 0 everywhere. "Where is motion stored" is therefore a question about *accessibility/geometry* (linear decodability, MDL, selectivity, variance share) or about *causal use*, not about information content.

### Gaps
- No study was found that validates MDL probing at n of about 100. The stability claims in Voita & Titov come from datasets with thousands of examples. Small-n behaviour is an untested inference.
- ACL Anthology IDs (2020.emnlp-main.14; 2020.acl-main.420) were not fetched **[UNVERIFIED]**. The arXiv records are verified, and the arXiv comment for Pimentel et al. states "Accepted for publication at ACL 2020."
- The exact form of Pimentel et al.'s upper and lower variational bounds (Corollaries 2-3) was not transcribed.

---

## Q3. Permutation tests for probe accuracy (Ojala & Garriga 2010) and how to get p-values

### Takeaway
Use Ojala & Garriga's "Test 1": retrain the probe on label-permuted data B times, and compute p = (1 + #{acc_perm >= acc_obs}) / (B + 1). In this design the permutation must respect the factorial structure. Permute the probed factor's labels *within blocks of the other factors*, and apply the same relabelling to both scenes. Otherwise exchangeability fails and the null is too lenient or too strict.

### Cited Findings
- Ojala & Garriga, "Permutation Tests for Studying Classifier Performance", JMLR 11:1833-1863, 2010. — [JMLR](https://www.jmlr.org/papers/v11/ojala10a.html)
  - **Test 1** "checks whether the classifier has found a real class structure", with the null distribution obtained by permuting labels.
  - **Test 2** permutes features within classes, testing whether the classifier exploits dependencies between features.
- The p-value for randomly drawn permutations should be computed as (b + 1)/(B + 1), where b is the number of permutations at least as extreme. This keeps p > 0 and gives exact Type-I control. — [Phipson & Smyth 2010, Stat. Appl. Genet. Mol. Biol. 9(1)](https://doi.org/10.2202/1544-6115.1585) (title verified: "Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn"; formula from memory **[UNVERIFIED in full text]**)

### Inferences
- **Restricted (blocked) permutation for the factorial design.**
  - **Verb probe:** inside each (subject, seed) block, apply a random permutation of the 4 verb labels, and use the same permutation for scene A and scene B. That preserves the cross-scene type structure and the balance across subjects/seeds.
  - **Subject probe:** inside each (verb, seed) block, permute the 15 subject labels.
  - **Scene probe:** inside each (subject, verb, seed) block, swap or keep the scene label.

  The number of distinct permutations is huge (4!^45 for verb), so B = 1000 random draws suffices. The smallest attainable p is 1/1001.
- **Cost.** Each permutation retrains the probe: 95 x 25 x 1000 logistic-regression fits, roughly 2.4M small fits. That is CPU-feasible in hours, and the cost multiplies for MLPs. A two-stage approach saves effort: permutation-test only a coarse grid (for example every 5th timestep) or the cells that survive the cluster test (Q6).
- **Family-wise testing across the 2-D map.** Reuse the same label permutation for all 2375 module-timestep cells within one permutation draw. The permutation distribution of the *maximum* accuracy (or cluster mass) then gives FWER control directly; see Q6.
- **Testing selectivity.** The null for "task accuracy > control accuracy" is not a label permutation. Use a bootstrap or paired sign-flip over test units (subjects) of per-unit (task-correct - control-correct). Alternatively, compare the observed task accuracy to the distribution of control accuracies over many control draws C.
- **Test 2 in this context.** Permuting sketch features within class asks whether the probe relies on feature covariation. This is of limited value with PCA in front of the probe.

### Gaps
- The full text of Ojala & Garriga was not read, so details such as their small-sample recommendations were not checked.

---

## Q4. Comparing layers: CKA (linear, RBF, debiased), SVCCA, RSA with multiple-regression / variance partitioning

### Takeaway
At n = 60-120 and p = 512 (or full width), **standard (biased) linear CKA is unusable**. Two *independent* random matrices give CKA of about 0.90 at n = 60, p = 512 in our simulation. Use **debiased CKA**, which is about 0 for independent data. CCA and SVCCA are invariant to invertible linear maps and therefore degenerate when width >= n; avoid them unless they are run on a handful of PCs. For "content vs motion", the most interpretable tool is **RSA with a multiple regression of each layer's RDM on model RDMs**, plus commonality analysis. Build categorical same-subject/same-verb/same-scene RDMs alongside CLIP/optical-flow RDMs, and test by permuting condition labels.

### Cited Findings
- **CKA (Kornblith, Norouzi, Lee & Hinton, ICML 2019).** — [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)
  - HSIC(K, L) = tr(K H L H)/(n-1)^2, with centering matrix H = I_n - (1/n) 1 1^T.
  - CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L)).
  - Linear CKA (centered columns): ||Y^T X||_F^2 / ( ||X^T X||_F ||Y^T Y||_F ).
  - RBF kernel: k(x_i, x_j) = exp(-||x_i - x_j||^2 / (2 sigma^2)), with "sigma as a fraction of the median distance between examples" (0.2, 0.4 and 0.8 shown).
  - The paper notes that Song et al. "provide an unbiased estimator" of HSIC.
- **Width >= n problem.** "any invariant similarity index gives the same result for any representation of width greater than or equal to the dataset size." This applies to similarity indices invariant to invertible linear transformation (CCA, SVCCA), and motivates CKA. — [Kornblith et al. 2019](https://arxiv.org/abs/1905.00414)
- **Debiased CKA in the small-n regime.** Neural data often sit in a "low-data high-dimensionality" regime where biased CKA gives high similarity even for random matrices. Biased CKA can be driven to its maximum with independent random data. Shuffling sample-feature pairs barely changes biased CKA. Only debiased CKA showed alignment specific to stimulus-driven responses (THINGS fMRI/MEG). — [Murphy, Zylberberg & Fyshe, ICLR 2024 Re-Align Workshop, arXiv:2405.01012](https://arxiv.org/abs/2405.01012)
- **Other CKA pitfalls.** CKA reliability problems, including sensitivity to manipulation and outliers, are covered by [Davari et al. 2022, arXiv:2210.16156](https://arxiv.org/abs/2210.16156) (venue not verified **[UNVERIFIED]**). Testing CKA and CCA against functional ground truth found that they can miss functionally relevant differences: [Ding, Denain & Steinhardt, NeurIPS 2021, arXiv:2108.01661](https://arxiv.org/abs/2108.01661).
- **SVCCA.** Raghu, Gilmer, Yosinski & Sohl-Dickstein, NIPS 2017. Step 1: SVD of each layer's (n x neurons) activation matrix, keeping the top directions. Step 2: CCA between the reduced subspaces, giving canonical correlations rho_1..rho_c, summarised by their mean. — [arXiv:1706.05806](https://arxiv.org/abs/1706.05806). The "99% variance" retention threshold is from memory **[UNVERIFIED]**. **PWCCA** (projection-weighted CCA) is the follow-up: [Morcos, Raghu & Bengio, NIPS 2018, arXiv:1806.05759](https://arxiv.org/abs/1806.05759).
- **RSA.** Kriegeskorte 2008, "Representational similarity analysis – connecting the branches of systems neuroscience", Frontiers in Systems Neuroscience 2:4. — [doi:10.3389/neuro.06.004.2008](https://doi.org/10.3389/neuro.06.004.2008). Crossref lists only N. Kriegeskorte; the paper's author list is Kriegeskorte, Mur & Bandettini from memory **[co-authors UNVERIFIED via Crossref]**. Toolbox and inference procedures, including condition-label randomisation tests and RDM comparison by rank correlation: [Nili et al. 2014, PLoS Comput Biol 10(4):e1003553](https://doi.org/10.1371/journal.pcbi.1003553). A unifying framework of encoding models, pattern-component modelling and RSA: [Diedrichsen & Kriegeskorte 2017, PLoS Comput Biol 13(4):e1005508](https://doi.org/10.1371/journal.pcbi.1005508).
- **Variance partitioning of RDMs with several model RDMs.** Used by Groen et al. 2018 (eLife 7:e32962), "Distinct contributions of functional and deep neural network features to representational similarity of scenes in human brain and behavior". — [doi:10.7554/eLife.32962](https://doi.org/10.7554/eLife.32962). Title and authors are verified. That the method is variance partitioning by multiple regression is from memory of the paper **[method detail UNVERIFIED]**.

### Inferences
- **Our simulation (scratch code; independent Gaussian X, Y; 20 draws).** True CKA is 0.

  | n | p | biased linear CKA | debiased CKA (mean +/- SD) |
  |---|---|---|---|
  | 60 | 64 | 0.515 | -0.003 +/- 0.031 |
  | 60 | 512 | 0.896 | +0.003 +/- 0.024 |
  | 60 | 4096 | 0.986 | — |
  | 120 | 512 | 0.809 | -0.003 +/- 0.011 |
  | 360 | 512 | 0.587 | — |
  | 360 | 4096 | 0.919 | — |

  Debiased CKA was within +/- 0.004 of 0 in every condition. Conclusion: comparisons between layers of different width (320/640/1280 channels x spatial x frames) with biased CKA are dominated by the p/n ratio, not by representation.
- **Debiased linear CKA formula (implemented and checked in the simulation).**
  - Let K~ = K with its diagonal set to 0 (K = X X^T, unnormalised Gram).
  - HSIC_u(K, L) = [ tr(K~ L~) + (1^T K~ 1)(1^T L~ 1)/((n-1)(n-2)) - (2/(n-2)) 1^T K~ L~ 1 ] / ( n(n-3) ).
  - dCKA = HSIC_u(K, L) / sqrt( HSIC_u(K, K) HSIC_u(L, L) ).

  The formula is commonly attributed to Song et al. 2012 (JMLR, "Feature selection via dependence maximization") **[UNVERIFIED citation]**. The simulation confirms that it is unbiased under independence. dCKA can be negative, so report it as is. For RBF CKA, set sigma as a fraction of the median pairwise distance computed *within each layer* (Kornblith).
- **CKA cost.** On sketches, CKA needs only n x n Gram matrices (n <= 360). All 95 x 95 module pairs, per timestep, are trivial on CPU. Gram matrices can also be built exactly from full activations by streaming inner products, without sketch noise.
- **SVCCA/CCA at n = 60-120.** With width >> n, CCA correlations are trivially about 1 (the Kornblith invariance argument). Do not use it, or restrict to at most about n/5 top PCs and validate on held-out clips.
- **RSA for content vs motion, multiple-regression form.**
  - Per (module, timestep), build the layer RDM D_L over N conditions. Use condition means over seeds (N = 120 conditions) or individual clips (N = 360), with cross-validated distances where possible.
  - Fit on the lower triangle, standardised or rank-transformed:

    vec(D_L) = beta_0 + beta_c vec(D_content) + beta_m vec(D_motion) + beta_s vec(D_scene) + epsilon

    Use OLS or non-negative least squares.
  - Model RDMs:
    - (i) categorical: D_subj(i, j) = 1[subject_i != subject_j], and likewise D_verb and D_scene;
    - (ii) feature-based: D_content = 1 - cos(CLIP image embedding of the mean frame); D_motion = distance between optical-flow descriptors, for example the L2 distance of time-averaged flow histograms.

    Categorical RDMs are exactly orthogonal in a balanced factorial design. Feature RDMs are correlated with each other because subject size and shape change optical flow. That correlation is why commonality analysis is needed.
- **Commonality / variance partitioning.**
  - Two predictors (c, m):
    - Unique_c = R^2_{c,m} - R^2_m
    - Unique_m = R^2_{c,m} - R^2_c
    - Common_{cm} = R^2_c + R^2_m - R^2_{c,m}
    - The common term can be negative under suppression.
  - Three predictors (c, m, s):
    - Unique_c = R^2_{cms} - R^2_{ms}
    - Common_{cm} = R^2_{cs} + R^2_{ms} - R^2_s - R^2_{cms}
    - Common_{cms} = R^2_c + R^2_m + R^2_s - R^2_{cm} - R^2_{cs} - R^2_{ms} + R^2_{cms}

  These are standard commonality-analysis identities written from the definitions, not transcribed from a cited source.
- **Inference for RSA regression.** RDM entries are not independent, so do not use parametric regression p-values. Permute condition labels (rows and columns of D_L together), restricted within seed, refit, and take the max over layers for FWER. Alternatively, bootstrap subjects.
- **Noise ceiling.** Use the seed replicates. The lower bound is the correlation of a seed's RDM with the mean RDM of the other 2 seeds; the upper bound uses the mean RDM over all 3. This places each layer's R^2 relative to the reliability its noise allows.

### Gaps
- The exact debiased-CKA formula was implemented from memory. Its source (Song et al. 2012 JMLR) was not fetched; the simulation is the only check.
- No study was found that applies CKA/RSA across the timesteps of a video diffusion U-Net with these model RDMs.
- No source was found for a recommended optical-flow RDM for gait/verb in generated video. The choice of flow descriptor (RAFT flow histograms, trajectory features) is open.

---

## Q5. Demixed PCA, ANOVA-style variance decomposition by task parameters, and disentanglement metrics (DCI, MIG)

### Takeaway
The study's design is fully balanced, which is exactly the case where Kobak et al.'s marginalisation is an exact orthogonal (ANOVA) decomposition. Per module and timestep, it gives the fraction of activation variance due to subject, verb, scene, seed, frame (time) and each interaction. The study's static/dynamic split corresponds to marginalising over the frame index: the static part comes from frame-averaged terms, and the dynamic part is everything that interacts with frame. Report this variance-fraction map with a noise/replicate correction (3 seeds means bias). dPCA proper (the decoder axes) is an optional extra. DCI and MIG add little here. They assume axis-aligned codes, and a count sketch destroys axis alignment, so use them, if at all, on channel-pooled activations.

### Cited Findings
- **Marginalisation (Kobak et al., eLife 2016).** — [eLife 5:e10989](https://doi.org/10.7554/eLife.10989); [full text](https://elifesciences.org/articles/10989); preprint [arXiv:1410.6031](https://arxiv.org/abs/1410.6031). Ten authors: Kobak, Brendel, Constantinidis, Feierstein, Kepecs, Mainen, Qi, Romo, Uchida & Machens (verified via Crossref).
  - Single-neuron decomposition, for time t, stimulus s, decision d and trial k:

    x_tsdk = x̄ + x̄_t + x̄_s + x̄_d + x̄_ts + x̄_td + x̄_sd + x̄_tsd + eps_tsdk

  - Each term is the average over the other parameters after subtracting lower-order terms:
    - x̄_t = <x_tsdk - x̄>_{sdk}
    - x̄_s = <x_tsdk - x̄>_{tdk}
    - x̄_ts = <x_tsdk - x̄ - x̄_t - x̄_s - x̄_d>_{dk}
  - In matrix form, "X = X_t + X_ts + X_td + X_tsd + X_noise". Here each parameter's terms are grouped with its time interactions; for example the stimulus marginalisation includes s and ts.
  - The covariance "is linearly decomposed into the sum of covariance matrices from each marginalization": C = sum_phi C_phi + C_noise.
- **dPCA loss.** L_phi = ||X_phi - F_phi D_phi X||_F^2, regularised as L_phi = ||X_phi - F D X||^2 + mu ||F D||^2 with mu = (lambda ||X||)^2. Closed form: A_RR = X_phi X^T (X X^T + mu I)^{-1}, then F = U_q and D = U_q^T A_RR, where U_q holds the top-q left singular vectors of A_RR X. "we used cross-validation to select the optimal value of lambda", and "a shuffling procedure" assessed whether decoding axes were above chance. — [Kobak et al. 2016](https://elifesciences.org/articles/10989)
- **Balance.** A dataset "is called balanced when there is the same number of data points for each combination of predictors". For unbalanced data, a re-balanced loss is given. — [Kobak et al. 2016](https://elifesciences.org/articles/10989)
- **Explained variance.**
  - Total: R^2 = (||X~||^2 - ||X~ - F D X~||^2)/||X~||^2.
  - Per marginalisation: R^2_phi = (||X~_phi||^2 - ||X~_phi - F D X~_phi||^2)/||X~||^2.
  - Noise correction: Theta = S Q T * sum_n C~_nn / K~_n, giving a signal fraction of 1 - Theta/||X~||^2. The authors "always reported fractions of the PSTH variance (as opposed to the total PSTH+noise variance)".

  — [Kobak et al. 2016](https://elifesciences.org/articles/10989)
- **DCI (Eastwood & Williams, ICLR 2018).** Disentanglement, completeness and informativeness are computed from a regressor's importance matrix R_ij (code dimension i, factor j) when the ground-truth factors are known. — [ICLR 2018 PDF (Williams' homepage)](https://homepages.inf.ed.ac.uk/ckiw/postscript/iclr_final.pdf); [ICLR 2018 poster page](https://iclr.cc/virtual/2018/poster/55). The equations below are from memory **[UNVERIFIED transcription]**:
  - P_ij = R_ij / sum_k R_ik
  - D_i = 1 - H_K(P_i.), with entropy in base K = number of factors
  - rho_i = sum_j R_ij / sum_ij R_ij
  - D = sum_i rho_i D_i
  - C_j = 1 - H_D(P~_.j), where P~ is normalised over code dimensions
  - I = prediction error
- **MIG (Chen, Li, Grosse & Duvenaud, NeurIPS 2018).** — [arXiv:1802.04942](https://arxiv.org/abs/1802.04942) (title and authors verified; the NeurIPS venue is from memory **[UNVERIFIED]**).

  MIG = (1/K) sum_k (1/H(v_k)) ( I(z_{j^(k)}; v_k) - max_{j != j^(k)} I(z_j; v_k) )

  Here j^(k) = argmax_j I(z_j; v_k). The formula is from memory **[UNVERIFIED transcription]**.
- **Caveat on disentanglement metrics.** Their large-scale study questions the reliability of unsupervised disentanglement and of metric agreement. — [Locatello et al., ICML 2019, arXiv:1811.12359](https://arxiv.org/abs/1811.12359)

### Inferences
- **Concrete per-layer decomposition (balanced, multivariate ANOVA via traces).**
  - For one module and timestep, let x_{s,v,c,r,f} in R^p be the activation of clip (subject s, verb v, scene c, seed r) at frame f. p can be the full activation or its sketch; inner products are approximately preserved (Q7).
  - Grand mean: x̄ = <x>_{svcrf}.
  - Main effect of subject: X_S(s) = <x>_{vcrf}(s) - x̄. Likewise X_V, X_C and X_R, and X_F(f) = <x>_{svcr}(f) - x̄.
  - Two-way terms, for example X_SV(s, v) = <x>_{crf}(s, v) - x̄ - X_S(s) - X_V(v). Continue up to the 5-way term. With one clip per cell there are no within-cell trials, so the highest-order term involving seed acts as residual/noise.
  - Variance fraction: frac_phi = sum_{cells} ||X_phi||^2 / sum_{cells} ||x - x̄||^2. These sum exactly to 1 in a balanced design.
  - Grouping for the study:
    - Content-static = S + C + S x C (frame-invariant terms).
    - Motion = V x F + S x V x F + ... (terms that vary with frame and involve verb).
    - Static-verb = V. This is verb information present in the frame mean, for example pose or posture.
    - Seed = R + its interactions; this shows how much the fixed noise dominates at each timestep.

  The study's frame-mean/residual split equals the partition {terms without F} versus {terms with F}. The ANOVA view refines it into which factor drives each part.
- **Small-replicate bias.** With only 3 seeds, the noise variance inflates each marginal. Under an i.i.d.-noise model, E||X̂_phi||^2 = ||X_phi||^2 + df_phi * sigma^2_noise per unit. A bias-corrected (omega^2-style) estimate is

  frac_phi_corr = ( SS_phi - df_phi * MS_res ) / ( SS_tot + MS_res )

  computed on traces. This is a standard ANOVA effect-size correction written from first principles **[no source verified]**. Seed is not a classic noise term here, because the same noise is reused across prompts. It is a crossed random factor, so treat seed interactions as the error term for the fixed factors (a mixed-model ANOVA).
- **Statistics for variance fractions.** Use restricted permutations as in Q3. For example, permute verb labels within (subject, seed) blocks, recompute frac_V and frac_VxF, and take the max over the module x timestep map (Q6).
- **dPCA axes.** These are optional and mostly useful for visualisation, for example a "verb" axis whose projection shows trajectory-like dynamics over frames. Choose lambda by CV across seeds. Cost is trivial at p = 512.
- **DCI/MIG.** A count sketch sums random signed subsets of coordinates into each bucket, so individual sketch dimensions are not meaningful units. Axis-based metrics (MIG, DCI-disentanglement) on sketches are uninterpretable. If wanted, compute them on per-channel activations averaged over spatial and frame axes (320-1280 channels) with the 4 discrete factors. MI for MIG can be estimated by discretisation; with n = 360 these estimates are noisy. Recommendation: the ANOVA variance fractions answer the "fraction due to subject/verb/scene/seed" question more directly.

### Gaps
- The DCI and MIG formulas were not re-read from the papers in this session (marked above).
- The Kobak et al. shuffle-test details and their CV procedure for lambda sit in the part of the eLife page that was not read.

---

## Q6. Statistics over a 2-D layer x timestep map: cluster-based permutation, BH-FDR, hierarchical bootstrap

### Takeaway
2375 correlated tests (95 modules x 25 steps). Use one of three tools depending on the claim:
- **Max-statistic or cluster-mass permutation tests** (Maris & Oostenveld) give strong FWER control for "there is an effect somewhere in this region". Adjacency is natural along timesteps and must be defined explicitly along modules.
- **BH-FDR** answers "which individual cells". Use Benjamini-Yekutieli if dependence is arbitrary.
- **Hierarchical/cluster bootstrap** gives CIs on layer-level summaries such as study-S, accuracy or variance fraction.

Choose the resampling or permutation *unit* to match the dependence structure. Seed is crossed, not nested, and has only 3 levels. Subject, with 15 levels, is the most defensible resampling unit.

### Cited Findings
- **Cluster-based permutation test (Maris & Oostenveld 2007).** "Nonparametric statistical testing of EEG- and MEG-data", J Neurosci Methods 164(1):177-190. — [doi:10.1016/j.jneumeth.2007.03.024](https://doi.org/10.1016/j.jneumeth.2007.03.024). The procedure is from memory of the paper **[detail UNVERIFIED in full text]**:
  1. Compute a test statistic at each sample.
  2. Threshold it at a cluster-forming level.
  3. Form clusters of adjacent supra-threshold samples.
  4. Cluster statistic = sum of statistics in the cluster.
  5. Compare the largest observed cluster statistic with the permutation distribution of the maximum cluster statistic.
- **TFCE.** Threshold-free cluster enhancement avoids choosing a cluster-forming threshold. — [Smith & Nichols 2009, NeuroImage 44(1):83-98](https://doi.org/10.1016/j.neuroimage.2008.03.061). The formula TFCE(p) = integral_{h=h0}^{h_p} e(h)^E h^H dh with E = 0.5 and H = 2 is from memory **[UNVERIFIED]**.
- **BH-FDR.** Benjamini & Hochberg 1995, JRSS-B 57(1):289-300. — [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)
  - Procedure: order p_(1) <= ... <= p_(m), find the largest k with p_(k) <= (k/m) q, and reject H_(1)..H_(k).
  - Under arbitrary dependence, Benjamini & Yekutieli replace q with q / sum_{i=1}^m (1/i). — [Benjamini & Yekutieli 2001, Ann. Stat. 29(4):1165-1188](https://doi.org/10.1214/aos/1013699998)
  - The procedure statement and the BY constant are from memory of these standard results; titles and metadata are verified.
- **Hierarchical bootstrap (Saravanan, Berman & Sober 2020).** "Application of the Hierarchical Bootstrap to Multi-Level Data in Neuroscience", Neurons, Behavior, Data Analysis, and Theory 3(5). — [NBDT](https://nbdt.scholasticahq.com/article/13927); [PMC7906290](https://pmc.ncbi.nlm.nih.gov/articles/PMC7906290); preprint [bioRxiv 10.1101/819334](https://doi.org/10.1101/819334). Findings, from search-result summaries of the abstract:
  - Traditional tests on nested data gave false-positive rates above 45% at nominal alpha = 5%.
  - Averaging within levels controls this but loses power.
  - Resampling with replacement sequentially at each level of the hierarchy keeps Type-I error within bounds with more power.
  - A later version compared against linear mixed models and found them comparable in simulation.

### Inferences
- **Which tool for which claim.**
  - (i) "Motion selectivity is significantly > 0 in a contiguous band of up-block temporal modules at mid timesteps": cluster-mass permutation, which gives FWER for the cluster's existence. Do not claim the cluster's precise boundaries are significant; this is a known interpretive limit of cluster tests **[source not verified in this session]**.
  - (ii) "Which specific cells are significant": BH, or BY because neighbouring cells are positively and possibly arbitrarily dependent.
  - (iii) "Module A stores more motion than module B": a paired bootstrap CI of the difference over subjects.
- **Defining adjacency for clusters.**
  - Timestep neighbours (t +/- 1) are natural.
  - For modules, use network order *within* each resolution block (down_k / mid / up_k) and treat block boundaries as non-adjacent. Optionally link each up-block module to its skip-connected down-block partner.
  - Alternatively run separate 1-D cluster tests along timestep for each module family, with max-stat correction across families.
- **Permutation scheme for maps of the study-S or of paired distances.**
  - Statistic per cell: the mean over pairs of the paired contrast, for example d_c - d_m for a matched content-pair and motion-pair sharing a base clip.
  - Null: sign-flip of contrasts, with flips made per subject (15 units, 2^15 = 32768 sign patterns; exact enumeration is possible). The same flip is applied to that subject's whole 95 x 25 map, which preserves spatial and temporal dependence.
  - For each flip, record the max |t| (max-stat FWER) and the max cluster mass.
  - For probe-accuracy maps, use the blocked label permutations of Q3 with the same permutation applied across all cells.
- **Hierarchical bootstrap in this design.**
  - The data are pairs within subject x verb x scene, crossed with seed. Seed has only 3 levels, so resampling seeds with replacement yields only 10 distinct multisets. Any CI driven by seed resampling is coarse and unreliable.
  - Recommended scheme: a two-level bootstrap. Resample subjects with replacement (15). Within each chosen subject, resample its clip pairs (verbs x scenes x seeds) with replacement. Compute the statistic, for example study-S at each cell or the argmax module. Use B >= 2000.
  - Treat seed as a fixed blocking factor: always compare within seed, then average. Use a mixed model (subject and seed random intercepts) as a cross-check, in line with Saravanan et al.'s comparison.
- **Statistics on "which layer stores X".**
  - "argmax over modules" is itself a random variable. Bootstrap it with the scheme above and report the distribution of the peak location, for example "the peak verb-x-frame variance fraction falls in up_blocks.1 temporal modules in 87% of bootstrap samples".
  - Also report the profile with a CI, not just the peak.
- **Study-S pitfall.** S = (d_c - d_m)/(d_c + d_m) is a ratio and is unstable when both distances are small, as at late timesteps or in low-variance layers. Bootstrap S with resampling at the subject level. Also report d_c - d_m and its normalisation by a noise floor separately, for example the distance between clips with identical prompts and different seeds.

### Gaps
- The full text of Maris & Oostenveld was not fetched. The exact algorithm and its Type-I claims are summarised from memory and marked.
- No literature was found on cluster tests over neural-network layer x timestep maps specifically. The adjacency choice is a design decision.
- The published NBDT page of Saravanan et al. was found only via search results, which give volume 3, issue 5, pages 1-25. The article itself was not opened.

---

## Q7. Count sketch / random projection guarantees for inner products and cosine; choosing k

### Takeaway
A count sketch with a shared (h, s) is an unbiased inner-product estimator, with variance

Var = (1/k) [ ||x||^2 ||y||^2 + <x,y>^2 - 2 sum_i x_i^2 y_i^2 ]

This is Weinberger et al.'s Lemma 2, and our simulation matches it. For unit vectors with cosine c, the cosine estimate from sketches has SD of about (1 - c^2)/sqrt(k). So k = 512 gives SD of about 0.044 at c = 0, 0.0082 at c = 0.9 and 0.0018 at c = 0.98. That is fine for probes. It may be marginal for study-S when the d's are small, for example d of about 0.02 (cos 0.98) gives a relative error of about 9% per pair. Compute paired cosine distances exactly on full activations (cheap streaming inner products), and keep sketches for probes, CKA and RSA.

### Cited Findings
- **Feature hashing (Weinberger, Dasgupta, Attenberg, Langford & Smola).** "Lemma 2: The hash kernel is unbiased, that is E_phi[<x, x'>_phi] = <x, x'>. Moreover, the variance is sigma^2_{x,x'} = (1/m) sum_{i != j} (x_i^2 x'_j^2 + x_i x'_i x_j x'_j)", where m is the hash dimension. The paper also gives exponential tail bounds for hashed inner products. — [arXiv:0902.2206](https://arxiv.org/abs/0902.2206). The ICML 2009 venue is from memory **[UNVERIFIED; arXiv record lacks venue]**.
- **Count sketch (Charikar, Chen & Farach-Colton).** "Finding frequent items in data streams", Theoretical Computer Science 312(1):3-15, 2004. This is the journal version; the conference version is ICALP 2002 **[conference venue UNVERIFIED]**. — [doi:10.1016/S0304-3975(03)00400-6](https://doi.org/10.1016/S0304-3975(03)00400-6)
- **Johnson-Lindenstrauss, elementary proof.** Dasgupta & Gupta, Random Structures & Algorithms 22(1):60-65 (published online 2002; issue 2003). — [doi:10.1002/rsa.10073](https://doi.org/10.1002/rsa.10073). The bound below is from memory of the theorem **[UNVERIFIED transcription]**: a random projection to k >= 4 (eps^2/2 - eps^3/3)^{-1} ln N dimensions preserves all pairwise squared distances among N points within (1 +/- eps) with high probability.
- **Sparse/binary JL.** ±1 (and sparse) projection matrices give JL guarantees. — [Achlioptas 2003, J. Comput. Syst. Sci. 66(4):671-687](https://doi.org/10.1016/S0022-0000(03)00025-4)

### Inferences
- **Equivalent closed form.** The sum over i != j equals (1/k)[ ||x||^2 ||y||^2 + <x,y>^2 - 2 sum_i x_i^2 y_i^2 ]. For dense activations (p of 10^5-10^6, energy spread out), the last term is negligible. Then Var(<Sx,Sy>) ≈ (||x||^2 ||y||^2 + <x,y>^2)/k, the same as a Gaussian projection.
- **Simulation check (d = 10^4, unit vectors, 300 sketch draws).** The empirical variance of <Sx,Sy> matched the Lemma 2 value within Monte-Carlo error, for example 1.85e-3 empirical versus 1.95e-3 predicted at c = 0 and k = 512. The empirical SD of the sketched cosine matched (1 - c^2)/sqrt(k), and bias was below 0.007 everywhere:

  | c | k = 128 | k = 512 | k = 2048 |
  |---|---|---|---|
  | 0 | 0.092 | 0.043 | 0.022 |
  | 0.52 | 0.067 | 0.030 | 0.0165 |
  | 0.90 | 0.0165 | 0.0086 | 0.0040 |
  | 0.98 | 0.0037 | 0.0018 | 0.0009 |
- **Choosing k.** To estimate one cosine with SD eps at true cosine c: k ≈ ((1 - c^2)/eps)^2. Examples:
  - c = 0.9, eps = 0.005: k ≈ 1444.
  - c = 0.98, eps = 0.001: k ≈ 1560.
  - c = 0.5, eps = 0.01: k ≈ 5625.

  For *all* pairwise distances among N = 360 clips to be within (1 +/- eps), the JL bound gives about k >= 4 ln 360 / (eps^2/2 - eps^3/3), roughly 5045 for eps = 0.1 and roughly 1358 for eps = 0.2. JL worst-case guarantees are conservative; per-pair variance is the practical guide.
- **Averaging over pairs.** Study-S averages over many pairs, so per-pair sketch noise shrinks roughly as 1/sqrt(#pairs). This is not exact, because one fixed sketch induces correlated errors across pairs that share clips. Sketch noise is small relative to between-pair (biological) variability unless the distances themselves are of order 1/sqrt(k).
- **Implementation must-haves.**
  - Use the *same* (h, s) for all clips, timesteps and both members of a pair; otherwise inner products are not estimated at all.
  - Use a separate hash per module, or the same one; both are fine.
  - Sketch the static (frame-mean) and dynamic (residual) parts *after* the split. The sketch is linear, so sketching first and splitting after is equivalent: S(x - x̄) = Sx - Sx̄.
  - Heavy-tailed activations (a few massive channels) raise sum x_i^2 y_i^2, which lowers variance, but they weaken tail bounds. Weinberger et al.'s tail bounds depend on the ratio ||x||_inf / ||x||_2 **[from memory, UNVERIFIED]**. Normalising each channel (z-scoring across clips) before sketching mitigates this.
- **Effect on probes, CKA and RSA.** Linear CKA on sketches equals CKA on S X, and the Gram matrix is approximated entrywise with the variance above. With k = 512 and n <= 360, Gram error is small relative to debiasing issues. Exact Gram matrices from full activations are also cheap and avoid the question.

### Gaps
- The exact constants of Weinberger et al.'s tail bound (Theorem 3) and of Charikar et al.'s count-sketch error bound were not transcribed.
- How heavy-tailed AnimateDiff activations are (and hence how large the sum x_i^2 y_i^2 correction is) is unknown and must be measured.
- Sources not verified in this session, and so not cited as facts:
  - Sassenhagen & Draschkow on cluster-test localisation;
  - Song et al. 2012 (unbiased HSIC);
  - the ACL Anthology pages for Hewitt & Liang, Voita & Titov and Pimentel et al.;
  - the NeurIPS proceedings page for Chen et al. (MIG).

### Bibliography and verification status
Verified means metadata was confirmed via the arXiv API, Crossref, or the official page in this session.
- Hewitt, J. & Liang, P. (2019). Designing and Interpreting Probes with Control Tasks. EMNLP 2019. https://arxiv.org/abs/1909.03368 — verified (arXiv comment "EMNLP 2019"; full text read).
- Voita, E. & Titov, I. (2020). Information-Theoretic Probing with Minimum Description Length. EMNLP 2020. https://arxiv.org/abs/2003.12298 — arXiv verified and full text read; the EMNLP venue is from memory.
- Pimentel, T., Valvoda, J., Hall Maudslay, R., Zmigrod, R., Williams, A. & Cotterell, R. (2020). Information-Theoretic Probing for Linguistic Structure. ACL 2020. https://arxiv.org/abs/2004.03061 — verified (arXiv comment "Accepted for publication at ACL 2020").
- Belinkov, Y. (2022). Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics 48(1):207-219. https://doi.org/10.1162/coli_a_00422 — verified (Crossref).
- Elazar, Y., Ravfogel, S., Jacovi, A. & Goldberg, Y. (2021). Amnesic Probing. TACL 9:160-175. https://doi.org/10.1162/tacl_a_00359 — verified.
- Ojala, M. & Garriga, G. C. (2010). Permutation Tests for Studying Classifier Performance. JMLR 11:1833-1863. https://www.jmlr.org/papers/v11/ojala10a.html — verified.
- Phipson, B. & Smyth, G. K. (2010). Permutation P-values Should Never Be Zero. Stat. Appl. Genet. Mol. Biol. 9(1). https://doi.org/10.2202/1544-6115.1585 — verified.
- Kornblith, S., Norouzi, M., Lee, H. & Hinton, G. (2019). Similarity of Neural Network Representations Revisited. ICML 2019. https://arxiv.org/abs/1905.00414 — verified; full text read.
- Murphy, A., Zylberberg, J. & Fyshe, A. (2024). Correcting Biased Centered Kernel Alignment Measures in Biological and Artificial Neural Networks. ICLR 2024 Re-Align Workshop. https://arxiv.org/abs/2405.01012 — verified.
- Davari, M., Horoi, S., Natik, A., Lajoie, G., Wolf, G. & Belilovsky, E. (2022). Reliability of CKA as a Similarity Measure in Deep Learning. https://arxiv.org/abs/2210.16156 — arXiv verified; venue not verified.
- Ding, F., Denain, J.-S. & Steinhardt, J. (2021). Grounding Representation Similarity with Statistical Testing. NeurIPS 2021. https://arxiv.org/abs/2108.01661 — verified (arXiv comment).
- Raghu, M., Gilmer, J., Yosinski, J. & Sohl-Dickstein, J. (2017). SVCCA. NIPS 2017. https://arxiv.org/abs/1706.05806 — verified (arXiv comment).
- Morcos, A. S., Raghu, M. & Bengio, S. (2018). Insights on representational similarity in neural networks with canonical correlation. NIPS 2018. https://arxiv.org/abs/1806.05759 — verified.
- Kriegeskorte, N. (et al.) (2008). Representational similarity analysis – connecting the branches of systems neuroscience. Front. Syst. Neurosci. 2:4. https://doi.org/10.3389/neuro.06.004.2008 — title, venue and year verified; co-authors (Mur, Bandettini) from memory.
- Nili, H. et al. (2014). A Toolbox for Representational Similarity Analysis. PLoS Comput Biol 10(4):e1003553. https://doi.org/10.1371/journal.pcbi.1003553 — verified.
- Diedrichsen, J. & Kriegeskorte, N. (2017). Representational models. PLoS Comput Biol 13(4):e1005508. https://doi.org/10.1371/journal.pcbi.1005508 — verified.
- Groen, I. I. A., Greene, M. R., Baldassano, C., Fei-Fei, L., Beck, D. M. & Baker, C. I. (2018). eLife 7:e32962. https://doi.org/10.7554/eLife.32962 — verified.
- Kobak, D. et al. (2016). Demixed principal component analysis of neural population data. eLife 5:e10989. https://doi.org/10.7554/eLife.10989 — verified; Methods read in part.
- Eastwood, C. & Williams, C. K. I. (2018). A Framework for the Quantitative Evaluation of Disentangled Representations. ICLR 2018. https://homepages.inf.ed.ac.uk/ckiw/postscript/iclr_final.pdf — verified via ICLR poster page and search results; formulas not re-read.
- Chen, R. T. Q., Li, X., Grosse, R. & Duvenaud, D. (2018). Isolating Sources of Disentanglement in VAEs. https://arxiv.org/abs/1802.04942 — arXiv verified; NeurIPS 2018 venue from memory.
- Locatello, F. et al. (2019). Challenging Common Assumptions in the Unsupervised Learning of Disentangled Representations. ICML 2019. https://arxiv.org/abs/1811.12359 — verified (journal_ref).
- Maris, E. & Oostenveld, R. (2007). Nonparametric statistical testing of EEG- and MEG-data. J Neurosci Methods 164(1):177-190. https://doi.org/10.1016/j.jneumeth.2007.03.024 — verified.
- Smith, S. M. & Nichols, T. E. (2009). Threshold-free cluster enhancement. NeuroImage 44(1):83-98. https://doi.org/10.1016/j.neuroimage.2008.03.061 — verified.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate. JRSS-B 57(1):289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x — verified.
- Benjamini, Y. & Yekutieli, D. (2001). The control of the FDR in multiple testing under dependency. Ann. Stat. 29(4):1165-1188. https://doi.org/10.1214/aos/1013699998 — verified.
- Saravanan, V., Berman, G. J. & Sober, S. J. (2020). Application of the Hierarchical Bootstrap to Multi-Level Data in Neuroscience. NBDT 3(5). https://nbdt.scholasticahq.com/article/13927 — verified via search results, PMC entry and the bioRxiv DOI 10.1101/819334.
- Charikar, M., Chen, K. & Farach-Colton, M. (2004). Finding frequent items in data streams. Theor. Comput. Sci. 312(1):3-15. https://doi.org/10.1016/S0304-3975(03)00400-6 — verified; ICALP 2002 conference version not verified.
- Weinberger, K., Dasgupta, A., Attenberg, J., Langford, J. & Smola, A. (2009). Feature Hashing for Large Scale Multitask Learning. https://arxiv.org/abs/0902.2206 — arXiv verified; Lemma 2 read; ICML 2009 venue from memory.
- Dasgupta, S. & Gupta, A. (2003). An elementary proof of a theorem of Johnson and Lindenstrauss. Random Struct. Algorithms 22(1):60-65. https://doi.org/10.1002/rsa.10073 — verified.
- Achlioptas, D. (2003). Database-friendly random projections. J. Comput. Syst. Sci. 66(4):671-687. https://doi.org/10.1016/S0022-0000(03)00025-4 — verified.
