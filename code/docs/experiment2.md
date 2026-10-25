# Experiment 2 · where AnimateDiff computes motion (novelty phase)

Design and literature: [`reports/Content motion probing experiment design.md`](../../reports/Content%20motion%20probing%20experiment%20design.md).
Pre-registered rules: [protocol.md](protocol.md) D-34 (probe controls) and D-35 (everything else).
Run log: `what_we_did.md`, sessions 5–.

| Phase | What | Script(s) | Runs on |
|---|---|---|---|
| 0 | Refined sweep views: all 117 coordinates (95 + 22 ResNets; 114 on v1-5-3) sketched, exact distances at every coordinate of `up_blocks.0/1`, channel and head statistics, subject/verb cross-attention, RAFT flow, CLIP embeddings, latent trajectory | `probe_sweep.py --exp2` | GPU |
| 1 | Quality gate over the 2×2 → primary cell and ladder verbs | `exp2_quality.py` | CPU |
| 2 | 2×2 sweep (seeds 42, 1337 × 2 scenes × 60 clips per cell); per-cell analysis; paired contrasts and cross-model CKA | `probe_sweep.py --exp2 [--config configs/variants/<cell>.yaml]`, `exp2_analysis.py`, `exp2_grid.py` | GPU, CPU |
| 3 | Content ladder (8 bases × 2 verbs × 7 variants × 2 seeds) | `probe_sweep.py --exp2 --set ladder`, `exp2_ladder.py` | GPU, CPU |
| — | Selection by the D-35 rules → Phase 5 candidates, LoRA target | `exp2_select.py` | CPU |
| 4 | LoRA timing pilot (cartwheel, debiased loss) | `train_motion_lora.py --variant modules`, `eval_motion_lora.py` | GPU |
| 5b | Patching A→B / B→A / C→B, 4 candidates × 50 pairs (seed 2024) | `exp2_patching.py --candidates` | GPU |
| 5c | LEACE vs random erasure, verb and subject | `exp2_erasure.py` | GPU |
| 6 | LoRA arms: T ×3 seeds, all-temporal (budget- and rank-matched), 19 matched random sets; MFS / X-CLIP / CLIP-T / leakage | `train_motion_lora.py`, `eval_motion_lora.py` | GPU |

Kaggle: `scripts/kaggle_run.py --user <account>`; at most 2 GPU batch sessions run at once, so
kernels are queued. The cartwheel clips are the private dataset `<account>/vdprobe-cartwheel`
(`--dataset vdprobe-cartwheel`, seen by jobs at `input/vdprobe-cartwheel`).

## New modules

- `ladder.py` (content ladder), `quality.py` (VBench subject consistency, dynamic degree, X-CLIP),
  `exp2.py` (sign-flip tests, ANOVA fractions, RDMs, flow decoding gain, channel selectivity),
  `erasure.py` (LEACE, random eraser, erase/sample probes), `motion_eval.py` (Motion Fidelity Score).
- Extended: `coords` (optional ResNet coordinates, `in_blocks`), `lockstep` (`ChannelStats`,
  `HeadStats`, sketch parts), `harness.CrossAttentionMaps`, `attention.temporal_head_stats`,
  `metrics` (RAFT, camera removal, masked flow fidelity, CLIP text), `analysis` (probe controls,
  debiased CKA, RSA commonality), `lora` (module sets, parameter counts, matched random subsets),
  `edit.run_arms(extra_probes=...)`.

## Report notes
  Limitations and context for the report (temporal identity drift, recent video models): `docs/report_notes.md`.
