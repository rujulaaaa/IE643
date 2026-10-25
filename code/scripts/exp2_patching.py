"""Experiment 2 Phase 5b · patching in both directions with an unrelated-clip control (D-35).

    python -u scripts/exp2_patching.py --candidates results/exp2/candidates.json --shard 0/4 \
        [--config configs/variants/<primary cell>.yaml] [--pairs N]

Pairs: 25 motion + 25 content pairs of the probing set at the held-out seed 2024, drawn with
numpy.default_rng(2024), each with the unrelated-clip control (D-35). For each pair (A = anchor clip, B = partner) and each candidate
{name, coords, steps, component}:
    B<-A   B's prompt receiving A's activations      sufficiency
    A<-B   A's prompt receiving B's activations      necessity
    B<-C   B's prompt receiving an unrelated clip C   disruption control
plus the plain clips A, B, C and, as the Experiment 1 baseline, B<-A temporal-attention injection
over the first third. Every output is scored with the contrastive metrics of the design report:
    D_c(x) = CLIP(x, prompt_A) - CLIP(x, prompt_B)
    D_m(x) = s(x, x_A) - s(x, x_B),  s = camera-compensated flow cosine inside the A/B subject masks
    PE_P = (D_P(x_patch) - D_P(x_B)) / (D_P(x_A) - D_P(x_B))       (0: no transfer, 1: full)
    NE_P = (D_P(x_A) - D_P(x_A<-B)) / (D_P(x_A) - D_P(x_B))
Writes <out>/rows.csv (one row per pair x candidate x direction) and gifs for the first pairs.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vdprobe import metrics as M
from vdprobe import runio
from vdprobe.coords import enumerate_coordinates
from vdprobe.edit import Arm, AttnSpec, EditSpec, run_arms, temporal_attention_modules
from vdprobe.harness import CrossAttentionMaps
from vdprobe.patching import PatchSpec, window
from vdprobe.pipeline import make_noise, phrase_token_positions
from vdprobe.probing_set import Clip, ProbingSet

SEED = 2024


def draw_pairs(ps: ProbingSet, n: int):
    rng = np.random.default_rng(SEED)
    mo, co = ps.motion_pairs(SEED), ps.content_pairs(SEED)
    pick = lambda xs: [xs[i] for i in sorted(rng.choice(len(xs), size=min(n, len(xs)), replace=False))]
    return pick(mo) + pick(co)


def unrelated(ps: ProbingSet, a: Clip, b: Clip) -> Clip:
    """Same family and scene, a subject that is neither A's nor B's, a verb that is neither."""
    fam = ps.families[b.family]["subjects"]
    subj = next(s for s in fam if s not in (a.subject, b.subject))
    verb = next(v for v in ps.verbs if v not in (a.verb, b.verb))
    return ps.clip(subj, verb, b.scene, b.seed)


def mask_probe(loaded, clip: Clip, coords):
    prompt = clip.prompt(loaded.cfg)
    tok = phrase_token_positions(loaded.pipe.tokenizer, prompt, {"subject": clip.subject})
    return CrossAttentionMaps(["subject"], coords=[c for c in coords if c.kind == "spatial_attn2"],
                              grid=8, token_index=tok)


def main():
    ap = runio.parser(__doc__, "exp2_patching")
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=25, help="pairs per kind (D-35)")
    ap.add_argument("--ctrl-pairs", type=int, default=25,
                    help="pairs per kind that also get the unrelated-clip control (D-35: all of them)")
    ap.add_argument("--shard", default="0/1")
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))
    cands = json.loads(args.candidates.read_text())["candidates"]
    loaded = runio.load(args)
    cfg = loaded.cfg
    coords = enumerate_coordinates(loaded.unet, resnets=True)   # a candidate may be a ResNet block
    ps = ProbingSet()
    drawn = draw_pairs(ps, args.pairs)
    with_ctrl = {id(p) for kind in ("motion", "content") for p in [q for q in drawn if q.kind == kind][:args.ctrl_pairs]}
    pairs = drawn[i::n]
    clip_scorer = M.ClipScorer(device=loaded.device)
    attn_mods = temporal_attention_modules(coords)
    first_third = window("first_third", cfg.steps)
    out_rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    noise = make_noise(cfg, SEED)
    t0 = time.perf_counter()
    for k, p in enumerate(pairs):
        a, b = p.a, p.b
        c = unrelated(ps, a, b)
        pa, pb, pc = (x.prompt(cfg) for x in (a, b, c))
        spec = lambda cd: EditSpec(patch=PatchSpec(tuple(cd["coords"]), tuple(cd["steps"]), component=cd["component"]))
        runs = {
            "AtoB": (pa, [Arm("B", pb, EditSpec())] + [Arm(f"B<-A|{cd['name']}", pb, spec(cd)) for cd in cands]
                     + [Arm("B<-A|attn", pb, EditSpec(attn=AttnSpec(attn_mods, first_third)))]),
            "BtoA": (pb, [Arm(f"A<-B|{cd['name']}", pa, spec(cd)) for cd in cands]),
            "CtoB": (pc, [Arm(f"B<-C|{cd['name']}", pb, spec(cd)) for cd in cands]),
        }
        if id(p) not in with_ctrl:
            del runs["CtoB"]
        frames, masks = {}, {}
        for run, (src_prompt, arms) in runs.items():
            extra = {}
            if run == "AtoB":
                extra = {"source": [mask_probe(loaded, a, coords)], "B": [mask_probe(loaded, b, coords)]}
            res = run_arms(loaded, coords, src_prompt, arms, noise, extra_probes=extra)
            src_name = {"AtoB": "A", "BtoA": "B_src", "CtoB": "C"}[run]
            frames[src_name] = res.frames["source"]
            frames.update({arm.id: res.frames[arm.id] for arm in arms})
            if run == "AtoB":
                masks["A"], masks["B"] = (extra[x][0].mask("subject").numpy() for x in ("source", "B"))
        union = masks["A"] | masks["B"]
        flows = {kk: M.remove_camera_motion(M.raft_flow(v, size=256, device=loaded.device), union)
                 for kk, v in frames.items()}
        txt = clip_scorer.text([pa, pb])
        Dc = {kk: float((clip_scorer.image(v) @ txt.T).mean(0) @ torch.tensor([1.0, -1.0], device=txt.device))
              for kk, v in frames.items()}
        s = lambda x, y: M.masked_motion_fidelity(flows[x], flows[y], union, union)
        Dm = {kk: s(kk, "A") - s(kk, "B") for kk in frames}
        den = {"c": Dc["A"] - Dc["B"], "m": Dm["A"] - Dm["B"]}
        for cd in cands:
            nm = cd["name"]
            for direction, key, f in (("suff", f"B<-A|{nm}", lambda D, x: (D[x] - D["B"])),
                                      ("nec", f"A<-B|{nm}", lambda D, x: (D["A"] - D[x])),
                                      ("ctrl", f"B<-C|{nm}", lambda D, x: (D[x] - D["B"]))):
                if key not in frames:
                    continue
                out_rows.append({"pair": f"{a.id}->{b.id}", "kind": p.kind, "subject_b": b.subject, "family": b.family,
                                 "candidate": nm, "direction": direction,
                                 "eff_c": f(Dc, key) / den["c"] if abs(den["c"]) > 1e-6 else np.nan,
                                 "eff_m": f(Dm, key) / den["m"] if abs(den["m"]) > 1e-6 else np.nan,
                                 "den_c": den["c"], "den_m": den["m"], "Dc": Dc[key], "Dm": Dm[key]})
        out_rows.append({"pair": f"{a.id}->{b.id}", "kind": p.kind, "subject_b": b.subject, "family": b.family,
                         "candidate": "attn_first_third", "direction": "suff",
                         "eff_c": (Dc["B<-A|attn"] - Dc["B"]) / den["c"] if abs(den["c"]) > 1e-6 else np.nan,
                         "eff_m": (Dm["B<-A|attn"] - Dm["B"]) / den["m"] if abs(den["m"]) > 1e-6 else np.nan,
                         "den_c": den["c"], "den_m": den["m"], "Dc": Dc["B<-A|attn"], "Dm": Dm["B<-A|attn"]})
        if k < 3:
            for kk, v in frames.items():
                runio.save_gif(v, args.out / "gifs" / f"{k:02d}_{kk.replace('|', '_').replace('<-', '_from_')}.gif")
        pd.DataFrame(out_rows).to_csv(args.out / "rows.csv", index=False)
        print(f"pair {k + 1}/{len(pairs)} {p.kind} {a.id} -> {b.id}  {time.perf_counter() - t0:.0f} s", flush=True)
    runio.write_json(args.out / "manifest.json", runio.record(
        "exp2_patching", loaded, shard=args.shard, candidates=cands, n_pairs=len(pairs), seed=SEED,
        seconds=time.perf_counter() - t0))


if __name__ == "__main__":
    main()
