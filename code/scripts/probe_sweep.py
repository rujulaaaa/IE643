"""Stage 2-3 · The probing sweep: every clip of the frozen probing set, every pair scored.

One lockstep group per (seed, scene): the 60 clips of that scene (15 subjects x 4 verbs)
share the seed's noise tensor and are denoised step by step together. Anchor clips run
first in each step and keep their step activations (StepCache); every partner clip is
scored against its anchors (PairScorer, D-01 distances on the full flattened tensors, at
the 16 coarse coordinates of D-08/D-25, all 25 steps). Every clip also gets a count
sketch of x_tilde and x_bar for the linear probes (D-21). Each clip is generated once.

Per group, written to <out>/<seed>__<scene>/:
  pairs.json        the pair table (index, kind, anchor, partner, family)
  distances.csv     one row per (pair, coordinate, step): d_static, d_dynamic, d_raw,
                    l2_static, l2_dynamic, static_energy_a/b
  per_channel.pt    per-channel dynamic distance (D-10) at the D-07 steps
  sketches.pt       {clip id: (coords, 25, 2, dim) fp16}, [x_tilde, x_bar]
  latents.pt        {clip id: final latents fp16}
  gifs/<clip>.gif   256 px previews
  manifest.json     config, revisions (D-20), timings; written last, marks the group done

    python scripts/probe_sweep.py                       # all 3 seeds x 2 scenes
    python scripts/probe_sweep.py --seeds 42 --scenes "on a beach" --subjects 2 --verbs 2   # smoke test

Experiment 2 (docs/experiment2.md) adds, with --exp2 (or the individual flags):
  --config configs/variants/<cell>.yaml     the 2x2 model grid
  --set ladder                               the content ladder (one group per seed)
  --refined-blocks up_blocks.0 up_blocks.1   exact distances at every coordinate in these blocks
  --resnets                                  ResNet blocks as coordinates too
and writes per group, besides the files above:
  sketches_all.pt       {clip: (all coords, 25, 2, 512)}           every coordinate, both parts
  sketches_refined.pt   {clip: (refined coords, 15, 1, 2048)}       x_tilde, steps 0-14
  channel_stats.pt      {clip: (refined coords, 15, 2, C)}          [static mean, dynamic energy]
  head_stats.pt         {clip: (temporal attns, 25, heads, 3)}      entropy, off-diagonal, offset
  xattn.pt              {clip: {"subject"/"verb": (F, 64, 64), "cells": (attn2, 25, 2, 8, 8)}}
  flows.pt              {clip: (F-1, 64, 64, 2)}  RAFT at 256 px, pooled to 64 (pixels at 256)
  clip_emb.pt           {clip: (F, 512)}          CLIP ViT-B/32 frame embeddings
  latent_traj.pt        {clip: (25, 4, F, 16, 16)} the latent entering each step, 4x4-pooled
"""

import csv
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch

from vdprobe import runio
from vdprobe.coords import D07_STEPS, coarse_coordinates, enumerate_coordinates, in_blocks
from vdprobe.harness import CrossAttentionMaps
from vdprobe.ladder import LadderSet
from vdprobe.lockstep import (ChannelStats, HeadStats, PairScorer, Sketcher, StepCache, Track,
                              run_lockstep)
from vdprobe.pipeline import decode, make_noise, phrase_token_positions
from vdprobe.probing_set import ProbingSet

REFINED_STEPS = range(15)   # dense window of the refined views (report: steps 0-14)


def slug(s):
    return s.replace(" ", "-")


def topo_order(clips, pairs):
    """Anchors before the clips scored against them; otherwise the probing-set order."""
    preds = defaultdict(set)
    for p in pairs:
        preds[p.b.id].add(p.a.id)
    done, order, pending = set(), [], list(clips)
    while pending:
        nxt = next(c for c in pending if preds[c.id] <= done)
        order.append(nxt)
        done.add(nxt.id)
        pending.remove(nxt)
    return order


def probing_groups(ps, args):
    subj_keep = {s for f in ps.families.values() for s in f["subjects"][:args.subjects]} if args.subjects else None
    verb_keep = set(ps.verbs[:args.verbs]) if args.verbs else None
    for seed in args.seeds or ps.seeds:
        for scene in args.scenes or ps.scenes:
            keep = lambda c, scene=scene: c.scene == scene and (subj_keep is None or c.subject in subj_keep) and \
                (verb_keep is None or c.verb in verb_keep)
            clips = [c for c in ps.clips(seed) if keep(c)]
            pairs = [p for p in ps.content_pairs(seed) + ps.motion_pairs(seed) + ps.cross_family_pairs(seed)
                     if keep(p.a) and keep(p.b)]
            yield f"s{seed}__{slug(scene)}", seed, scene, clips, pairs


def ladder_groups(ls, args):
    for seed in args.seeds or ls.seeds:
        clips, pairs = ls.clips(seed), ls.pairs(seed)
        if args.subjects:     # smoke test: first N bases
            bases = {b["breed"] for b in ls.bases[:args.subjects]}
            keep = lambda c: ls.base_of(c) in bases
            clips, pairs = [c for c in clips if keep(c)], [p for p in pairs if keep(p.a)]
        yield f"s{seed}__ladder", seed, "ladder", clips, pairs


def run_group(loaded, coords, coarse, name, seed, scene, clips, pairs, args):
    out = args.out / name
    if (out / "manifest.json").exists() and not args.force:
        print(f"[skip] {out.name} already done")
        return
    out.mkdir(parents=True, exist_ok=True)
    scored = list({c.name: c for c in coarse + in_blocks(coords, args.refined_blocks)}.values())
    refined = in_blocks(coords, args.refined_blocks)
    order = topo_order(clips, pairs)
    anchors = {p.a.id for p in pairs}
    kinds = defaultdict(int)
    for p in pairs:
        kinds[p.kind] += 1
    print(f"\n=== {name}: {len(clips)} clips, {len(pairs)} pairs {dict(kinds)}, {len(anchors)} anchors, "
          f"{len(scored)} scored coordinates ({len(refined)} refined)")

    steps = range(loaded.cfg.steps)
    caches = {a: StepCache(scored) for a in anchors}
    refs = defaultdict(list)
    for i, p in enumerate(pairs):
        refs[p.b.id].append((i, caches[p.a.id]))
    sk_coords = coords if args.sketch_all else coarse
    sketch = {c.id: Sketcher(sk_coords, steps=steps, dim=args.sketch_dim) for c in clips}
    extra = defaultdict(dict)        # clip id -> {view: probe}
    scorers = {}
    tracks = []
    for c in order:
        probes = [sketch[c.id]]
        if c.id in caches:
            probes.insert(0, caches[c.id])
        if refs[c.id]:
            scorers[c.id] = PairScorer(scored, refs[c.id], per_channel_steps=D07_STEPS)
            probes.append(scorers[c.id])
        prompt = c.prompt(loaded.cfg)
        if refined and args.refined_sketch_dim:
            extra[c.id]["sk_ref"] = Sketcher(refined, steps=REFINED_STEPS, dim=args.refined_sketch_dim, parts=("tilde",))
        if refined and args.channel_stats:
            extra[c.id]["chan"] = ChannelStats(refined, steps=REFINED_STEPS)
        if args.head_stats:
            extra[c.id]["head"] = HeadStats(coords, steps=steps)
        if args.xattn:
            tok = phrase_token_positions(loaded.pipe.tokenizer, prompt, {"subject": c.subject, "verb": c.verb})
            extra[c.id]["xattn"] = CrossAttentionMaps(["subject", "verb"], coords=[x for x in coords
                                                      if x.kind == "spatial_attn2"], grid=8, token_index=tok)
        probes += list(extra[c.id].values())
        tracks.append(Track(c.id, prompt, probes, meta={"clip": c}))

    # free an anchor's step activations as soon as its last partner in this step has run
    n_partners = defaultdict(int)
    for p in pairs:
        n_partners[p.a.id] += 1
    remaining = {}

    lat_traj = defaultdict(list)    # clip id -> per step, the latent *entering* the step, pooled 4x4

    def after_track(step, tr):
        if args.post:
            lat_traj[tr.id].append(tr.latents_in)
        if tr.id in caches and n_partners[tr.id] == 0:
            caches[tr.id].clear()
        for _, cache in refs[tr.id]:
            a = next(k for k, v in caches.items() if v is cache)
            remaining[a] = remaining.get(a, n_partners[a]) - 1
            if remaining[a] == 0:
                cache.clear()

    def after_step(step):
        remaining.clear()
        for c in caches.values():
            c.clear()

    def before_track(step, tr):
        if args.post:      # the flow-decoding baseline (latent-only control) needs the input latent
            tr.latents_in = torch.nn.functional.avg_pool3d(tr.latents.float(), (1, 4, 4))[0].half().cpu()

    noise = make_noise(loaded.cfg, seed)
    t = run_lockstep(loaded, coords, tracks, noise, after_track=after_track, after_step=after_step,
                     before_track=before_track, log_every=5)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"  denoised in {t['seconds'] / 60:.1f} min ({t['seconds'] / len(tracks):.1f} s per clip), "
          f"peak host RSS {rss_gb:.1f} GB")

    t1 = time.perf_counter()
    latents, flows, clip_emb = {}, {}, {}
    clip_scorer = None
    if args.post:
        from vdprobe import metrics as M
        clip_scorer = M.ClipScorer(device=loaded.device)
    for tr in tracks:
        latents[tr.id] = tr.latents.half()
        frames = decode(loaded, tr.latents)
        runio.save_gif(frames, out / "gifs" / f"{tr.id}.gif")
        if args.post:
            f = torch.from_numpy(M.raft_flow(frames, size=256, device=loaded.device))     # (F-1, 256, 256, 2)
            flows[tr.id] = torch.nn.functional.avg_pool2d(f.permute(0, 3, 1, 2), 4).permute(0, 2, 3, 1).half()
            clip_emb[tr.id] = clip_scorer.image(frames).half().cpu()
    decode_s = time.perf_counter() - t1

    def stack_steps(d, names_, steps_):
        return torch.stack([torch.stack([d[(n, s)] for s in steps_]) for n in names_])

    all_names, ref_names = [c.name for c in coords], [c.name for c in refined]
    if args.sketch_all:
        torch.save({cid: stack_steps(sk.out, all_names, steps) for cid, sk in sketch.items()}, out / "sketches_all.pt")
    views = {"sk_ref": ("sketches_refined.pt", ref_names, REFINED_STEPS),
             "chan": ("channel_stats.pt", ref_names, REFINED_STEPS),
             "head": ("head_stats.pt", [c.name for c in coords if c.kind.startswith("temporal_attn")], steps)}
    for key, (fname, names_, steps_) in views.items():
        got = {cid: stack_steps(e[key].out, names_, steps_) for cid, e in extra.items() if key in e}
        if got:
            torch.save(got, out / fname)
    if args.xattn:
        x_names = [c.name for c in coords if c.kind == "spatial_attn2"]
        torch.save({cid: {"subject": e["xattn"].total["subject"].half(), "verb": e["xattn"].total["verb"].half(),
                          "cells": torch.stack([torch.stack([torch.stack([e["xattn"].cells[(n, s)][ph] for ph in
                                                ("subject", "verb")]) for s in steps]) for n in x_names]).half()}
                    for cid, e in extra.items()}, out / "xattn.pt")
    if args.post:
        torch.save({k: torch.stack(v) for k, v in lat_traj.items()}, out / "latent_traj.pt")
        torch.save(flows, out / "flows.pt")
        torch.save(clip_emb, out / "clip_emb.pt")

    names = [c.name for c in coarse]
    with open(out / "distances.csv", "w", newline="") as f:
        rows = [r for s in scorers.values() for r in s.rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    torch.save({k: v for s in scorers.values() for k, v in s.per_channel.items()}, out / "per_channel.pt")
    if not args.sketch_all:     # with --sketch-all the coarse sketches are a slice of sketches_all.pt
        torch.save({cid: stack_steps(sk.out, names, steps) for cid, sk in sketch.items()}, out / "sketches.pt")
    torch.save(latents, out / "latents.pt")
    (out / "pairs.json").write_text(json.dumps(
        [{"pair": i, "kind": p.kind, "a": p.a.id, "b": p.b.id, "family_a": p.a.family, "family_b": p.b.family,
          "subject_a": p.a.subject, "subject_b": p.b.subject, "verb_a": p.a.verb, "verb_b": p.b.verb,
          "scene_a": p.a.scene, "scene_b": p.b.scene}
         for i, p in enumerate(pairs)], indent=1))
    runio.write_json(out / "manifest.json", runio.record(
        "probe_sweep", loaded, seed=seed, scene=scene, n_clips=len(clips), n_pairs=len(pairs),
        coords=names, scored_coords=[c.name for c in scored], refined_coords=ref_names, all_coords=all_names,
        set=args.set, sketch_dim=args.sketch_dim, refined_sketch_dim=args.refined_sketch_dim,
        refined_steps=list(REFINED_STEPS), clips=[c.id for c in clips],
        clip_meta={c.id: {"subject": c.subject, "verb": c.verb, "scene": c.scene, "family": c.family} for c in clips},
        prompts={tr.id: tr.prompt for tr in tracks}, denoise_seconds=t["seconds"], decode_seconds=decode_s,
        peak_rss_gb=rss_gb, subset={"subjects": args.subjects, "verbs": args.verbs}))


def main():
    p = runio.parser(__doc__, "sweep")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--subjects", type=int, default=0, help="smoke test: first N subjects per family")
    p.add_argument("--verbs", type=int, default=0, help="smoke test: first N verbs")
    p.add_argument("--sketch-dim", type=int, default=512)
    p.add_argument("--force", action="store_true")
    p.add_argument("--set", choices=("probing", "ladder"), default="probing")
    p.add_argument("--refined-blocks", nargs="*", default=[])
    p.add_argument("--resnets", action="store_true", help="hook the ResNet blocks as coordinates too")
    p.add_argument("--sketch-all", action="store_true", help="sketch every coordinate, not only the coarse 16")
    p.add_argument("--refined-sketch-dim", type=int, default=0)
    p.add_argument("--channel-stats", action="store_true")
    p.add_argument("--head-stats", action="store_true")
    p.add_argument("--xattn", action="store_true", help="per-frame subject/verb cross-attention maps")
    p.add_argument("--post", action="store_true", help="RAFT flow and CLIP frame embeddings per clip")
    p.add_argument("--exp2", action="store_true", help="all Experiment 2 views (refined up0+up1, resnets, ...)")
    args = p.parse_args()
    if args.exp2:
        args.refined_blocks = args.refined_blocks or ["up_blocks.0", "up_blocks.1"]
        args.resnets = args.sketch_all = args.channel_stats = args.head_stats = args.xattn = args.post = True
        args.refined_sketch_dim = args.refined_sketch_dim or 2048

    if args.set == "ladder":
        ds = LadderSet()
        groups = ladder_groups(ds, args)
    else:
        ds = ProbingSet()
        groups = probing_groups(ds, args)
    ds.check_counts()
    loaded = runio.load(args)
    coords = enumerate_coordinates(loaded.unet, resnets=args.resnets)
    coarse = coarse_coordinates(coords)
    print(f"{len(coords)} coordinates, coarse {len(coarse)}")
    for name, seed, scene, clips, pairs in groups:
        run_group(loaded, coords, coarse, name, seed, scene, clips, pairs, args)


if __name__ == "__main__":
    main()
