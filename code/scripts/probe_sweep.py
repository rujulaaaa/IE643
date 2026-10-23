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
"""

import csv
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch

from vdprobe import runio
from vdprobe.coords import D07_STEPS, coarse_coordinates, enumerate_coordinates
from vdprobe.lockstep import PairScorer, Sketcher, StepCache, Track, run_lockstep
from vdprobe.pipeline import decode, make_noise
from vdprobe.probing_set import ProbingSet


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


def run_group(loaded, coords, coarse, ps, seed, scene, args):
    out = args.out / f"s{seed}__{slug(scene)}"
    if (out / "manifest.json").exists() and not args.force:
        print(f"[skip] {out.name} already done")
        return
    out.mkdir(parents=True, exist_ok=True)

    subj_keep = {s for f in ps.families.values() for s in f["subjects"][:args.subjects]} if args.subjects else None
    verb_keep = set(ps.verbs[:args.verbs]) if args.verbs else None
    keep = lambda c: c.scene == scene and (subj_keep is None or c.subject in subj_keep) and \
        (verb_keep is None or c.verb in verb_keep)
    clips = [c for c in ps.clips(seed) if keep(c)]
    pairs = [p for p in ps.content_pairs(seed) + ps.motion_pairs(seed) + ps.cross_family_pairs(seed)
             if keep(p.a) and keep(p.b)]
    order = topo_order(clips, pairs)
    anchors = {p.a.id for p in pairs}
    print(f"\n=== seed {seed} · {scene}: {len(clips)} clips, {len(pairs)} pairs "
          f"({sum(p.kind == 'content' for p in pairs)} content, {sum(p.kind == 'motion' for p in pairs)} motion, "
          f"{sum(p.kind == 'cross_family' for p in pairs)} cross-family), {len(anchors)} anchors")

    caches = {a: StepCache(coarse) for a in anchors}
    refs = defaultdict(list)
    for i, p in enumerate(pairs):
        refs[p.b.id].append((i, caches[p.a.id]))
    sketch = {c.id: Sketcher(coarse, steps=range(loaded.cfg.steps), dim=args.sketch_dim) for c in clips}
    scorers = {}
    tracks = []
    for c in order:
        probes = [sketch[c.id]]
        if c.id in caches:
            probes.insert(0, caches[c.id])
        if refs[c.id]:
            scorers[c.id] = PairScorer(coarse, refs[c.id], per_channel_steps=D07_STEPS)
            probes.append(scorers[c.id])
        tracks.append(Track(c.id, c.prompt(loaded.cfg), probes, meta={"clip": c}))

    # free an anchor's step activations as soon as its last partner in this step has run
    n_partners = defaultdict(int)
    for p in pairs:
        n_partners[p.a.id] += 1
    remaining = {}

    def after_track(step, tr):
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

    noise = make_noise(loaded.cfg, seed)
    t = run_lockstep(loaded, coords, tracks, noise, after_track=after_track, after_step=after_step, log_every=5)
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"  denoised in {t['seconds'] / 60:.1f} min ({t['seconds'] / len(tracks):.1f} s per clip), "
          f"peak host RSS {rss_gb:.1f} GB")

    t1 = time.perf_counter()
    latents = {}
    for tr in tracks:
        latents[tr.id] = tr.latents.half()
        runio.save_gif(decode(loaded, tr.latents), out / "gifs" / f"{tr.id}.gif")
    decode_s = time.perf_counter() - t1

    names = [c.name for c in coarse]
    with open(out / "distances.csv", "w", newline="") as f:
        rows = [r for s in scorers.values() for r in s.rows]
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    torch.save({k: v for s in scorers.values() for k, v in s.per_channel.items()}, out / "per_channel.pt")
    torch.save({cid: torch.stack([torch.stack([sk.out[(n, s)] for s in range(loaded.cfg.steps)]) for n in names])
                for cid, sk in sketch.items()}, out / "sketches.pt")
    torch.save(latents, out / "latents.pt")
    (out / "pairs.json").write_text(json.dumps(
        [{"pair": i, "kind": p.kind, "a": p.a.id, "b": p.b.id, "family_a": p.a.family, "family_b": p.b.family,
          "subject_a": p.a.subject, "subject_b": p.b.subject, "verb_a": p.a.verb, "verb_b": p.b.verb}
         for i, p in enumerate(pairs)], indent=1))
    runio.write_json(out / "manifest.json", runio.record(
        "probe_sweep", loaded, seed=seed, scene=scene, n_clips=len(clips), n_pairs=len(pairs),
        coords=names, sketch_dim=args.sketch_dim, clips=[c.id for c in clips],
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
    args = p.parse_args()

    ps = ProbingSet()
    ps.check_counts()
    loaded = runio.load(args)
    coords = enumerate_coordinates(loaded.unet)
    coarse = coarse_coordinates(coords)
    for seed in args.seeds or ps.seeds:
        for scene in args.scenes or ps.scenes:
            run_group(loaded, coords, coarse, ps, seed, scene, args)


if __name__ == "__main__":
    main()
