"""Stage 6c (D-38) · Probe-timed prompt switching on the edit pairs, next to the Stage 6 arms.

Per pair, two lockstep groups from the pair's shared noise, merged into one pair directory that
score_edits.py scores as one pool:
  1. edit.run_arms: source A, B_ref, both prompt-only baselines, both selected component patches
     (configs/edit_selection.json) and temporal attention injection (the D-17 baseline)
  2. switching.run_switch: A again (its trajectory and DAAM subject map), then the switch arms
     (switching.switch_arms: both directions at tau in --taus, masked motion arms at --masked-taus)

    python scripts/switch_sweep.py --split test --shard 0/4
    python scripts/score_edits.py <out>
"""

import json
import time

import torch

from vdprobe import runio
from vdprobe.coords import enumerate_coordinates
from vdprobe.edit import run_arms
from vdprobe.edit_plan import describe, load_pairs, selected_arms
from vdprobe.pipeline import decode, make_noise
from vdprobe.switching import describe_switch, run_switch, switch_arms

REFERENCE = ("B_ref", "motion.base", "content.base", "motion.sel", "content.sel", "motion.attn")


def main():
    ap = runio.parser(__doc__, "switch_sweep")
    ap.add_argument("--split", choices=["dev", "test"], default="test")
    ap.add_argument("--shard", default="0/1", help="i/n: run pairs i, i+n, i+2n, ...")
    ap.add_argument("--pairs", type=int, default=0, help="limit to the first N pairs (smoke test)")
    ap.add_argument("--taus", default="1,2,3,5,8")
    ap.add_argument("--masked-taus", default="2,3")
    ap.add_argument("--mask-tau", type=float, default=0.4, help="DAAM mask threshold")
    ap.add_argument("--dilate", type=int, default=7, help="mask dilation (max-pool kernel, latent px)")
    ap.add_argument("--cache-device", default="cuda")
    args = ap.parse_args()

    loaded = runio.load(args)
    cfg = loaded.cfg
    coords = enumerate_coordinates(loaded.unet)
    sel = json.load(open(runio.REPO_ROOT / "configs" / "edit_selection.json"))
    taus = tuple(int(t) for t in args.taus.split(",") if t)
    mtaus = tuple(int(t) for t in args.masked_taus.split(",") if t)
    pairs = load_pairs(args.split)
    if args.pairs:
        pairs = pairs[:args.pairs]
    i, n = map(int, args.shard.split("/"))
    pairs = pairs[i::n]

    for k, pair in enumerate(pairs):
        d = args.out / pair.id
        if (d / "manifest.json").exists():
            print(f"[{k + 1}/{len(pairs)}] {pair.id}: done, skipping")
            continue
        prompts = pair.prompts(cfg)
        ref = [a for a in selected_arms(pair, cfg, coords, sel, alphas=()) if a.id in REFERENCE]
        sw = switch_arms(prompts, taus, mtaus)
        print(f"[{k + 1}/{len(pairs)}] {pair.id}: source + {len(ref)} reference arms + {len(sw)} switch arms", flush=True)
        d.mkdir(parents=True, exist_ok=True)
        (d / "arms.json").write_text(json.dumps({"pair": pair.__dict__, "prompts": prompts,
                                                 "arms": [describe(a) for a in ref] +
                                                         [describe_switch(a, s) for a, s in sw.items()]}, indent=1))
        torch.cuda.reset_peak_memory_stats()
        noise = make_noise(cfg, pair.seed)
        r = run_arms(loaded, coords, prompts["A"], ref, noise, cache_device=args.cache_device, log_every=5)
        s = run_switch(loaded, coords, prompts["A"], pair.subject_a, sw, noise, mask_tau=args.mask_tau,
                       dilate=args.dilate, log_every=5)
        lat = {"source": r.source_latents.half(), **{a: x.half() for a, x in r.latents.items()},
               **{a: x.half() for a, x in s.latents.items()}}
        torch.save(lat, d / "latents.pt")
        if s.mask is not None:
            torch.save(s.mask.half().cpu(), d / "subject_mask.pt")
        for a, fr in r.frames.items():
            runio.save_gif(fr, d / "gifs" / f"{a}.gif")
        for a, x in s.latents.items():
            runio.save_gif(decode(loaded, x), d / "gifs" / f"{a}.gif")
        same = float((s.source_latents - r.source_latents).abs().max())
        rec = runio.record("switch_sweep", loaded, split=args.split, selection=sel, pair=pair.__dict__,
                           taus=taus, masked_taus=mtaus, mask_tau=args.mask_tau, dilate=args.dilate,
                           mask_area=None if s.mask is None else float(s.mask.float().mean()),
                           source_max_abs_diff_between_groups=same,
                           seconds=r.seconds + s.seconds, n_tracks=len(ref) + len(sw) + 2,
                           seconds_per_clip=(r.seconds + s.seconds) / (len(ref) + len(sw) + 2),
                           peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                           n_patched=r.n_patched, n_attn_injected=r.n_attn_injected,
                           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        runio.write_json(d / "manifest.json", rec)
        print(f"  {rec['seconds']:.0f} s ({rec['seconds_per_clip']:.1f} s per clip), mask area "
              f"{rec['mask_area']}, source diff between groups {same:.2e}", flush=True)


if __name__ == "__main__":
    main()
