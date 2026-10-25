"""Stages 5-6 · Run the edit arms for a set of edit pairs (one lockstep group per pair).

    --split dev  --arms coarse                 Stage 5 localisation (edit_plan.coarse_arms)
    --split test --arms selected --sel s.json  Stage 6 evaluation (edit_plan.selected_arms)
    --split dev  --arms lit                    Stage 6b: literature operators (edit_plan.literature_arms)
    --split test --arms lit --lit-sel l.json   Stage 6b evaluation (edit_plan.literature_test_arms, D-37)

Per pair, written to <out>/<pair id>/:
  arms.json       every arm's prompt and patch spec
  latents.pt      {"source": ..., arm id: ...} final latents, fp16
  gifs/*.gif      256 px previews
  manifest.json   timings, counts of applied patches, revisions (D-20); written last = done

Scoring is a separate CPU/GPU step (scripts/score_edits.py) over the saved latents.

    python -u scripts/edit_sweep.py --split dev --arms coarse --shard 0/2
"""

import json
import time

import torch

from vdprobe import runio
from vdprobe.coords import enumerate_coordinates
from vdprobe.edit import run_arms
from vdprobe.edit_plan import (coarse_arms, describe, literature_arms, literature_test_arms, load_pairs,
                               selected_arms)
from vdprobe.pipeline import make_noise


def main():
    ap = runio.parser(__doc__, "edit_sweep")
    ap.add_argument("--split", choices=["dev", "test"], required=True)
    ap.add_argument("--arms", choices=["coarse", "selected", "lit"], required=True)
    ap.add_argument("--sel", default=None, help="selection JSON for --arms selected")
    ap.add_argument("--sel-json", default=None, help="the same, inline")
    ap.add_argument("--lit-sel", default=None, help="Stage 6b selection JSON (D-37) for --split test --arms lit")
    ap.add_argument("--shard", default="0/1", help="i/n: run pairs i, i+n, i+2n, ...")
    ap.add_argument("--pairs", type=int, default=0, help="limit to the first N pairs (smoke test)")
    ap.add_argument("--only", default="", help="comma-separated arm ids to keep (smoke test)")
    ap.add_argument("--cache-device", default="cuda")
    ap.add_argument("--lora", action="append", default=[], help="motion LoRA file(s) for --arms selected")
    args = ap.parse_args()

    loaded = runio.load(args)
    loras = []
    for path in args.lora:
        from safetensors import safe_open

        from vdprobe.lora import load_motion_lora
        with safe_open(path, "pt") as f:
            name = "lora_" + f.metadata()["variant"]
        load_motion_lora(loaded.unet, path, name=name)
        loras.append(name)
    if loras:
        loaded.unet.disable_adapters()
    coords = enumerate_coordinates(loaded.unet)
    pairs = load_pairs(args.split)
    if args.pairs:
        pairs = pairs[:args.pairs]
    i, n = map(int, args.shard.split("/"))
    pairs = pairs[i::n]
    sel = None
    if args.arms == "selected" or (args.arms == "lit" and args.lit_sel):
        sel = json.loads(args.sel_json) if args.sel_json else json.load(open(args.sel or runio.REPO_ROOT / "configs" / "edit_selection.json"))
    lit_sel = json.load(open(args.lit_sel)) if args.lit_sel else None

    for k, pair in enumerate(pairs):
        d = args.out / pair.id
        if (d / "manifest.json").exists():
            print(f"[{k + 1}/{len(pairs)}] {pair.id}: done, skipping")
            continue
        if args.arms == "coarse":
            arms = coarse_arms(pair, loaded.cfg, coords)
        elif args.arms == "selected":
            arms = selected_arms(pair, loaded.cfg, coords, sel, loras=tuple(loras))
        elif lit_sel is None:
            arms = literature_arms(pair, loaded.cfg, coords)
        else:
            arms = literature_test_arms(pair, loaded.cfg, coords, sel, lit_sel)
        if args.only:
            keep = set(args.only.split(","))
            arms = [a for a in arms if a.id in keep]
        prompts = pair.prompts(loaded.cfg)
        print(f"[{k + 1}/{len(pairs)}] {pair.id}: source + {len(arms)} arms", flush=True)
        d.mkdir(parents=True, exist_ok=True)
        (d / "arms.json").write_text(json.dumps({"pair": pair.__dict__, "prompts": prompts,
                                                 "arms": [describe(a) for a in arms]}, indent=1))
        torch.cuda.reset_peak_memory_stats()
        r = run_arms(loaded, coords, prompts["A"], arms, make_noise(loaded.cfg, pair.seed),
                     cache_device=args.cache_device, log_every=5)
        lat = {"source": r.source_latents.half(), **{a: x.half() for a, x in r.latents.items()}}
        torch.save(lat, d / "latents.pt")
        for a, fr in r.frames.items():
            runio.save_gif(fr, d / "gifs" / f"{a}.gif")
        rec = runio.record("edit_sweep", loaded, split=args.split, arm_set=args.arms, selection=sel, lit_selection=lit_sel,
                           pair=pair.__dict__, loras=args.lora, seconds=r.seconds, n_tracks=len(arms) + 1,
                           seconds_per_clip=r.seconds / (len(arms) + 1),
                           peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                           n_patched=r.n_patched, n_attn_injected=r.n_attn_injected,
                           n_kv_injected=r.n_kv_injected,
                           finished=time.strftime("%Y-%m-%d %H:%M:%S"))
        runio.write_json(d / "manifest.json", rec)
        print(f"  {r.seconds:.0f} s ({r.seconds / (len(arms) + 1):.1f} s per clip)", flush=True)


if __name__ == "__main__":
    main()
