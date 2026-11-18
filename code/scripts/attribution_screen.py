"""Part D1: attribution patching screen over every site of the U-Net (src/vdprobe/attribution.py).

Pairs as in scripts/camera_patch.py (held-out seed, one noise tensor): A = (scene X, camera c), B = (scene X, camera
off). Both run once, unpatched, in one lockstep group; the latent entering each screened step is kept. At each
screened step, A's activations come from A's own latent and B's from B's (as in the patching runs), the
text-conditional branch of B runs with gradients, and every site gets

    frac(site) = atp(site) / |eps_A - eps_B|^2         estimated share of the A-B gap closed by patching that site

Outputs: atp.csv (model, pair, camera, scene, step, site, kind, block, score, frac, gap), manifest.json.
`--compare <dir of camera_patch rows.csv>` adds atp_vs_patch.json: per block, the sum of `frac` over the block's
motion modules (mean over pairs and steps 0-7) against the measured `eff_cam` of the `mm_<block>` candidate
(sufficiency, same model), and their Spearman correlation.

    python scripts/kaggle_run.py --user <account> --slug vdprobe-cc-atp --prefetch-cameractrl \\
        --job "scripts/attribution_screen.py --model cc --shard 0/2" --job "scripts/attribution_screen.py --model cc --shard 1/2"
"""

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from vdprobe import runio
from vdprobe.attribution import atp_step, spearman
from vdprobe.camera_models import FAMILIES, MODELS, default_config, load_camera_model
from vdprobe.camera_set import CameraClip
from vdprobe.coords import enumerate_coordinates
from vdprobe.lockstep import Track, run_lockstep
from vdprobe.pipeline import make_noise

STEPS = (0, 2, 4, 8, 16)          # 0 %, 8 %, 16 %, 32 %, 64 % of 25 steps


def block_sums(rows, steps=range(8)) -> dict:
    """{block: mean over (pair, step) of the summed frac of that block's motion modules}."""
    acc = {}
    for r in rows:
        if r["kind"] != "temporal_module" or int(r["step"]) not in steps:
            continue
        acc.setdefault(r["block"], {}).setdefault((r["pair"], int(r["step"])), 0.0)
        acc[r["block"]][(r["pair"], int(r["step"]))] += float(r["frac"])
    return {b: float(np.mean(list(v.values()))) for b, v in acc.items()}


def compare(atp_rows, patch_dir: Path, model: str) -> dict:
    import pandas as pd
    files = [f for f in patch_dir.rglob("rows.csv")]
    if not files:
        return {"error": f"no rows.csv under {patch_dir}"}
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[(df.model == model) & (df.direction == "suff") & df.candidate.str.startswith("mm_")]
    eff = df.groupby(df.candidate.str[3:]).eff_cam.mean().to_dict()
    atp = block_sums(atp_rows)
    blocks = sorted(set(eff) & set(atp))
    return {"blocks": {b: {"atp_frac": atp[b], "eff_cam": float(eff[b])} for b in blocks},
            "spearman": spearman([atp[b] for b in blocks], [eff[b] for b in blocks]), "n_blocks": len(blocks)}


def main():
    p = runio.parser(__doc__, "attribution_screen")
    p.set_defaults(config=None)
    p.add_argument("--model", choices=MODELS, default="cc")
    p.add_argument("--seed", type=int, default=None, help="default: the set's held-out seed")
    p.add_argument("--scenes", type=int, default=6)
    p.add_argument("--cameras", type=int, default=4)
    p.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    p.add_argument("--frames", type=int, default=None, help="fewer frames if a pair does not fit (record it)")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--compare", type=Path, default=None)
    args = p.parse_args()

    f = FAMILIES[args.model]
    cameras = [f["zoom"][0], f["pan"][0], f["tilt"][0], f["roll"][0]][:args.cameras]
    cfg_path = args.config or default_config(args.model)
    cm = load_camera_model(args.model, cfg_path, args.profile, cameras=cameras)
    loaded = cm.loaded
    if args.frames:
        import dataclasses
        loaded.cfg = dataclasses.replace(loaded.cfg, frames=args.frames)
    cfg, unet = loaded.cfg, loaded.unet
    unet.requires_grad_(False)                      # no parameter gradients: only activation gradients are needed
    coords = enumerate_coordinates(unet)
    kind = {c.name: (c.kind, c.block) for c in coords}
    names = [c.name for c in coords]
    seed = args.seed if args.seed is not None else cm.cameras.heldout_seed
    pairs = [(c, s) for s in cm.cameras.scenes[:args.scenes] for c in cameras]
    i, n = (int(v) for v in args.shard.split("/"))
    noise = make_noise(cfg, seed)
    sched = loaded.pipe.scheduler
    rows, t0 = [], time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    for camera, scene in pairs[i::n]:
        prompt = CameraClip(scene, camera, seed).prompt(cfg)
        tracks = [Track("A", prompt, meta={"camera": camera}), Track("B", prompt, meta={"camera": cm.off})]
        kept = {}

        def before(step, tr):
            cm.before_track(step, tr)
            if step in args.steps:
                kept[(tr.id, step)] = tr.latents.detach().clone()

        run_lockstep(loaded, [], tracks, noise, before_track=before)
        cm.reset()
        timesteps = sched.timesteps
        cond = {tr.id: tr.embeds[cfg.frames:] for tr in tracks}             # [negative, positive] -> positive
        for step in args.steps:
            t = timesteps[step]

            def call(tid):
                def run():
                    cm.before_track(step, next(tr for tr in tracks if tr.id == tid))
                    x = sched.scale_model_input(kept[(tid, step)], t)
                    if tid == "B":          # weights are frozen; the graph runs from the input through the activations
                        x = x.detach().requires_grad_(True)
                    return unet(x, t, encoder_hidden_states=cond[tid]).sample
                return run

            torch.cuda.reset_peak_memory_stats()
            res = atp_step(unet, names, call("A"), call("B"))
            cm.reset()
            for site, s in res["scores"].items():
                rows.append({"model": cm.name, "pair": f"{camera}__{scene}", "camera": camera, "scene": scene,
                             "step": step, "site": site, "kind": kind[site][0], "block": kind[site][1],
                             "score": s, "frac": res["frac"][site], "gap": res["gap"]})
            top = sorted(res["frac"].items(), key=lambda kv: -abs(kv[1]))[:3]
            print(f"{camera:<18} {scene[:24]:<24} step {step:>2}  gap {res['gap']:.3g}  peak "
                  f"{torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB  top " +
                  ", ".join(f"{k.replace('_blocks', '')}={v:+.3f}" for k, v in top), flush=True)
        with open(args.out / "atp.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    out = {"block_frac_steps_0_7": block_sums(rows)}
    if args.compare:
        out["vs_patch"] = compare(rows, args.compare, cm.name)
        (args.out / "atp_vs_patch.json").write_text(json.dumps(out["vs_patch"], indent=2))
    runio.write_json(args.out / "manifest.json", runio.record(
        "attribution_screen", loaded, model=cm.name, seed=seed, steps=args.steps, frames=cfg.frames, shard=args.shard,
        pairs=[f"{c}__{s}" for c, s in pairs[i::n]], n_sites=len(names), seconds=time.perf_counter() - t0, **out))


if __name__ == "__main__":
    main()
