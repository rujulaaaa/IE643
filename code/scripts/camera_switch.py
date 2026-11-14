"""When and where is camera motion fixed? Switching the camera by step and by block (Tasks A4, B4).

    --model lora   camera MotionLoRA on / off per step; LoRA kept in one block only         (D-40, HC1)
    --model cc     CameraCtrl trajectory on / off per step; per block group: only / without /
                   a different camera in that group                                          (D-41, HK1, HK2)

One lockstep group per (scene, camera), held-out seed: every track has the same prompt and noise.

    on, off                    camera at every step / never
    on_until_<tau>             camera for steps < tau                      (--arms switch)
    on_from_<tau>              camera for steps >= tau
    only_<block>               camera in the modules of one block only     (--arms only)
    without_<block>            camera everywhere except that block         (--arms without)
    on_B, swap_<block>,        cc only: camera B in that block (or every block: swap_all, which must
    swap_all                   equal on_B), camera A elsewhere              (--arms swap)

    retained   = (sig - sig_off) . (sig_on - sig_off) / |sig_on - sig_off|^2        (1 for on, 0 for off)
    retained_b = (sig - sig_on)  . (sig_B  - sig_on)  / |sig_B  - sig_on|^2         (swap arms: 1 = reads as B)

Per group <out>/<scene>__<camera>/: rows.csv, gifs/, latents.pt, manifest.json (written last).

    python scripts/kaggle_run.py --user <account> --slug vdprobe-cam-switch \
        --job "scripts/camera_switch.py --shard 0/2" --job "scripts/camera_switch.py --shard 1/2"
    python scripts/camera_switch.py --scenes 1 --cameras 1 --taus 3            # smoke
"""

import csv
import time

import numpy as np
import torch

from vdprobe import runio
from vdprobe.camera_models import FAMILIES, MODELS, default_config, load_camera_model
from vdprobe.coords import enumerate_coordinates
from vdprobe.lockstep import Track, run_lockstep
from vdprobe.metrics import ClipScorer, camera_signature, content_fidelity, latent_mad, raft_flow
from vdprobe.patching import NULL_PATCH_MAX_LATENT_MAD
from vdprobe.pipeline import decode, make_noise

TAUS = (1, 2, 3, 5, 8, 12, 18)
BLOCKS = {"lora": ("down_blocks", "mid_block", "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"),
          "cc": ("down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3",
                 "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3")}
SWAP_TO = {"pan-left": "pan-right", "zoom-in": "zoom-out", "tilt-up": "roll-cw"}      # cc: camera A -> camera B
ARMS = ("switch", "only", "without", "swap")
MIN_EFFECT = 1e-4      # |sig_on - sig_off| below this: the camera did nothing measurable, retained is NaN


def until(camera, tau):
    return lambda step: camera if step < tau else None


def since(camera, tau):
    return lambda step: camera if step >= tau else None


def default_cameras(model):
    """One camera per motion type: zoom-in, pan-left, tilt-up and the clockwise roll."""
    f = FAMILIES[model]
    return [f["zoom"][0], f["pan"][0], f["tilt"][0], f["roll"][0]]


def build_tracks(cm, camera, taus, arms, blocks, swap_sets=()):
    """{arm name: Track.meta} for one (scene, camera) group, in run order."""
    metas = {"on": {"camera": camera}, "off": {"camera": cm.off}}
    if "switch" in arms:
        for tau in taus:
            metas[f"on_until_{tau}"] = {"camera": until(camera, tau)}
        for tau in taus:
            metas[f"on_from_{tau}"] = {"camera": since(camera, tau)}
    if "only" in arms:
        for b in blocks:
            metas[f"only_{b}"] = cm.block_only(camera, b)
    if "without" in arms:
        for b in blocks:
            metas[f"without_{b}"] = cm.block_without(camera, b)
    if "swap" in arms and cm.name == "cc" and camera in SWAP_TO:
        from vdprobe.cameractrl.adapter import CameraSpec
        other = SWAP_TO[camera]
        metas["on_B"] = {"camera": other}
        for b in blocks:
            metas[f"swap_{b}"] = {"camera": CameraSpec(camera, block_trajectories={b: other})}
        for group in swap_sets:
            metas["swap_" + "+".join(group)] = {"camera": CameraSpec(camera, block_trajectories={b: other for b in group})}
        metas["swap_all"] = {"camera": CameraSpec(camera, block_trajectories={b: other for b in blocks})}
    return metas


def retained(sig, base, full):
    d = full - base
    n = float(d @ d)
    return float((sig - base) @ d / n) if n > MIN_EFFECT ** 2 else float("nan")


def arm_fields(arm):
    """(kind, tau, block) of an arm name."""
    for kind in ("on_until", "on_from"):
        if arm.startswith(kind + "_"):
            return kind, int(arm[len(kind) + 1:]), ""
    for kind in ("only", "without", "swap"):
        if arm.startswith(kind + "_"):
            return kind, "", arm[len(kind) + 1:]
    return arm, "", ""


def score(sigs, clip_sim):
    """Rows of one group from {arm: signature} and {arm: CLIP similarity to off}."""
    rows = []
    for arm, sig in sigs.items():
        kind, tau, block = arm_fields(arm)
        rows.append({"arm": arm, "kind": kind, "tau": tau, "block": block,
                     "tx": sig[0], "ty": sig[1], "zoom": sig[2], "roll": sig[3],
                     "retained": retained(sig, sigs["off"], sigs["on"]),
                     "retained_b": retained(sig, sigs["on"], sigs["on_B"]) if "on_B" in sigs else float("nan"),
                     "clip_sim_to_off": clip_sim[arm]})
    return rows


def run_group(cm, coords, scene, camera, seed, clip_scorer, args):
    loaded = cm.loaded
    out = args.out / f"{scene.replace(' ', '-')}__{camera}"
    if (out / "manifest.json").exists() and not args.force:
        print(f"[skip] {out.name} already done")
        return True
    prompt = loaded.cfg.prompt_template.format(scene=scene) if "{subject}" not in loaded.cfg.prompt_template \
        else f"{scene}, high quality, detailed"
    metas = build_tracks(cm, camera, args.taus, args.arms, BLOCKS[cm.name], args.swap_sets)
    tracks = [Track(arm, prompt, meta=dict(meta)) for arm, meta in metas.items()]
    print(f"\n=== {out.name}: {len(tracks)} tracks", flush=True)
    t = run_lockstep(loaded, coords, tracks, make_noise(loaded.cfg, seed), before_track=cm.before_track, log_every=5)
    cm.reset()

    sigs, embs, latents = {}, {}, {}
    for tr in tracks:
        frames = decode(loaded, tr.latents)
        runio_gif(frames, out / "gifs" / f"{tr.id}.gif")
        sigs[tr.id] = camera_signature(raft_flow(frames, size=cm.flow_size, device=loaded.device))
        embs[tr.id] = clip_scorer.image(frames)
        latents[tr.id] = tr.latents.half()
    rows = score(sigs, {a: content_fidelity(e, embs["off"]) for a, e in embs.items()})
    for r in rows:
        r.update(scene=scene, camera=camera, seed=seed, model=cm.name,
                 mad_to_on=latent_mad(latents[r["arm"]].float(), latents["on"].float()))
        print(f"  {r['arm']:<24} retained {r['retained']:+.3f}  b {r['retained_b']:+.3f}  "
              f"clip {r['clip_sim_to_off']:.3f}", flush=True)

    checks = {"camera_changes_clip": latent_mad(latents["on"].float(), latents["off"].float()) > 1e-2}
    if "swap_all" in latents:
        mad = latent_mad(latents["swap_all"].float(), latents["on_B"].float())
        checks["swap_all_equals_on_B"] = mad < NULL_PATCH_MAX_LATENT_MAD
        checks["swap_all_mad"] = mad
    ok = all(v for k, v in checks.items() if isinstance(v, bool))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "scene", "camera", "seed"] +
                           [k for k in rows[0] if k not in ("model", "scene", "camera", "seed")])
        w.writeheader()
        w.writerows(rows)
    torch.save(latents, out / "latents.pt")
    runio.write_json(out / "manifest.json", runio.record(
        "camera_switch", loaded, model=cm.name, scene=scene, camera=camera, seed=seed, prompt=prompt,
        arms=list(metas), taus=list(args.taus), blocks=list(BLOCKS[cm.name]), checks=checks, passed=ok,
        weak_camera=bool(np.linalg.norm(sigs["on"] - sigs["off"]) <= MIN_EFFECT),
        denoise_seconds=t["seconds"], seconds_per_clip=t["seconds"] / len(tracks),
        peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
    print(f"{'PASS' if ok else 'FAIL'}  checks {checks}", flush=True)
    return ok


def runio_gif(frames, path, height=192, fps=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    small = [f.resize((round(w * height / h), height)) for f in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0)


def main():
    p = runio.parser(__doc__, "camera_switch")
    p.set_defaults(config=None)
    p.add_argument("--model", choices=MODELS, default="lora")
    p.add_argument("--seed", type=int, default=None, help="default: the set's held-out seed")
    p.add_argument("--scenes", type=int, default=4, help="first N scenes of the set")
    p.add_argument("--cameras", type=int, default=4, help="first N of zoom-in, pan-left, tilt-up, roll (clockwise)")
    p.add_argument("--taus", type=int, nargs="+", default=list(TAUS))
    p.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    p.add_argument("--swap-sets", nargs="*", default=[], help="cc: extra swap arms over several blocks, each 'b1+b2+b3'")
    p.add_argument("--shard", default="0/1", help="i/n: run groups i, i+n, ...")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    args.swap_sets = [tuple(s.split("+")) for s in args.swap_sets]
    for group in args.swap_sets:
        if not set(group) <= set(BLOCKS[args.model]):
            raise SystemExit(f"--swap-sets: unknown block in {group}")

    cameras = default_cameras(args.model)[:args.cameras]
    cm = load_camera_model(args.model, args.config or default_config(args.model), args.profile, cameras=cameras)
    coords = enumerate_coordinates(cm.loaded.unet)
    seed = args.seed if args.seed is not None else cm.cameras.heldout_seed
    groups = [(s, c) for s in cm.cameras.scenes[:args.scenes] for c in cameras]
    i, n = (int(x) for x in args.shard.split("/"))
    clip_scorer = ClipScorer(device=cm.loaded.device)
    t0, ok = time.perf_counter(), True
    for scene, camera in groups[i::n]:
        ok = run_group(cm, coords, scene, camera, seed, clip_scorer, args) and ok
    print(f"{len(groups[i::n])} groups in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
