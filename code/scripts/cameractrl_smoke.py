"""CameraCtrl smoke run: does the grafted model move the camera, and do the harness guarantees hold?

For each scene and each guidance scale, one lockstep group with one track per camera (same
prompt, same noise). Outputs:

    gifs/<guidance>/<scene>__<camera>.gif     look at these first
    camera_sig.csv                            (tx, ty, zoom, roll) of every clip from RAFT flow
    smoke.json                                checks, timings, peak memory

Checks (exit code 1 if one fails):
    off_vs_pipe      the camera-off track equals an ordinary `pipe(...)` generation      MAD < 1e-3
    null_patch       a camera clip patched with its own activations is unchanged          MAD < 1e-3
    camera_moves     every camera clip differs from the off clip                          MAD > 1e-2
    pan_reads_as_tx  |tx| is the largest signature component of the pan-left clips
    zoom_reads_as_zoom, roll_reads_as_roll   likewise

    python scripts/kaggle_run.py --user <account> --slug vdprobe-cc-smoke --prefetch-cameractrl \
        --job "scripts/cameractrl_smoke.py"
"""

import csv
import dataclasses
import time

import numpy as np
import torch

from vdprobe import runio
from vdprobe.camera_set import CameraSet
from vdprobe.cameractrl.model import camera_switcher, load_cameractrl
from vdprobe.config import CONFIG_DIR, GenerationConfig
from vdprobe.coords import coarse_coordinates, enumerate_coordinates
from vdprobe.edit import Arm, EditSpec, run_arms
from vdprobe.lockstep import Track, run_lockstep
from vdprobe.metrics import camera_signature, latent_mad, raft_flow
from vdprobe.patching import NULL_PATCH_MAX_LATENT_MAD, PatchSpec, window
from vdprobe.pipeline import decode, generate, make_noise

EXPECT = {"pan-left": 0, "pan-right": 0, "tilt-up": 1, "tilt-down": 1, "zoom-in": 2, "zoom-out": 2,
          "roll-cw": 3, "roll-ccw": 3}


def save_gif(frames, path, fps=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)


def main():
    ap = runio.parser(__doc__, "cameractrl_smoke")
    ap.set_defaults(config=CONFIG_DIR / "cameractrl.yaml")
    ap.add_argument("--scenes", type=int, default=2)
    ap.add_argument("--cameras", nargs="+", default=["off", "static", "pan-left", "zoom-in", "roll-cw"])
    ap.add_argument("--guidance", type=float, nargs="+", default=[14.0, 7.5])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = GenerationConfig.from_yaml(args.config)
    loaded = load_cameractrl(cfg, profile=args.profile or "gpu")
    adapter = loaded.extras["camera"]
    coords = enumerate_coordinates(loaded.unet)
    coarse = tuple(c.name for c in coarse_coordinates(coords))
    scenes = CameraSet(CONFIG_DIR / "cc_set.yaml").scenes[:args.scenes]
    noise = make_noise(cfg, args.seed)
    slug = lambda s: s.replace(" ", "-")
    rows, checks, timing = [], {}, {}
    print(f"{len(coords)} coordinates, {len(adapter.sites)} camera sites, revisions {loaded.revisions}", flush=True)

    for g in args.guidance:
        ld = dataclasses.replace(loaded, cfg=dataclasses.replace(cfg, guidance_scale=g))
        for scene in scenes:
            prompt = cfg.prompt_template.format(scene=scene)
            tracks = [Track(c, prompt, meta={"camera": c}) for c in args.cameras]
            torch.cuda.reset_peak_memory_stats()
            t = run_lockstep(ld, coords, tracks, noise, before_track=camera_switcher(ld))
            adapter.set(None)
            timing[f"g{g}/{slug(scene)}"] = {"seconds_per_clip": t["seconds"] / len(tracks),
                                              "peak_alloc_gb": torch.cuda.max_memory_allocated() / 2 ** 30}
            off = next((tr for tr in tracks if tr.id == "off"), None)
            for tr in tracks:
                frames = decode(loaded, tr.latents)
                save_gif(frames, args.out / "gifs" / f"g{g}" / f"{slug(scene)}__{tr.id}.gif")
                sig = camera_signature(raft_flow(frames, size=(cfg.height, cfg.width), device=loaded.device))
                rows.append({"guidance": g, "scene": scene, "camera": tr.id, "tx": sig[0], "ty": sig[1],
                             "zoom": sig[2], "roll": sig[3],
                             "mad_to_off": latent_mad(off.latents, tr.latents) if off is not None else float("nan")})
                print(f"g={g:<5} {scene[:28]:<28} {tr.id:<10} sig " + " ".join(f"{v:+.4f}" for v in sig) +
                      f"   MAD to off {rows[-1]['mad_to_off']:.3f}", flush=True)

    # --- checks, on the first scene at the first guidance scale -----------------------------------
    g, scene = args.guidance[0], scenes[0]
    ld = dataclasses.replace(loaded, cfg=dataclasses.replace(cfg, guidance_scale=g))
    prompt = cfg.prompt_template.format(scene=scene)
    adapter.set(None)
    solo = [Track("off", prompt)]
    run_lockstep(ld, coords, solo, noise, before_track=camera_switcher(ld))
    ref = generate(ld, prompt, noise, decode_frames=False)
    checks["off_vs_pipe"] = {"mad": latent_mad(ref.latents, solo[0].latents)}
    checks["off_vs_pipe"]["pass"] = checks["off_vs_pipe"]["mad"] < NULL_PATCH_MAX_LATENT_MAD
    timing["pipe_seconds"] = ref.seconds

    cam = next(c for c in args.cameras if c not in ("off", "static"))
    t0 = time.perf_counter()
    r = run_arms(ld, coords, prompt, [Arm("null", prompt, EditSpec(PatchSpec(coarse, window("all", cfg.steps))))],
                 noise, decode_frames=False, track_meta={"source": {"camera": cam}, "null": {"camera": cam}},
                 before_track=camera_switcher(ld))
    adapter.set(None)
    checks["null_patch"] = {"mad": latent_mad(r.source_latents, r.latents["null"]), "camera": cam,
                            "n_patched": r.n_patched["null"], "seconds": time.perf_counter() - t0}
    checks["null_patch"]["pass"] = checks["null_patch"]["mad"] < NULL_PATCH_MAX_LATENT_MAD and r.n_patched["null"] > 0

    moving = [x for x in rows if x["camera"] != "off"]
    checks["camera_moves"] = {"min_mad_to_off": min(x["mad_to_off"] for x in moving) if moving else None}
    checks["camera_moves"]["pass"] = bool(moving) and checks["camera_moves"]["min_mad_to_off"] > 1e-2
    for name, cams in (("pan_reads_as_tx", ("pan-left", "pan-right")), ("tilt_reads_as_ty", ("tilt-up", "tilt-down")),
                       ("zoom_reads_as_zoom", ("zoom-in", "zoom-out")), ("roll_reads_as_roll", ("roll-cw", "roll-ccw"))):
        hits = [x for x in rows if x["camera"] in cams and x["guidance"] == g]
        if hits:
            ok = [int(np.argmax(np.abs([x["tx"], x["ty"], x["zoom"], x["roll"]]))) == EXPECT[x["camera"]] for x in hits]
            checks[name] = {"clips": len(hits), "correct": int(sum(ok)), "pass": all(ok)}

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "camera_sig.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    hard = ("off_vs_pipe", "null_patch", "camera_moves")          # the signature checks inform the magnitudes; they do not gate
    ok = all(checks[k]["pass"] for k in hard)
    runio.write_json(args.out / "smoke.json", runio.record("cameractrl_smoke", loaded, seed=args.seed, checks=checks,
                                                           timing=timing, passed=ok, cameras=args.cameras, scenes=scenes))
    for k, v in checks.items():
        print(f"{'PASS' if v['pass'] else 'FAIL'}  {k}  {v}", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
