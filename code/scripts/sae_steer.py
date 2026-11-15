"""Part C: steer the camera with a direction at the motion site, and ablate camera features (HS2).

Needs the outputs of scripts/sae_train.py (`directions.pt`, `default/sae.safetensors`). Model: the
MotionLoRA cell; steering arms run **without** a LoRA. Held-out seed, one lockstep group per scene.

Steering arms, for each camera in --cameras (site and steps of the SAE: update of
up_blocks.1.motion_modules.2, steps 0-7), each direction scaled to |mean-difference vector| x strength:
    base                        nothing added
    mean_<cam>_x<s>             the mean-difference vector of that camera (mean token of the camera minus
                                mean token of no camera, training seed)
    feat_<cam>_x<s>             decoder direction of the camera's top SAE feature
    rand_<cam>_x<s>             a random direction (same for every scene)
Ablation arms (the LoRA is on):
    lora_<cam>                  the camera clip
    lora_<cam>_abl_top          the camera's top --top features zeroed in the update
    lora_<cam>_abl_rand         as many random features with matched firing frequency

rows.csv: scene, camera, arm, kind, strength, tx, ty, zoom, roll, shift (signature change from base,
projected on the LoRA clip's signature direction and divided by its length: 1 = the full LoRA camera),
clip_sim_to_base.

    python scripts/sae_steer.py --sae input/vdprobe-cam-sae --scenes 1 --strengths 2      # smoke
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from vdprobe import runio
from vdprobe import sae as S
from vdprobe.camera_models import default_config, load_camera_model
from vdprobe.coords import enumerate_coordinates
from vdprobe.lockstep import Track, run_lockstep
from vdprobe.metrics import ClipScorer, camera_signature, content_fidelity, raft_flow
from vdprobe.pipeline import decode, make_noise
from vdprobe.steer import AddDirection, FeatureAblate

SITE = "up_blocks.1.motion_modules.2"
STEPS = tuple(range(8))


def matched_random(freq: np.ndarray, top: list[int], rng) -> list[int]:
    """For each feature of `top`, a different feature with the closest firing frequency among a random half
    of the remaining ones (so the control removes about as much activity)."""
    pool = [i for i in rng.permutation(len(freq)) if i not in set(top)][:max(len(freq) // 2, len(top))]
    out = []
    for t in top:
        j = min(pool, key=lambda i: abs(freq[i] - freq[t]))
        out.append(int(j))
        pool.remove(j)
    return out


def steering_arms(directions: dict, cameras, strengths, n_channels: int, seed: int = 643) -> dict:
    """{arm: (kind, camera, strength, direction (C,))}: every direction has the mean-difference vector's norm."""
    g = torch.Generator().manual_seed(seed)
    arms = {}
    for cam in cameras:
        mean = directions["mean_diff"][cam].float()
        norm = mean.norm()
        unit = lambda v: v.float() / v.float().norm() * norm
        feat = directions["features"][cam]["decoder"][0]
        rand = torch.randn(n_channels, generator=g)
        for s in strengths:
            arms[f"mean_{cam}_x{s:g}"] = ("mean", cam, s, mean)
            arms[f"feat_{cam}_x{s:g}"] = ("feature", cam, s, unit(feat))
            arms[f"rand_{cam}_x{s:g}"] = ("random", cam, s, unit(rand))
    return arms


def shift(sig, base, full):
    d = full - base
    n = float(d @ d)
    return float((sig - base) @ d / n) if n > 1e-8 else float("nan")


def main():
    p = runio.parser(__doc__, "sae_steer")
    p.set_defaults(config=None)
    p.add_argument("--sae", type=Path, required=True, help="output directory of scripts/sae_train.py")
    p.add_argument("--cameras", nargs="+", default=["pan-left", "zoom-in"])
    p.add_argument("--scenes", type=int, default=4)
    p.add_argument("--strengths", type=float, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    hits = sorted(args.sae.rglob("directions.pt"))
    if not hits:
        raise SystemExit(f"no directions.pt under {args.sae}")
    root = hits[0].parent
    directions = torch.load(root / "directions.pt", map_location="cpu", weights_only=True)
    sae, norm, _ = S.load(root / "default" / "sae.safetensors")
    import pandas as pd
    freq = pd.read_csv(root / "features.csv").frequency.values

    cm = load_camera_model("lora", args.config or default_config("lora"), args.profile, cameras=args.cameras)
    loaded, cfg = cm.loaded, cm.loaded.cfg
    coords = enumerate_coordinates(loaded.unet)
    site = next(c for c in coords if c.name == SITE)
    seed = args.seed if args.seed is not None else cm.cameras.heldout_seed
    steer = steering_arms(directions, args.cameras, args.strengths, sae.d_in)
    rng = np.random.default_rng(643)
    ablate = {}
    for cam in args.cameras:
        top = directions["features"][cam]["index"][:args.top]
        ablate[cam] = {"top": top, "rand": matched_random(freq, top, rng)}
    clip_scorer = ClipScorer(device=loaded.device)
    from vdprobe.camera_set import CameraClip

    ok = True
    for scene in cm.cameras.scenes[:args.scenes]:
        out = args.out / scene.replace(" ", "-")
        if (out / "manifest.json").exists() and not args.force:
            print(f"[skip] {out.name}")
            continue
        prompt = CameraClip(scene, "none", seed).prompt(cfg)
        tracks = [Track("base", prompt, meta={"camera": cm.off})]
        probes = {}
        for arm, (kind, cam, s, d) in steer.items():
            probes[arm] = AddDirection(SITE, STEPS, d, s)
            tracks.append(Track(arm, prompt, [probes[arm]], meta={"camera": cm.off}))
        for cam in args.cameras:
            tracks.append(Track(f"lora_{cam}", prompt, meta={"camera": cam}))
            for which, feats in ablate[cam].items():
                arm = f"lora_{cam}_abl_{which}"
                probes[arm] = FeatureAblate(SITE, STEPS, sae, norm, feats)
                tracks.append(Track(arm, prompt, [probes[arm]], meta={"camera": cam}))
        print(f"\n=== {scene}: {len(tracks)} tracks", flush=True)
        t = run_lockstep(loaded, [site], tracks, make_noise(cfg, seed), before_track=cm.before_track, log_every=5)
        cm.reset()

        sigs, embs = {}, {}
        for tr in tracks:
            frames = decode(loaded, tr.latents)
            runio.save_gif(frames, out / "gifs" / f"{tr.id}.gif", size=192)
            sigs[tr.id] = camera_signature(raft_flow(frames, size=cm.flow_size, device=loaded.device))
            embs[tr.id] = clip_scorer.image(frames)
        rows = []
        for tr in tracks:
            kind, cam, s = ("base", "", 0.0) if tr.id == "base" else steer[tr.id][:3] if tr.id in steer else \
                ("lora" if "_abl_" not in tr.id else "ablate_" + tr.id.rsplit("_", 1)[1], tr.id.split("_")[1], 0.0)
            full = sigs.get(f"lora_{cam}")
            sig = sigs[tr.id]
            rows.append({"scene": scene, "camera": cam, "arm": tr.id, "kind": kind, "strength": s,
                         "tx": sig[0], "ty": sig[1], "zoom": sig[2], "roll": sig[3],
                         "shift": shift(sig, sigs["base"], full) if full is not None else float("nan"),
                         "clip_sim_to_base": content_fidelity(embs[tr.id], embs["base"]),
                         "applied": getattr(probes.get(tr.id), "applied", 0),
                         "removed_norm": getattr(probes.get(tr.id), "removed_norm", 0.0)})
            print(f"  {tr.id:<32} shift {rows[-1]['shift']:+.3f}  clip {rows[-1]['clip_sim_to_base']:.3f}", flush=True)
        checks = {"probes_applied": all(r["applied"] > 0 for r in rows if r["arm"] in probes)}
        ok = ok and all(checks.values())
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        runio.write_json(out / "manifest.json", runio.record(
            "sae_steer", loaded, scene=scene, seed=seed, site=SITE, steps=list(STEPS), cameras=args.cameras,
            strengths=args.strengths, ablate=ablate, checks=checks,
            direction_norm={c: float(directions["mean_diff"][c].norm()) for c in args.cameras},
            denoise_seconds=t["seconds"], n_tracks=len(tracks)))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
