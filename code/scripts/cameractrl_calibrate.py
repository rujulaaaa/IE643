"""CameraCtrl: choose the trajectory magnitudes (one step size per motion type) before they are frozen in D-41.

The smoke run showed that one step for every translation does not fit: at 0.1 scene units per frame pans
are strong and smear the picture, zooms and tilts barely register. Here each translation runs at several
step sizes and each roll at several angles, all from the same noise and prompt (one lockstep group per scene).

Per clip, in calib.csv:
    tx, ty, zoom, roll   camera signature (metrics.camera_signature, RAFT at 256 x 384)
    main                 the component the camera should drive, signed so that the commanded direction is positive
    off_axis             largest of the other three components (absolute)
    sharpness            mean variance of the Laplacian over frames, relative to the static clip (1 = as sharp)
    clip_sim_to_off      CLIP image similarity to the camera-off clip (content kept)

Rule used to pick (written before the run): per motion type, the smallest step whose median `main` over scenes
is >= 0.03 (a pan of 1.5 % of the frame height per frame; the released 2 degrees per frame of roll gives 0.035),
provided median sharpness >= 0.8; otherwise the step with the largest `main` among those with sharpness >= 0.8.

    python scripts/kaggle_run.py --user <account> --slug vdprobe-cc-calib --prefetch-cameractrl \\
        --job "scripts/cameractrl_calibrate.py --scenes 0 1" --job "scripts/cameractrl_calibrate.py --scenes 2 3"
"""

import csv

import numpy as np
import torch

from vdprobe import runio
from vdprobe.camera_set import CameraSet
from vdprobe.cameractrl.model import camera_switcher, load_cameractrl
from vdprobe.cameractrl.rays import trajectory
from vdprobe.config import CONFIG_DIR, GenerationConfig
from vdprobe.coords import enumerate_coordinates
from vdprobe.lockstep import Track, run_lockstep
from vdprobe.metrics import ClipScorer, camera_signature, content_fidelity, raft_flow
from vdprobe.pipeline import decode, make_noise

STEPS = (0.025, 0.05, 0.1, 0.2, 0.4)
ROLLS = (1.0, 2.0, 4.0)
# camera -> (signature component, sign of the commanded direction); signs as measured in the smoke run
AXIS = {"pan-left": (0, 1), "pan-right": (0, -1), "tilt-up": (1, 1), "tilt-down": (1, -1),
        "zoom-in": (2, 1), "zoom-out": (2, -1), "roll-cw": (3, -1), "roll-ccw": (3, 1)}
MIN_MAIN, MIN_SHARP = 0.03, 0.8


def sharpness(frames) -> float:
    """Mean variance of a 3x3 Laplacian of the grey frames."""
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])[None, None]
    x = torch.stack([torch.from_numpy(np.asarray(f.convert("L"), np.float32) / 255) for f in frames])[:, None]
    return float(torch.nn.functional.conv2d(x, k).var(dim=(1, 2, 3)).mean())


def tracks_for(prompt, steps, rolls, frames):
    out = [Track("off", prompt, meta={"camera": None}), Track("static", prompt, meta={"camera": "static"})]
    for cam, (_, _) in AXIS.items():
        for v in (rolls if cam.startswith("roll") else steps):
            t = trajectory(cam, frames, step=v, roll_deg=v) if cam.startswith("roll") else trajectory(cam, frames, step=v)
            out.append(Track(f"{cam}@{v:g}", prompt, meta={"camera": t, "cam": cam, "value": v}))
    return out


def choose(rows) -> dict:
    """The rule of the docstring, per motion type (pan, tilt, zoom on steps; roll on degrees)."""
    import pandas as pd
    df = pd.DataFrame([r for r in rows if r["camera"] in AXIS])
    df["family"] = df.camera.str.split("-").str[0]
    out = {}
    for fam, d in df.groupby("family"):
        m = d.groupby("value").agg(main=("main", "median"), sharp=("sharpness", "median")).sort_index()
        ok = m[m.sharp >= MIN_SHARP]
        hit = ok[ok.main >= MIN_MAIN]
        pick = hit.index.min() if len(hit) else (ok.main.idxmax() if len(ok) else None)
        out[fam] = {"value": None if pick is None else float(pick), "reached_target": bool(len(hit)),
                    "table": {float(k): {"main": float(v.main), "sharpness": float(v.sharp)} for k, v in m.iterrows()}}
    return out


def main():
    ap = runio.parser(__doc__, "cameractrl_calibrate")
    ap.set_defaults(config=CONFIG_DIR / "cameractrl.yaml")
    ap.add_argument("--scenes", type=int, nargs="+", default=[0, 1], help="indices into the cc_set scenes")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=float, nargs="+", default=list(STEPS))
    ap.add_argument("--rolls", type=float, nargs="+", default=list(ROLLS))
    args = ap.parse_args()

    cfg = GenerationConfig.from_yaml(args.config)
    loaded = load_cameractrl(cfg, profile=args.profile or "gpu")
    adapter = loaded.extras["camera"]
    coords = enumerate_coordinates(loaded.unet)
    scenes = [CameraSet(CONFIG_DIR / "cc_set.yaml").scenes[i] for i in args.scenes]
    clip = ClipScorer(device=loaded.device)
    noise = make_noise(cfg, args.seed)
    rows = []
    for scene in scenes:
        prompt = cfg.prompt_template.format(scene=scene)
        tracks = tracks_for(prompt, args.steps, args.rolls, cfg.frames)
        print(f"\n=== {scene}: {len(tracks)} tracks", flush=True)
        t = run_lockstep(loaded, coords, tracks, noise, before_track=camera_switcher(loaded), log_every=5)
        adapter.set(None)
        sig, sharp, emb = {}, {}, {}
        for tr in tracks:
            frames = decode(loaded, tr.latents)
            w = round(192 * cfg.width / cfg.height)
            small = [f.resize((w, 192)) for f in frames]
            path = args.out / "gifs" / f"{scene.replace(' ', '-')}__{tr.id}.gif"
            path.parent.mkdir(parents=True, exist_ok=True)
            small[0].save(path, save_all=True, append_images=small[1:], duration=125, loop=0)
            sig[tr.id] = camera_signature(raft_flow(frames, size=(256, 384), device=loaded.device))
            sharp[tr.id], emb[tr.id] = sharpness(frames), clip.image(frames)
        for tr in tracks:
            cam = tr.meta.get("cam", tr.id)
            s = sig[tr.id]
            axis, sign = AXIS.get(cam, (None, 1))
            main = float(sign * s[axis]) if axis is not None else 0.0
            off_axis = float(np.abs(np.delete(s, axis)).max()) if axis is not None else float(np.abs(s).max())
            rows.append({"scene": scene, "track": tr.id, "camera": cam, "value": tr.meta.get("value", 0.0),
                         "tx": s[0], "ty": s[1], "zoom": s[2], "roll": s[3], "main": main, "off_axis": off_axis,
                         "sharpness": sharp[tr.id] / sharp["static"],
                         "clip_sim_to_off": content_fidelity(emb[tr.id], emb["off"]),
                         "seconds_per_clip": t["seconds"] / len(tracks)})
            r = rows[-1]
            print(f"  {tr.id:<16} main {r['main']:+.4f}  off-axis {r['off_axis']:.4f}  sharp {r['sharpness']:.2f}  "
                  f"clip {r['clip_sim_to_off']:.3f}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "calib.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    pick = choose(rows)
    runio.write_json(args.out / "calib.json", runio.record("cameractrl_calibrate", loaded, seed=args.seed, scenes=scenes,
                                                           steps=args.steps, rolls=args.rolls, rule={"min_main": MIN_MAIN,
                                                           "min_sharpness": MIN_SHARP}, choice=pick))
    for fam, v in pick.items():
        print(fam, v["value"], "reached" if v["reached_target"] else "NOT reached", v["table"], flush=True)


if __name__ == "__main__":
    main()
