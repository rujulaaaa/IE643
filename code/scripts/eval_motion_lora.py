"""Experiment 2 Phase 6 · Evaluate motion-customisation LoRA arms on new subjects.

    python scripts/eval_motion_lora.py --arms results/.../motion_lora/T_s0 results/.../R01 ... \
        --data data/hmdb51_cartwheel [--base]

Every arm (a directory holding motion_lora.safetensors; `--base` adds the model without a LoRA)
generates the same 12 clips at the frozen 512 x 512 protocol: 3 new subjects x 2 scenes x 2
seeds, "a {subject} {verb phrase} {scene}, high quality, detailed". Per clip:
  mfs_official / mfs_symmetric   Motion Fidelity Score against each reference (training +
                                 validation) clip, max over references (vdprobe.motion_eval)
  xclip_target                   X-CLIP zero-shot probability of the trained motion among 8 actions
  clip_t                         CLIP text alignment to the prompt (editability)
  leak                           max CLIP image similarity of the clip's mean frame to a reference
                                 clip's mean frame (appearance leakage; lower is better)
Writes <out>/eval.csv and <out>/eval_summary.json (per-arm means and medians).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from vdprobe import metrics as M
from vdprobe import motion_eval as ME
from vdprobe import quality as Q
from vdprobe import runio
from vdprobe.lora import load_motion_lora, remove_lora
from vdprobe.pipeline import generate, make_noise

SUBJECTS = ("young woman", "teenage boy", "robot")
SCENES = ("on a beach", "in a park")
SEEDS = (0, 1)
ACTIONS = ("doing a cartwheel", "doing a handstand", "doing a somersault", "running", "walking", "jumping",
           "standing still", "dancing")


def reference_frames(data: Path, size: int = 256) -> tuple[list[list[Image.Image]], dict]:
    import cv2

    man = json.loads((data / "manifest.json").read_text())
    out = []
    for c in man["clips"]:
        cap, frames = cv2.VideoCapture(str(data / c["path"])), []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        n = len(frames)
        idx = np.linspace(0, n - 1, 16).round().astype(int)
        h, w = frames[0].shape[:2]
        x0 = (w - h) // 2
        out.append([Image.fromarray(frames[i][:, x0:x0 + h]).resize((size, size)) for i in idx])
    return out, man


def main():
    p = runio.parser(__doc__, "motion_lora_eval")
    p.add_argument("--arms", nargs="*", type=Path, default=[])
    p.add_argument("--base", action="store_true")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--grid", type=int, default=30, help="CoTracker grid (points per side)")
    args = p.parse_args()
    loaded = runio.load(args)
    cfg, dev = loaded.cfg, loaded.device
    refs, man = reference_frames(args.data)
    verb_phrase = next(iter(man["classes"].values()))
    clip = M.ClipScorer(device=dev)
    xclip = Q.XClipVerbs(list(ACTIONS), device=dev)
    ref_tracks = [ME.tracklets(r, args.grid, dev) for r in refs]
    ref_emb = torch.stack([clip.image(r).mean(0) for r in refs])
    ref_emb = torch.nn.functional.normalize(ref_emb, dim=-1)

    arms = ([("base", None)] if args.base else []) + [(a.name, a / "motion_lora.safetensors") for a in args.arms]
    rows = []
    for name, path in arms:
        if path is not None:
            load_motion_lora(loaded.unet, path)
        for subj in SUBJECTS:
            for scene in SCENES:
                for seed in SEEDS:
                    prompt = cfg.prompt(subj, verb_phrase, scene)
                    frames = generate(loaded, prompt, make_noise(cfg, seed)).frames
                    small = [f.resize((256, 256)) for f in frames]
                    tr = ME.tracklets(small, args.grid, dev)
                    mfs = [ME.motion_fidelity_score(tr, r) for r in ref_tracks]
                    emb = clip.image(frames)
                    probs = xclip(frames, subj)
                    rows.append({"arm": name, "subject": subj, "scene": scene, "seed": seed,
                                 "mfs_official": max(m["mfs_official"] for m in mfs),
                                 "mfs_symmetric": max(m["mfs_symmetric"] for m in mfs),
                                 "xclip_target": float(probs[0]),
                                 "clip_t": float((emb @ clip.text([prompt])[0]).mean()),
                                 "leak": float((torch.nn.functional.normalize(emb.mean(0), dim=-1) @ ref_emb.T).max())})
                    if subj == SUBJECTS[0] and scene == SCENES[0] and seed == SEEDS[0]:
                        runio.save_gif(frames, args.out / "gifs" / f"{name}.gif")
        print(f"{name}: " + ", ".join(f"{k} {np.mean([r[k] for r in rows if r['arm'] == name]):.3f}"
                                     for k in ("mfs_official", "mfs_symmetric", "xclip_target", "clip_t", "leak")),
              flush=True)
        if path is not None:
            remove_lora(loaded.unet)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "eval.csv", index=False)
    summ = df.groupby("arm")[["mfs_official", "mfs_symmetric", "xclip_target", "clip_t", "leak"]].agg(["mean", "median"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    runio.write_json(args.out / "eval_summary.json", runio.record(
        "motion_lora_eval", loaded, arms=[a for a, _ in arms], data=str(args.data), verb_phrase=verb_phrase,
        subjects=SUBJECTS, scenes=SCENES, seeds=SEEDS, actions=ACTIONS, grid=args.grid,
        summary=json.loads(summ.to_json(orient="index"))))


if __name__ == "__main__":
    main()
