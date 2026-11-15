"""Camera patching: does a region of the U-Net carry the camera, and only the camera? (Tasks A5, B4)

    --model lora   camera = MotionLoRA  (D-40, HC2, HC3)      --model cc   camera = CameraCtrl trajectory (D-41, HK3)

One pair = (camera c, scene X, scene Y), held-out seed, one noise tensor for every clip:

    A  = (X, camera c)     B  = (X, no camera)     B2 = (Y, no camera)

Arms (the camera of a target is off unless stated; the patched activations are the only way it can
learn about the camera):

    B<-A|cand     same scene                              sufficiency
    B2<-A|cand    other scene                             transfer and content leak
    A<-B|cand     A keeps its camera, receives B          necessity
    B<-B2|cand    no camera in the source (named candidates only)      control

Candidates (patch steps 0-7 unless stated):
    exp1_region             up_blocks.0.motion_modules.2, steps 0-2, dynamic part     named
    up1_motion_modules      every motion module of up_blocks.1, dynamic part          named
    up1_motion_modules_all  the same, whole tensor                                    named
    mm_<block>              every motion module of one block, whole tensor            first --block-pairs pairs
    attn_first_third        temporal attention probabilities, first third of the steps (B<-A, B2<-A)

With d = sig(A) - sig(B) (camera signatures, metrics.camera_signature) and Dc(x) = CLIP(x, prompt X) -
CLIP(x, prompt Y):

    eff_cam      suff / ctrl: (sig(x) - sig(B)) . d / |d|^2      transfer: (sig(x) - sig(B2)) . d / |d|^2
                 nec: (sig(A) - sig(x)) . d / |d|^2              (0: nothing, 1: the full camera)
    eff_cam_cos  the same with D(x) = cos(sig(x), sig(A)) - cos(sig(x), sig(target))  (secondary)
    eff_content  transfer: (Dc(x) - Dc(B2)) / (Dc(A) - Dc(B2));  ctrl: (Dc(x) - Dc(B)) / (Dc(B2) - Dc(B));
                 same-scene arms: NaN

Per pair <out>/<camera>__<scene X>/: rows.csv, sheet.png, sigs.json, manifest.json (written last).

    python scripts/camera_patch.py --pairs 1 --block-pairs 0 --null          # smoke
"""

import json
import time

import numpy as np
import pandas as pd
import torch

from vdprobe import metrics as M
from vdprobe import runio
from vdprobe.camera_models import FAMILIES, MODELS, default_config, load_camera_model
from vdprobe.coords import block_order, enumerate_coordinates
from vdprobe.edit import Arm, AttnSpec, EditSpec, run_arms, temporal_attention_modules
from vdprobe.patching import NULL_PATCH_MAX_LATENT_MAD, PatchSpec, window
from vdprobe.pipeline import make_noise

NAMED = ("exp1_region", "up1_motion_modules", "up1_motion_modules_all")
STEPS = tuple(range(8))
MIN_EFFECT = 1e-4


def candidates(coords) -> list[dict]:
    """Patch candidates for this U-Net: the three named ones, then one per block that has motion modules."""
    mm = lambda block: tuple(c.name for c in coords if c.kind == "temporal_module" and c.block == block)
    up1 = mm("up_blocks.1")
    out = [{"name": "exp1_region", "coords": ("up_blocks.0.motion_modules.2",), "steps": (0, 1, 2), "component": "dynamic"},
           {"name": "up1_motion_modules", "coords": up1, "steps": STEPS, "component": "dynamic"},
           {"name": "up1_motion_modules_all", "coords": up1, "steps": STEPS, "component": "all"}]
    out += [{"name": f"mm_{b}", "coords": mm(b), "steps": STEPS, "component": "all"} for b in block_order(coords) if mm(b)]
    return out


def draw_pairs(cs, cameras, n_scenes):
    """(camera, scene X, scene Y): Y is the next scene of the set. Scene-major, so a shard mixes cameras."""
    scenes = cs.scenes
    return [(c, scenes[i], scenes[(i + 1) % len(scenes)]) for i in range(n_scenes) for c in cameras]


def proj(v, d):
    n = float(d @ d)
    return float(v @ d / n) if n > MIN_EFFECT ** 2 else float("nan")


def score_pair(sigs: dict, dc: dict, cand_names) -> list[dict]:
    """Rows of one pair from {clip or arm: signature} and {clip or arm: Dc}. Arms missing from `sigs` are skipped."""
    d = sigs["A"] - sigs["B"]
    cos = M.camera_fidelity
    dcos = lambda x, target: cos(sigs[x], sigs["A"]) - cos(sigs[x], sigs[target])
    den_cos = dcos("A", "B") - dcos("B", "B")
    ratio = lambda num, den: num / den if abs(den) > 1e-6 else float("nan")
    rows = []
    for nm in cand_names:
        arms = {"suff": (f"B<-A|{nm}", "B"), "transfer": (f"B2<-A|{nm}", "B2"), "nec": (f"A<-B|{nm}", "A"),
                "ctrl": (f"B<-B2|{nm}", "B")}
        for direction, (key, target) in arms.items():
            if key not in sigs:
                continue
            if direction == "nec":
                eff, eff_cos = proj(sigs["A"] - sigs[key], d), ratio(dcos("A", "B") - dcos(key, "B"), den_cos)
            else:
                eff = proj(sigs[key] - sigs[target], d)
                eff_cos = ratio(dcos(key, target) - dcos(target, target), dcos("A", target) - dcos(target, target))
            content = float("nan")
            if direction == "transfer":
                content = ratio(dc[key] - dc["B2"], dc["A"] - dc["B2"])
            elif direction == "ctrl":
                content = ratio(dc[key] - dc["B"], dc["B2"] - dc["B"])
            rows.append({"candidate": nm, "direction": direction, "arm": key, "eff_cam": eff, "eff_cam_cos": eff_cos,
                         "eff_content": content, "den_cam": float(np.linalg.norm(d)), "den_content": dc["A"] - dc["B2"],
                         "Dc": dc[key]})
    return rows


def run_pair(cm, coords, cands, camera, x, y, seed, with_blocks, clip_scorer, args):
    loaded, cfg = cm.loaded, cm.loaded.cfg
    out = args.out / f"{camera}__{x.replace(' ', '-')}"
    if (out / "manifest.json").exists() and not args.force:
        print(f"[skip] {out.name} already done")
        return True
    from vdprobe.camera_set import CameraClip
    px, py = CameraClip(x, camera, seed).prompt(cfg), CameraClip(y, camera, seed).prompt(cfg)
    use = [c for c in cands if c["name"] in NAMED or with_blocks]
    named = [c for c in use if c["name"] in NAMED]
    spec = lambda cd: EditSpec(patch=PatchSpec(tuple(cd["coords"]), tuple(cd["steps"]), component=cd["component"]))
    attn = EditSpec(attn=AttnSpec(temporal_attention_modules(coords), window("first_third", cfg.steps)))
    on, off = {"camera": camera}, {"camera": cm.off}
    noise = make_noise(cfg, seed)
    runs = {   # run name: (source prompt, source meta, [(arm, target meta)])
        "A": (px, on, [(Arm("B", px, EditSpec()), off), (Arm("B2", py, EditSpec()), off)]
              + [(Arm(f"B<-A|{c['name']}", px, spec(c)), off) for c in named]
              + [(Arm(f"B2<-A|{c['name']}", py, spec(c)), off) for c in use]
              + [(Arm("B<-A|attn_first_third", px, attn), off), (Arm("B2<-A|attn_first_third", py, attn), off)]),
        "B": (px, off, [(Arm(f"A<-B|{c['name']}", px, spec(c)), on) for c in use]
              + ([(Arm("null", px, spec(next(c for c in cands if c["name"] == "up1_motion_modules_all"))), off)]
                 if args.null else [])),
        "B2": (py, off, [(Arm(f"B<-B2|{c['name']}", px, spec(c)), off) for c in named]),
    }
    frames, latents, seconds, checks = {}, {}, 0.0, {}
    for name, (src_prompt, src_meta, arms) in runs.items():
        meta = {"source": src_meta, **{a.id: m for a, m in arms}}
        res = run_arms(loaded, coords, src_prompt, [a for a, _ in arms], noise, track_meta=meta,
                       before_track=cm.before_track)
        cm.reset()
        seconds += res.seconds
        frames[name + "_src"] = res.frames["source"]
        latents[name + "_src"] = res.source_latents
        for a, _ in arms:
            frames[a.id], latents[a.id] = res.frames[a.id], res.latents[a.id]
            if a.spec.patch is not None and res.n_patched[a.id] == 0:
                checks[f"patched:{a.id}"] = False
        print(f"  run {name}: {len(arms) + 1} tracks, {res.seconds / 60:.1f} min", flush=True)
    frames["A"] = frames.pop("A_src")
    # the three unpatched clips exist twice (as a source and as a baseline arm): they must agree
    checks["B_arm_equals_B_source"] = M.latent_mad(latents["B"], latents["B_src"]) < NULL_PATCH_MAX_LATENT_MAD
    checks["B2_arm_equals_B2_source"] = M.latent_mad(latents["B2"], latents["B2_src"]) < NULL_PATCH_MAX_LATENT_MAD
    if args.null:
        checks["null_mad"] = M.latent_mad(latents["null"], latents["B_src"])
        checks["null_patch"] = checks["null_mad"] < NULL_PATCH_MAX_LATENT_MAD
    for k in ("B_src", "B2_src", "null"):
        frames.pop(k, None)

    sigs = {k: M.camera_signature(M.raft_flow(v, size=cm.flow_size, device=loaded.device)) for k, v in frames.items()}
    txt = clip_scorer.text([px, py])
    sign = torch.tensor([1.0, -1.0], device=txt.device)
    dc = {k: float((clip_scorer.image(v) @ txt.T).mean(0) @ sign) for k, v in frames.items()}
    names = [c["name"] for c in use] + ["attn_first_third"]
    rows = score_pair(sigs, dc, names)
    for r in rows:
        r.update(model=cm.name, pair=out.name, camera=camera, scene_a=x, scene_b=y, seed=seed)
        print(f"  {r['arm']:<36} cam {r['eff_cam']:+.3f}  content {r['eff_content']:+.3f}", flush=True)
    ok = all(v for v in checks.values() if isinstance(v, bool))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
    (out / "sigs.json").write_text(json.dumps({k: [float(z) for z in v] for k, v in sigs.items()}, indent=1))
    best = "up1_motion_modules_all"
    M.contact_sheet({k: frames[k] for k in ("A", "B", f"B<-A|{best}", "B2", f"B2<-A|{best}", f"A<-B|{best}")
                     if k in frames}).save(out / "sheet.png")
    if args.gifs:
        for k, v in frames.items():
            runio.save_gif(v, out / "gifs" / (k.replace("|", "_").replace("<-", "_from_") + ".gif"), size=192)
    runio.write_json(out / "manifest.json", runio.record(
        "camera_patch", loaded, model=cm.name, camera=camera, scene_a=x, scene_b=y, seed=seed, prompts=[px, py],
        candidates=[{**c, "coords": list(c["coords"]), "steps": list(c["steps"])} for c in use], checks=checks,
        passed=ok, sheet_rows=["A", "B", f"B<-A|{best}", "B2", f"B2<-A|{best}", f"A<-B|{best}"],
        denoise_seconds=seconds, n_tracks=sum(len(a) + 1 for _, _, a in runs.values()),
        peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30))
    print(f"{'PASS' if ok else 'FAIL'}  checks {checks}", flush=True)
    return ok


def main():
    p = runio.parser(__doc__, "camera_patch")
    p.set_defaults(config=None)
    p.add_argument("--model", choices=MODELS, default="lora")
    p.add_argument("--seed", type=int, default=None, help="default: the set's held-out seed")
    p.add_argument("--scenes", type=int, default=6, help="first N scenes as scene X")
    p.add_argument("--cameras", type=int, default=4, help="first N of zoom-in, pan-left, tilt-up, roll (clockwise)")
    p.add_argument("--pairs", type=int, default=0, help="smoke test: first N pairs only")
    p.add_argument("--block-pairs", type=int, default=12, help="the mm_<block> candidates run on the first N pairs")
    p.add_argument("--null", action="store_true", help="add the null patch (B receiving B) to every pair")
    p.add_argument("--gifs", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    f = FAMILIES[args.model]
    cameras = [f["zoom"][0], f["pan"][0], f["tilt"][0], f["roll"][0]][:args.cameras]
    cm = load_camera_model(args.model, args.config or default_config(args.model), args.profile, cameras=cameras)
    coords = enumerate_coordinates(cm.loaded.unet)
    cands = candidates(coords)
    print("candidates: " + ", ".join(f"{c['name']} ({len(c['coords'])})" for c in cands), flush=True)
    seed = args.seed if args.seed is not None else cm.cameras.heldout_seed
    pairs = draw_pairs(cm.cameras, cameras, args.scenes)
    pairs = [(k, *pr) for k, pr in enumerate(pairs)][:args.pairs or None]
    i, n = (int(v) for v in args.shard.split("/"))
    clip_scorer = M.ClipScorer(device=cm.loaded.device)
    t0, ok = time.perf_counter(), True
    for k, camera, x, y in pairs[i::n]:
        print(f"\n=== pair {k}: {camera} | {x} -> {y}", flush=True)
        ok = run_pair(cm, coords, cands, camera, x, y, seed, k < args.block_pairs, clip_scorer, args) and ok
    print(f"{len(pairs[i::n])} pairs in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
