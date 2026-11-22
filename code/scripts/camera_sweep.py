"""Camera probing sweep: static scenes x cameras, every pair scored (Tasks A3 and B4; D-40, D-41).

    --model lora   SD1.5 + AnimateDiff v2 + the eight camera MotionLoRAs   (configs/camera_set.yaml)
    --model cc     CameraCtrl, the camera as an input trajectory            (configs/cc_set.yaml)

One lockstep group per seed: all clips (scene x camera) share the seed's noise. A camera pair
holds the scene and changes the camera (kind "motion"); a content pair holds the camera and
changes the scene (kind "content"). Anchors (the no-camera clip of each scene; the anchor scene
under each camera) run first in every step. The prompt never mentions the camera.

Per group, written to <out>/s<seed>__camera/ in the format of scripts/probe_sweep.py, so
scripts/analyze_sweep.py reads it unchanged (subject = family = scene, verb = camera):
  pairs.json, distances.csv, per_channel.pt, sketches.pt, latents.pt, gifs/, manifest.json
plus
  camera_sig.csv   clip, scene, camera, tx, ty, zoom, roll   (RAFT flow, 256 px high)
  tokens.pt        --tokens: {clip id: {step: (F * 64, C) fp16}}, the update of --token-site
  verify.json      --verify: the no-camera track against a solo lockstep run and against pipe()

    python scripts/kaggle_run.py --user <account> --slug vdprobe-cam-smoke \
        --job "scripts/camera_sweep.py --seeds 42 --scenes 2 --cameras 3 --tokens --verify"
"""

import csv
import json
import resource
import time
from collections import defaultdict
from pathlib import Path

import torch

from vdprobe import runio
from vdprobe.camera_models import MODELS, default_config, load_camera_model
from vdprobe.coords import D07_STEPS, coarse_coordinates, enumerate_coordinates
from vdprobe.lockstep import PairScorer, Sketcher, StepCache, TokenSampler, Track, run_lockstep
from vdprobe.metrics import camera_signature, latent_mad, raft_flow
from vdprobe.patching import NULL_PATCH_MAX_LATENT_MAD
from vdprobe.pipeline import decode, generate, make_noise

TOKEN_SITE = "up_blocks.1.motion_modules.2"
TOKEN_STEPS = (0, 1, 2, 4, 8)


def topo_order(clips, pairs):
    """Anchors before the clips scored against them; otherwise the set's order."""
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


def save_gif(frames, path, height=256, fps=8):
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = frames[0].size
    small = [f.resize((round(w * height / h), height)) for f in frames]
    small[0].save(path, save_all=True, append_images=small[1:], duration=int(1000 / fps), loop=0)


def subset(cs, seed, n_scenes, n_cameras):
    scenes = set(cs.scenes[:n_scenes] if n_scenes else cs.scenes)
    cams = set(cs.cameras[:n_cameras] if n_cameras else cs.cameras)
    if cs.anchor_scene not in scenes or cs.anchor_camera not in cams:
        raise SystemExit("the subset must keep the anchor scene and the anchor camera (both are first in the config)")
    keep = lambda c: c.scene in scenes and c.camera in cams
    clips = [c for c in cs.clips(seed) if keep(c)]
    pairs = [p for p in cs.content_pairs(seed) + cs.camera_pairs(seed) if keep(p.a) and keep(p.b)]
    return clips, pairs


def verify(cm, coords, clip, group_latents, noise):
    """The no-camera clip of the group against (1) the same clip alone in a lockstep run and (2) pipe()."""
    loaded = cm.loaded
    prompt = clip.prompt(loaded.cfg)
    solo = [Track(clip.id, prompt, meta={"camera": clip.camera})]
    run_lockstep(loaded, coords, solo, noise, before_track=cm.before_track)
    cm.reset()
    ref = generate(loaded, prompt, noise, decode_frames=False)
    out = {"clip": clip.id,
           "group_vs_solo_max_abs": float((group_latents - solo[0].latents).abs().max()),
           "group_vs_solo_mad": latent_mad(group_latents, solo[0].latents),
           "group_vs_pipe_max_abs": float((group_latents - ref.latents).abs().max()),
           "group_vs_pipe_mad": latent_mad(group_latents, ref.latents)}
    out["pass"] = out["group_vs_solo_mad"] < NULL_PATCH_MAX_LATENT_MAD and out["group_vs_pipe_mad"] < NULL_PATCH_MAX_LATENT_MAD
    return out


def run_group(cm, coords, coarse, seed, args):
    loaded, cs = cm.loaded, cm.cameras
    out = args.out / f"s{seed}__camera"
    if (out / "manifest.json").exists() and not args.force:
        print(f"[skip] {out.name} already done")
        return True
    out.mkdir(parents=True, exist_ok=True)
    clips, pairs = subset(cs, seed, args.scenes, args.cameras)
    order = topo_order(clips, pairs)
    anchors = {p.a.id for p in pairs}
    print(f"\n=== {out.name}: {len(clips)} clips, {len(pairs)} pairs, {len(anchors)} anchors, {len(coarse)} coordinates",
          flush=True)

    steps = range(loaded.cfg.steps)
    caches = {a: StepCache(coarse) for a in anchors}
    refs = defaultdict(list)
    for i, p in enumerate(pairs):
        refs[p.b.id].append((i, caches[p.a.id]))
    token_coords = [c for c in coords if c.name == args.token_site]
    if args.tokens and not token_coords:
        raise SystemExit(f"--token-site {args.token_site} is not a coordinate of this U-Net")
    sketch, scorers, samplers, tracks = {}, {}, {}, []
    for c in order:
        sketch[c.id] = Sketcher(coarse, steps=steps, dim=args.sketch_dim)
        probes = [sketch[c.id]]
        if c.id in caches:
            probes.insert(0, caches[c.id])
        if refs[c.id]:
            scorers[c.id] = PairScorer(coarse, refs[c.id], per_channel_steps=D07_STEPS)
            probes.append(scorers[c.id])
        if args.tokens:
            samplers[c.id] = TokenSampler(token_coords, steps=args.token_steps)
            probes.append(samplers[c.id])
        tracks.append(Track(c.id, c.prompt(loaded.cfg), probes, meta={"clip": c, "camera": c.camera}))

    # free an anchor's step activations as soon as its last partner in this step has run
    n_partners = defaultdict(int)
    for p in pairs:
        n_partners[p.a.id] += 1
    owner = {id(v): k for k, v in caches.items()}
    remaining = {}

    def after_track(step, tr):
        if tr.id in caches and n_partners[tr.id] == 0:
            caches[tr.id].clear()
        for _, cache in refs[tr.id]:
            a = owner[id(cache)]
            remaining[a] = remaining.get(a, n_partners[a]) - 1
            if remaining[a] == 0:
                cache.clear()

    def after_step(step):
        remaining.clear()
        for c in caches.values():
            c.clear()

    noise = make_noise(loaded.cfg, seed)
    t = run_lockstep(loaded, coords, tracks, noise, after_track=after_track, after_step=after_step,
                     before_track=cm.before_track, log_every=5)
    cm.reset()
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
    print(f"  denoised in {t['seconds'] / 60:.1f} min ({t['seconds'] / len(tracks):.1f} s per clip), "
          f"peak host RSS {rss_gb:.1f} GB", flush=True)

    t1 = time.perf_counter()
    latents, sig_rows = {}, []
    for tr in tracks:
        c = tr.meta["clip"]
        latents[tr.id] = tr.latents.half()
        frames = decode(loaded, tr.latents)
        save_gif(frames, out / "gifs" / f"{tr.id}.gif")
        sig = camera_signature(raft_flow(frames, size=cm.flow_size, device=loaded.device))
        sig_rows.append({"clip": tr.id, "scene": c.scene, "camera": c.camera, "seed": seed,
                         "tx": sig[0], "ty": sig[1], "zoom": sig[2], "roll": sig[3]})
        print(f"  {c.camera:<22} {c.scene[:30]:<30} sig " + " ".join(f"{v:+.4f}" for v in sig), flush=True)
    decode_s = time.perf_counter() - t1

    names = [c.name for c in coarse]
    with open(out / "camera_sig.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sig_rows[0]))
        w.writeheader()
        w.writerows(sig_rows)
    rows = [r for s in scorers.values() for r in s.rows]
    if rows:
        with open(out / "distances.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    torch.save({k: v for s in scorers.values() for k, v in s.per_channel.items()}, out / "per_channel.pt")
    torch.save({cid: torch.stack([torch.stack([sk.out[(n, s)] for s in steps]) for n in names])
                for cid, sk in sketch.items()}, out / "sketches.pt")
    torch.save(latents, out / "latents.pt")
    if args.tokens:
        torch.save({cid: {s: sm.store[(args.token_site, s)] for s in args.token_steps} for cid, sm in samplers.items()},
                   out / "tokens.pt")
    (out / "pairs.json").write_text(json.dumps(
        [{"pair": i, "kind": p.kind, "a": p.a.id, "b": p.b.id, "family_a": p.a.scene, "family_b": p.b.scene,
          "subject_a": p.a.scene, "subject_b": p.b.scene, "verb_a": p.a.camera, "verb_b": p.b.camera,
          "scene_a": p.a.scene, "scene_b": p.b.scene, "camera_a": p.a.camera, "camera_b": p.b.camera}
         for i, p in enumerate(pairs)], indent=1))

    ok, ver = True, None
    if args.verify:
        anchor = next(c for c in clips if c.camera == cs.anchor_camera and c.scene == cs.anchor_scene)
        ver = verify(cm, coords, anchor, next(tr.latents for tr in tracks if tr.id == anchor.id), noise)
        ok = ver["pass"]
        runio.write_json(out / "verify.json", ver)
        print(f"{'PASS' if ok else 'FAIL'}  verify  {ver}", flush=True)

    runio.write_json(out / "manifest.json", runio.record(
        "camera_sweep", loaded, model=cm.name, seed=seed, scene="camera", n_clips=len(clips), n_pairs=len(pairs),
        coords=names, all_coords=[c.name for c in coords], sketch_dim=args.sketch_dim, clips=[c.id for c in clips],
        clip_meta={c.id: {"scene": c.scene, "camera": c.camera, "subject": c.scene, "verb": c.camera,
                          "family": c.scene} for c in clips},
        prompts={tr.id: tr.prompt for tr in tracks}, cameras=cs.cameras, anchor_camera=cs.anchor_camera,
        anchor_scene=cs.anchor_scene, token_site=args.token_site if args.tokens else None,
        token_steps=list(args.token_steps) if args.tokens else None, denoise_seconds=t["seconds"],
        seconds_per_clip=t["seconds"] / len(tracks), decode_seconds=decode_s, peak_rss_gb=rss_gb,
        peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30, verify=ver,
        subset={"scenes": args.scenes, "cameras": args.cameras}))
    return ok


def main():
    p = runio.parser(__doc__, "camera_sweep")
    p.set_defaults(config=None)
    p.add_argument("--model", choices=MODELS, default="lora")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--scenes", type=int, default=0, help="smoke test: first N scenes")
    p.add_argument("--cameras", type=int, default=0, help="smoke test: first N cameras (the first is no camera)")
    p.add_argument("--sketch-dim", type=int, default=512)
    p.add_argument("--tokens", action="store_true", help="store token samples of --token-site for the SAE")
    p.add_argument("--token-site", default=TOKEN_SITE)
    p.add_argument("--token-steps", type=int, nargs="+", default=list(TOKEN_STEPS))
    p.add_argument("--verify", action="store_true", help="check the no-camera track against a solo run and pipe()")
    p.add_argument("--force", action="store_true")
    p.add_argument("--set", type=Path, default=None, help="camera set file (default: the model's own; D-45 uses configs/av_set.yaml)")
    args = p.parse_args()

    from vdprobe.camera_set import CameraSet
    from vdprobe.camera_models import CONFIGS
    from vdprobe.config import CONFIG_DIR
    names = CameraSet(args.set or CONFIG_DIR / CONFIGS[args.model][1])
    attach = names.cameras[:args.cameras] if args.cameras else None
    cm = load_camera_model(args.model, args.config or default_config(args.model), args.profile,
                           cameras=[c for c in attach if c != names.anchor_camera] if attach else None,
                           camera_set=args.set)
    coords = enumerate_coordinates(cm.loaded.unet)
    coarse = coarse_coordinates(coords)
    print(f"model {cm.name}: {len(coords)} coordinates, coarse {len(coarse)}, revisions {cm.loaded.revisions}", flush=True)
    ok = True
    for seed in args.seeds or cm.cameras.seeds:
        ok = run_group(cm, coords, coarse, seed, args) and ok
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
