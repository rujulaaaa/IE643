"""MVDream (multi-view image diffusion, SD2.1 base, 4 views): subject vs pose vs viewpoint (D-46).

The novelty-assessment feedback asked to extend the matched-pair design to a multi-view generation model, hold
subject and action fixed while varying the viewpoint, and test whether the action-sensitive layers also respond to
the viewpoint; and for a patch that moves the action while keeping the target's subject and viewpoint.

MVDream takes the viewpoint as an input: a camera matrix per view (4 views, azimuths start + 0/90/180/270,
one elevation), added to the timestep embedding. An image model has no motion, so "action" here is the pose a verb
prompt produces ("a horse jumping" vs "a horse running"), the same verbs as Experiment 1.

    sweep   prompts of configs/av_set.yaml (11) x views {az0, az30, az60, el30}; per seed one lockstep group.
            Pairs: action / subject / text-view (prompt swap vs the anchor prompt, same camera) and camera
            (az0 vs the other three, same prompt). Cosine distance of the conditional branch at every U-Net block
            output (input_blocks 0-11, middle, output_blocks 0-11) and every step -> distances.csv.
    patch   pair (X, v, cam_s) -> (Y, running, cam_t): S source, T target, Ts = (Y, running, cam_s),
            I = (Y, v, cam_t) ideal; arms T<-S|<block group> at steps 0-9 of 30. Scores (0 = nothing moved over):
              act_clip   (Dv(P) - Dv(T)) / (Dv(I) - Dv(T)),  Dv = CLIP(x, "Y v") - CLIP(x, "Y running")
              leak_subj  (Ds(P) - Ds(T)) / (Ds(S) - Ds(T)),  Ds = CLIP(x, "X") - CLIP(x, "Y")
              leak_view  (Dw(P) - Dw(T)) / (Dw(Ts) - Dw(T)), Dw = per-view CLIP image cos to Ts minus to T
            HV3 pass: some group with act_clip CI lower bound > 0.3, |leak_subj| <= 0.1, |leak_view| <= 0.1.

    python scripts/mvdream_probe.py sweep --seeds 42 --prompts 2 --views 2 --out ~/out/mv_smoke
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
MV_REPO = Path("/scratch/mvdream_diffusers")
VIEWS = {"az0": (0.0, 0.0), "az30": (0.0, 30.0), "az60": (0.0, 60.0), "el30": (30.0, 0.0)}   # (elevation, azimuth start)
STEPS, GUIDANCE, SIZE = 30, 7.5, 256
NEG = "ugly, blurry, pixelated obscure, unnatural colors, poor lighting, dull, unclear, cropped, lowres, low quality"


def load():
    sys.path.insert(0, str(MV_REPO))
    from mv_unet import get_camera
    from pipeline_mvdream import MVDreamPipeline
    pipe = MVDreamPipeline.from_pretrained("ashawkey/mvdream-sd2.1-diffusers", torch_dtype=torch.float16,
                                           trust_remote_code=True).to("cuda")
    from diffusers import DDIMScheduler
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    return pipe, get_camera


def sites(unet):
    out = [(f"input_blocks.{i}", m) for i, m in enumerate(unet.input_blocks)]
    out.append(("middle_block", unet.middle_block))
    out += [(f"output_blocks.{i}", m) for i, m in enumerate(unet.output_blocks)]
    return out


class Clip:
    """One 4-view sample: prompt, viewpoint, latents; advanced one DDIM step at a time."""

    def __init__(self, cid, prompt, view, pipe, get_camera, noise):
        self.id, self.prompt, self.view = cid, prompt, view
        emb = pipe._encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1,
                                  do_classifier_free_guidance=True, negative_prompt=NEG)
        neg, pos = emb.chunk(2)
        self.context = torch.cat([neg] * 4 + [pos] * 4)
        el, az = VIEWS[view]
        cam = get_camera(4, elevation=el, azimuth_start=az).to(dtype=torch.float16, device="cuda")
        self.camera = torch.cat([cam] * 2)
        self.latents = noise.clone()


class Hooks:
    """Forward hooks on the block outputs. mode 'record' keeps the output; 'patch' replaces it with a stored one."""

    def __init__(self, named):
        self.named, self.handles = named, []
        self.record, self.patch, self.store = set(), {}, {}
        for name, m in named:
            self.handles.append(m.register_forward_hook(self._hook(name)))

    def _hook(self, name):
        def fn(module, args, output):
            if name in self.record:
                self.store[name] = output.detach().clone()
            if name in self.patch:
                return self.patch[name].to(output.dtype)
        return fn


@torch.no_grad()
def unet_step(pipe, clip, t, hooks=None, record=(), patch=None):
    if hooks is not None:
        hooks.record, hooks.patch, hooks.store = set(record), patch or {}, {}
    x = pipe.scheduler.scale_model_input(torch.cat([clip.latents] * 2), t)
    eps = pipe.unet.forward(x=x, timesteps=torch.tensor([t] * 8, dtype=x.dtype, device="cuda"), context=clip.context,
                            num_frames=4, camera=clip.camera)
    u, c = eps.chunk(2)
    eps = u + GUIDANCE * (c - u)
    clip.latents = pipe.scheduler.step(eps, t, clip.latents, return_dict=False)[0]
    return dict(hooks.store) if hooks is not None else {}


def cos_dist(a, b):
    """Mean over the 4 views of the cosine distance of the conditional branch (second half of the batch)."""
    a, b = a[4:].flatten(1).float(), b[4:].flatten(1).float()
    return float((1 - F.cosine_similarity(a, b, dim=1)).mean())


@torch.no_grad()
def decode(pipe, clip):
    imgs = pipe.decode_latents(clip.latents)            # (4, H, W, 3) in [0, 1]
    from PIL import Image
    return [Image.fromarray((i * 255).round().astype(np.uint8)) for i in imgs]


def grid(images):
    from PIL import Image
    w, h = images[0].size
    g = Image.new("RGB", (w * len(images), h))
    for i, im in enumerate(images):
        g.paste(im, (i * w, 0))
    return g


def make_noise(seed):
    g = torch.Generator("cuda").manual_seed(seed)
    return torch.randn((4, 4, SIZE // 8, SIZE // 8), generator=g, device="cuda", dtype=torch.float16)


def sweep(args):
    pipe, get_camera = load()
    named = sites(pipe.unet)
    names = [n for n, _ in named]
    hooks = Hooks(named)
    cfg = yaml.safe_load((ROOT / "configs" / "av_set.yaml").read_text())
    prompts = cfg["scenes"][:args.prompts or None]
    anchor = cfg["anchor_scene"]
    kind_of = {s: k for k, ss in cfg["pair_kinds"].items() for s in ss}
    views = list(VIEWS)[:args.views or None]
    slug = lambda p: p.replace(", side view", "").replace(" on a beach", "").replace(", ", "-").replace(" ", "-")
    for seed in args.seeds:
        out = args.out / f"s{seed}"
        out.mkdir(parents=True, exist_ok=True)
        noise = make_noise(seed)
        clips = {(p, v): Clip(f"{slug(p)}__{v}", p + ", high quality, detailed", v, pipe, get_camera, noise)
                 for p in prompts for v in views}
        pairs = [(kind_of[p], (anchor, v), (p, v)) for v in views for p in prompts if p != anchor]
        pairs += [("camera", (p, views[0]), (p, v)) for p in prompts for v in views[1:]]
        anchors = {a for _, a, _ in pairs}
        order = sorted(clips, key=lambda k: k not in anchors)
        pipe.scheduler.set_timesteps(STEPS, device="cuda")
        rows, t0 = [], time.perf_counter()
        for step, t in enumerate(pipe.scheduler.timesteps):
            cache = {}
            for k in order:
                store = unet_step(pipe, clips[k], t, hooks, record=names)
                if k in anchors:
                    cache[k] = store
                for i, (kind, a, b) in enumerate(pairs):
                    if b == k:
                        for n in names:
                            rows.append({"pair": i, "kind": kind, "a": clips[a].id, "b": clips[b].id, "view": b[1],
                                         "site": n, "step": step, "d": cos_dist(cache[a][n], store[n])})
            if step % 5 == 0:
                print(f"seed {seed} step {step}: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        with open(out / "distances.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        (out / "images").mkdir(exist_ok=True)
        for k, c in clips.items():
            grid(decode(pipe, c)).save(out / "images" / f"{c.id}.png")
        json.dump({"seed": seed, "prompts": prompts, "views": {v: VIEWS[v] for v in views}, "steps": STEPS,
                   "guidance": GUIDANCE, "size": SIZE, "sites": names, "n_pairs": len(pairs),
                   "seconds": time.perf_counter() - t0}, open(out / "manifest.json", "w"), indent=1)
        print(f"seed {seed}: {len(clips)} samples, {len(pairs)} pairs, {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


GROUPS = {"in_0-3": range(0, 4), "in_4-6": range(4, 7), "in_7-9": range(7, 10), "in_10-11": range(10, 12),
          "middle": [], "out_0-2": range(0, 3), "out_3-5": range(3, 6), "out_6-8": range(6, 9), "out_9-11": range(9, 12), "all": []}
SUBJECT_PAIRS = [("horse", "tiger"), ("wolf", "deer"), ("tiger", "brown bear"), ("deer", "horse")]
VERBS = ["jumping", "walking"]
VIEW_PAIRS = [("az0", "el30"), ("el30", "az0"), ("az0", "az0")]


def group_sites(g):
    if g == "all":
        return [f"input_blocks.{i}" for i in range(12)] + ["middle_block"] + [f"output_blocks.{i}" for i in range(12)]
    if g == "middle":
        return ["middle_block"]
    pre = "input_blocks" if g.startswith("in") else "output_blocks"
    return [f"{pre}.{i}" for i in GROUPS[g]]


def patch(args):
    from vdprobe.metrics import ClipScorer
    pipe, get_camera = load()
    named = sites(pipe.unet)
    hooks = Hooks(named)
    scorer = ClipScorer(device="cuda")
    pr = lambda s, v: f"a {s} {v} on a beach, side view, high quality, detailed"
    pairs = [(x, y, v, vs, vt) for vs, vt in VIEW_PAIRS for x, y in SUBJECT_PAIRS for v in VERBS][:args.pairs or None]
    noise = make_noise(args.seed)
    all_rows = []
    for x, y, v, vs, vt in pairs:
        name = f"{vs}_to_{vt}__{x.replace(' ', '-')}_to_{y.replace(' ', '-')}__{v}"
        out = args.out / name
        out.mkdir(parents=True, exist_ok=True)
        base = {"S": Clip("S", pr(x, v), vs, pipe, get_camera, noise), "T": Clip("T", pr(y, "running"), vt, pipe, get_camera, noise),
                "Ts": Clip("Ts", pr(y, "running"), vs, pipe, get_camera, noise), "I": Clip("I", pr(y, v), vt, pipe, get_camera, noise)}
        arms = {f"T<-S|{g}": Clip(f"T<-S|{g}", pr(y, "running"), vt, pipe, get_camera, noise) for g in GROUPS}
        if args.null:
            arms["null"] = Clip("null", pr(x, v), vs, pipe, get_camera, noise)
        pipe.scheduler.set_timesteps(STEPS, device="cuda")
        all_sites = [n for n, _ in named]
        t0 = time.perf_counter()
        for step, t in enumerate(pipe.scheduler.timesteps):
            on = args.patch_from <= step < args.patch_steps
            src = unet_step(pipe, base["S"], t, hooks, record=all_sites if on else ())
            for k in ("T", "Ts", "I"):
                unet_step(pipe, base[k], t, hooks)
            for k, c in arms.items():
                ss = all_sites if k == "null" else group_sites(k.split("|")[1])
                unet_step(pipe, c, t, hooks, patch={n: src[n] for n in ss} if on else None)
        hooks.patch = {}
        imgs = {k: decode(pipe, c) for k, c in {**base, **arms}.items()}
        checks = {}
        if args.null:
            checks["null_mad"] = float((arms["null"].latents - base["S"].latents).abs().mean())
            checks["null_patch"] = checks["null_mad"] < 1e-3
        txt = scorer.text([pr(y, v), pr(y, "running"), f"a {x} on a beach", f"a {y} on a beach"])
        emb = {k: scorer.image(im) for k, im in imgs.items()}            # (4, d) per view
        dv = {k: float(e.mean(0) @ (txt[0] - txt[1])) for k, e in emb.items()}
        ds = {k: float(e.mean(0) @ (txt[2] - txt[3])) for k, e in emb.items()}
        dw = {k: float(((e * emb["Ts"]).sum(1) - (e * emb["T"]).sum(1)).mean()) for k, e in emb.items()}
        r_ = lambda n, d: n / d if abs(d) > 1e-6 else float("nan")
        rows = []
        for k in arms:
            if k == "null":
                continue
            rows.append({"pair": name, "candidate": k.split("|")[1], "subject_s": x, "subject_t": y, "verb": v,
                         "view_s": vs, "view_t": vt, "seed": args.seed,
                         "act_clip": r_(dv[k] - dv["T"], dv["I"] - dv["T"]),
                         "leak_subj": r_(ds[k] - ds["T"], ds["S"] - ds["T"]),
                         "leak_view": r_(dw[k] - dw["T"], dw["Ts"] - dw["T"]) if vs != vt else float("nan"),
                         "den_act": dv["I"] - dv["T"], "den_subj": ds["S"] - ds["T"], "den_view": dw["Ts"] - dw["T"]})
            r = rows[-1]
            print(f"  {k:<16} act {r['act_clip']:+.3f} subj {r['leak_subj']:+.3f} view {r['leak_view']:+.3f}", flush=True)
        all_rows += rows
        import pandas as pd
        pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
        from PIL import Image
        sheet_keys = ["S", "T", "Ts", "I", "T<-S|in_0-3", "T<-S|in_4-6", "T<-S|middle", "T<-S|all"]
        gs = [grid(imgs[k]) for k in sheet_keys]
        sheet = Image.new("RGB", (gs[0].width, gs[0].height * len(gs)))
        for i, g in enumerate(gs):
            sheet.paste(g, (0, i * gs[0].height))
        sheet.save(out / "sheet.png")
        if args.images:
            (out / "images").mkdir(exist_ok=True)
            for k, im in imgs.items():
                grid(im).save(out / "images" / f"{k.replace('|', '_').replace('<-', '_from_')}.png")
        json.dump({"pair": [x, y, v, vs, vt], "checks": checks, "passed": all(c for c in checks.values() if isinstance(c, bool)),
                   "patch_steps": args.patch_steps, "patch_from": args.patch_from, "seconds": time.perf_counter() - t0}, open(out / "manifest.json", "w"), indent=1)
        print(f"{name}: {(time.perf_counter() - t0):.0f} s, checks {checks}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("sweep", "patch"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 1337])
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--prompts", type=int, default=0)
    p.add_argument("--views", type=int, default=0)
    p.add_argument("--pairs", type=int, default=0)
    p.add_argument("--patch-steps", type=int, default=10, help="patch steps < this")
    p.add_argument("--patch-from", type=int, default=0, help="patch steps >= this (D-47)")
    p.add_argument("--null", action="store_true")
    p.add_argument("--images", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    (sweep if a.what == "sweep" else patch)(a)


if __name__ == "__main__":
    main()
