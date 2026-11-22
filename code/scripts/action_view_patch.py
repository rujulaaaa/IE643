"""Action transfer that keeps the target's subject and viewpoint (D-45, HV2), on CameraCtrl.

The novelty-assessment feedback asked for "a patching example that transfers the action while preserving
the target subject and viewpoint". One pair = (subject X -> subject Y, source verb v, camera c_s -> c_t),
held-out seed, one noise tensor for every clip:

    S  = (X, v,       c_s)   source
    T  = (Y, running, c_t)   target (unpatched)
    I  = (Y, v,       c_t)   ideal: what a perfect action-only transfer would look like
    T<-S|cand                target receiving the source's activations at `cand`

Scores of a patched clip P (all 0 = nothing moved over, 1 = everything moved over):
    act_flow   object motion (RAFT flow, median background flow removed) moves from T's towards I's:
               (m(P, I) - m(T, I)) / (1 - m(T, I)),  m = metrics.motion_fidelity
    act_clip   CLIP verb score Dv = CLIP(x, "Y v") - CLIP(x, "Y running"): (Dv(P) - Dv(T)) / (Dv(I) - Dv(T))
    leak_subj  CLIP subject score Ds = CLIP(x, "X") - CLIP(x, "Y"): (Ds(P) - Ds(T)) / (Ds(S) - Ds(T))
    leak_cam   camera signature moves towards the source camera: proj(sig(P) - sig(T), sig(S) - sig(T))
               (NaN when c_s = c_t)
Pass (HV2): some candidate with act_flow CI lower bound > 0.3, and |leak_subj| <= 0.1 and |leak_cam| <= 0.1
(means over pairs; verdict in scripts/action_view_verdict.py).

    python scripts/action_view_patch.py --pairs 1 --null      # smoke
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vdprobe import metrics as M
from vdprobe import runio
from vdprobe.camera_models import load_camera_model, default_config
from vdprobe.config import CONFIG_DIR
from vdprobe.coords import block_order, enumerate_coordinates
from vdprobe.edit import Arm, AttnSpec, EditSpec, run_arms, temporal_attention_modules
from vdprobe.patching import NULL_PATCH_MAX_LATENT_MAD, PatchSpec, window
from vdprobe.pipeline import make_noise

STEPS = tuple(range(8))
SUBJECT_PAIRS = [("horse", "tiger"), ("wolf", "deer"), ("tiger", "brown bear"), ("deer", "horse")]
VERBS = ["jumping", "walking"]
CAMERA_PAIRS = [("pan-left", "zoom-in"), ("zoom-in", "pan-left"), ("off", "off")]
TARGET_VERB = "running"
SET = CONFIG_DIR / "av_set.yaml"


def prompt(subject, verb, view="side view"):
    return f"a {subject} {verb} on a beach, {view}, high quality, detailed"


def candidates(coords):
    mm = lambda block: tuple(c.name for c in coords if c.kind == "temporal_module" and c.block == block)
    every = tuple(c.name for c in coords if c.kind == "temporal_module")
    out = [{"name": "exp1_region", "coords": ("up_blocks.0.motion_modules.2",), "steps": (0, 1, 2), "component": "dynamic"},
           {"name": "up1_motion_modules", "coords": mm("up_blocks.1"), "steps": STEPS, "component": "dynamic"},
           {"name": "all_motion_modules_dynamic", "coords": every, "steps": STEPS, "component": "dynamic"}]
    out += [{"name": f"mm_{b}", "coords": mm(b), "steps": STEPS, "component": "all"} for b in block_order(coords) if mm(b)]
    return out


def late_candidates(coords, n_steps):
    """D-47: the camera is fixed in steps 0-2 (D-45 sweep), the action keeps growing after; patch later windows only."""
    mm = lambda blocks: tuple(c.name for c in coords if c.kind == "temporal_module" and c.block in blocks)
    every = mm({c.block for c in coords})
    up = mm({"up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"})
    down = mm({"down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3"})
    up23 = mm({"up_blocks.2", "up_blocks.3"})
    w = lambda a, b: tuple(range(a, min(b, n_steps - 1) + 1))
    spec = [("mm_all", every, "all", (0, 7)), ("mm_all", every, "all", (3, 10)), ("mm_all", every, "all", (5, 15)),
            ("mm_all", every, "all", (8, 24)), ("mm_all", every, "all", (3, 24)),
            ("mm_all_dyn", every, "dynamic", (3, 10)), ("mm_all_dyn", every, "dynamic", (3, 24)),
            ("mm_up", up, "all", (3, 10)), ("mm_up", up, "all", (3, 24)), ("mm_up23", up23, "all", (3, 24)),
            ("mm_down", down, "all", (3, 10)), ("mm_down", down, "all", (3, 24))]
    return [{"name": f"{n}@{a}-{b}", "coords": co, "steps": w(a, b), "component": comp} for n, co, comp, (a, b) in spec]


MATCHED_CAMERAS = [("pan-left", "pan-left"), ("zoom-in", "zoom-in"), ("off", "off")]


def combo_candidates(coords):
    """D-49: the action-carrying up blocks together (D-48: each of up_blocks.1-3 alone moved 0.23-0.26)."""
    mm = lambda blocks: tuple(c.name for c in coords if c.kind == "temporal_module" and c.block in blocks)
    u123, u23 = mm({"up_blocks.1", "up_blocks.2", "up_blocks.3"}), mm({"up_blocks.2", "up_blocks.3"})
    r = lambda a, b: tuple(range(a, b + 1))
    return [{"name": "mm_up_blocks.1", "coords": mm({"up_blocks.1"}), "steps": STEPS, "component": "all"},
            {"name": "mm_up123@0-7", "coords": u123, "steps": STEPS, "component": "all"},
            {"name": "mm_up123@0-12", "coords": u123, "steps": r(0, 12), "component": "all"},
            {"name": "mm_up123_dyn@0-12", "coords": u123, "steps": r(0, 12), "component": "dynamic"},
            {"name": "mm_up123_dyn@0-24", "coords": u123, "steps": r(0, 24), "component": "dynamic"},
            {"name": "mm_up23@0-7", "coords": u23, "steps": STEPS, "component": "all"}]


XCLIP = {}


def xclip_verb(frames, texts, device):
    """X-CLIP (video-text) probabilities of `texts` for a clip, 8 frames evenly spaced; None if unavailable."""
    try:
        from transformers import AutoProcessor, XCLIPModel
        if "m" not in XCLIP:
            XCLIP["p"] = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
            XCLIP["m"] = XCLIPModel.from_pretrained("microsoft/xclip-base-patch32").to(device).eval()
        idx = np.linspace(0, len(frames) - 1, 8).round().astype(int)
        inp = XCLIP["p"](text=texts, videos=[[np.asarray(frames[i].convert("RGB")) for i in idx]],
                         return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            return XCLIP["m"](**inp).logits_per_video.softmax(-1)[0].float().cpu().numpy()
    except Exception as e:      # noqa: BLE001 - secondary metric only
        print("xclip unavailable:", repr(e)[:200], flush=True)
        return None


def draw_pairs(matched=False):
    """matched (D-48): the action donor is rendered under the target's camera, so only action and subject differ."""
    cams = MATCHED_CAMERAS if matched else CAMERA_PAIRS
    return [(x, y, v, cs, ct) for cs, ct in cams for x, y in SUBJECT_PAIRS for v in VERBS]


def proj(v, d):
    n = float(d @ d)
    return float(v @ d / n) if n > 1e-8 else float("nan")


def ratio(num, den):
    return float(num / den) if abs(den) > 1e-6 else float("nan")


def run_pair(cm, coords, cands, pair, seed, clip_scorer, args):
    x, y, v, cs, ct = pair
    loaded = cm.loaded
    name = f"{cs}_to_{ct}__{x.replace(' ', '-')}_to_{y.replace(' ', '-')}__{v}"
    out = args.out / name
    if (out / "manifest.json").exists() and not args.force:
        print(f"[skip] {name}")
        return True
    ps, pt, pi = prompt(x, v), prompt(y, TARGET_VERB), prompt(y, v)
    spec = lambda cd: EditSpec(patch=PatchSpec(tuple(cd["coords"]), tuple(cd["steps"]), component=cd["component"]))
    attn = EditSpec(attn=AttnSpec(temporal_attention_modules(coords), window("first_third", loaded.cfg.steps)))
    arms = [Arm("T", pt, EditSpec()), Arm("I", pi, EditSpec())]
    arms += [Arm(f"T<-S|{c['name']}", pt, spec(c)) for c in cands]
    if args.combo:
        pass
    elif args.late:
        for a, b in ((3, 12), (3, 24)):
            arms.append(Arm(f"T<-S|attn@{a}-{b}", pt, EditSpec(attn=AttnSpec(temporal_attention_modules(coords), tuple(range(a, b + 1))))))
    else:
        arms.append(Arm("T<-S|attn_first_third", pt, attn))
    meta = {"source": {"camera": cs}, "I": {"camera": ct}, **{a.id: {"camera": ct} for a in arms if a.id != "I"}}
    if args.null:       # S receiving S at the widest candidate must equal S
        widest = max(cands, key=lambda c: len(c["coords"]))
        arms.append(Arm("null", ps, EditSpec(patch=PatchSpec(tuple(widest["coords"]), STEPS, component="all"))))
        meta["null"] = {"camera": cs}
    res = run_arms(loaded, coords, ps, arms, make_noise(loaded.cfg, seed), track_meta=meta, before_track=cm.before_track)
    cm.reset()
    frames = {"S": res.frames["source"], **{a.id: res.frames[a.id] for a in arms}}
    checks = {f"patched:{a.id}": False for a in arms if a.spec.patch is not None and res.n_patched[a.id] == 0}
    if args.null:
        checks["null_mad"] = M.latent_mad(res.latents["null"], res.source_latents)
        checks["null_patch"] = checks["null_mad"] < NULL_PATCH_MAX_LATENT_MAD
        frames.pop("null")

    flows = {k: M.raft_flow(f, size=cm.flow_size, device=loaded.device) for k, f in frames.items()}
    sigs = {k: M.camera_signature(f) for k, f in flows.items()}
    obj = {k: M.remove_camera_motion(f) for k, f in flows.items()}
    m = lambda a, b: M.motion_fidelity(obj[a], obj[b])
    txt = clip_scorer.text([prompt(y, v), prompt(y, TARGET_VERB), f"a {x} on a beach", f"a {y} on a beach"])
    img = {k: clip_scorer.image(f).mean(0) for k, f in frames.items()}
    dv = {k: float(e @ txt[0] - e @ txt[1]) for k, e in img.items()}
    ds = {k: float(e @ txt[2] - e @ txt[3]) for k, e in img.items()}
    d_cam = sigs["S"] - sigs["T"] if cs != ct else np.zeros(4)
    xc = {}
    if args.combo:
        vt = [f"a {y} {v}", f"a {y} {TARGET_VERB}"]
        xc = {k: xclip_verb(f, vt, loaded.device) for k, f in frames.items()}
    rows = []
    for a in arms:
        if not a.id.startswith("T<-S|"):
            continue
        k = a.id
        rows.append({"pair": name, "candidate": k.split("|")[1], "subject_s": x, "subject_t": y, "verb": v,
                     "camera_s": cs, "camera_t": ct, "seed": seed,
                     "act_flow": ratio(m(k, "I") - m("T", "I"), 1 - m("T", "I")),
                     "act_clip": ratio(dv[k] - dv["T"], dv["I"] - dv["T"]),
                     "leak_subj": ratio(ds[k] - ds["T"], ds["S"] - ds["T"]),
                     "leak_cam": proj(sigs[k] - sigs["T"], d_cam) if cs != ct else float("nan"),
                     "act_src": ratio(m(k, "S") - m("T", "S"), 1 - m("T", "S")),
                     "m_T_I": m("T", "I"), "m_P_I": m(k, "I"), "m_P_S": m(k, "S"), "m_T_S": m("T", "S"),
                     "den_act_clip": dv["I"] - dv["T"], "den_subj": ds["S"] - ds["T"],
                     "den_cam": float(np.linalg.norm(d_cam)), "speed": M.motion_magnitude(obj[k])})
        if xc.get(k) is not None and xc.get("T") is not None and xc.get("I") is not None:
            rows[-1].update(xclip_P=float(xc[k][0]), xclip_T=float(xc["T"][0]), xclip_I=float(xc["I"][0]),
                            xclip_S=float(xc["S"][0]) if xc.get("S") is not None else float("nan"),
                            act_xclip=ratio(xc[k][0] - xc["T"][0], xc["I"][0] - xc["T"][0]))
        r = rows[-1]
        print(f"  {k:<36} act_flow {r['act_flow']:+.3f} act_clip {r['act_clip']:+.3f} "
              f"leak_subj {r['leak_subj']:+.3f} leak_cam {r['leak_cam']:+.3f}", flush=True)
    ok = all(v_ for v_ in checks.values() if isinstance(v_, bool))
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "rows.csv", index=False)
    (out / "sigs.json").write_text(json.dumps({k: [float(z) for z in s] for k, s in sigs.items()}, indent=1))
    show = ("S", "T", "I", "T<-S|mm_up_blocks.1", "T<-S|mm_up123@0-7", "T<-S|mm_up123@0-12") if args.combo else \
        ("S", "T", "I", "T<-S|mm_all@3-10", "T<-S|mm_all@5-15", "T<-S|mm_all@3-24", "T<-S|mm_up@3-24") if args.late else \
        ("S", "T", "I", "T<-S|exp1_region", "T<-S|all_motion_modules_dynamic", "T<-S|mm_up_blocks.1")
    M.contact_sheet({k: frames[k] for k in show
                     if k in frames}).save(out / "sheet.png")
    if args.gifs:
        for k, f in frames.items():
            runio.save_gif(f, out / "gifs" / (k.replace("|", "_").replace("<-", "_from_") + ".gif"), size=192)
    runio.write_json(out / "manifest.json", runio.record(
        "action_view_patch", loaded, pair=list(pair), seed=seed, prompts={"S": ps, "T": pt, "I": pi},
        candidates=[{**c, "coords": list(c["coords"]), "steps": list(c["steps"])} for c in cands], checks=checks,
        passed=ok, denoise_seconds=res.seconds, n_tracks=len(arms) + 1))
    print(f"{'PASS' if ok else 'FAIL'} {name} ({res.seconds / 60:.1f} min) checks {checks}", flush=True)
    return ok


def main():
    p = runio.parser(__doc__, "action_view_patch")
    p.set_defaults(config=None)
    p.add_argument("--seed", type=int, default=None, help="default: the set's held-out seed")
    p.add_argument("--pairs", type=int, default=0, help="smoke test: first N pairs only")
    p.add_argument("--null", action="store_true")
    p.add_argument("--gifs", action="store_true")
    p.add_argument("--shard", default="0/1")
    p.add_argument("--force", action="store_true")
    p.add_argument("--late", action="store_true", help="D-47: late-window candidates instead of the D-45 set")
    p.add_argument("--combo", action="store_true", help="D-49: matched donor, combined up-block candidates, X-CLIP")
    p.add_argument("--matched", action="store_true", help="D-48: donor rendered under the target camera; D-45 + D-47 candidates")
    args = p.parse_args()
    cm = load_camera_model("cc", args.config or default_config("cc"), args.profile, camera_set=SET)
    coords = enumerate_coordinates(cm.loaded.unet)
    cands = late_candidates(coords, cm.loaded.cfg.steps) if args.late else candidates(coords)
    if args.combo:
        args.matched = True
        cands = combo_candidates(coords)
    elif args.matched:
        cands = candidates(coords) + [c for c in late_candidates(coords, cm.loaded.cfg.steps) if "dyn" in c["name"]]
    print("candidates: " + ", ".join(f"{c['name']} ({len(c['coords'])})" for c in cands), flush=True)
    seed = args.seed if args.seed is not None else cm.cameras.heldout_seed
    pairs = draw_pairs(args.matched)[:args.pairs or None]
    i, n = (int(z) for z in args.shard.split("/"))
    clip_scorer = M.ClipScorer(device=cm.loaded.device)
    t0, ok = time.perf_counter(), True
    for k, pr in list(enumerate(pairs))[i::n]:
        print(f"\n=== pair {k}: {pr}", flush=True)
        ok = run_pair(cm, coords, cands, pr, seed, clip_scorer, args) and ok
    print(f"done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
