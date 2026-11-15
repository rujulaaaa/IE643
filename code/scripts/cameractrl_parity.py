"""Parity gate: is the graft in vdprobe.cameractrl the authors' CameraCtrl? (design doc, "Parity gate")

The same latent, timestep, text embedding and camera trajectory go through

  ours        diffusers UNetMotionModel (v1-5-3 motion adapter) + our LoRA merge + CameraAdapter hooks
  reference   the authors' code at commit af45a7a, with the diffusers version it was written for
              (0.24.0), run in a child process with its own package directory: their
              tools/merge_lora2unet.py, their UNet3DConditionModelPoseCond with v3_sd15_mm.ckpt,
              their CameraPoseEncoder and PoseAdaptorAttnProcessor with CameraCtrl.ckpt

in fp32, and the noise predictions are compared: camera off (checks the base model, the motion
module and the LoRA merge) and two cameras (check the encoder, the key map and the injection).
Pass: max |diff| < 1e-3 and relative L2 < 1e-3 for every case. Exit code 1 otherwise.

    python scripts/kaggle_run.py --user <account> --slug vdprobe-cc-parity --prefetch-cameractrl \
        --job "scripts/cameractrl_parity.py"
"""

import gc
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from vdprobe import runio
from vdprobe.cameractrl.model import load_cameractrl, named_trajectory
from vdprobe.cameractrl.adapter import CameraSpec
from vdprobe.cameractrl.rays import plucker_embedding
from vdprobe.config import CONFIG_DIR, GenerationConfig

COMMIT = "af45a7a0dbeedb7f6ed66aac4a9e5411a279f0d2"
REPO = "https://github.com/hehao13/CameraCtrl.git"
# transformers is pinned too: diffusers 0.24 imports it, and the image's transformers 5 needs a newer huggingface_hub
REF_DEPS = ["diffusers==0.24.0", "huggingface_hub==0.25.2", "transformers==4.46.3", "einops", "omegaconf"]
CAMERAS = ["pan-left", "roll-cw"]
TIMESTEPS = [801, 201]
MAX_ABS, REL_L2 = 1e-3, 1e-3

TRACE = r'''
# --- layer-by-layer trace, the same code in both processes -----------------------------------------
import re as _re
_TRACE_RE = _re.compile(r"^(conv_in|conv_norm_out|conv_out|(?:(?:down_blocks|up_blocks)\.\d+|mid_block)"
                        r"\.(?:resnets|attentions|motion_modules|downsamplers|upsamplers)\.\d+)$")
def _canon(x, frames):
    x = getattr(x, "sample", x)
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not torch.is_tensor(x):
        return None
    if x.ndim == 4 and frames > 1 and x.shape[0] % frames == 0:            # (b*f, c, h, w) -> (b, c, f, h, w)
        x = x.reshape(x.shape[0] // frames, frames, *x.shape[1:]).permute(0, 2, 1, 3, 4)
    return x
def _digest(x):
    if x is None:
        return None
    v = x.detach().float().flatten()
    n = min(v.numel(), 8192)                                                # integer arithmetic: float32 linspace
    idx = torch.arange(n, device=v.device) * (v.numel() - 1) // max(n - 1, 1)  # overshoots the end of large tensors
    return {"sub": v[idx].cpu(), "shape": list(x.shape)}
def trace(unet, frames, run):
    rec, handles = {}, []
    for name, m in unet.named_modules():
        if _TRACE_RE.match(name):
            def hook(mod, args, kwargs, out, name=name):
                first = args[0] if args else next(iter(kwargs.values()))
                rec[name] = {"in": _digest(_canon(first, frames)), "out": _digest(_canon(out, frames))}
            handles.append(m.register_forward_hook(hook, with_kwargs=True))
    try:
        with torch.no_grad():
            run()
    finally:
        for h in handles:
            h.remove()
    return rec
'''

REFERENCE = r'''
import json, subprocess, sys
from pathlib import Path
import torch
from einops import rearrange
from omegaconf import OmegaConf

work, repo, mm_ckpt, lora_ckpt, cc_ckpt = (Path(a) for a in sys.argv[1:6])
import diffusers
print("reference: diffusers", diffusers.__version__, "torch", torch.__version__, flush=True)

# 1. the authors' LoRA merge, their script unchanged
merged = work / "sd" / "unet_webvidlora_v3"
subprocess.run([sys.executable, str(repo / "tools" / "merge_lora2unet.py"), "--lora_ckpt_path", str(lora_ckpt),
                "--unet_ckpt_path", str(work / "sd"), "--save_path", str(merged),
                "--unet_config_path", str(work / "sd" / "unet" / "config.json")], check=True, stdout=subprocess.DEVNULL)

# 2. the model, built as get_pipeline() in their inference.py builds it (no image LoRA)
from cameractrl.models.unet import UNet3DConditionModelPoseCond
from cameractrl.models.pose_adaptor import CameraPoseEncoder
from cameractrl.models.attention_processor import PoseAdaptorAttnProcessor

cfg = OmegaConf.load(repo / "configs" / "train_cameractrl" / "adv3_256_384_cameractrl_relora.yaml")
unet = UNet3DConditionModelPoseCond.from_pretrained_2d(
    str(work / "sd"), subfolder="unet_webvidlora_v3",
    unet_additional_kwargs=OmegaConf.to_container(cfg.unet_additional_kwargs))
encoder = CameraPoseEncoder(**OmegaConf.to_container(cfg.pose_encoder_kwargs))
unet.set_all_attn_processor(add_spatial_lora=False, add_motion_lora=False,
                            lora_kwargs={"lora_rank": 2, "lora_scale": 1.0},
                            motion_lora_kwargs={"lora_rank": -1, "lora_scale": 1.0},
                            **OmegaConf.to_container(cfg.attention_processor_kwargs))
mm = torch.load(mm_ckpt, map_location="cpu")
_, unexpected = unet.load_state_dict(mm, strict=False)
assert len(unexpected) == 0, unexpected[:5]
cc = torch.load(cc_ckpt, map_location="cpu")
missing, unexpected = encoder.load_state_dict(cc["pose_encoder_state_dict"])
assert len(missing) == 0 and len(unexpected) == 0
_, unexpected = unet.load_state_dict(cc["attention_processor_state_dict"], strict=False)
assert len(unexpected) == 0, unexpected[:5]
# unet.attn_processors leaves the motion modules out, which is where these processors are
procs = [m.processor for m in unet.modules() if isinstance(getattr(m, "processor", None), PoseAdaptorAttnProcessor)]
print("reference: pose processors", len(procs), flush=True)
assert len(procs) == 20
unet, encoder = unet.cuda().eval(), encoder.cuda().eval()

# 3. the forward passes: with the weights as released (keys "fp32/...", reported only), then with every
#    weight rounded to fp16 and held in fp32 (the gate). The project stores all weights in fp16, so the
#    second pass has the graft's weights exactly and any difference left is a difference in the computation.
inp = torch.load(work / "inputs.pt", map_location="cuda")
out = {}
def forward_all(prefix):
    with torch.no_grad():
        for name, plucker in inp["plucker"].items():
            feats = encoder(plucker)                                              # as in pipeline_animation.py
            feats = [rearrange(x, "(b f) c h w -> b c f h w", b=1) for x in feats]
            out[f"{prefix}features/{name}"] = [f.cpu() for f in feats]
            for scale, tag in ((1.0, name), (0.0, "off")):
                if tag == "off" and any(k.startswith(f"{prefix}eps/off/") for k in out):
                    continue
                for p in procs:
                    p.scale = scale                                                # scale 0: h + 0 * merge(h + pose) = h
                for t in inp["timesteps"]:
                    eps = unet(inp["latent"], torch.tensor(t, device="cuda"), inp["text"], pose_embedding_features=feats).sample
                    out[f"{prefix}eps/{tag}/{t}"] = eps.cpu()
forward_all("fp32/")
n = 0
with torch.no_grad():
    for m in (unet, encoder):
        for t in list(m.parameters()) + list(m.buffers()):
            if t.is_floating_point():
                t.copy_(t.half().float())
                n += 1
print("reference: rounded", n, "tensors to fp16 values", flush=True)
forward_all("")
for p in procs:
    p.scale = 0.0
_feats = [rearrange(x, "(b f) c h w -> b c f h w", b=1) for x in encoder(next(iter(inp["plucker"].values())))]
out["trace"] = trace(unet, inp["latent"].shape[2], lambda: unet(
    inp["latent"], torch.tensor(inp["timesteps"][0], device="cuda"), inp["text"], pose_embedding_features=_feats))
torch.save(out, work / "reference.pt")
print("reference: done", sorted(k for k in out if k.startswith("eps")), flush=True)
'''


def run(cmd, **kw):
    print("$", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, **kw)


def main():
    ap = runio.parser(__doc__, "cameractrl_parity")
    ap.set_defaults(config=CONFIG_DIR / "cameractrl.yaml")
    ap.add_argument("--work", type=Path, default=Path("/tmp/parity_work"),
                    help="scratch directory; outside /kaggle/working, so it is not saved as kernel output")
    ap.add_argument("--ref-gpu", default=None, help="CUDA_VISIBLE_DEVICES of the reference process")
    args = ap.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import hf_hub_download

    # --- ours, fp32 ------------------------------------------------------------------------------
    cfg = GenerationConfig.from_yaml(args.config)
    loaded = load_cameractrl(cfg)
    adapter = loaded.extras["camera"]
    pipe = loaded.pipe
    with torch.no_grad():
        pos, _ = pipe.encode_prompt("a mountain lake at sunrise", loaded.device, 1, False)
    text = pos.float()                                                          # (1, 77, 768)
    g = torch.Generator().manual_seed(0)
    latent = torch.randn(cfg.latent_shape, generator=g).to(loaded.device)       # (1, 4, 16, 32, 48) fp32
    pipe.unet.float()
    adapter.float()
    adapter.clear_cache()
    trajs = {c: named_trajectory(cfg, c) for c in CAMERAS}
    ours = {}
    with torch.no_grad():
        for tag in ["off"] + CAMERAS:
            adapter.set(None if tag == "off" else CameraSpec(trajs[tag]), cfg.height, cfg.width, cfg.frames)
            for t in TIMESTEPS:
                ours[f"eps/{tag}/{t}"] = pipe.unet(latent, torch.tensor(t, device=loaded.device),
                                                   encoder_hidden_states=text.repeat_interleave(cfg.frames, dim=0)).sample.cpu()
        for c in CAMERAS:
            emb = plucker_embedding(trajs[c], cfg.height, cfg.width).to(loaded.device)
            ours[f"features/{c}"] = [f.cpu() for f in adapter.encoder(emb)]
    adapter.set(None)
    ns = {"torch": torch}
    exec(TRACE, ns)
    ours_trace = ns["trace"](pipe.unet, cfg.frames, lambda: pipe.unet(
        latent, torch.tensor(TIMESTEPS[0], device=loaded.device),
        encoder_hidden_states=text.repeat_interleave(cfg.frames, dim=0)))
    torch.save({"latent": latent.cpu(), "text": text.cpu(), "timesteps": TIMESTEPS,
                "plucker": {c: plucker_embedding(trajs[c], cfg.height, cfg.width) for c in CAMERAS}}, work / "inputs.pt")

    # --- the reference's inputs on disk: an fp32 SD1.5 U-Net in diffusers layout, and the checkpoints ---
    from diffusers import UNet2DConditionModel
    from vdprobe.pipeline import _resolve_base
    _, snap, variant = _resolve_base(cfg)
    unet2d = UNet2DConditionModel.from_pretrained(snap, subfolder="unet", variant=variant, torch_dtype=torch.float32)
    unet2d.save_pretrained(work / "sd" / "unet")
    # the repository's own config.json, as the authors use it: the one save_pretrained writes names
    # mid_block_type "UNetMidBlock2DCrossAttn", which their 3D U-Net does not know
    shutil.copy(Path(snap) / "unet" / "config.json", work / "sd" / "unet" / "config.json")
    del unet2d
    x = cfg.extra
    mm_ckpt = hf_hub_download("guoyww/animatediff", "v3_sd15_mm.ckpt")
    lora_ckpt = hf_hub_download(x["domain_lora_repo"], x["domain_lora_file"])
    cc_ckpt = hf_hub_download(x["cameractrl_repo"], x["cameractrl_file"])
    pipe.to("cpu")                      # the reference needs the card: fp32, attention without slicing
    adapter.to("cpu")
    del loaded, pipe, adapter
    gc.collect()
    torch.cuda.empty_cache()
    print(f"our model released: {torch.cuda.memory_allocated() / 2 ** 30:.2f} GB still allocated", flush=True)

    # --- the reference, in a child process with its own packages ------------------------------------
    repo, deps = work / "CameraCtrl", work / "deps"
    if not repo.exists():
        run(["git", "clone", "-q", REPO, repo])
    run(["git", "-C", repo, "checkout", "-q", COMMIT])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "--target", deps, *REF_DEPS])
    (work / "reference.py").write_text("import torch\n" + TRACE + REFERENCE)
    env = dict(os.environ, PYTHONPATH=f"{deps}{os.pathsep}{repo}", PYTHONUNBUFFERED="1")
    if args.ref_gpu is not None:        # the kernel's second T4 is idle when this is the only job
        env["CUDA_VISIBLE_DEVICES"] = args.ref_gpu
    run([sys.executable, work / "reference.py", work, repo, mm_ckpt, lora_ckpt, cc_ckpt], env=env)
    ref = torch.load(work / "reference.pt", map_location="cpu")

    # --- compare --------------------------------------------------------------------------------
    report, ok = {}, True
    for k in sorted(ours):
        if k.startswith("features/"):
            # ours: (B, C, F, h, w) as well; one entry per level
            d = [float((a - b).abs().max()) for a, b in zip(ours[k], ref[k])]
            report[k] = {"max_abs_per_level": d, "pass": max(d) < MAX_ABS}
        else:
            a, b = ours[k], ref[k]
            report[k] = {"max_abs": float((a - b).abs().max()), "rel_l2": float((a - b).norm() / b.norm()),
                         "ref_norm": float(b.norm())}
            report[k]["pass"] = report[k]["max_abs"] < MAX_ABS and report[k]["rel_l2"] < REL_L2
        ok &= report[k]["pass"]
        print(f"{'PASS' if report[k]['pass'] else 'FAIL'}  {k:<24} {report[k]}", flush=True)
    # where the two computations part: per module, relative difference of its input and of its output
    # (camera off, first timestep). A module whose output differs far more than its input is a suspect.
    rel = lambda a, b: float((a - b).norm() / b.norm().clamp_min(1e-12))
    layers = []
    for name, o in ours_trace.items():
        r = ref.get("trace", {}).get(name)
        if r is None:
            layers.append({"module": name, "note": "not in the reference"})
            continue
        row = {"module": name}
        for side in ("in", "out"):
            a, b = o[side], r[side]
            if a is None or b is None:
                row[side] = None
            elif a["shape"] != b["shape"]:
                row[side], row[f"{side}_shapes"] = None, [a["shape"], b["shape"]]
            else:
                row[side] = rel(a["sub"], b["sub"])
        layers.append(row)
        f = lambda v: "   n/a  " if v is None else f"{v:.2e}"
        flag = "  <--" if row.get("out") and row["out"] > 1e-4 and (row.get("in") is None or row["out"] > 5 * row["in"]) else ""
        print(f"layer {name:<34} in {f(row.get('in'))}  out {f(row.get('out'))}{flag}"
              + (f"  shapes {row.get('in_shapes') or row.get('out_shapes')}" if row.get("in_shapes") or row.get("out_shapes") else ""), flush=True)
    missing = [k for k in ref.get("trace", {}) if k not in ours_trace]
    if missing:
        print("in the reference only:", missing, flush=True)
    report["layers"] = layers
    # not part of the gate: the same comparison against the reference with its weights as released (fp32),
    # which measures what storing weights in fp16 costs
    for k in sorted(ours):
        if k.startswith("eps/") and f"fp32/{k}" in ref:
            a, b = ours[k], ref[f"fp32/{k}"]
            report[f"fp32_weights/{k}"] = {"max_abs": float((a - b).abs().max()), "rel_l2": float((a - b).norm() / b.norm())}
            print(f"info  vs fp32 weights  {k:<24} {report[f'fp32_weights/{k}']}", flush=True)
    # power of the test: the camera must change the reference prediction by far more than the tolerance
    for c in CAMERAS:
        for t in TIMESTEPS:
            eff = float((ref[f"eps/{c}/{t}"] - ref[f"eps/off/{t}"]).abs().max())
            report[f"camera_effect/{c}/{t}"] = {"max_abs": eff, "pass": eff > 10 * MAX_ABS}
            ok &= eff > 10 * MAX_ABS
            print(f"camera effect in the reference  {c} t={t}: max |on - off| = {eff:.4f}", flush=True)
    (args.out / "parity.json").write_text(json.dumps({"commit": COMMIT, "reference_deps": REF_DEPS, "max_abs": MAX_ABS,
                                                      "rel_l2": REL_L2, "passed": bool(ok), "cases": report}, indent=2))
    print("PARITY", "PASSED" if ok else "FAILED", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
