"""Stage 6 · Train one motion LoRA (lora.VARIANTS: full / probe / anti) on the HMDB51 clips.

Data: data/hmdb51_lora (scripts/fetch_lora_data.py), 19 clips of run / walk / jump / turn.
The last clip of each class (sorted by name) is held out for validation: 15 train, 4 val.
Caption: "a person {verb}" with the verb from the probing vocabulary (turn -> spinning).

Each example: 16 frames at stride min(2, (n-1)//15) from a random start, a random square crop
of the full height (240 px), resized to 256 x 256, encoded once per step by the frozen VAE.
Loss: epsilon-prediction MSE on the whole clip under the frozen SD1.5 + motion adapter UNet
(fp16 weights, fp32 LoRA parameters, autocast, gradient checkpointing incl. motion modules). DDPM training
noise schedule = the inference scheduler's betas (linear, 1000 steps).

Validation loss: every held-out clip at 8 fixed timesteps x 2 fixed noises, fixed frame
window and centre crop, identical for every variant and for the base model (step 0, before
any update). Lower = the adapter explains real motion of unseen clips better.

Writes <out>/<variant>/motion_lora.safetensors (metadata: variant, rank, alpha, steps, lr)
and <out>/<variant>/train_log.json (loss curve, val losses, parameter count, timings).

    python -u scripts/train_motion_lora.py --variant full --steps 800
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vdprobe import runio
from vdprobe.config import REPO_ROOT
from vdprobe.lora import VARIANTS, add_motion_lora, n_params, save_motion_lora

DATA = REPO_ROOT / "data" / "hmdb51_lora"
FRAMES, SIZE = 16, 256
VAL_T = (50, 150, 250, 400, 550, 700, 850, 950)


def read_clip(path: Path) -> np.ndarray:
    import cv2

    cap, out = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return np.stack(out)                                  # (n, H, W, 3) uint8


def window(clip: np.ndarray, rng: random.Random | None) -> torch.Tensor:
    """(1, 3, F, SIZE, SIZE) in [-1, 1]; random start/crop when rng is given, else centred."""
    n, h, w, _ = clip.shape
    stride = max(1, min(2, (n - 1) // (FRAMES - 1)))
    span = stride * (FRAMES - 1) + 1
    start = rng.randint(0, max(0, n - span)) if rng else max(0, (n - span) // 2)
    idx = [min(start + i * stride, n - 1) for i in range(FRAMES)]
    x0 = rng.randint(0, w - h) if rng else (w - h) // 2
    x = torch.from_numpy(clip[idx][:, :, x0:x0 + h]).permute(0, 3, 1, 2).float() / 127.5 - 1
    x = F.interpolate(x, size=(SIZE, SIZE), mode="bilinear", antialias=True, align_corners=False)
    if rng and rng.random() < 0.5:
        x = x.flip(-1)
    return x.permute(1, 0, 2, 3).unsqueeze(0)


@torch.no_grad()
def encode_video(vae, x: torch.Tensor) -> torch.Tensor:
    f = x.shape[2]
    frames = x[0].permute(1, 0, 2, 3).to(vae.device, vae.dtype)        # (F, 3, H, W)
    lat = vae.encode(frames).latent_dist.sample() * vae.config.scaling_factor
    return lat.permute(1, 0, 2, 3).unsqueeze(0).float()                  # (1, 4, F, h, w)


def checkpoint_motion_modules(unet) -> int:
    """diffusers' gradient checkpointing covers the resnets only; the motion modules (and the
    LoRA inside them) keep all activations, which overflows 8 GB for the full variant. Wrap each
    motion module in non-reentrant checkpointing too: same gradients, recomputed forward."""
    from diffusers.models.unets.unet_motion_model import AnimateDiffTransformer3D
    from torch.utils.checkpoint import checkpoint

    n = 0
    for m in unet.modules():
        if isinstance(m, AnimateDiffTransformer3D):
            fwd = m.forward
            m.forward = lambda *a, fwd=fwd, **k: (checkpoint(fwd, *a, use_reentrant=False, **k)
                                                  if torch.is_grad_enabled() else fwd(*a, **k))
            n += 1
    return n


def main():
    ap = runio.parser(__doc__, "motion_lora")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=5e-5)     # roadmap §5.5 motion LoRA
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out / args.variant
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    loaded = runio.load(args)
    pipe = loaded.pipe
    unet, vae = pipe.unet, pipe.vae
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    params = add_motion_lora(unet, args.variant, rank=args.rank, alpha=args.rank)
    unet.enable_gradient_checkpointing()
    n_ckpt = checkpoint_motion_modules(unet)
    unet.train()
    n_train = sum(p.numel() for p in params)
    print(f"{args.variant}: {n_train / 1e6:.2f} M trainable LoRA parameters, {n_ckpt} motion modules checkpointed")

    manifest = json.loads((DATA / "manifest.json").read_text())
    by_class = {}
    for c in manifest["clips"]:
        by_class.setdefault(c["class"], []).append(c)
    train, val = [], []
    for cls, cs in by_class.items():
        cs = sorted(cs, key=lambda c: c["path"])
        val.append(cs[-1])
        train += cs[:-1]
    clips = {c["path"]: read_clip(DATA / c["path"]) for c in train + val}

    @torch.no_grad()
    def embed(verb):
        return pipe.encode_prompt(f"a person {verb}", loaded.device, 1, False)[0].repeat_interleave(FRAMES, 0)
    emb = {v: embed(v) for v in set(manifest["classes"].values())}

    sched = pipe.scheduler
    acp = sched.alphas_cumprod.to(loaded.device)

    def loss_at(lat, verb, t, noise):
        a = acp[t].view(1, 1, 1, 1, 1)
        noisy = a.sqrt() * lat + (1 - a).sqrt() * noise
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(noisy.half(), t, encoder_hidden_states=emb[verb]).sample
        return F.mse_loss(pred.float(), noise)

    val_lat = [(encode_video(vae, window(clips[c["path"]], None)), c["verb"]) for c in val]
    g = torch.Generator(device="cpu").manual_seed(1234)
    val_noise = [[torch.randn(lat.shape, generator=g).to(loaded.device) for _ in range(2)] for lat, _ in val_lat]

    @torch.no_grad()
    def validate():
        unet.eval()
        ls = [float(loss_at(lat, verb, torch.tensor(t, device=loaded.device), nz))
              for (lat, verb), nzs in zip(val_lat, val_noise) for t in VAL_T for nz in nzs]
        unet.train()
        return float(np.mean(ls))

    log = {"variant": args.variant, "steps": args.steps, "lr": args.lr, "rank": args.rank,
           "trainable_params": n_train, "train_clips": [c["path"] for c in train],
           "val_clips": [c["path"] for c in val], "loss": [], "val": []}
    v0 = validate()
    log["val"].append([0, v0])
    print(f"step    0  val {v0:.5f}", flush=True)

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=1e-2)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        c = rng.choice(train)
        lat = encode_video(vae, window(clips[c["path"]], rng))
        t = torch.randint(0, sched.config.num_train_timesteps, (1,), device=loaded.device)
        loss = loss_at(lat, c["verb"], t, torch.randn_like(lat))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if step == 1 and not any(p.grad is not None and p.grad.abs().sum() > 0 for p in params):
            raise RuntimeError("no gradient reaches the LoRA parameters (checkpointing?)")
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        log["loss"].append(float(loss.detach()))
        if step % args.val_every == 0 or step == args.steps:
            v = validate()
            log["val"].append([step, v])
            el = time.perf_counter() - t0
            print(f"step {step:>4}  train(avg100) {np.mean(log['loss'][-100:]):.5f}  val {v:.5f}  "
                  f"{el:.0f} s ({el / step:.2f} s/step)", flush=True)
    log["seconds"] = time.perf_counter() - t0
    log["peak_alloc_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
    meta = {"variant": args.variant, "rank": args.rank, "alpha": args.rank, "steps": args.steps, "lr": args.lr}
    unet.eval()
    from peft.utils import get_peft_model_state_dict
    log["saved_params"] = n_params(get_peft_model_state_dict(unet, adapter_name="motion"))
    save_motion_lora(unet, out / "motion_lora.safetensors", meta)
    runio.write_json(out / "train_log.json", runio.record("motion_lora", loaded, **log))


if __name__ == "__main__":
    main()
