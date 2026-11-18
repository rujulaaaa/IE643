"""Part C: train a TopK sparse autoencoder on the motion site's tokens and label its features (HS1).

Tokens come from `scripts/camera_sweep.py --tokens` (`<group>/tokens.pt`, {clip: {step: (1024, C)}}).
The SAE is trained on the tokens of --train-seed and every statistic is computed on --eval-seed.

    <out>/<variant>/sae.safetensors, train_log.json      variants: default (5120 features, k 32),
                                                          half (2560, k 32), k64 (5120, k 64)
    <out>/features.csv      default variant, per feature: camera_auroc, camera (best class), scene_auroc,
                            camera_eta2, scene_eta2, frequency, mean activation per step and per frame
    <out>/summary.json      HS1 count for the SAE features and, as baselines, for the raw channels and
                            for as many random unit directions; explained variance on the eval seed
    <out>/directions.pt     decoder directions of the top camera features and the mean-difference
                            vector of every camera (activation scale), for scripts/sae_steer.py

    python scripts/sae_train.py input/vdprobe-cam-sweep --out results/camera/sae
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vdprobe import sae as S
from vdprobe import sae_features as SF

VARIANTS = {"default": (5120, 32), "half": (2560, 32), "k64": (5120, 64)}


def load_tokens(roots, seed):
    files = sorted({f for r in roots for f in Path(r).resolve().rglob("tokens.pt")})   # resolve: input/<slug> is a symlink
    out = {}
    for f in files:
        out.update({cid: v for cid, v in torch.load(f, map_location="cpu", weights_only=True).items() if SF.parse_clip_id(cid)["seed"] == seed})
    if not out:
        raise SystemExit(f"no tokens of seed {seed} under {roots}")
    return out


def flat(tokens):
    return torch.cat([t for cid in sorted(tokens) for _, t in sorted(tokens[cid].items())]).float()


@torch.no_grad()
def explained_variance(sae, x, batch=8192, device="cpu"):
    sae = sae.to(device)
    sq = sum(float((sae(x[i:i + batch].to(device))[0] - x[i:i + batch].to(device)).pow(2).sum()) for i in range(0, len(x), batch))
    sae.cpu()
    return 1.0 - sq / float((x - x.mean(0)).pow(2).sum())


@torch.no_grad()
def profiles(sae, norm, tokens, frames, device, batch=8192):
    """Per feature: firing frequency, mean activation per step and per frame index (rows are frame-major)."""
    sae = sae.to(device)
    steps = sorted(next(iter(tokens.values())))
    by_step = torch.zeros(len(steps), sae.n_features)
    by_frame = torch.zeros(frames, sae.n_features)
    fired, n = torch.zeros(sae.n_features), 0
    for cid in sorted(tokens):
        for si, s in enumerate(steps):
            x = tokens[cid][s].float()
            z = torch.cat([sae.encode(norm(x[i:i + batch].to(device))).cpu() for i in range(0, len(x), batch)])
            by_step[si] += z.mean(0)
            by_frame += z.view(frames, -1, sae.n_features).mean(1)
            fired += (z > 0).float().sum(0)
            n += len(z)
    sae.cpu()
    k = len(tokens)
    return fired / n, by_step / k, by_frame / (k * len(steps)), steps


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/camera/sae"))
    p.add_argument("--train-seed", type=int, default=42)
    p.add_argument("--eval-seed", type=int, default=1337)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--top", type=int, default=10, help="top features per camera kept in directions.pt")
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    train, test = load_tokens(args.roots, args.train_seed), load_tokens(args.roots, args.eval_seed)
    x_train = flat(train)
    norm = S.Normalizer.fit(x_train)
    xn_train, xn_test = norm(x_train), norm(flat(test))
    print(f"train {tuple(xn_train.shape)} from {len(train)} clips, eval {tuple(xn_test.shape)} from {len(test)} clips", flush=True)

    logs, default = {}, None
    for name in args.variants:
        n_features, k = VARIANTS[name]
        sae, log = S.train_sae(xn_train, n_features=n_features, k=k, epochs=args.epochs, batch=args.batch, lr=args.lr, device=device)
        log.update(epochs=args.epochs, batch=args.batch, lr=args.lr, steps=args.epochs * -(-len(xn_train) // args.batch))
        log["explained_variance_eval"] = explained_variance(sae, xn_test, device=device)
        S.save(args.out / name / "sae.safetensors", sae, norm, {"variant": name, "train_seed": args.train_seed})
        (args.out / name / "train_log.json").write_text(json.dumps(log, indent=1))
        logs[name] = {k_: log[k_] for k_ in ("explained_variance", "explained_variance_eval", "dead_fraction")}
        print(name, logs[name], flush=True)
        if name == "default" or default is None:
            default = sae

    # --- label on the eval seed ---------------------------------------------------------------------
    meta = [SF.parse_clip_id(c) for c in sorted(test)]
    cams, scenes = [m["camera"] for m in meta], [m["scene"] for m in meta]
    enc = lambda x: default.encode(norm(x))
    default.to(device)
    _, z = SF.clip_means(test, encode=enc, device=device)
    default.cpu()
    _, raw = SF.clip_means(test)
    g = torch.Generator().manual_seed(643)
    rand = torch.nn.functional.normalize(torch.randn(raw.shape[1], raw.shape[1], generator=g), dim=0).numpy()
    lab = {"sae": SF.label_directions(z, cams, scenes), "channels": SF.label_directions(raw, cams, scenes),
           "random": SF.label_directions(raw @ rand, cams, scenes)}
    freq, by_step, by_frame, steps = profiles(default, norm, test, args.frames, device)
    L = lab["sae"]
    df = pd.DataFrame({"feature": np.arange(len(freq)), "camera_auroc": L["camera_auroc"],
                       "camera": [L["classes"][i] for i in L["camera_argmax"]], "scene_auroc": L["scene_auroc"],
                       "camera_eta2": L["camera_eta2"], "scene_eta2": L["scene_eta2"], "camera_only": L["camera_only"],
                       "frequency": freq.numpy()})
    for si, s in enumerate(steps):
        df[f"step_{s}"] = by_step[si].numpy()
    for f in range(args.frames):
        df[f"frame_{f}"] = by_frame[f].numpy()
    df.to_csv(args.out / "features.csv", index=False)
    for kind in ("channels", "random"):
        pd.DataFrame({k: lab[kind][k] for k in ("camera_auroc", "scene_auroc", "camera_eta2", "scene_eta2")}) \
            .to_csv(args.out / f"baseline_{kind}.csv", index=False)

    # --- directions for steering: top camera-only features per camera, and mean-difference vectors (train seed)
    ids, raw_train = SF.clip_means(train)
    cam_train = np.array([SF.parse_clip_id(c)["camera"] for c in ids])
    off = next(c for c in ("none", "off") if c in cam_train)
    directions = {"mean_diff": {c: torch.from_numpy(raw_train[cam_train == c].mean(0) - raw_train[cam_train == off].mean(0))
                                for c in sorted(set(cam_train.tolist())) if c != off}, "features": {}, "scale": float(norm.scale)}
    for c in L["classes"]:
        d = df[(df.camera == c) & (df.frequency > 0)].sort_values(["camera_only", "camera_auroc"], ascending=False).head(args.top)
        directions["features"][str(c)] = {"index": [int(i) for i in d.feature], "camera_auroc": [float(v) for v in d.camera_auroc],
                                     "decoder": default.decoder.weight[:, d.feature.tolist()].T.clone() * norm.scale}
    torch.save(directions, args.out / "directions.pt")

    summary = {"train_seed": args.train_seed, "eval_seed": args.eval_seed, "n_tokens_train": len(xn_train),
               "variants": logs, "rule": f"camera AUROC >= {SF.CAMERA_MIN} and scene eta2 <= {SF.SCENE_ETA2_MAX}",
               "n_camera_only": {k: v["n_camera_only"] for k, v in lab.items()},
               "n_directions": {k: int(len(v["camera_auroc"])) for k, v in lab.items()},
               "camera_only_by_camera": df[df.camera_only].camera.value_counts().to_dict(),
               "HS1": {"min_features": SF.HS1_MIN_FEATURES, "pass": lab["sae"]["n_camera_only"] >= SF.HS1_MIN_FEATURES}}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
