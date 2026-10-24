"""Experiment 2 Phase 5c · LEACE erasure vs a same-rank random eraser at a candidate coordinate.

    python -u scripts/exp2_erasure.py --coord up_blocks.0.motion_modules.2 --steps 0 1 2 3 4 \
        [--config configs/variants/<primary cell>.yaml] [--n-fit 32] [--n-test 20]

1. Fit: generate n_fit probing clips (seed 1337, scene "on a beach", subjects x verbs balanced),
   sampling x_tilde channel vectors at the coordinate over the step window (erasure.TildeSampler).
   Fit LEACE for the verb (rank 3) and for the subject (selectivity control), each with a random
   eraser of the same rank.
2. Test: n_test held-out clips (seed 2024), each generated unerased and with each of the four
   erasers applied to x_tilde over the window (erasure.EraseProbe). Per clip and eraser:
     motion_change = 1 - s(erased, unerased)    s = camera-compensated flow cosine in the subject mask
     verb_drop     = X-CLIP probability of the clip's verb, unerased minus erased
Writes <out>/rows.csv and <out>/manifest.json (means; LEACE - random differences per concept).
"""

import time

import numpy as np
import pandas as pd
import torch

from vdprobe import metrics as M
from vdprobe import quality as Q
from vdprobe import runio
from vdprobe.coords import by_name, enumerate_coordinates
from vdprobe.erasure import EraseProbe, TildeSampler, fit_leace, random_eraser
from vdprobe.harness import CrossAttentionMaps, Harness
from vdprobe.pipeline import generate, make_noise, phrase_token_positions
from vdprobe.probing_set import ProbingSet


def main():
    ap = runio.parser(__doc__, "exp2_erasure")
    ap.add_argument("--coord", required=True)
    ap.add_argument("--steps", type=int, nargs="+", required=True)
    ap.add_argument("--n-fit", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=20)
    args = ap.parse_args()
    loaded = runio.load(args)
    cfg = loaded.cfg
    coords = enumerate_coordinates(loaded.unet, resnets=True)   # a candidate may be a ResNet block
    coord = by_name(coords)[args.coord]
    ps = ProbingSet()
    t0 = time.perf_counter()

    # 1. fit
    rng = np.random.default_rng(1337)
    fit = [c for c in ps.clips(1337) if c.scene == "on a beach"]
    fit = [fit[i] for i in sorted(rng.choice(len(fit), size=min(args.n_fit, len(fit)), replace=False))]
    X, verbs, subjects = [], [], []
    for c in fit:
        smp = TildeSampler(coord, args.steps, n_pos=256, seed=len(X))
        with Harness(loaded.unet, coords, [smp]):
            generate(loaded, c.prompt(cfg), make_noise(cfg, c.seed), decode_frames=False)
        x = torch.cat(smp.samples).float().numpy()
        X.append(x)
        verbs += [c.verb] * len(x)
        subjects += [c.subject] * len(x)
    X = np.concatenate(X)
    onehot = lambda labels: (np.array(labels)[:, None] == np.array(sorted(set(labels)))[None]).astype(float)
    erasers = {}
    for concept, labels in (("verb", verbs), ("subject", subjects)):
        e = fit_leace(X, onehot(labels))
        erasers[f"leace_{concept}"] = e
        erasers[f"random_{concept}"] = random_eraser(X, e.rank, seed=7)
    print(f"fitted on {len(fit)} clips, {len(X)} samples, ranks "
          f"{ {k: e.rank for k, e in erasers.items()} }, {time.perf_counter() - t0:.0f} s", flush=True)

    # 2. test
    xclip = Q.XClipVerbs(ps.verbs, device=loaded.device)
    rng = np.random.default_rng(2024)
    test = ps.clips(2024)
    test = [test[i] for i in sorted(rng.choice(len(test), size=min(args.n_test, len(test)), replace=False))]
    attn2 = [c for c in coords if c.kind == "spatial_attn2"]
    rows = []
    for k, c in enumerate(test):
        prompt = c.prompt(cfg)
        tok = phrase_token_positions(loaded.pipe.tokenizer, prompt, {"subject": c.subject})
        xa = CrossAttentionMaps(["subject"], coords=attn2, grid=8, token_index=tok)
        with Harness(loaded.unet, coords, [xa]):
            base = generate(loaded, prompt, make_noise(cfg, c.seed)).frames
        mask = xa.mask("subject").numpy()
        f0 = M.remove_camera_motion(M.raft_flow(base, 256, loaded.device), mask)
        p0 = xclip(base, c.subject)[ps.verbs.index(c.verb)]
        for name, e in erasers.items():
            probe = EraseProbe(coord, e, args.steps)
            with Harness(loaded.unet, coords, [probe]):
                fr = generate(loaded, prompt, make_noise(cfg, c.seed)).frames
            f1 = M.remove_camera_motion(M.raft_flow(fr, 256, loaded.device), mask)
            rows.append({"clip": c.id, "subject": c.subject, "verb": c.verb, "eraser": name,
                         "motion_change": 1 - M.masked_motion_fidelity(f1, f0, mask, mask),
                         "verb_drop": float(p0 - xclip(fr, c.subject)[ps.verbs.index(c.verb)]),
                         "applied": probe.applied})
            if k < 2:
                runio.save_gif(fr, args.out / "gifs" / f"{c.id}__{name}.gif")
        if k < 2:
            runio.save_gif(base, args.out / "gifs" / f"{c.id}__none.gif")
        pd.DataFrame(rows).to_csv(args.out / "rows.csv", index=False)
        print(f"test {k + 1}/{len(test)} {c.id}  {time.perf_counter() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows)
    summ = df.groupby("eraser")[["motion_change", "verb_drop"]].mean()
    diffs = {}
    for concept in ("verb", "subject"):
        a = df[df.eraser == f"leace_{concept}"].set_index("clip")
        b = df[df.eraser == f"random_{concept}"].set_index("clip")
        diffs[concept] = {m: float((a[m] - b[m]).mean()) for m in ("motion_change", "verb_drop")}
    runio.write_json(args.out / "manifest.json", runio.record(
        "exp2_erasure", loaded, coord=args.coord, steps=args.steps, n_fit=len(fit), n_samples=len(X),
        ranks={k: e.rank for k, e in erasers.items()}, summary=summ.to_dict(), leace_minus_random=diffs,
        seconds=time.perf_counter() - t0))


if __name__ == "__main__":
    main()
