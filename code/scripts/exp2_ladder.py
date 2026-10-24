"""Experiment 2 Phase 3 · the content ladder: dose-response of every layer to measured content change.

    python scripts/exp2_ladder.py results/kaggle/vdprobe-exp2-ladder/out/<cell> --out results/exp2/ladder

Per pair (base clip A, one-slot variant B):
    dc  = 1 - mean_f cos(CLIP(A_f), CLIP(B_f))                       measured content change
    df  = 1 - s(A, B)  (camera-compensated flow cosine in the subject masks)   measured motion change
    d~  = d(coord, step) / d_null(coord, step)       the layer's distance over the null-rung floor
          of the same base, verb and seed (rung 0: "on a beach" -> "on the beach")
Per base (the subject unit; 8 bases) and per (coordinate, step, part):
    d~ = alpha + beta * dc + gamma * df  (OLS over its 24 non-null pairs)
beta = content sensitivity, gamma = sensitivity to the motion side effect of a content swap; mean
over bases, exact sign flip (2^8 patterns), max-|t| FWER over the map. Also: ladder validity
(Spearman of rung index 1-4 with dc, per base) and the P+ prediction as a per-base contrast of
the colour / species response ratio between 64x64 and 8x8 coordinates.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from vdprobe import analysis as A
from vdprobe import exp2 as E
from vdprobe import metrics as M
from vdprobe.coords import _block_and_level
from vdprobe.ladder import RUNGS, LadderSet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-flip", type=int, default=10000)
    args = p.parse_args()
    ls = LadderSet()
    groups, long = A.load_all([args.root])
    man = groups[0].manifest
    coords, n_steps = man["scored_coords"], man["steps"]
    emb, flows, xat = {}, {}, {}
    for g in groups:
        emb.update({k: v.float().numpy() for k, v in torch.load(g.path / "clip_emb.pt").items()})
        flows.update({k: v.float().numpy() for k, v in torch.load(g.path / "flows.pt").items()})
        xat.update({k: v["subject"].float().numpy() for k, v in torch.load(g.path / "xattn.pt").items()})
    masks = {k: v >= 0.4 * v.max(axis=(-1, -2), keepdims=True) for k, v in xat.items()}

    # measured content and motion change per pair
    pr = pd.concat([g.pairs.assign(seed=g.seed) for g in groups], ignore_index=True)
    meta = {k: v for g in groups for k, v in g.manifest["clip_meta"].items()}
    pr["base"] = pr.a.map(lambda c: ls.base_of(SimpleNamespace(subject=meta[c]["subject"])))
    pr["verb"] = pr.a.map(lambda c: meta[c]["verb"])
    def dc(a, b):
        ea, eb = emb[a] / np.linalg.norm(emb[a], axis=-1, keepdims=True), emb[b] / np.linalg.norm(emb[b], axis=-1, keepdims=True)
        return float(1 - (ea * eb).sum(-1).mean())
    def dfl(a, b):
        u = masks[a] | masks[b]
        fa, fb = M.remove_camera_motion(flows[a], u), M.remove_camera_motion(flows[b], u)
        return 1 - M.masked_motion_fidelity(fa, fb, u, u)
    pr["dc"] = [dc(a, b) for a, b in zip(pr.a, pr.b)]
    pr["df"] = [dfl(a, b) for a, b in zip(pr.a, pr.b)]
    res = {"n_pairs": len(pr), "rung_means": pr.groupby("kind")[["dc", "df"]].mean().round(4).to_dict(orient="index")}
    order = {r: i for i, r in enumerate(RUNGS)}
    val = []
    for base, g in pr[pr.kind.isin(RUNGS[1:5])].groupby("base"):
        val.append(spearmanr(g.kind.map(order), g.dc).statistic)
    res["validity_spearman_rung1to4_vs_dc"] = {"mean": float(np.nanmean(val)), "per_base": [float(v) for v in val]}
    print("rungs:", res["rung_means"], "\nvalidity:", res["validity_spearman_rung1to4_vs_dc"], flush=True)

    # layer distances normalised by the null rung of the same base / verb / seed
    d = long.merge(pr[["pair", "seed", "base", "verb", "dc", "df"]], on=["pair", "seed"])
    out_maps, res["dose"] = {}, {}
    for part in ("d_dynamic", "d_static"):
        null = d[d.kind == "null"].set_index(["base", "verb", "seed", "coord", "step"])[part]
        x = d[d.kind != "null"].copy()
        x["floor"] = null.reindex(pd.MultiIndex.from_frame(x[["base", "verb", "seed", "coord", "step"]])).to_numpy()
        x["dn"] = x[part] / np.maximum(x["floor"], 1e-6)
        bases = sorted(x.base.unique())
        B = np.full((len(bases), len(coords), n_steps), np.nan)
        G = np.full_like(B, np.nan)
        for bi, base in enumerate(bases):
            xb = x[x.base == base]
            for (coord, step), g in xb.groupby(["coord", "step"]):
                if coord not in coords:
                    continue
                Xd = np.column_stack([np.ones(len(g)), g.dc, g.df])
                beta, *_ = np.linalg.lstsq(Xd, g.dn.to_numpy(), rcond=None)
                B[bi, coords.index(coord), step], G[bi, coords.index(coord), step] = beta[1], beta[2]
        for name, arr in (("beta", B), ("gamma", G)):
            Xf = arr.reshape(len(bases), -1)
            ok = ~np.isnan(Xf).any(0)
            r = E.sign_flip_test(Xf[:, ok], n_flip=args.n_flip)
            t_full, pf = np.full(Xf.shape[1], np.nan), np.full(Xf.shape[1], np.nan)
            t_full[ok], pf[ok] = r["t"], r["p_fwer"]
            out_maps[f"{part}_{name}_mean"] = np.nanmean(arr, 0)
            out_maps[f"{part}_{name}_t"] = t_full.reshape(len(coords), n_steps)
            out_maps[f"{part}_{name}_p_fwer"] = pf.reshape(len(coords), n_steps)
            res["dose"][f"{part}_{name}"] = {"cells_fwer_05": int((pf < .05).sum()), "n_bases": len(bases)}
        # P+ contrast: colour / species response ratio, 64x64 coordinates vs 8x8 coordinates, steps 0-14
        lvl = {c: _block_and_level(c, 4)[1] for c in coords}
        rat = []
        for base in bases:
            xb = x[(x.base == base) & (x.step < 15)]
            r = xb[xb.kind == "colour"].groupby("coord").dn.mean() / xb[xb.kind == "species"].groupby("coord").dn.mean()
            hi = r[[c for c in r.index if lvl[c] == 0]].mean()
            lo = r[[c for c in r.index if lvl[c] == 3]].mean()
            rat.append(hi - lo)
        rat = np.array(rat)[:, None]
        rr = E.sign_flip_test(rat[~np.isnan(rat[:, 0])], n_flip=args.n_flip, alternative="greater")
        res["dose"][f"{part}_pplus_colour_over_species_64_minus_8"] = {
            "mean": float(rr["mean"][0]), "p_one_sided": float(rr["p"][0]), "n_bases": rr["n"],
            "prediction": "> 0: colour loads relatively more on fine (64x64) layers than on coarse (8x8)"}
    print("dose:", res["dose"], flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    pr.to_csv(args.out / "pairs_measured.csv", index=False)
    (args.out / "ladder.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(args.out / "ladder_maps.npz", coords=np.array(coords), **out_maps)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
