"""Experiment 2 Phase 2 · the 2x2 model grid as paired contrasts (CPU).

    python scripts/exp2_grid.py --cells sd15_mm2=<dir> epicrealism_mm2=<dir> sd15_mm3=<dir> \
        epicrealism_mm3=<dir> --out results/exp2/grid

Every cell shares prompts and noise, so each contrast is within subject: for subject s and
(coordinate, step), Delta_s = S_dyn,s(cell a) - S_dyn,s(cell b), tested by sign flip over
subjects with max-|t| FWER over the map (coordinates present in every cell; v1-5-3 has no
mid-block motion module). Contrasts: base swap at each adapter, adapter swap at each base, the
interaction (difference of differences) and the diagonal. Also the confirmatory region per cell
and cross-cell debiased CKA per (coordinate, step) on matched clips (same prompt, same noise).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from vdprobe import analysis as A
from vdprobe import exp2 as E

CONTRASTS = {
    "base_at_mm2": ("epicrealism_mm2", "sd15_mm2"),
    "base_at_mm3": ("epicrealism_mm3", "sd15_mm3"),
    "adapter_at_sd15": ("sd15_mm3", "sd15_mm2"),
    "adapter_at_epic": ("epicrealism_mm3", "epicrealism_mm2"),
    "diagonal": ("epicrealism_mm3", "sd15_mm2"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", nargs="+", required=True, help="name=directory")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-flip", type=int, default=10000)
    p.add_argument("--cka-steps", type=int, nargs="*", default=[0, 2, 5, 10, 15, 20, 24])
    args = p.parse_args()
    cells = dict(c.split("=", 1) for c in args.cells)
    units, mans, sk = {}, {}, {}
    for name, d in cells.items():
        groups, long = A.load_all([Path(d)])
        meta = {k: v for g in groups for k, v in g.manifest["clip_meta"].items()}
        u = A.selectivity_units(long)
        u["subject"] = u.unit.map(lambda c: meta[c]["subject"])
        units[name], mans[name] = u, groups[0].manifest
        sk[name] = {k: v for g in groups for k, v in torch.load(g.path / "sketches_all.pt").items()}
    scored = sorted(set.intersection(*(set(m["scored_coords"]) for m in mans.values())))
    n_steps = mans[next(iter(mans))]["steps"]
    S = {n: E.subject_means(u, "dyn", scored, range(n_steps)) for n, u in units.items()}
    res, maps = {"scored_coords": scored, "contrasts": {}}, {}
    for cname, (a, b) in list(CONTRASTS.items()) + [("interaction", (None, None))]:
        if cname == "interaction":
            need = ["epicrealism_mm3", "sd15_mm3", "epicrealism_mm2", "sd15_mm2"]
            if not all(n in S for n in need):
                continue
            X = (S["epicrealism_mm3"][0] - S["sd15_mm3"][0]) - (S["epicrealism_mm2"][0] - S["sd15_mm2"][0])
        else:
            if a not in S or b not in S:
                continue
            assert S[a][1] == S[b][1], "subject order differs between cells"
            X = S[a][0] - S[b][0]
        X = X[~np.isnan(X).any(1)]
        r = E.sign_flip_test(X, n_flip=args.n_flip)
        maps.update({f"{cname}_{k}": r[k].reshape(len(scored), n_steps) for k in ("mean", "t", "p_fwer")})
        res["contrasts"][cname] = {"cells": [a, b] if a else "dd", "n_subjects": r["n"],
                                   "cells_fwer_05": int((r["p_fwer"] < .05).sum()),
                                   "more_motion_cells": int(((r["p_fwer"] < .05) & (r["t"] < 0)).sum()),
                                   "more_content_cells": int(((r["p_fwer"] < .05) & (r["t"] > 0)).sum())}
        print(cname, res["contrasts"][cname], flush=True)
    # cross-cell debiased CKA, x_tilde sketches, matched clips
    names = list(cells)
    shared = sorted(set.intersection(*(set(m["all_coords"]) for m in mans.values())))
    clips = sorted(set.intersection(*(set(s) for s in sk.values())))
    idx = {n: [mans[n]["all_coords"].index(c) for c in shared] for n in names}
    cka = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            M_ = np.zeros((len(shared), len(args.cka_steps)))
            for li in range(len(shared)):
                for j, t in enumerate(args.cka_steps):
                    Xa = np.stack([sk[a][c][idx[a][li], t, 0].float().numpy() for c in clips])
                    Xb = np.stack([sk[b][c][idx[b][li], t, 0].float().numpy() for c in clips])
                    M_[li, j] = A.debiased_cka(Xa, Xb)
            cka[f"{a}|{b}"] = M_
            maps[f"cka_{a}|{b}"] = M_
    res["cka"] = {k: {"mean": float(np.nanmean(v)), "min": float(np.nanmin(v))} for k, v in cka.items()}
    res["cka_coords"], res["cka_steps"], res["n_matched_clips"] = shared, args.cka_steps, len(clips)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "grid.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(args.out / "grid_maps.npz", **maps)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
