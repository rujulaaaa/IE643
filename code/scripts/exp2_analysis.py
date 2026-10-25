"""Experiment 2 Phase 2 · per-cell analysis of one refined sweep (CPU).

    python scripts/exp2_analysis.py results/kaggle/vdprobe-exp2-<cell>/out/<cell> --out results/exp2/<cell>

Sections (docs/experiment2.md; reports/Content motion probing experiment design.md):
  confirm    the pre-registered Experiment 1 region (up_blocks.0's last motion module, steps 0-2):
             per-subject mean S_dyn, one-sided exact sign flip (S < 0 = motion-leaning)
  smap       S_dyn / S_stat on every scored coordinate (coarse + refined) x 25 steps: per-subject
             means, sign-flip t, per-cell p and max-|t| FWER p over the map
  anova      variance fractions of each coordinate's sketch (x_tilde, x_bar) by subject, verb, scene,
             seed and subject x verb (balanced factorial)
  rsa        per group: layer RDM (x_tilde sketch) on a content RDM (CLIP) and a motion RDM (camera-
             compensated, subject-masked RAFT flow); unique shares, permutation within group,
             averaged over groups; FWER from the max over the map
  decode     within-prompt flow decoding gain over the latent-only control (leave-one-subject-out)
  channels   MOFT-style channel selection on seed A, evaluated on seed B with per-channel distances
  heads      verb eta^2 of each temporal-attention head's entropy / off-diagonal mass / frame offset
Writes <out>/exp2_analysis.json and <out>/maps.npz.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vdprobe import analysis as A
from vdprobe import exp2 as E

REGION = ("up_blocks.0.motion_modules.2", (0, 1, 2))   # Experiment 1's motion site (pre-registered)


def load_cell(root: Path):
    groups, long = A.load_all([root])
    meta = {}
    for g in groups:
        meta.update(g.manifest["clip_meta"])
    return groups, long, meta


def units_with_subject(long: pd.DataFrame, meta: dict) -> pd.DataFrame:
    u = A.selectivity_units(long)
    u["subject"] = u.unit.map(lambda c: meta[c]["subject"])
    return u


def section_smap(units, coords, n_steps, n_flip):
    out, maps = {}, {}
    for chan in ("dyn", "stat"):
        X, subjects = E.subject_means(units, chan, coords, range(n_steps))
        keep = ~np.isnan(X).any(1)
        r = E.sign_flip_test(X[keep], n_flip=n_flip)
        for k in ("mean", "t", "p", "p_fwer"):
            maps[f"S_{chan}_{k}"] = r[k].reshape(len(coords), n_steps)
        out[chan] = {"n_subjects": int(keep.sum()), "n_patterns": r["n_patterns"],
                     "cells_fwer_05": int((r["p_fwer"] < .05).sum()),
                     "motion_cells_fwer_05": int(((r["p_fwer"] < .05) & (r["t"] < 0)).sum()),
                     "content_cells_fwer_05": int(((r["p_fwer"] < .05) & (r["t"] > 0)).sum())}
    return out, maps


def section_confirm(units, n_flip):
    coord, steps = REGION
    X, subjects = E.subject_means(units, "dyn", [coord], steps)
    x = np.nanmean(X, 1, keepdims=True)
    r = E.sign_flip_test(x[~np.isnan(x[:, 0])], n_flip=n_flip, alternative="less")
    return {"coord": coord, "steps": list(steps), "mean_S_dyn": float(r["mean"][0]), "t": float(r["t"][0]),
            "p_one_sided": float(r["p"][0]), "n_subjects": r["n"], "n_patterns": r["n_patterns"],
            "replicates": bool(r["p"][0] < .05),
            "rule": "per-subject mean S_dyn over the region < 0, exact sign flip, one-sided p < 0.05"}


def load_views(groups):
    sk, lat, flows, emb, xat = {}, {}, {}, {}, {}
    for g in groups:
        p = g.path
        sk.update({k: v.float().numpy() for k, v in torch.load(p / "sketches_all.pt").items()})
        if (p / "latent_traj.pt").exists():
            lat.update({k: v.float().numpy() for k, v in torch.load(p / "latent_traj.pt").items()})
        flows.update({k: v.float().numpy() for k, v in torch.load(p / "flows.pt").items()})
        emb.update({k: v.float().numpy() for k, v in torch.load(p / "clip_emb.pt").items()})
        xat.update({k: v["subject"].float().numpy() for k, v in torch.load(p / "xattn.pt").items()})
    return sk, lat, flows, emb, xat


def subject_masks(xat: dict, tau: float = 0.4) -> dict:
    return {k: v >= tau * v.max(axis=(-1, -2), keepdims=True) for k, v in xat.items()}


def section_anova(sk, meta, coords, n_steps):
    clips = sorted(sk)
    d = pd.DataFrame([{**meta[c], "seed": c.split("__")[0]} for c in clips])
    factors = ["subject", "verb", "scene", "seed"]
    out = {f: np.zeros((len(coords), n_steps, 2)) for f in factors + ["subjectxverb", "residual"]}
    X = np.stack([sk[c] for c in clips])                     # (n, L, T, 2, dim)
    for li in range(len(coords)):
        for t in range(n_steps):
            for part in (0, 1):
                fr = E.anova_fractions(X[:, li, t, part], d, factors, pairs=[("subject", "verb")])
                for k, v in fr.items():
                    out[k][li, t, part] = v
    return out


RSA_PREDICTORS = ("content", "motion", "text")


def _rsa_group(Xg, Dc, Dm, Dt, n_perm, seed):
    """Xg: (n, L, T, dim) x_tilde sketches of one group. Model RDMs as in the design report: content (CLIP image),
    motion (masked, camera-compensated flow) and the prompt's text embedding, so the unique motion share is what
    flow explains beyond both what is shown and what the prompt says (the verb reaches every layer through the
    text). Returns observed unique shares (L, T, 3) and permutation draws (R, L, T, 3), one shared permutation
    per draw."""
    from vdprobe.analysis import rsa_commonality

    n, L, T, _ = Xg.shape
    iu = np.triu_indices(n, 1)
    preds = {"content": Dc[iu], "motion": Dm[iu], "text": Dt[iu]}
    rng = np.random.default_rng(seed)
    perms = [np.arange(n)] + [rng.permutation(n) for _ in range(n_perm)]
    res = np.zeros((len(perms), L, T, len(RSA_PREDICTORS)))
    for li in range(L):
        for t in range(T):
            D = E.cosine_rdm(Xg[:, li, t])
            for r, pm in enumerate(perms):
                u = rsa_commonality(D[np.ix_(pm, pm)][iu], preds)["unique"]
                res[r, li, t] = [u[k] for k in RSA_PREDICTORS]
    return res[0], res[1:]


def section_rsa(groups, sk, flows, emb, masks, coords, n_steps, n_perm, n_jobs):
    from joblib import Parallel, delayed

    from vdprobe.metrics import ClipScorer

    clip_text = ClipScorer(device="cpu")
    obs, null = [], []
    for g in groups:
        clips = g.manifest["clips"]
        Xg = np.stack([sk[c][:, :, 0] for c in clips])        # x_tilde
        Dc = E.content_rdm(np.stack([emb[c] for c in clips]))
        Dm = E.flow_rdm(np.stack([flows[c] for c in clips]), np.stack([masks[c] for c in clips]))
        Dt = E.cosine_rdm(clip_text.text([g.manifest["prompts"][c] for c in clips]).numpy())
        # split the coordinates across workers; every worker uses the same permutations (same seed)
        chunks = np.array_split(np.arange(Xg.shape[1]), max(1, n_jobs))
        res = Parallel(n_jobs=n_jobs)(delayed(_rsa_group)(Xg[:, ch], Dc, Dm, Dt, n_perm, 643)
                                      for ch in chunks if len(ch))
        obs.append(np.concatenate([r[0] for r in res], axis=0))
        null.append(np.concatenate([r[1] for r in res], axis=1))
    o, nl = np.mean(obs, 0), np.mean(null, 0)                 # averaged over groups, per permutation draw
    p = (1 + (nl >= o[None] - 1e-12).sum(0)) / (1 + n_perm)
    mx = nl.reshape(n_perm, -1, nl.shape[-1]).max(1)          # max over the map, per draw and predictor
    p_fwer = (1 + (mx[:, None, None, :] >= o[None] - 1e-12).sum(0)) / (1 + n_perm)
    return {"unique": o, "p": p, "p_fwer": p_fwer}


def section_decode(sk, lat, flows, masks, meta, coords, n_steps, n_jobs):
    from joblib import Parallel, delayed

    clips = sorted(c for c in sk if c in lat)
    if not clips:
        return None
    pid = np.array(["|".join((meta[c]["subject"], meta[c]["verb"], meta[c]["scene"])) for c in clips])
    groups = np.array([meta[c]["subject"] for c in clips])
    Y = np.stack([E.flow_descriptor(flows[c], masks[c]) for c in clips])
    rng = np.random.default_rng(0)
    Lz = np.stack([lat[c].reshape(lat[c].shape[0], -1) for c in clips])     # (n, T, d_lat)
    proj = rng.normal(0, 1 / np.sqrt(Lz.shape[-1]), (Lz.shape[-1], 512))   # fixed random projection

    def cell(li, t):
        r = E.decoding_gain(np.stack([sk[c][li, t, 0] for c in clips]), Lz[:, t] @ proj, Y, pid, groups)
        return r["r2_layer"], r["r2_latent"]

    res = Parallel(n_jobs=n_jobs)(delayed(cell)(li, t) for li in range(len(coords)) for t in range(n_steps))
    res = np.array(res).reshape(len(coords), n_steps, 2)
    return {"r2_layer": res[..., 0], "r2_latent": res[..., 1],
            "gain": np.maximum(res[..., 0], 0) - np.maximum(res[..., 1], 0)}     # floored as in exp2.decoding_gain


def section_channels(groups, meta, top: float = 0.04):
    """Select top channels per (refined coord, step) on the first seed by verb eta^2 of dynamic energy;
    evaluate S_dyn from per-channel distances on the other seed, all channels vs selected."""
    ch, pc, pairs = {}, {}, []
    for g in groups:
        ch.update(torch.load(g.path / "channel_stats.pt"))
        d = torch.load(g.path / "per_channel.pt")
        for (pi, coord, step), v in d.items():
            pc[(g.seed, g.scene, pi, coord, step)] = v.float().numpy()
        pp = g.pairs.copy()
        pp["seed"], pp["scene"] = g.seed, g.scene
        pairs.append(pp)
    pairs = pd.concat(pairs)
    man = groups[0].manifest
    ref, steps = man["refined_coords"], man["refined_steps"]
    seeds = sorted({g.seed for g in groups})
    if len(seeds) < 2:
        return None
    sel_seed, ev_seed = seeds[0], seeds[1]
    clips = [c for c in ch if c.startswith(f"s{sel_seed}__")]
    verbs = np.array([meta[c]["verb"] for c in clips])
    E_all = np.stack([ch[c] for c in clips])                  # (n, L, S, 2, C)
    out = {}
    for li, coord in enumerate(ref):
        for t in (0, 4, 8, 12):
            if t not in steps:
                continue
            ti = steps.index(t)
            eta = E.channel_selectivity(E_all[:, li, ti, 1], verbs)
            k = max(1, int(round(top * len(eta))))
            idx = np.argsort(eta)[-k:]
            rows = []
            for r in pairs[pairs.seed == ev_seed].itertuples():
                v = pc.get((r.seed, r.scene, r.pair, coord, t))
                if v is not None:
                    rows.append({"kind": r.kind, "b": r.b, "all": float(v.mean()), "top": float(v[idx].mean())})
            if not rows:
                continue
            df = pd.DataFrame(rows)
            c, m = df[df.kind == "content"], df[df.kind == "motion"]
            s = lambda col: float((c[col].median() - m[col].median()) / max(c[col].median() + m[col].median(), 1e-12))
            out[f"{coord}|{t}"] = {"S_dyn_all": s("all"), "S_dyn_top": s("top"), "eta_top_mean": float(eta[idx].mean()),
                                   "k": k, "select_seed": sel_seed, "eval_seed": ev_seed}
    return out


def section_heads(groups, meta):
    hs = {}
    for g in groups:
        hs.update(torch.load(g.path / "head_stats.pt"))
    clips = sorted(hs)
    verbs = np.array([meta[c]["verb"] for c in clips])
    H = np.stack([hs[c].float().numpy() for c in clips])       # (n, L_attn, T, heads, 3)
    n, L, T, nh, k = H.shape
    eta = np.zeros((L, T, nh, k))
    for li in range(L):
        for t in range(T):
            eta[li, t] = E.channel_selectivity(H[:, li, t].reshape(n, -1), verbs).reshape(nh, k)
    return eta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path, help="the cell's output directory (holds one directory per group)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-flip", type=int, default=10000)
    p.add_argument("--n-perm", type=int, default=49, help="RSA permutation draws")
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--skip", nargs="*", default=[], help="sections to skip")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    groups, long, meta = load_cell(args.root)
    man = groups[0].manifest
    scored, all_coords, n_steps = man["scored_coords"], man["all_coords"], man["steps"]
    res = {"cell": args.root.name, "groups": [g.path.name for g in groups], "n_clips": len(meta),
           "revisions": man["revisions"], "region": {"coord": REGION[0], "steps": list(REGION[1])}}
    maps = {}
    units = units_with_subject(long, meta) if (long.kind == "content").any() else None
    if units is not None and "confirm" not in args.skip:
        res["confirm"] = section_confirm(units, args.n_flip)
        print("confirm:", res["confirm"], flush=True)
    if units is not None and "smap" not in args.skip:
        res["smap"], m = section_smap(units, scored, n_steps, args.n_flip)
        maps.update(m)
        print("smap:", res["smap"], flush=True)
    sk, lat, flows, emb, xat = load_views(groups)
    masks = subject_masks(xat)
    if "anova" not in args.skip:
        an = section_anova(sk, meta, all_coords, n_steps)
        maps.update({f"anova_{k}": v for k, v in an.items()})
        res["anova"] = {k: {"max_tilde": float(np.nanmax(v[..., 0])), "mean_tilde": float(np.nanmean(v[..., 0])),
                            "mean_bar": float(np.nanmean(v[..., 1]))} for k, v in an.items()}
        print("anova:", res["anova"], flush=True)
    if "rsa" not in args.skip:
        r = section_rsa(groups, sk, flows, emb, masks, all_coords, n_steps, args.n_perm, args.n_jobs)
        maps.update({f"rsa_{k}": v for k, v in r.items()})
        res["rsa"] = {"n_perm": args.n_perm,
                      "motion_cells_fwer_05": int((r["p_fwer"][..., 1] < .05).sum()),
                      "content_cells_fwer_05": int((r["p_fwer"][..., 0] < .05).sum()),
                      "text_cells_fwer_05": int((r["p_fwer"][..., 2] < .05).sum()),
                      "predictors": list(RSA_PREDICTORS),
                      "best_motion": all_coords[int(np.unravel_index(r["unique"][..., 1].argmax(), r["unique"].shape[:2])[0])]}
        print("rsa:", res["rsa"], flush=True)
    if "decode" not in args.skip:
        d = section_decode(sk, lat, flows, masks, meta, all_coords, n_steps, args.n_jobs)
        if d is not None:
            maps.update({f"decode_{k}": v for k, v in d.items()})
            li, t = np.unravel_index(d["gain"].argmax(), d["gain"].shape)
            res["decode"] = {"best_gain": float(d["gain"].max()), "best_coord": all_coords[li], "best_step": int(t),
                             "cells_gain_pos": int((d["gain"] > 0).sum())}
            print("decode:", res["decode"], flush=True)
    if "channels" not in args.skip:
        res["channels"] = section_channels(groups, meta)
    if "heads" not in args.skip:
        maps["head_eta"] = section_heads(groups, meta)
    res["seconds"] = time.perf_counter() - t0
    (args.out / "exp2_analysis.json").write_text(json.dumps(res, indent=1, default=float))
    np.savez_compressed(args.out / "maps.npz", scored=np.array(scored), all_coords=np.array(all_coords), **maps)
    print(f"wrote {args.out} in {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
