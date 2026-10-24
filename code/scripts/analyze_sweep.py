"""Stage 3-4 · From sweep distances to selectivity maps, the H1-H3 verdicts and linear probes.

    python scripts/analyze_sweep.py results/kaggle/vdprobe-sweep [more roots ...] --out results/analysis

Writes <out>/analysis.json (verdicts, block profiles, robustness numbers) and
<out>/maps.npz (S_dyn, S_stat, S_raw, S_l2dyn maps with CIs; probe accuracy maps; with
--probe-controls also control-task accuracy, selectivity and permutation p per probe), which the
figures and the interface read. Tests follow docs/hypotheses.md exactly (D-22, D-23); the
matched-unit definition of S is D-28 (see vdprobe/analysis.py).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from vdprobe import analysis as A
from vdprobe.coords import D07_STEPS, Coordinate, _block_and_level, classify

N_LEVELS = 4  # SD1.5: block_out_channels has 4 entries


def coord_meta(names):
    out = []
    for n in names:
        kind = classify(n)
        block, level = _block_and_level(n, N_LEVELS)
        c = Coordinate(n, kind, block, level)
        out.append({"name": n, "kind": kind, "block": block, "level": level, "short": c.short})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/analysis"))
    p.add_argument("--n-boot", type=int, default=A.N_BOOT)
    p.add_argument("--no-probes", action="store_true")
    p.add_argument("--probe-controls", action="store_true",
                   help="D-34: control task, MLP probe and permutation null next to the linear probes")
    p.add_argument("--n-perm", type=int, default=99, help="permutation draws for the D-34 null")
    p.add_argument("--mlp-hidden", type=int, default=10)
    p.add_argument("--mlp-alpha", type=float, default=1.0, help="L2 penalty of the MLP probe")
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    groups, long = A.load_all(args.roots)
    names = groups[0].manifest["coords"]
    meta = coord_meta(names)
    n_steps = int(long.step.max()) + 1
    print(f"{len(groups)} groups: " + ", ".join(f"s{g.seed}/{g.scene}" for g in groups))
    print(f"{long.pair.nunique()} pair ids per group (max), {len(long):,} distance rows")

    units = A.selectivity_units(long)
    n_units = units[["seed", "scene", "unit"]].drop_duplicates().groupby("seed").size().to_dict()
    print(f"matched selectivity units per seed (scene x clip): "
          f"{units[['seed', 'scene', 'unit']].drop_duplicates().groupby('seed').size().to_dict()}")

    maps = {ch: A.selectivity_map(units, ch, names, n_steps, args.n_boot) for ch in ("dyn", "stat", "raw", "l2dyn")}

    temporal_names = [m["name"] for m in meta if m["kind"] == "temporal_module"]
    h1 = A.test_h1(maps["dyn"], meta)
    win = next(m["name"] for m in meta if m["kind"] == "temporal_module" and m["block"] == h1["argmax_block"])
    h2 = A.test_h2(maps["dyn"], names, win)
    h3 = A.test_h3(maps["dyn"], maps["stat"], meta, D07_STEPS)

    # Static energy share F*||x_bar||^2/||x||^2 per coordinate (H3 prediction 3), anchors of content pairs
    se = long[long.kind == "content"].groupby("coord").static_energy_a.median().reindex(names)

    # Cross-family hard subset (D-05): content distance within vs across families, never pooled
    cf = long[long.kind.isin(["content", "cross_family"])].groupby(["kind", "coord"])[["d_static", "d_dynamic"]] \
        .median().unstack(0)

    # Robustness: does the L2 map rank coordinates like the cosine map? (D-01 appendix)
    rob = {"spearman_dyn_cos_vs_l2": A.spearman(maps["dyn"]["median"].ravel(), maps["l2dyn"]["median"].ravel()),
           "spearman_dyn_median_ratio_vs_ratio_of_medians":
               A.spearman(maps["dyn"]["median"].ravel(), maps["dyn"]["ratio_of_medians"].ravel())}
    seeds = sorted(maps["dyn"]["per_seed"])
    if len(seeds) > 1:
        rob["seed_agreement_dyn"] = {f"{a}-{b}": A.spearman(maps["dyn"]["per_seed"][a].ravel(),
                                                            maps["dyn"]["per_seed"][b].ravel())
                                     for i, a in enumerate(seeds) for b in seeds[i + 1:]}

    probes = {}
    if not args.no_probes:
        t0 = time.perf_counter()
        feats, clips, pnames = A.load_sketches(groups)
        assert pnames == names
        for key, target, chan in (("verb_tilde", "verb", 0), ("verb_bar", "verb", 1),
                                  ("family_tilde", "family", 0), ("family_bar", "family", 1)):
            probes[key] = A.probe_accuracy(feats, clips, target, chan, len(names), n_steps)
            print(f"  probe {key:<13} best {probes[key]['acc'].max():.3f}  mean {probes[key]['acc'].mean():.3f}  "
                  f"(chance {probes[key]['chance']:.2f})", flush=True)
        print(f"  probes done in {time.perf_counter() - t0:.0f} s")

    controls = {}
    if args.probe_controls and not args.no_probes:
        t0 = time.perf_counter()
        for target, chan, tag in (("verb", 0, "tilde"), ("verb", 1, "bar"), ("family", 0, "tilde"), ("family", 1, "bar")):
            for probe in A.PROBES:
                key = f"{target}_{tag}_{probe}"
                controls[key] = A.probe_scores(feats, clips, target, chan, len(names), n_steps, probe=probe,
                                               n_perm=args.n_perm, hidden=args.mlp_hidden, alpha=args.mlp_alpha,
                                               n_jobs=args.n_jobs)
                r = controls[key]
                print(f"  control {key:<20} acc {r['acc'].mean():.3f}  control {r['control_acc'].mean():.3f}  "
                      f"selectivity max {r['selectivity'].max():+.3f}  cells p<.05 {(r['p_perm'] < .05).mean():.2f} "
                      f"(FWER {(r['p_fwer'] < .05).mean():.2f})",
                      flush=True)
        print(f"  probe controls done in {time.perf_counter() - t0:.0f} s")

    def control_summary(r):
        sel = r["selectivity"]
        li, t = np.unravel_index(sel.argmax(), sel.shape)
        return {"acc_mean": float(r["acc"].mean()), "control_mean": float(r["control_acc"].mean()),
                "selectivity_max": float(sel.max()), "selectivity_max_coord": meta[int(li)]["short"],
                "selectivity_max_step": int(t), "frac_cells_p05": float((r["p_perm"] < .05).mean()),
                "frac_cells_p05_fwer": float((r["p_fwer"] < .05).mean()),
                "selectivity_by_coord": {meta[i]["short"]: float(sel[i].mean()) for i in range(len(meta))},
                "chance": r["chance"]}

    def block_profile(mp):
        return {m["short"]: {"median": float(mp["median"][i].mean()),
                             "lo": float(mp["lo"][i].mean()), "hi": float(mp["hi"][i].mean())}
                for i, m in enumerate(meta)}

    result = {
        "groups": [{"seed": g.seed, "scene": g.scene, "path": str(g.path), "n_clips": g.manifest["n_clips"],
                    "torch": g.manifest.get("torch"), "gpu": g.manifest.get("gpu")} for g in groups],
        "coords": meta, "n_steps": n_steps, "n_boot": args.n_boot,
        "units_per_seed": {int(k): int(v) for k, v in n_units.items()},
        "H1": h1, "H2": h2, "H3": h3,
        "profile_dyn": block_profile(maps["dyn"]), "profile_stat": block_profile(maps["stat"]),
        "static_energy_share": se.round(4).to_dict(),
        "cross_family": {f"{a}|{b}": v for (a, b), v in cf.round(4).to_dict().items()} if len(cf) else {},
        "robustness": rob,
        "probes": {k: {"best": float(v["acc"].max()), "mean": float(v["acc"].mean()), "chance": v["chance"],
                       "best_coord": meta[int(np.unravel_index(v["acc"].argmax(), v["acc"].shape)[0])]["short"],
                       "best_step": int(np.unravel_index(v["acc"].argmax(), v["acc"].shape)[1]),
                       "by_coord": {meta[i]["short"]: float(v["acc"][i].mean()) for i in range(len(meta))}}
                   for k, v in probes.items()},
        "probe_controls": {"n_perm": args.n_perm, "mlp_hidden": args.mlp_hidden, "mlp_alpha": args.mlp_alpha,
                           **{k: control_summary(v) for k, v in controls.items()}} if controls else {},
    }
    (args.out / "analysis.json").write_text(json.dumps(result, indent=2, default=float))
    np.savez_compressed(args.out / "maps.npz", coords=np.array(names),
                        **{f"S_{ch}_{k}": maps[ch][k] for ch in maps for k in ("median", "lo", "hi", "ratio_of_medians")},
                        **{f"probe_{k}": v["acc"] for k, v in probes.items()},
                        **{f"ctrl_{k}_{f}": v[f] for k, v in controls.items()
                           for f in ("acc", "control_acc", "selectivity", "p_perm", "p_fwer")})

    print("\nH1 (layer)    ", "SURVIVES" if h1["survives"] else "FALSIFIED",
          f"| argmax {h1['argmax_block']} | 95% set {h1['set95']}")
    print("H2 (timestep) ", "SURVIVES" if h2["survives"] else "FALSIFIED",
          f"| tau {h2['tau']} CI {h2['tau_ci']} | early {h2['mean_early']:.3f} late {h2['mean_late']:.3f}")
    print("H3 (static)   ", "SURVIVES" if h3["survives"] else "FALSIFIED",
          f"| f_stat {h3['f_stat']:.2f} n_flip {h3['n_flip']}")
    print(f"robustness {rob}")
    print(f"wrote {args.out / 'analysis.json'} and maps.npz")


if __name__ == "__main__":
    main()
