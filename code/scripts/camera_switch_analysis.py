"""Analysis of scripts/camera_switch.py: how much camera motion each arm keeps (HC1 of D-40; HK1, HK2 of D-41).

Reads every <root>/**/rows.csv, bootstraps the mean of `retained` (and `retained_b` for swap arms)
over groups (scene x camera), and writes

    <out>/switch.json            per arm: mean, 95 % CI, n; verdicts
    <out>/camera_switch.png      retained against tau, on_until and on_from
    <out>/camera_blocks.png      retained per block: only / without (/ swap, CameraCtrl)

    python scripts/camera_switch_analysis.py results/kaggle/vdprobe-cam-switch --out results/camera/switch_lora
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

N_BOOT = 10_000
HC1_TAU, HC1_MIN = 3, 0.5
HK2_MAX_BLOCKS, HK2_MIN = 3, 0.5


def load_rows(roots) -> pd.DataFrame:
    files = sorted({f for r in roots for f in Path(r).rglob("rows.csv")})
    if not files:
        raise SystemExit(f"no rows.csv under {roots}")
    df = pd.concat([pd.read_csv(f, keep_default_na=True) for f in files], ignore_index=True)
    df["group"] = df.scene + "|" + df.camera
    return df


def boot_mean(values, rng, n_boot=N_BOOT) -> dict:
    """Mean and percentile CI over groups; NaN groups (camera had no measurable effect) are dropped."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    draws = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(draws, 0.025)), "hi": float(np.quantile(draws, 0.975)),
            "n": int(len(v))}


def summarise(df: pd.DataFrame, n_boot=N_BOOT, seed=643) -> dict:
    rng = np.random.default_rng(seed)
    arms = {}
    for arm, g in df.groupby("arm", sort=False):
        arms[arm] = {"kind": g.kind.iloc[0], "retained": boot_mean(g.retained, rng, n_boot),
                     "clip_sim_to_off": float(g.clip_sim_to_off.mean())}
        if g.retained_b.notna().any():
            arms[arm]["retained_b"] = boot_mean(g.retained_b, rng, n_boot)
    per_camera = {cam: {arm: float(g.retained.mean()) for arm, g in d.groupby("arm", sort=False)}
                  for cam, d in df.groupby("camera", sort=False)}
    out = {"model": str(df.model.iloc[0]), "n_groups": int(df.group.nunique()), "n_boot": n_boot,
           "weak_groups": sorted(df[(df.arm == "on_until_1") & df.retained.isna()].group.unique().tolist())
           if (df.arm == "on_until_1").any() else [],
           "arms": arms, "per_camera": per_camera, "verdicts": {}}

    key = f"on_until_{HC1_TAU}"
    if key in arms and arms[key]["retained"]["n"]:
        r = arms[key]["retained"]
        out["verdicts"]["when"] = {"rule": f"mean retained of {key} >= {HC1_MIN}", **r, "pass": r["mean"] >= HC1_MIN}
        # the first tau at which on_until keeps half, and the last tau at which on_from still keeps half
        taus = sorted(int(a.split("_")[-1]) for a in arms if a.startswith("on_until_"))
        half_until = next((t for t in taus if arms[f"on_until_{t}"]["retained"]["mean"] >= 0.5), None)
        half_from = [t for t in taus if f"on_from_{t}" in arms and arms[f"on_from_{t}"]["retained"]["mean"] >= 0.5]
        out["verdicts"]["when"].update(first_tau_until_half=half_until, last_tau_from_half=max(half_from) if half_from else None)

    swaps = {a: v["retained_b"] for a, v in arms.items() if v["kind"] == "swap" and a != "swap_all" and "retained_b" in v}
    if swaps:
        n_blocks = lambda a: len(a[len("swap_"):].split("+"))
        ok = {a: v for a, v in swaps.items() if n_blocks(a) <= HK2_MAX_BLOCKS and v["mean"] is not None}
        best = max(ok, key=lambda a: ok[a]["mean"]) if ok else None
        out["verdicts"]["where"] = {
            "rule": f"some swap of at most {HK2_MAX_BLOCKS} block groups has mean retained_b >= {HK2_MIN}",
            "best_arm": best, **(ok[best] if best else {}), "pass": bool(best and ok[best]["mean"] >= HK2_MIN),
            "single_blocks_ranked": sorted(((a[len("swap_"):], round(v["mean"], 4)) for a, v in ok.items()
                                            if n_blocks(a) == 1), key=lambda x: -x[1])}
    return out


def figures(summary: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = summary["arms"]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    for kind, label in (("on_until", "camera on for steps < tau"), ("on_from", "camera on for steps >= tau")):
        pts = sorted((int(a.split("_")[-1]), v["retained"]) for a, v in arms.items() if v["kind"] == kind and v["retained"]["n"])
        if pts:
            t = [p[0] for p in pts]
            m = np.array([p[1]["mean"] for p in pts])
            ax.errorbar(t, m, yerr=[m - [p[1]["lo"] for p in pts], [p[1]["hi"] for p in pts] - m], marker="o",
                        capsize=3, label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("tau (denoising step, of 25)")
    ax.set_ylabel("retained camera signature")
    ax.set_title(f"{summary['model']}: when is camera motion fixed?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "camera_switch.png", dpi=160)
    plt.close(fig)

    kinds = [(k, f) for k, f in (("only", "retained"), ("without", "retained"), ("swap", "retained_b"))
             if any(v["kind"] == k and "+" not in a and a != "swap_all" for a, v in arms.items())]
    if not kinds:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.4), squeeze=False)
    for ax, (kind, field) in zip(axes[0], kinds):
        items = [(a[len(kind) + 1:], v[field]) for a, v in arms.items()
                 if v["kind"] == kind and "+" not in a and a != "swap_all" and v.get(field, {}).get("n")]
        m = np.array([v["mean"] for _, v in items])
        ax.bar(range(len(items)), m, yerr=[m - [v["lo"] for _, v in items], [v["hi"] for _, v in items] - m], capsize=3)
        ax.set_xticks(range(len(items)), [b.replace("_blocks", "") for b, _ in items], rotation=45, ha="right", fontsize=8)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title({"only": "camera in one block only", "without": "camera removed from one block",
                      "swap": "camera B in one block (towards B)"}[kind], fontsize=9)
        ax.set_ylabel(field)
    fig.tight_layout()
    fig.savefig(out / "camera_blocks.png", dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roots", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("results/camera/switch"))
    p.add_argument("--n-boot", type=int, default=N_BOOT)
    args = p.parse_args()
    df = load_rows(args.roots)
    summary = summarise(df, args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "switch.json").write_text(json.dumps(summary, indent=2))
    df.to_csv(args.out / "rows.csv", index=False)
    figures(summary, args.out)
    for arm, v in summary["arms"].items():
        r = v["retained"]
        if r["n"]:
            b = v.get("retained_b")
            print(f"{arm:<28} retained {r['mean']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] n={r['n']}" +
                  (f"   towards B {b['mean']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]" if b and b["n"] else ""))
    print(json.dumps(summary["verdicts"], indent=2))


if __name__ == "__main__":
    main()
