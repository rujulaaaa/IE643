"""Data and live runs for the novelty-phase tabs of the interface (scripts/app.py).

Saved results (no GPU):
    results/autoresearch/av_combo/<pair>/    viewpoint-matched action transfer: rows.csv, sigs.json, gifs/ (every arm)
    results/autoresearch/av_patch/           donor with a different camera (gifs of one pair in _gifs/)
    results/autoresearch/mv_matched_sheets/  MVDream, same view ring: donor / target / ideal / patched sheets
    results/autoresearch/mv_patch/<pair>/    MVDream, all view pairs: sheet.png, rows.csv
    results/action_view/                     sweep sensitivities (sweep_cells.csv) and the verdict tables
    results/autoresearch/mv_sweep/           MVDream sweep distances

Live run (GPU): one viewpoint-matched action transfer on CameraCtrl with the same code as
scripts/action_view_patch.py (donor, target, ideal and the patched target, one noise tensor).
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .demo import RESULTS, playable

AR = RESULTS / "autoresearch"
AV = RESULTS / "action_view"

CANDIDATES = {   # arm id in the saved runs -> label
    "mm_up_blocks.1": "Up-block 1 motion modules, steps 0-7 (best: identity kept)",
    "mm_up123@0-7": "Up-blocks 1-3, steps 0-7",
    "mm_up123_dyn@0-12": "Up-blocks 1-3, frame-varying part, steps 0-12",
    "mm_up123@0-12": "Up-blocks 1-3, steps 0-12 (identity leaks)",
    "mm_up123_dyn@0-24": "Up-blocks 1-3, frame-varying part, all steps (identity leaks)",
}
CAMERA_LABEL = {"off": "no camera motion", "pan-left": "pan left", "zoom-in": "zoom in"}
BLOCKS = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3", "mid_block",
          "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3"]


def _gif(d: Path, key: str) -> str | None:
    p = d / "gifs" / (key.replace("|", "_").replace("<-", "_from_") + ".gif")
    return playable(str(p)) if p.exists() else None


# ------------------------------------------------------------------ action transfer (viewpoint-matched)
def matched_pairs() -> list[str]:
    return sorted(p.parent.name for p in (AR / "av_combo").glob("*/rows.csv"))


def pair_label(name: str) -> str:
    cam, subj, verb = name.split("__")
    x, y = subj.split("_to_")
    c = CAMERA_LABEL.get(cam.split("_to_")[0], cam.split("_to_")[0])
    return f"{x.replace('-', ' ')} {verb} -> {y.replace('-', ' ')} running  ({c})"


def matched_pair(name: str, candidate: str) -> dict:
    d = AR / "av_combo" / name
    rows = pd.read_csv(d / "rows.csv").set_index("candidate")
    r = rows.loc[candidate]
    cam, subj, verb = name.split("__")
    x, y = (s.replace("-", " ") for s in subj.split("_to_"))
    clips = [(_gif(d, "S"), f"Donor: a {x} {verb}"),
             (_gif(d, "T"), f"Target: a {y} running"),
             (_gif(d, "I"), f"Reference: a {y} {verb} (generated directly)"),
             (_gif(d, f"T<-S|{candidate}"), f"Target patched: {CANDIDATES.get(candidate, candidate)}")]
    scores = pd.DataFrame([{
        "Motion transfer (0 = target's, 1 = donor's)": r["act_src"],
        "Identity leak (0 = target kept, 1 = donor)": r["leak_subj"],
        "X-CLIP verb probability, patched": r.get("xclip_P", np.nan),
        "X-CLIP verb probability, target / reference": f"{r.get('xclip_T', np.nan):.2f} / {r.get('xclip_I', np.nan):.2f}",
    }]).round(3)
    return {"clips": [c for c in clips if c[0]], "scores": scores,
            "caption": f"Camera: **{CAMERA_LABEL.get(cam.split('_to_')[0], cam)}** for donor and target (viewpoint-matched)."}


def matched_summary() -> pd.DataFrame | None:
    rows = [pd.read_csv(f) for f in (AR / "av_combo").glob("*/rows.csv")]
    if not rows:
        return None
    r = pd.concat(rows)
    out = []
    for c, g in r.groupby("candidate"):
        if c not in CANDIDATES:
            continue
        a = _boot(g["act_src"])
        s = _boot(g["leak_subj"])
        out.append({"Patch": CANDIDATES[c], "Motion transfer": f"{a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}]",
                    "Identity leak": f"{s[0]:.2f} [{s[1]:.2f}, {s[2]:.2f}]", "_k": a[0]})
    return pd.DataFrame(out).sort_values("_k", ascending=False).drop(columns="_k")


def _boot(x, n=5000, seed=643):
    x = np.asarray([v for v in x if np.isfinite(v)])
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------ donor with a different camera
MISMATCHED = "pan-left_to_zoom-in__horse_to_tiger__jumping"   # the D-45 pair whose clips were kept


def mismatched_example() -> dict | None:
    d = AR / "av_patch" / "_gifs"
    if not (d / "gifs").exists() or not (AR / "av_patch" / MISMATCHED / "rows.csv").exists():
        return None
    rows = pd.read_csv(AR / "av_patch" / MISMATCHED / "rows.csv").set_index("candidate")
    keys = [("S", "Donor: a horse jumping, camera pans left"), ("T", "Target: a tiger running, camera zooms in"),
            ("T<-S|mm_up_blocks.1", "Target patched, up-block 1"),
            ("T<-S|all_motion_modules_dynamic", "Target patched, all motion modules")]
    clips = [(_gif(d, k), lab) for k, lab in keys]
    sc = pd.DataFrame([{"Patch": lab, "Donor camera carried into target (0-1)": rows.loc[k.split("|")[1], "leak_cam"],
                        "Identity leak": rows.loc[k.split("|")[1], "leak_subj"]}
                       for k, lab in keys[2:] if k.split("|")[1] in rows.index]).round(2)
    return {"clips": [c for c in clips if c[0]], "scores": sc}


# ------------------------------------------------------------------ sweeps
def cc_cells() -> pd.DataFrame | None:
    f = AV / "sweep_cells.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    p = df["coord"].str.split(".")
    df["block"] = [x[0] if x[0] == "mid_block" else ".".join(x[:2]) for x in p]
    return df


def plot_block(block: str):
    """Action / camera / text-view sensitivity of one CameraCtrl block across denoising steps."""
    import matplotlib.pyplot as plt
    df = cc_cells()
    g = df[df["block"] == block]
    m = g.groupby("step")[["action", "camera", "view", "subject"]].mean()
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for k, col, lab in (("action", "#B4553C", "action change"), ("camera", "#2E6F95", "camera change"),
                        ("view", "#C9963F", "text-viewpoint change")):
        ax.plot(m.index, m[k] / m["subject"], color=col, lw=2.2, label=lab)
    ax.axhline(1, color="#8A939C", lw=1, ls="--")
    ax.text(m.index.max(), 1.03, "= subject change", ha="right", va="bottom", fontsize=9, color="#5E6873")
    ax.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    lo = min(0.25, float((m[["action", "camera", "view"]].div(m["subject"], axis=0)).min().min()) * 0.8)
    ax.set_ylim(bottom=lo)
    hi = ax.get_ylim()[1]
    ax.set_yticks([t for t in (0.25, 0.5, 1, 2, 5, 10, 20, 50, 100) if lo <= t <= hi])
    ax.set_xlabel("denoising step")
    ax.set_ylabel("response / response to a subject change")
    ax.set_title(f"CameraCtrl, {block.replace('_', ' ')}", fontsize=12, loc="left")
    ax.legend(frameon=False)
    ax.grid(True, color="#E3E7EB")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def figures() -> list[tuple[str, str]]:
    out = []
    for f, cap in (("fig_f2_cc_maps.png", "CameraCtrl: response to action, camera and text-view changes, per block and step"),
                   ("fig_f3_mv_maps.png", "MVDream: response to pose, camera and text-view changes, per block and step"),
                   ("fig_f6_tradeoff.png", "Viewpoint-matched donor: motion transfer vs identity leak, every patch"),
                   ("fig_f5_leak.png", "Donor with a different camera: share of its camera carried into the target")):
        if (AV / f).exists():
            out.append((str(AV / f), cap))
    return out


# ------------------------------------------------------------------ MVDream
def mv_sheets() -> list[tuple[str, str]]:
    out = []
    for f in sorted((AR / "mv_matched_sheets").glob("*.png")):
        _, subj, verb = f.stem.split("__")
        x, y = (s.replace("-", " ") for s in subj.split("_to_"))
        out.append((str(f), f"{x} {verb} -> {y} running, same view ring. Rows: donor, target, reference, "
                             f"patched encoder 0-3, patched middle block"))
    return out


def verdicts() -> pd.DataFrame:
    """One row per pre-registered hypothesis of the novelty phase (from the verdict files)."""
    rows = [("HV1", "CameraCtrl: action map differs from camera and text-view maps (rho < 0.5)", "fail",
             "rho(action, camera) 0.29; rho(action, text view) 0.62"),
            ("HV1-MV", "MVDream: action map differs from camera map", "pass", "rho 0.30 overall; top cells shared"),
            ("HV2", "Donor with another camera: action moves without camera or identity", "fail", "camera leak 0.13-0.8"),
            ("HV3", "MVDream, other view ring: pose moves without viewpoint", "fail", "view leak 0.40"),
            ("HV4/HV5", "Late steps only separate action from viewpoint", "fail", "camera leak >= 0.32"),
            ("HV6", "Viewpoint-matched donor: transfer CI lower bound > 0.2, identity kept", "fail by 0.02",
             "up-block 1: 0.26 [0.18, 0.34], leak 0.05"),
            ("HV7", "Combined up blocks beat one block", "fail", "0.23 vs 0.26")]
    return pd.DataFrame(rows, columns=["Rule", "Hypothesis", "Verdict", "Result"])


# ------------------------------------------------------------------ live run (GPU)
class LiveTransfer:
    """CameraCtrl loaded once; run() = one viewpoint-matched transfer at up-block 1 (and the D-49 variants)."""

    def __init__(self):
        self.cm = None

    def ensure(self):
        if self.cm is None:
            from .camera_models import default_config, load_camera_model
            from .config import CONFIG_DIR
            from .coords import enumerate_coordinates
            self.cm = load_camera_model("cc", default_config("cc"), None, camera_set=CONFIG_DIR / "av_set.yaml")
            self.coords = enumerate_coordinates(self.cm.loaded.unet)
        return self.cm

    def run(self, donor: str, target: str, verb: str, camera: str, seed: int, out: Path) -> dict:
        import importlib.util
        import types
        from .config import REPO_ROOT
        from .metrics import ClipScorer
        spec = importlib.util.spec_from_file_location("avp", REPO_ROOT / "scripts" / "action_view_patch.py")
        avp = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(avp)
        cm = self.ensure()
        if not hasattr(self, "clip"):
            self.clip = ClipScorer(device=cm.loaded.device)
        cands = [c for c in avp.combo_candidates(self.coords) if c["name"] in ("mm_up_blocks.1", "mm_up123@0-12")]
        args = types.SimpleNamespace(out=out, force=True, null=False, gifs=True, combo=True, late=False)
        avp.run_pair(cm, self.coords, cands, (donor, target, verb, camera, camera), seed, self.clip, args)
        name = next(p for p in out.iterdir() if p.is_dir() and (p / "rows.csv").exists())
        return {"dir": name, "rows": pd.read_csv(name / "rows.csv")}
