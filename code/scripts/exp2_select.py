"""Experiment 2 · apply the pre-registered selection rules (D-35) to the primary cell's analysis.

    python scripts/exp2_select.py results/exp2/<primary cell> --out results/exp2/selection

Writes candidates.json (Phase 5 patching / erasure) and lora_target.json (Phase 6).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from vdprobe.coords import _block_and_level, classify

FIVE = 2   # half-width of the 5-step window


def window_at(t: int, n_steps: int) -> list[int]:
    return [s for s in range(t - FIVE, t + FIVE + 1) if 0 <= s < n_steps]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cell", type=Path, help="exp2_analysis.py output directory of the primary cell")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    an = json.loads((args.cell / "exp2_analysis.json").read_text())
    m = np.load(args.cell / "maps.npz")
    all_coords = [str(c) for c in m["all_coords"]]
    uniq = m["rsa_unique"]                       # (L, T, 2): [content, motion]
    n_steps = uniq.shape[1]
    refined = [c for c in all_coords if c.split(".")[0] == "up_blocks" and c.split(".")[1] in ("0", "1")]
    ri = [all_coords.index(c) for c in refined]
    li, t = np.unravel_index(np.argmax(uniq[ri, :, 1]), (len(ri), n_steps))
    motion_peak = (refined[li], int(t))
    ci, tc = np.unravel_index(np.argmax(uniq[:, :, 0]), uniq.shape[:2])
    content_peak = (all_coords[ci], int(tc))
    up1_mm = [c for c in all_coords if c.startswith("up_blocks.1.motion_modules.") and c.count(".") == 3]
    cands = [
        {"name": "exp1_region", "coords": ["up_blocks.0.motion_modules.2"], "steps": [0, 1, 2], "component": "dynamic",
         "why": "Experiment 1's motion site (pre-registered region)"},
        {"name": "rsa_motion_peak", "coords": [motion_peak[0]], "steps": window_at(motion_peak[1], n_steps),
         "component": "dynamic", "why": f"highest RSA unique motion share among refined coordinates, peak step {motion_peak[1]}"},
        {"name": "up1_motion_modules", "coords": up1_mm, "steps": list(range(8)), "component": "dynamic",
         "why": "the literature's motion site (MOFT, MotionClone, DMT)"},
        {"name": "rsa_content_peak", "coords": [content_peak[0]], "steps": window_at(content_peak[1], n_steps),
         "component": "static", "why": f"highest RSA unique content share, peak step {content_peak[1]}"},
    ]
    # LoRA target: the 3 width-1280 motion modules with the highest mean unique motion share over steps 0-14
    mm = [c for c in all_coords if classify(c) == "temporal_module"]
    wide = [c for c in mm if _block_and_level(c, 4)[1] >= 2]          # levels 2-3: 1280 channels in SD1.5
    score = {c: float(uniq[all_coords.index(c), :15, 1].mean()) for c in wide}
    target = sorted(score, key=score.get, reverse=True)[:3]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "candidates.json").write_text(json.dumps({"rule": "D-35", "cell": an["cell"], "candidates": cands}, indent=1))
    (args.out / "lora_target.json").write_text(json.dumps(
        {"rule": "D-35", "cell": an["cell"], "target": target, "scores": score}, indent=1))
    print(json.dumps(cands, indent=1))
    print("LoRA target:", target)


if __name__ == "__main__":
    main()
