"""Stage 6 · Four rankings of the 9 temporal blocks, and how far they agree (roadmap §5.5).

    selectivity   M(b) = −mean_t median S_dyn, from the probing sweep (analysis.json, H1)
    presence      scene-split linear-probe verb accuracy on x̃ of the block's coarse motion module
    use           patching: median ΔD over dev pairs of the motion-direction x̃ patch on that
                  block (window "all", then "ft"), from the Stage 5 dev run (summary.csv)
    learning      where gradient descent put the motion: relative ‖ΔW‖ of the full-coverage
                  motion LoRA, sqrt(Σ‖ΔW‖²) / sqrt(Σ‖W‖²) over the block's adapted projections

Prints the table and the pairwise Spearman correlations; adds the held-out validation losses
of the three LoRA variants (full / probe / anti). Writes <out>/layer_rankings.json.

    python scripts/layer_rankings.py --analysis results/analysis/analysis.json \
        --dev results/kaggle/stage5/out/job0/dev --lora results/kaggle/stage5/out --out results/rankings
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vdprobe.lora import delta_w_norms


def short(block: str) -> str:
    return block.replace("down_blocks.", "down").replace("up_blocks.", "up").replace("mid_block", "mid")


def block_of(module: str) -> str:
    m = re.match(r"(down_blocks\.\d+|up_blocks\.\d+|mid_block)", module)
    return m.group(1)


def base_weight_norms(modules: list[str]) -> dict[str, float]:
    """‖W‖_F of the frozen motion-adapter projections that the LoRA adapted."""
    import torch
    from diffusers import MotionAdapter

    from vdprobe.config import GenerationConfig
    from vdprobe.pipeline import ADAPTER_FILES, _snapshot

    cfg = GenerationConfig.from_yaml()
    ad = MotionAdapter.from_pretrained(_snapshot(cfg.adapter_id, ADAPTER_FILES), torch_dtype=torch.float32)
    # adapter paths: down_blocks.i.motion_modules.j... / mid_block.motion_modules.0... / up_blocks.i...
    params = dict(ad.named_parameters())
    out = {}
    for m in modules:
        w = params.get(m.replace("base_model.model.", "") + ".weight")
        if w is None:
            raise KeyError(f"no base weight for {m}")
        out[m] = float(w.norm())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--dev", type=Path, required=True, help="Stage 5 dev run (has summary.csv)")
    ap.add_argument("--lora", type=Path, required=True, help="directory searched for <variant>/motion_lora.safetensors")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    an = json.loads(args.analysis.read_text())
    blocks = list(an["H1"]["M_by_block"])
    tab = pd.DataFrame({"block": blocks})
    tab["selectivity"] = tab.block.map(an["H1"]["M_by_block"])
    probe = an["probes"]["verb_tilde"]["by_coord"]
    tab["presence"] = [next(v for k, v in probe.items() if k.startswith(short(b) + ".mm")) for b in blocks]

    summ = pd.read_csv(args.dev / "summary.csv")
    use = {}
    for b in blocks:
        for w in ("all", "ft"):
            r = summ[summ.arm == f"motion.dynamic.{b}.{w}"]
            use[(b, w)] = float(r.median_dD.iloc[0]) if len(r) else np.nan
    tab["use_all"] = [use[(b, "all")] for b in blocks]
    tab["use_ft"] = [use[(b, "ft")] for b in blocks]

    from safetensors import safe_open
    from safetensors.torch import load_file

    loras, val = {}, {}
    for f in sorted(args.lora.rglob("motion_lora.safetensors")):
        with safe_open(str(f), "pt") as h:
            meta = h.metadata()
        loras[meta["variant"]] = (f, meta)
        log = json.loads((f.parent / "train_log.json").read_text())
        val[meta["variant"]] = {"val_base": log["val"][0][1], "val_final": log["val"][-1][1],
                                "trainable_params": log["trainable_params"], "val_curve": log["val"]}
    if "full" in loras:
        f, meta = loras["full"]
        dw = delta_w_norms(load_file(str(f)), int(meta["alpha"]), int(meta["rank"]))
        dw = {k.replace("base_model.model.", ""): v for k, v in dw.items()}
        wn = base_weight_norms(list(dw))
        num, den = {}, {}
        for m, v in dw.items():
            b = block_of(m)
            num[b] = num.get(b, 0) + v ** 2
            den[b] = den.get(b, 0) + wn[m] ** 2
        tab["learning"] = [np.sqrt(num[b] / den[b]) if b in num else np.nan for b in blocks]

    cols = [c for c in ("selectivity", "presence", "use_all", "use_ft", "learning") if c in tab]
    rho = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            ok = tab[a].notna() & tab[b].notna()
            r, p = spearmanr(tab[a][ok], tab[b][ok])
            rho[f"{a}~{b}"] = {"rho": float(r), "p": float(p), "n": int(ok.sum())}
    pd.set_option("display.width", 200)
    print(tab.round(4).to_string(index=False))
    for k, v in rho.items():
        print(f"  spearman {k:<26} {v['rho']:+.2f}  (p {v['p']:.3f}, n {v['n']})")
    for k, v in val.items():
        print(f"  LoRA {k:<6} params {v['trainable_params'] / 1e6:6.2f} M   val loss {v['val_base']:.5f} -> {v['val_final']:.5f}")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "layer_rankings.json").write_text(json.dumps(
        {"table": tab.to_dict(orient="records"), "spearman": rho, "lora": val}, indent=1))
    print(f"wrote {args.out / 'layer_rankings.json'}")


if __name__ == "__main__":
    main()
