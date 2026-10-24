"""Motion LoRAs on AnimateDiff's temporal path (Deliverable 5; D-19 as amended 2026-10-20).

Three adapters with the same rank, data and steps, differing only in where they sit:

    full    every temporal attention projection: motion_modules.*.attn{1,2}.to_{q,k,v,out.0}
            (the MotionDirector-style temporal LoRA, the trained baseline)
    probe   the same projections, only inside the block the Deliverable 2 map ranks most
            motion-selective: up_blocks.0, the whole H1 95% bootstrap set (D-32); ~20% of the
            parameters of `full`
    anti    control: the two blocks the map ranks LEAST motion-selective, down_blocks.2 and
            down_blocks.3 (lowest M(b)); ~27% of `full`, i.e. more parameters than `probe`

If `probe` comes closer to `full` than `anti` does on held-out clips, the selectivity map
predicts where motion can be learned, not only where it is represented.

Experiment 2 (docs/experiment2.md) adds LoRAs on an arbitrary set of motion modules (`modules`),
so the probe-selected set can be tested against parameter-matched random sets: a rank-r LoRA on
q/k/v/out of both temporal attentions of a motion module of width C costs 16 r C parameters.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

PROJ = r"transformer_blocks\.\d+\.attn[12]\.(to_q|to_k|to_v|to_out\.0)"
BLOCKS = {"probe": ("up_blocks.0",), "anti": ("down_blocks.2", "down_blocks.3")}
VARIANTS = ("full", "probe", "anti")


def target_regex(variant: str) -> str:
    if variant == "full":
        return rf".*motion_modules\.\d+\.{PROJ}"
    if variant in BLOCKS:
        blocks = "|".join(re.escape(b) for b in BLOCKS[variant])
        return rf"({blocks})\.motion_modules\.\d+\.{PROJ}"
    raise ValueError(variant)


def modules_regex(modules) -> str:
    """Projections inside exactly these motion modules, e.g. ["up_blocks.0.motion_modules.1"]."""
    alt = "|".join(re.escape(m) for m in modules)
    return rf"({alt})\.{PROJ}"


def lora_config(variant: str, rank: int = 32, alpha: int = 32, modules=None):
    from peft import LoraConfig

    target = modules_regex(modules) if modules else target_regex(variant)
    return LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=target)


def motion_module_widths(unet) -> dict[str, int]:
    """{motion module path: channel width C}, in module order."""
    from diffusers.models.unets.unet_motion_model import AnimateDiffTransformer3D

    return {n: m.proj_in.in_features for n, m in unet.named_modules() if isinstance(m, AnimateDiffTransformer3D)}


def lora_param_count(widths, rank: int) -> int:
    """16 r sum(C): q/k/v/out (4 C x C matrices, r(C + C) each) in both temporal attentions."""
    return 16 * rank * sum(widths)


def matched_random_subsets(widths: dict[str, int], target: list[str], n: int, seed: int = 643) -> list[list[str]]:
    """n distinct random module sets with exactly the target's multiset of widths, disjoint from
    the target (so equal rank gives an equal parameter count): the localisation null."""
    import random
    from collections import Counter

    rng = random.Random(seed)
    need = Counter(widths[m] for m in target)
    pools = {c: [m for m, w in widths.items() if w == c and m not in target] for c in need}
    seen, out = set(), []
    for _ in range(100 * n):
        pick = tuple(sorted(m for c, k in need.items() for m in rng.sample(pools[c], k)))
        if pick not in seen:
            seen.add(pick)
            out.append(list(pick))
        if len(out) == n:
            return out
    raise ValueError(f"only {len(out)} distinct matched subsets exist; asked for {n}")


def add_motion_lora(unet, variant: str, rank: int = 32, alpha: int = 32, name: str = "motion",
                    modules=None) -> list:
    """Attach a fresh adapter (a named `variant`, or an explicit list of motion `modules`);
    returns its trainable parameters (fp32)."""
    unet.add_adapter(lora_config(variant, rank, alpha, modules), adapter_name=name)
    params = []
    for n, p in unet.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()
            p.requires_grad_(True)
            params.append(p)
    return params


def save_motion_lora(unet, path: Path, meta: dict, name: str = "motion") -> None:
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    sd = {k: v.detach().half().contiguous() for k, v in get_peft_model_state_dict(unet, adapter_name=name).items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(path), metadata={k: str(v) for k, v in meta.items()})


def load_motion_lora(unet, path: Path, name: str = "motion", scale: float = 1.0) -> None:
    """Attach a saved adapter (variant, rank and alpha are read from the file's metadata)."""
    from peft.utils import set_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), "pt") as f:
        meta = f.metadata()
    modules = meta["modules"].split(",") if meta.get("modules") else None
    cfg = lora_config(meta["variant"], int(meta["rank"]), int(meta["alpha"]), modules)
    unet.add_adapter(cfg, adapter_name=name)
    sd = {k: v.to(unet.dtype) for k, v in load_file(str(path)).items()}
    res = set_peft_model_state_dict(unet, sd, adapter_name=name)
    if getattr(res, "unexpected_keys", None):
        raise RuntimeError(f"unexpected LoRA keys: {res.unexpected_keys[:5]}")
    unet.set_adapters([name], weights=[scale])


def remove_lora(unet, name: str = "motion") -> None:
    unet.delete_adapters(name)


def delta_w_norms(state_dict: dict, alpha: int, rank: int) -> dict[str, float]:
    """‖ΔW‖_F = ‖(alpha/r) · B A‖_F per adapted projection, keyed by the projection's module path."""
    out = {}
    for k, a in state_dict.items():
        if ".lora_A." not in k:
            continue
        b = state_dict[k.replace(".lora_A.", ".lora_B.")]
        mod = k.split(".lora_A.")[0]
        out[mod] = float(torch.linalg.matrix_norm((alpha / rank) * (b.float() @ a.float())))
    return out


def n_params(state_dict: dict) -> int:
    return sum(v.numel() for v in state_dict.values())
