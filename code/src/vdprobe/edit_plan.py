"""Which edits are run: the frozen edit pairs (D-15, D-30) and the arm sets of Stages 5-6.

Every arm is an `edit.Arm` with an id of the form "<direction>.<name>", direction in
{"motion", "content"}, plus three unpatched reference arms:

    B_ref          B's own prompt (the clip whose content / motion the edit should take)
    motion.base    (B.subject, A.verb), no patch: the prompt-only baseline, motion direction
    content.base   (A.subject, B.verb), no patch: the prompt-only baseline, content direction

The source track is clip A itself.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .config import CONFIG_DIR, GenerationConfig
from .coords import Coordinate, block_order
from .edit import Arm, AttnSpec, EditSpec
from .patching import PatchSpec, window


@dataclass(frozen=True)
class EditPair:
    subject_a: str
    verb_a: str
    subject_b: str
    verb_b: str
    scene: str
    seed: int

    @property
    def id(self) -> str:
        s = lambda x: x.replace(" ", "-")
        return f"{self.seed}__{s(self.scene)}__{s(self.subject_a)}-{self.verb_a}__{s(self.subject_b)}-{self.verb_b}"

    def prompts(self, cfg: GenerationConfig) -> dict:
        p = lambda subj, verb: cfg.prompt(subj, verb, self.scene)
        return {"A": p(self.subject_a, self.verb_a), "B": p(self.subject_b, self.verb_b),
                "motion": p(self.subject_b, self.verb_a), "content": p(self.subject_a, self.verb_b)}


def load_pairs(split: str, path: str | Path = CONFIG_DIR / "edit_pairs.yaml") -> list[EditPair]:
    rows = yaml.safe_load(Path(path).read_text())[split]
    return [EditPair(*r[:5], int(r[5])) for r in rows]


# --- coordinates -----------------------------------------------------------------------

def motion_modules(coords: list[Coordinate], block: str | None = None) -> tuple[str, ...]:
    return tuple(c.name for c in coords if c.kind == "temporal_module" and (block is None or c.block == block))


def temporal_attention(coords: list[Coordinate], block: str | None = None) -> tuple[str, ...]:
    return tuple(c.name for c in coords if c.kind.startswith("temporal_attn") and (block is None or c.block == block))


def temporal_blocks(coords: list[Coordinate]) -> list[str]:
    return [b for b in block_order(coords) if motion_modules(coords, b)]


WINDOWS = {"ft": window("first_third"), "all": window("all"),
           "one0": window("one", center=0), "five1": window("five", center=1)}


def reference_arms(pair: EditPair, cfg: GenerationConfig) -> list[Arm]:
    p = pair.prompts(cfg)
    return [Arm("B_ref", p["B"], EditSpec(label="B, unpatched")),
            Arm("motion.base", p["motion"], EditSpec(label="prompt only")),
            Arm("content.base", p["content"], EditSpec(label="prompt only"))]


def _component(direction: str) -> str:
    # motion direction keeps A's motion: transplant A's x̃; content direction transplants A's x̄
    return "dynamic" if direction == "motion" else "static"


def coarse_arms(pair: EditPair, cfg: GenerationConfig, coords: list[Coordinate]) -> list[Arm]:
    """Stage 5 localisation (D-13 windows x D-08 blocks), both directions, on dev pairs.

    Per direction: the D-29 component patch on the motion-module outputs of one block, for
    every temporal block, windows {first third, all}; the same on all 21 motion modules for
    windows {one step, five steps, first third, all}. Motion direction also gets granularity 3
    (temporal attention injection, all modules) and granularity 1 (whole tensor) references.
    """
    p = pair.prompts(cfg)
    arms = reference_arms(pair, cfg)
    for d in ("motion", "content"):
        comp = _component(d)
        for b in temporal_blocks(coords):
            for w in ("ft", "all"):
                arms.append(Arm(f"{d}.{comp}.{b}.{w}", p[d],
                                EditSpec(PatchSpec(motion_modules(coords, b), WINDOWS[w], component=comp))))
        for w in ("one0", "five1", "ft", "all"):
            arms.append(Arm(f"{d}.{comp}.allmm.{w}", p[d],
                            EditSpec(PatchSpec(motion_modules(coords), WINDOWS[w], component=comp))))
        arms.append(Arm(f"{d}.tensor.allmm.ft", p[d], EditSpec(PatchSpec(motion_modules(coords), WINDOWS["ft"]))))
    for w in ("ft", "all"):
        arms.append(Arm(f"motion.attn.all.{w}", p["motion"], EditSpec(attn=AttnSpec(temporal_attention(coords), WINDOWS[w]))))
    return arms


def selected_arms(pair: EditPair, cfg: GenerationConfig, coords: list[Coordinate], sel: dict,
                  alphas=(0.25, 0.5, 0.75), loras: tuple[str, ...] = ()) -> list[Arm]:
    """Stage 6 evaluation arms from the Stage 5 selection.

    `sel[direction]` = {"blocks": [...] or "all", "window": key of WINDOWS}. Per direction: the
    selected component patch, its alpha dose-response (granularity 4), the whole-tensor patch at
    the same place (granularity 1); motion direction also gets attention injection at the same
    window (granularity 3, the D-17 training-free baseline), and one prompt-only arm per motion
    LoRA in `loras` (adapter names already attached to the UNet; the trained baseline, D-19).
    """
    p = pair.prompts(cfg)
    arms = reference_arms(pair, cfg)
    for d in ("motion", "content"):
        s = sel[d]
        mods = motion_modules(coords) if s["blocks"] == "all" else \
            tuple(m for b in s["blocks"] for m in motion_modules(coords, b))
        steps, comp = WINDOWS[s["window"]], _component(d)
        arms.append(Arm(f"{d}.sel", p[d], EditSpec(PatchSpec(mods, steps, component=comp), label="selected")))
        for a in alphas:
            arms.append(Arm(f"{d}.sel.a{a:g}", p[d], EditSpec(PatchSpec(mods, steps, alpha=a, component=comp))))
        arms.append(Arm(f"{d}.tensor", p[d], EditSpec(PatchSpec(mods, steps))))
    arms.append(Arm("motion.attn", p["motion"],
                    EditSpec(attn=AttnSpec(temporal_attention(coords), WINDOWS[sel["motion"]["window"]]))))
    for name in loras:
        arms.append(Arm(f"motion.{name}", p["motion"], EditSpec(label=f"prompt only + {name}"), lora=name))
    return arms


def lora_arms(pair: EditPair, cfg: GenerationConfig, coords: list[Coordinate],
              loras: tuple[str, ...]) -> list[Arm]:
    """Stage 6 add-on (D-34) on the test pairs: the motion-direction references, attention
    injection over all 25 steps (the best motion arm of the dev sweep, not in the selected set),
    and one prompt-only arm per motion LoRA (cut-free retrain, several seeds per variant).
    Scored in one pool with the selected-arm run of the same pairs (score_edits.py --with)."""
    p = pair.prompts(cfg)
    arms = [a for a in reference_arms(pair, cfg) if a.id != "content.base"]
    arms.append(Arm("motion.attn_all", p["motion"],
                    EditSpec(attn=AttnSpec(temporal_attention(coords), WINDOWS["all"]), label="attention, all steps")))
    for name in loras:
        arms.append(Arm(f"motion.{name}", p["motion"], EditSpec(label=f"prompt only + {name}"), lora=name))
    return arms


def adapter_name(meta: dict) -> str:
    """lora_<variant> for the first three LoRAs; lora_<variant>_s<seed> for the cut-free seeds."""
    return f"lora_{meta['variant']}" + (f"_s{meta['seed']}" if "seed" in meta else "")


def describe(arm: Arm) -> dict:
    s = arm.spec
    d = {"id": arm.id, "prompt": arm.prompt, "label": s.label}
    if arm.lora:
        d["lora"] = {"adapter": arm.lora, "scale": arm.lora_scale}
    if s.patch is not None:
        d["patch"] = {"coords": list(s.patch.coords), "steps": list(s.patch.steps), "alpha": s.patch.alpha,
                      "component": s.patch.component, "branches": s.patch.branches}
    if s.attn is not None:
        d["attn"] = {"modules": list(s.attn.modules), "steps": list(s.attn.steps), "alpha": s.attn.alpha}
    return d
