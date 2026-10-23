"""The controlled probing set: frozen vocabulary, clips, and the anchor pair design (D-03..D-06).

A pair is two clips that share a seed, a noise tensor, a sampler and every prompt
token but one. Content pairs swap the subject, motion pairs swap the verb.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import yaml

from .config import CONFIG_DIR, GenerationConfig


@dataclass(frozen=True)
class Clip:
    subject: str
    verb: str
    scene: str
    seed: int
    family: str = ""

    @property
    def id(self) -> str:
        slug = lambda s: re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
        return f"s{self.seed}__{slug(self.subject)}__{slug(self.verb)}__{slug(self.scene)}"

    def prompt(self, cfg: GenerationConfig) -> str:
        return cfg.prompt(self.subject, self.verb, self.scene)

    @property
    def phrases(self) -> dict[str, str]:
        """Prompt phrases whose cross-attention maps the harness can record."""
        return {"subject": self.subject, "verb": self.verb, "scene": self.scene}


@dataclass(frozen=True)
class Pair:
    kind: str  # "content" | "motion" | "cross_family"
    a: Clip    # the anchor clip
    b: Clip    # the swapped clip

    @property
    def swapped(self) -> tuple[str, str]:
        if self.kind == "motion":
            return self.a.verb, self.b.verb
        return self.a.subject, self.b.subject


class ProbingSet:
    def __init__(self, path: str | Path = CONFIG_DIR / "probing_set.yaml"):
        raw = yaml.safe_load(Path(path).read_text())
        self.raw = raw
        self.families: dict[str, dict] = raw["families"]
        self.verbs: list[str] = raw["verbs"]
        self.anchor_verb: str = raw["anchor_verb"]
        self.scenes: list[str] = raw["scenes"]
        self.seeds: list[int] = raw["seeds"]
        self.expected: dict[str, int] = raw.get("expected_counts", {})
        self._family_of = {s: f for f, spec in self.families.items() for s in spec["subjects"]}

    @property
    def subjects(self) -> list[str]:
        return [s for spec in self.families.values() for s in spec["subjects"]]

    def clip(self, subject: str, verb: str, scene: str, seed: int) -> Clip:
        return Clip(subject, verb, scene, seed, self._family_of.get(subject, ""))

    def clips(self, seed: int) -> list[Clip]:
        return [self.clip(s, v, sc, seed) for s, v, sc in product(self.subjects, self.verbs, self.scenes)]

    def content_pairs(self, seed: int) -> list[Pair]:
        """Within-family subject swaps against the family anchor (D-05)."""
        out = []
        for spec in self.families.values():
            anchor = spec["anchor"]
            for partner, verb, scene in product(spec["subjects"], self.verbs, self.scenes):
                if partner != anchor:
                    out.append(Pair("content", self.clip(anchor, verb, scene, seed),
                                    self.clip(partner, verb, scene, seed)))
        return out

    def motion_pairs(self, seed: int) -> list[Pair]:
        """Verb swaps against the anchor verb, for every subject."""
        return [Pair("motion", self.clip(s, self.anchor_verb, sc, seed), self.clip(s, v, sc, seed))
                for s, v, sc in product(self.subjects, self.verbs, self.scenes) if v != self.anchor_verb]

    def cross_family_pairs(self, seed: int) -> list[Pair]:
        """Quadruped anchor vs every human subject: the labelled hard subset, never pooled."""
        q_anchor = self.families["Q"]["anchor"]
        return [Pair("cross_family", self.clip(q_anchor, v, sc, seed), self.clip(h, v, sc, seed))
                for h, v, sc in product(self.families["H"]["subjects"], self.verbs, self.scenes)]

    def all_clips(self) -> list[Clip]:
        return [c for seed in self.seeds for c in self.clips(seed)]

    def check_counts(self) -> dict[str, int]:
        seed = self.seeds[0]
        counts = {
            "clips_per_seed": len(self.clips(seed)),
            "content_pairs": len(self.content_pairs(seed)),
            "motion_pairs": len(self.motion_pairs(seed)),
            "cross_family_pairs": len(self.cross_family_pairs(seed)),
        }
        for k, v in self.expected.items():
            if counts.get(k) != v:
                raise AssertionError(f"probing set drifted from the frozen protocol: {k}={counts.get(k)}, expected {v}")
        return counts
