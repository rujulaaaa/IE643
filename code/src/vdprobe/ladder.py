"""Experiment 2 content ladder: graded one-slot content swaps against a base clip.

For each base animal and verb, the base clip "a {coat} {colour} {breed} {verb} {scene}" is paired
with seven variants, each changing one slot:

    rung 0  null      scene "on a beach" -> "on the beach"     (text-perturbation floor)
    rung 1  colour    brown -> black
    rung 2  coat      smooth-coated -> shaggy
    rung 3  breed     breed -> another breed of the same family
    rung 4  species   breed -> another species of the same family
    rung 5  scene     on a beach -> in a snowy forest
    rung 6  lighting  on a beach -> on a beach at sunset

The verb never changes, so every pair is a content pair whose size is measured (CLIP), not named.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import yaml

from .config import CONFIG_DIR
from .probing_set import Clip, Pair

RUNGS = ("null", "colour", "coat", "breed", "species", "scene", "lighting")


class LadderSet:
    def __init__(self, path: str | Path = CONFIG_DIR / "ladder.yaml"):
        raw = yaml.safe_load(Path(path).read_text())
        self.raw = raw
        self.bases: list[dict] = raw["bases"]
        self.verbs: list[str] = raw["verbs"]
        self.seeds: list[int] = raw["seeds"]
        self.expected: dict[str, int] = raw.get("expected_counts", {})

    def _subject(self, base: dict, coat=None, colour=None, breed=None) -> str:
        r = self.raw
        return f"{coat or r['coat']} {colour or r['colour']} {breed or base['breed']}"

    def variants(self, base: dict, verb: str, seed: int) -> tuple[Clip, list[tuple[str, Clip]]]:
        r, fam = self.raw, base["family"]
        clip = lambda subject, scene: Clip(subject, verb, scene, seed, fam)
        b = clip(self._subject(base), r["scene"])
        out = [("null", clip(self._subject(base), r["scene_null"])),
               ("colour", clip(self._subject(base, colour=r["colour_swap"]), r["scene"])),
               ("coat", clip(self._subject(base, coat=r["coat_swap"]), r["scene"])),
               ("breed", clip(self._subject(base, breed=base["breed_swap"]), r["scene"])),
               ("species", clip(self._subject(base, breed=base["species_swap"]), r["scene"])),
               ("scene", clip(self._subject(base), r["scene_swap"])),
               ("lighting", clip(self._subject(base), r["lighting_swap"]))]
        return b, out

    def clips(self, seed: int) -> list[Clip]:
        out = []
        for base, verb in product(self.bases, self.verbs):
            b, vs = self.variants(base, verb, seed)
            out += [b] + [c for _, c in vs]
        return out

    def pairs(self, seed: int) -> list[Pair]:
        out = []
        for base, verb in product(self.bases, self.verbs):
            b, vs = self.variants(base, verb, seed)
            out += [Pair(rung, b, c) for rung, c in vs]
        return out

    def base_of(self, clip: Clip) -> str:
        """The base breed a ladder clip belongs to (its subject unit for sign-flip tests)."""
        for base in self.bases:
            if any(clip.subject.endswith(x) for x in (base["breed"], base["breed_swap"], base["species_swap"])):
                return base["breed"]
        raise KeyError(clip.subject)

    def check_counts(self) -> dict[str, int]:
        seed = self.seeds[0]
        counts = {"clips_per_seed": len(self.clips(seed)), "pairs_per_seed": len(self.pairs(seed))}
        if len({c.id for c in self.clips(seed)}) != counts["clips_per_seed"]:
            raise AssertionError("ladder clip ids collide")
        for k, v in self.expected.items():
            if counts.get(k) != v:
                raise AssertionError(f"ladder drifted from its config: {k}={counts.get(k)}, expected {v}")
        return counts
