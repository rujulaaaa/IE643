"""The content ladder: one slot changed per pair, counts as frozen, unique clip ids."""

from vdprobe.config import GenerationConfig
from vdprobe.ladder import RUNGS, LadderSet


def test_counts_and_ids():
    ls = LadderSet()
    assert ls.check_counts() == {"clips_per_seed": 128, "pairs_per_seed": 112}


def test_each_pair_changes_one_prompt_slot():
    ls, cfg = LadderSet(), GenerationConfig()
    for p in ls.pairs(42):
        assert p.kind in RUNGS and p.a.verb == p.b.verb and p.a.seed == p.b.seed
        wa, wb = p.a.prompt(cfg).split(), p.b.prompt(cfg).split()
        changed_subject = p.a.subject != p.b.subject
        changed_scene = p.a.scene != p.b.scene
        assert changed_subject != changed_scene                 # exactly one of the two slots
        if p.kind in ("colour", "coat"):
            assert sum(x != y for x, y in zip(wa, wb)) == 1 and len(wa) == len(wb)   # one word
        assert ls.base_of(p.a) == ls.base_of(p.b)
