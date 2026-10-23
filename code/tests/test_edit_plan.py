"""The frozen edit pairs obey the D-15 / D-30 design, and the arm builders produce valid specs."""

from collections import Counter

import yaml

from vdprobe.config import CONFIG_DIR, GenerationConfig
from vdprobe.coords import enumerate_coordinates
from vdprobe.edit_plan import coarse_arms, load_pairs, selected_arms, temporal_blocks


def _family():
    fam = yaml.safe_load((CONFIG_DIR / "probing_set.yaml").read_text())["families"]
    return {s: f for f, v in fam.items() for s in v["subjects"]}


def test_pairs_follow_the_design():
    fam = _family()
    dev, test = load_pairs("dev"), load_pairs("test")
    assert len(test) == 12 and len(dev) == 4
    for p in dev + test:
        assert fam[p.subject_a] == fam[p.subject_b]
        assert p.subject_a != p.subject_b and p.verb_a != p.verb_b
    assert Counter(fam[p.subject_a] for p in test) == {"Q": 6, "H": 6}
    verbs = {"running", "walking", "jumping", "spinning"}
    assert {p.verb_a for p in test} == verbs and {p.verb_b for p in test} == verbs
    assert not {p.id for p in dev} & {p.id for p in test}


def test_arm_builders(tiny_unet):
    coords = enumerate_coordinates(tiny_unet)
    cfg = GenerationConfig.from_yaml()
    pair = load_pairs("test")[0]
    arms = coarse_arms(pair, cfg, coords)
    ids = [a.id for a in arms]
    assert len(ids) == len(set(ids))
    nb = len(temporal_blocks(coords))
    assert len(arms) == 3 + 2 * (2 * nb + 4 + 1) + 2
    p = pair.prompts(cfg)
    for a in arms:
        d = a.id.split(".")[0]
        assert a.prompt == (p[d] if d in ("motion", "content") else p["B"])
        if a.spec.patch is not None and a.spec.patch.component != "all":
            assert a.spec.patch.component == ("dynamic" if d == "motion" else "static")
    sel = {"motion": {"blocks": [temporal_blocks(coords)[0]], "window": "ft"},
           "content": {"blocks": "all", "window": "all"}}
    arms = selected_arms(pair, cfg, coords, sel)
    assert {"motion.sel", "content.sel", "motion.attn", "motion.tensor", "content.sel.a0.5"} <= {a.id for a in arms}
