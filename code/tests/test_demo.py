"""The interface engine builds the same arms as the sweeps and scores them like score_edits.py."""

import numpy as np
import pytest
import torch

from vdprobe.config import GenerationConfig
from vdprobe.coords import enumerate_coordinates
from vdprobe.demo import EditRequest, build_arms, score_edit, vocabulary
from vdprobe.edit_plan import motion_modules, temporal_blocks


def _req(**kw):
    base = dict(subject_a="golden retriever", verb_a="running", subject_b="horse", verb_b="walking",
                scene="on a beach", seed=42)
    return EditRequest(**{**base, **kw})


def _sel(coords):
    b = temporal_blocks(coords)
    return {"motion": {"blocks": [b[-1]], "window": "ft"}, "content": {"blocks": [b[0]], "window": "ft"}}


def test_arms_follow_the_selection(tiny_unet):
    coords, cfg = enumerate_coordinates(tiny_unet), GenerationConfig.from_yaml()
    sel = _sel(coords)
    src, arms = build_arms(_req(), cfg, coords, sel)
    assert src == cfg.prompt("golden retriever", "running", "on a beach")
    assert [a.id for a in arms] == ["B_ref", "base", "edit"]
    assert arms[0].prompt == cfg.prompt("horse", "walking", "on a beach")
    # motion direction: B's subject with A's verb, x̃ transplanted at the selected block, first third
    assert arms[1].prompt == arms[2].prompt == cfg.prompt("horse", "running", "on a beach")
    p = arms[2].spec.patch
    assert p.component == "dynamic" and p.coords == motion_modules(coords, sel["motion"]["blocks"][0])
    assert p.steps == tuple(range(9)) and arms[1].spec.is_baseline

    _, arms = build_arms(_req(direction="content", with_reference=False), cfg, coords, sel)
    assert [a.id for a in arms] == ["base", "edit"]
    assert arms[1].prompt == cfg.prompt("golden retriever", "walking", "on a beach")
    assert arms[1].spec.patch.component == "static"


def test_other_methods(tiny_unet):
    coords, cfg = enumerate_coordinates(tiny_unet), GenerationConfig.from_yaml()
    sel = _sel(coords)
    _, arms = build_arms(_req(method="attn"), cfg, coords, sel)
    assert arms[-1].spec.attn is not None and arms[-1].spec.patch is None
    _, arms = build_arms(_req(method="attn_all"), cfg, coords, sel)
    assert len(arms[-1].spec.attn.steps) == 25 and arms[-1].spec.patch is None
    _, arms = build_arms(_req(method="lora_probe"), cfg, coords, sel)
    assert arms[-1].lora == "lora_probe" and arms[-1].spec.is_baseline
    _, arms = build_arms(_req(method="custom", block="all", window="all", alpha=0.5), cfg, coords, sel)
    p = arms[-1].spec.patch
    assert p.coords == motion_modules(coords) and len(p.steps) == 25 and p.alpha == 0.5
    _, arms = build_arms(_req(method="tensor"), cfg, coords, sel)
    assert arms[-1].spec.patch.component == "all"
    with pytest.raises(ValueError):
        build_arms(_req(direction="content", method="lora_full"), cfg, coords, sel)
    with pytest.raises(ValueError):
        build_arms(_req(method="custom", block="no_such_block"), cfg, coords, sel)
    # free-text subjects outside the frozen vocabulary are allowed in the interface
    src, _ = build_arms(_req(subject_a="red panda"), cfg, coords, sel)
    assert "red panda" in src and "red panda" not in vocabulary()["subjects"]


def test_scores_match_the_definition():
    rng = np.random.default_rng(0)
    emb = lambda: torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    flow = lambda: rng.normal(size=(3, 16, 16, 2)).astype(np.float32)
    A, B = {"emb": emb(), "flow": flow()}, {"emb": emb(), "flow": flow()}
    # a perfect motion-direction edit: A's flow, B's appearance
    feats = {"source": A, "B_ref": B, "base": {"emb": emb(), "flow": flow()},
             "edit": {"emb": B["emb"].clone(), "flow": A["flow"].copy()}}
    df = score_edit(feats, "motion", {"sc": [0.0, 1.0], "sm": [0.0, 1.0]}).set_index("clip")
    assert df.loc["edit", "sc_B"] == pytest.approx(1.0) and df.loc["edit", "sm_A"] == pytest.approx(1.0)
    assert df.loc["edit", "D"] > df.loc["base", "D"]
    assert np.isnan(df.loc["source", "D"]) and np.isnan(df.loc["B_ref", "D"])
    # without B's clip there is no D
    df = score_edit({k: v for k, v in feats.items() if k != "B_ref"}, "motion", None)
    assert "D" not in df and "sc_B" not in df


def test_shot_bounds_split_at_cuts():
    from vdprobe.inversion import shot_bounds

    rng = np.random.default_rng(0)
    base_a, base_b = rng.integers(0, 255, (48, 64, 3)), rng.integers(0, 255, (48, 64, 3))
    jitter = lambda b, k: np.clip(b + k, 0, 255).astype(np.uint8)
    frames = [jitter(base_a, k % 3) for k in range(10)] + [jitter(base_b, k % 3) for k in range(25)]
    assert shot_bounds(frames) == [(0, 10), (10, 35)]
    assert shot_bounds([jitter(base_a, k % 3) for k in range(20)]) == [(0, 20)]


def test_upload_start():
    import torch
    from vdprobe.demo import upload_start
    zi, g = torch.randn(1, 4, 2, 8, 8) * 0.9 + 0.1, torch.randn(1, 4, 2, 8, 8)
    assert torch.equal(upload_start(zi, g, "gaussian"), g.half())
    assert torch.equal(upload_start(zi, g, "inverted"), zi.half())
    b = upload_start(zi, g, "blend").float()
    assert abs(b.std().item() - 1) < 1e-2
    assert torch.corrcoef(torch.stack([b.flatten(), zi.flatten()]))[0, 1] > 0.5
    with pytest.raises(ValueError):
        upload_start(zi, g, "nope")
