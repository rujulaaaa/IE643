"""The frozen protocol: probing-set counts, distances, selectivity, storage arithmetic."""

import json
from pathlib import Path

import torch

from vdprobe.config import GenerationConfig
from vdprobe.probing_set import ProbingSet
from vdprobe.summaries import paired_distances, selectivity, split_static_dynamic

ROOT = Path(__file__).resolve().parents[1]


def test_probing_set_counts_match_roadmap():
    ps = ProbingSet()
    assert ps.check_counts() == {"clips_per_seed": 120, "content_pairs": 104,
                                 "motion_pairs": 90, "cross_family_pairs": 56}
    assert len(ps.all_clips()) == 360
    for pair in ps.content_pairs(42):
        assert pair.a.family == pair.b.family and pair.a.verb == pair.b.verb and pair.a.scene == pair.b.scene
    for pair in ps.motion_pairs(42):
        assert pair.a.subject == pair.b.subject and pair.a.verb == "running" != pair.b.verb
    assert len({c.id for c in ps.all_clips()}) == 360


def test_prompt_template_is_frozen():
    cfg = GenerationConfig.from_yaml()
    assert cfg.prompt("golden retriever", "running", "on a beach") == \
        "a golden retriever running on a beach, high quality, detailed"
    assert (cfg.frames, cfg.height, cfg.width, cfg.steps, cfg.guidance_scale) == (16, 512, 512, 25, 7.5)


def test_static_dynamic_split_is_orthogonal():
    x = torch.randn(16, 8, 10)
    x_bar, x_til = split_static_dynamic(x)
    assert torch.allclose(x_til.mean(0), torch.zeros_like(x_bar), atol=1e-6)
    energy = 16 * x_bar.pow(2).sum() + x_til.pow(2).sum()
    assert torch.allclose(energy, x.pow(2).sum(), rtol=1e-5)


def test_static_content_dominates_raw_distance():
    """H3 in miniature: a big shared static component makes raw distance blind to motion."""
    static = torch.randn(1, 8, 50) * 10
    motion_a, motion_b = torch.randn(16, 8, 50), torch.randn(16, 8, 50)
    d = paired_distances(static + motion_a, static + motion_b)
    assert d["d_raw"] < 0.05 < 0.5 < d["d_dynamic"]
    assert d["d_static"] < 0.05 and d["static_energy_a"] > 0.9


def test_selectivity_sign():
    assert selectivity(0.4, 0.1) > 0 > selectivity(0.1, 0.4)
    assert selectivity(0.0, 0.0) == 0.0


def test_e3_storage_arithmetic_reproduces_record():
    """The 87.9 GiB -> 44.6 MiB claim, recomputed from AnimateDiff's architecture alone."""
    # (S, C) per coordinate for SD1.5 at 512x512 (latent 64x64), per block
    spatial = [(4096, 320)] * 2 + [(1024, 640)] * 2 + [(256, 1280)] * 2 + [(64, 1280)] \
        + [(256, 1280)] * 3 + [(1024, 640)] * 3 + [(4096, 320)] * 3
    temporal = [(4096, 320)] * 2 + [(1024, 640)] * 2 + [(256, 1280)] * 2 + [(64, 1280)] * 2 \
        + [(64, 1280)] + [(64, 1280)] * 3 + [(256, 1280)] * 3 + [(1024, 640)] * 3 + [(4096, 320)] * 3
    assert (len(spatial), len(temporal)) == (16, 21)
    T, F, B, G = 25, 16, 2, 16
    full = T * B * F * 2 * (2 * sum(s * c for s, c in spatial) + 3 * sum(s * c for s, c in temporal))
    stats = T * 4 * (2 * sum(2 * c + F for _, c in spatial) + 3 * sum(2 * c + F for _, c in temporal))
    attn = T * 4 * F * G * G * (2 * len(spatial) + 2 * len(temporal))
    rec = json.loads((ROOT / "results/kaggle_t4/E3_result.json").read_text())
    assert round(full / 2 ** 30, 3) == rec["full_dump_gb_per_clip"]
    assert round(stats / 2 ** 20, 2) == rec["summary_split_mb"]["channel_stats"]
    assert round(attn / 2 ** 20, 2) == rec["summary_split_mb"]["attention_movies"]
    assert round(full / (stats + attn), 1) == rec["compression_ratio"]
