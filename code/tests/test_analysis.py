"""Analysis tests on a synthetic sweep with a planted answer: motion selectivity in one temporal
block early in the trajectory, content selectivity everywhere on the static channel, and verb
information in one coordinate's sketch. The tests check that the pipeline recovers exactly that."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

from vdprobe import analysis as A
from vdprobe.probing_set import ProbingSet

COORDS = ["down_blocks.0.motion_modules.1", "down_blocks.1.motion_modules.1",
          "up_blocks.1.motion_modules.2", "up_blocks.3.motion_modules.2", "up_blocks.1.attentions.2.transformer_blocks.0.attn2"]
MOTION_COORD, PROBE_COORD = 2, 1
T = 25


def _write_group(root, ps, seed, scene, rng):
    pairs = [p for p in ps.content_pairs(seed) + ps.motion_pairs(seed) + ps.cross_family_pairs(seed)
             if p.a.scene == scene]
    rows = []
    for i, p in enumerate(pairs):
        for li, c in enumerate(COORDS):
            for t in range(T):
                content = p.kind != "motion"
                d_stat = 0.30 if content else 0.10                     # static channel: content-selective
                d_dyn = 0.20
                if li == MOTION_COORD and t < 8 and not content:
                    d_dyn = 0.60                                         # planted: motion, early, one block
                rows.append({"pair": i, "coord": c, "step": t,
                             "d_static": d_stat + rng.normal(0, .02), "d_dynamic": d_dyn + rng.normal(0, .02),
                             "d_raw": d_stat, "l2_static": d_stat, "l2_dynamic": d_dyn,
                             "static_energy_a": .9, "static_energy_b": .9})
    out = root / f"s{seed}__{scene.replace(' ', '-')}"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "distances.csv", index=False)
    (out / "pairs.json").write_text(json.dumps(
        [{"pair": i, "kind": p.kind, "a": p.a.id, "b": p.b.id, "family_a": p.a.family, "family_b": p.b.family,
          "subject_a": p.a.subject, "subject_b": p.b.subject, "verb_a": p.a.verb, "verb_b": p.b.verb}
         for i, p in enumerate(pairs)]))
    verbs = {v: k for k, v in enumerate(ps.verbs)}
    sk = {}
    for c in ps.clips(seed):
        if c.scene != scene:
            continue
        x = rng.normal(0, 1, (len(COORDS), T, 2, 32)).astype(np.float32)
        x[PROBE_COORD, :, 0, verbs[c.verb]] += 4.0                      # planted verb signal on x_tilde
        sk[c.id] = torch.from_numpy(x).half()
    torch.save(sk, out / "sketches.pt")
    (out / "manifest.json").write_text(json.dumps({"seed": seed, "scene": scene, "coords": COORDS, "n_clips": 60}))


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    root = tmp_path_factory.mktemp("sweep")
    ps, rng = ProbingSet(), np.random.default_rng(0)
    for seed in (42, 1337):
        for scene in ps.scenes:
            _write_group(root, ps, seed, scene, rng)
    return root


def test_units_count(sweep):
    groups, long = A.load_all([sweep])
    units = A.selectivity_units(long)
    per = units[["seed", "scene", "unit"]].drop_duplicates().groupby("seed").size()
    assert (per == 78).all()          # (7 + 6) partners x 3 non-anchor verbs x 2 scenes


def test_planted_answer_is_recovered(sweep):
    groups, long = A.load_all([sweep])
    units = A.selectivity_units(long)
    dyn = A.selectivity_map(units, "dyn", COORDS, T, n_boot=200)
    stat = A.selectivity_map(units, "stat", COORDS, T, n_boot=200)
    assert dyn["median"][MOTION_COORD, :8].max() < -0.3       # motion-selective where planted
    assert abs(dyn["median"][0]).max() < 0.1                  # elsewhere neutral
    meta = [{"name": c, "kind": "temporal_module" if "motion" in c else "spatial_attn2",
             "block": ".".join(c.split(".")[:2]), "level": lvl, "short": c[:12]}
            for c, lvl in zip(COORDS, (0, 1, 2, 0, 2))]
    h1 = A.test_h1(dyn, meta)
    assert h1["argmax_block"] == "up_blocks.1" and h1["set95"] == ["up_blocks.1"]
    h2 = A.test_h2(dyn, COORDS, COORDS[MOTION_COORD])
    assert h2["tau"] == 8 and h2["survives"]
    h3 = A.test_h3(dyn, stat, meta)
    assert h3["f_stat"] == 1.0 and h3["n_flip"] >= 1 and h3["survives"]


def test_probe_finds_planted_coordinate(sweep):
    groups, _ = A.load_all([sweep])
    feats, clips, names = A.load_sketches(groups)
    res = A.probe_accuracy(feats, clips, "verb", 0, len(names), 2, max_pca=16)
    assert res["acc"][PROBE_COORD].min() > 0.7          # chance 0.25; PCA keeps 16 of 32 dims
    assert res["acc"][0].max() < 0.6                          # chance is 0.25
