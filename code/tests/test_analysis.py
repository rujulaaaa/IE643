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


# ---------------------------------------------------------------------------
# Probe controls (D-34)
# ---------------------------------------------------------------------------

def test_control_labels_keep_class_balance_and_type_structure(sweep):
    groups, _ = A.load_all([sweep])
    _, clips, _ = A.load_sketches(groups)
    cs = clips[clips.seed == 42].reset_index(drop=True)
    ctrl = A.control_labels(cs, "verb", np.random.default_rng(0))
    assert sorted(ctrl) == sorted(cs.verb)                          # same class frequencies
    per_type = pd.Series(ctrl).groupby([cs.subject, cs.verb]).nunique()
    assert (per_type == 1).all()                                    # one label per (subject, verb) type
    assert (ctrl != cs.verb.to_numpy()).mean() > 0.5                # not the real labels


def test_planted_verb_signal_is_selective(sweep):
    groups, _ = A.load_all([sweep])
    feats, clips, names = A.load_sketches(groups)
    for probe in A.PROBES:
        res = A.probe_scores(feats, clips, "verb", 0, len(names), 2, probe=probe, n_perm=19, max_pca=16, n_jobs=4)
        assert res["acc"][PROBE_COORD].min() > 0.7
        assert res["control_acc"][PROBE_COORD].max() < 0.45            # a verb code cannot fit random type labels
        assert res["selectivity"][PROBE_COORD].min() > 0.3
        assert res["p_perm"][PROBE_COORD].max() == pytest.approx(0.05)  # (1 + 0) / (1 + 19): beats every null draw
        assert res["p_perm"][0].min() > 0.05                           # noise coordinate: not significant
        assert res["p_fwer"][PROBE_COORD].max() == pytest.approx(0.05)  # survives the max-over-map null


def test_type_identity_code_is_not_selective():
    """A layer that only identifies the prompt (a random code per (subject, verb) type, the same in
    both scenes) lets a probe fit any type -> label map: high accuracy on the real task AND on the
    control task, so selectivity is ~0. This is the case the control task exists to expose."""
    rng = np.random.default_rng(1)
    ps = ProbingSet()
    rows, feats, code = [], {}, {}
    for seed in (42, 1337):
        for c in ps.clips(seed):
            key = (c.subject, c.verb)
            code.setdefault(key, rng.normal(0, 1, 64))
            x = np.zeros((1, 1, 2, 64), np.float32)
            x[0, 0, 0] = code[key] * 3 + rng.normal(0, 1, 64)
            feats[c.id] = x
            rows.append({"clip": c.id, "seed": seed, "scene": c.scene, "subject": c.subject, "verb": c.verb,
                         "family": c.family})
    clips = pd.DataFrame(rows)
    res = A.probe_scores(feats, clips, "verb", 0, 1, 1, probe="linear")
    assert res["acc"][0, 0] > 0.8 and res["control_acc"][0, 0] > 0.8
    assert abs(res["selectivity"][0, 0]) < 0.15


def test_probe_accuracy_unchanged_by_refactor(sweep):
    groups, _ = A.load_all([sweep])
    feats, clips, names = A.load_sketches(groups)
    old = A.probe_accuracy(feats, clips, "verb", 0, len(names), 2, max_pca=16)
    new = A.probe_scores(feats, clips, "verb", 0, len(names), 2, probe="linear", control=False, max_pca=16)
    np.testing.assert_array_equal(old["acc"], new["acc"])


def test_restricted_permutation_keeps_subject_structure(sweep):
    groups, _ = A.load_all([sweep])
    _, clips, _ = A.load_sketches(groups)
    cs = clips[clips.seed == 42].reset_index(drop=True)
    rng = np.random.default_rng(0)
    pv = A.permuted_labels(cs, "verb", rng)
    for _, g in cs.assign(pv=pv).groupby("subject"):
        assert sorted(g.pv) == sorted(g.verb)                       # each subject keeps its 4 verbs
    same_type = cs.assign(pv=pv).groupby(["subject", "verb"]).pv.nunique()
    assert (same_type == 1).all()                                   # both scenes relabelled alike
    pf = A.permuted_labels(cs, "family", rng)
    assert (pd.Series(pf).groupby(cs.subject).nunique() == 1).all()  # family stays a subject property
    assert sorted(pf) == sorted(cs.family)


# ---------------------------------------------------------------------------
# Layer comparison (Experiment 2)
# ---------------------------------------------------------------------------

def test_debiased_cka_is_zero_for_independent_data_and_one_for_rotations():
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(60, 512)), rng.normal(size=(60, 512))
    assert abs(A.debiased_cka(X, Y)) < 0.05
    Q, _ = np.linalg.qr(rng.normal(size=(512, 512)))
    assert A.debiased_cka(X, 3 * X @ Q) == pytest.approx(1.0, abs=1e-6)   # invariant to rotation + scale
    Z = X[:, :20] @ rng.normal(size=(20, 300)) + 0.1 * rng.normal(size=(60, 300))
    assert A.debiased_cka(X[:, :20], Z) > 0.8                              # shared low-dim structure


def test_rsa_commonality_separates_correlated_predictors():
    """Layer distances driven by content only, with a motion RDM that is correlated with content
    (as flow is with silhouette): the unique motion share must be ~0 and not significant."""
    rng = np.random.default_rng(0)
    N = 40
    pts_c = rng.normal(size=(N, 3))
    pts_m = 0.7 * pts_c + 0.7 * rng.normal(size=(N, 3))
    dist = lambda P: np.linalg.norm(P[:, None] - P[None], axis=-1)
    Dc, Dm = dist(pts_c), dist(pts_m)
    Dl = 2 * Dc + 0.3 * dist(rng.normal(size=(N, 3)))
    res = A.rsa_permutation(Dl, {"content": Dc, "motion": Dm}, n_perm=99, rng=rng)
    assert res["r2_single"]["motion"] > 0.1                   # alone, motion looks like it explains the layer
    assert res["unique"]["content"] > 0.3 and res["unique"]["motion"] < 0.02
    assert res["p_unique"]["content"] <= 0.02 and res["p_unique"]["motion"] > 0.05
