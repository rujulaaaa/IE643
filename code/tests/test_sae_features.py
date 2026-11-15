import numpy as np
import torch

from vdprobe import sae_features as SF


def test_auroc_matches_definition_with_ties():
    x = np.array([[0.0], [1.0], [1.0], [2.0]])
    pos = np.array([False, False, True, True])
    assert SF.auroc(x, pos)[0] == (1 + 1 + 0.5 + 1) / 4        # pairs: (1,0) (1,1 tie) (2,0) (2,1)
    assert np.isnan(SF.auroc(x, np.zeros(4, bool))[0])


def test_selectivity_is_two_sided_and_names_the_class():
    labels = np.array(["a"] * 4 + ["b"] * 4 + ["c"] * 4)
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3))
    x[:, 0] = (labels == "b") * 5.0 + rng.normal(size=12) * 0.01        # on for b
    x[:, 1] = (labels == "c") * -5.0 + rng.normal(size=12) * 0.01       # off for c
    best, arg = SF.selectivity(x, labels)
    assert best[0] == 1.0 and arg[0] == 1 and best[1] == 1.0 and arg[1] == 2 and best[2] < 1.0


def test_label_directions_counts_camera_only():
    cams = np.repeat(["none", "pan", "zoom"], 8)
    scenes = np.tile(np.arange(8), 3)
    x = np.zeros((24, 3))
    x[:, 0] = (cams == "pan") + 0.001 * np.random.default_rng(1).permutation(24)   # camera only
    x[:, 1] = (scenes == 3) * 1.0                                 # scene only
    x[:, 2] = (cams == "zoom") * 1.0 + (scenes == 2) * 3.0        # both
    out = SF.label_directions(x, cams, scenes)
    assert out["camera_only"].tolist() == [True, False, False] and out["n_camera_only"] == 1
    assert out["classes"][out["camera_argmax"][0]] == "pan"
    assert out["scene_eta2"][1] == 1.0 and out["camera_eta2"][0] > 0.99 and out["scene_eta2"][0] < 0.01


def test_noise_rarely_counts_as_camera_only():
    cams, scenes = np.repeat(np.arange(9), 8), np.tile(np.arange(8), 9)          # the real design: 72 clips
    out = SF.label_directions(np.random.default_rng(0).normal(size=(72, 2000)), cams, scenes)
    assert out["n_camera_only"] <= 2                       # camera AUROC >= 0.9 is rare under noise
    assert (out["scene_auroc"] > 0.6).mean() > 0.5         # why the scene side is not an AUROC rule


def test_clip_means_and_ids():
    tok = {"42__a-lake__pan-left": {0: torch.ones(4, 3), 1: torch.zeros(4, 3)},
           "42__a-lake__none": {0: torch.full((4, 3), 2.0), 1: torch.full((4, 3), 2.0)}}
    ids, m = SF.clip_means(tok)
    assert ids == sorted(tok) and m.shape == (2, 3) and np.allclose(m[ids.index("42__a-lake__pan-left")], 0.5)
    _, m0 = SF.clip_means(tok, steps=(0,))
    assert np.allclose(m0[ids.index("42__a-lake__pan-left")], 1.0)
    _, e = SF.clip_means(tok, encode=lambda x: x * 2, batch=3)
    assert np.allclose(e, m * 2)
    assert SF.parse_clip_id(ids[0]) == {"clip": ids[0], "seed": 42, "scene": "a-lake", "camera": "none"}
