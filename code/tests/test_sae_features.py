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


def test_sae_train_script_end_to_end(tmp_path, monkeypatch):
    """Planted tokens: channel 0 carries the camera, channel 1 the scene. The script must run and find it."""
    import importlib.util, json, sys
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("sae_train", Path(__file__).resolve().parents[1] / "scripts" / "sae_train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod, "VARIANTS", {"default": (48, 4)})
    cams, scenes, frames, n, C = ["none", "pan-left", "zoom-in"], ["lake", "castle", "street", "forest"], 4, 8, 16
    g = torch.Generator().manual_seed(0)
    for seed in (42, 1337):
        tok = {}
        for ci, cam in enumerate(cams):
            for si, sc in enumerate(scenes):
                base = torch.zeros(C)
                base[0], base[1] = 3.0 * ci, 3.0 * si
                tok[f"{seed}__{sc}__{cam}"] = {s: (base + 0.3 * torch.randn(frames * n, C, generator=g)).half() for s in (0, 1)}
        d = tmp_path / "in" / f"s{seed}__camera"
        d.mkdir(parents=True)
        torch.save(tok, d / "tokens.pt")
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["sae_train.py", str(tmp_path / "in"), "--out", str(out), "--variants", "default",
                                      "--epochs", "40", "--batch", "64", "--lr", "3e-3", "--frames", str(frames), "--top", "3"])
    mod.main()
    s = json.loads((out / "summary.json").read_text())
    assert s["variants"]["default"]["explained_variance_eval"] > 0.5
    assert s["n_camera_only"]["channels"] >= 1                 # the planted camera channel
    import pandas as pd
    df = pd.read_csv(out / "features.csv")
    assert len(df) == 48 and {"step_0", "step_1", "frame_3", "camera_eta2"} <= set(df.columns)
    dirs = torch.load(out / "directions.pt", weights_only=False)
    assert set(dirs["mean_diff"]) == {"pan-left", "zoom-in"} and dirs["mean_diff"]["zoom-in"][0] > 4.0
    assert dirs["features"]["zoom-in"]["decoder"].shape[1] == C
    assert (out / "default" / "sae.safetensors").exists()
