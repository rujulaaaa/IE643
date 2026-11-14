import pytest

from vdprobe.camera_set import CameraSet
from vdprobe.config import CONFIG_DIR, GenerationConfig


def test_counts():
    cs = CameraSet()
    assert len(cs.clips(42)) == 72
    assert len(cs.camera_pairs(42)) == 64 and len(cs.content_pairs(42)) == 63


def test_pair_rules():
    cs = CameraSet()
    for p in cs.camera_pairs(42):
        assert p.kind == "motion" and p.a.camera == "none" and p.a.scene == p.b.scene and p.b.camera != "none"
        assert p.a.seed == p.b.seed == 42 and p.a.prompt() == p.b.prompt()
    for p in cs.content_pairs(42):
        assert p.kind == "content" and p.a.camera == p.b.camera and p.a.scene == cs.anchor_scene != p.b.scene


def test_ids_unique():
    cs = CameraSet()
    ids = [c.id for s in cs.seeds for c in cs.clips(s)]
    assert len(ids) == len(set(ids)) == 144
    assert ids[0] == "42__a-mountain-lake-at-sunrise__none"


def test_prompt_never_names_the_camera():
    cs = CameraSet()
    cfg = GenerationConfig.from_yaml(CONFIG_DIR / "cameractrl.yaml")
    for c in cs.clips(42):
        assert c.prompt() == c.prompt(cfg) == f"{c.scene}, high quality, detailed"
        assert c.prompt(GenerationConfig.from_yaml()) == f"{c.scene}, high quality, detailed"   # the frozen subject/verb template is not used


def test_heldout_seed_is_not_a_sweep_seed():
    for path in ("camera_set.yaml", "cc_set.yaml"):
        cs = CameraSet(CONFIG_DIR / path)
        assert cs.heldout_seed not in cs.seeds


def test_cameractrl_set():
    from vdprobe.cameractrl.rays import CAMERAS
    cs = CameraSet(CONFIG_DIR / "cc_set.yaml")
    assert cs.anchor_camera == "off" and cs.moving_cameras == list(CAMERAS) and cs.lora_repo is None
    assert (len(cs.clips(42)), len(cs.camera_pairs(42)), len(cs.content_pairs(42))) == (80, 72, 70)
    assert cs.scenes == CameraSet().scenes and cs.seeds == CameraSet().seeds


def test_drift_is_caught(tmp_path):
    text = (CONFIG_DIR / "camera_set.yaml").read_text().replace("  - a tropical beach with palm trees\n", "")
    p = tmp_path / "set.yaml"
    p.write_text(text)
    with pytest.raises(AssertionError, match="drifted"):
        CameraSet(p)
