"""camera_models (the lora / cc interface) and the pure helpers of scripts/camera_sweep.py."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from vdprobe import camera_models as CM
from vdprobe.camera_set import CameraSet
from vdprobe.cameractrl.adapter import CameraSpec
from vdprobe.cameractrl.rays import CAMERAS
from vdprobe.config import CONFIG_DIR

ROOT = Path(__file__).resolve().parents[1]


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("model", CM.MODELS)
def test_families_cover_the_moving_cameras_of_the_set(model):
    cs = CameraSet(CONFIG_DIR / CM.CONFIGS[model][1])
    named = {c for pair in CM.FAMILIES[model].values() for c in pair}
    assert named == set(cs.moving_cameras) - {"static"}
    assert CM.family_of(model, cs.anchor_camera) is None
    assert CM.family_of(model, CM.FAMILIES[model]["roll"][0]) == "roll"


def test_cc_family_names_are_trajectories():
    assert {c for pair in CM.FAMILIES["cc"].values() for c in pair} <= set(CAMERAS)


def _model(name):
    cfg = SimpleNamespace(height=256, width=384) if name == "cc" else SimpleNamespace(height=512, width=512)
    return CM.CameraModel(name, SimpleNamespace(cfg=cfg), CameraSet(CONFIG_DIR / CM.CONFIGS[name][1]), None, None)


def test_flow_size_keeps_the_aspect_ratio():
    assert _model("cc").flow_size == (256, 384) and _model("lora").flow_size == (256, 256)


def test_block_meta_per_model():
    assert _model("lora").block_only("pan-left", "up_blocks.1") == \
        {"camera": "pan-left", "camera_block_scales": {"": 0.0, "up_blocks.1": 1.0}}
    m = _model("cc").block_without("pan-left", "up_blocks.1")
    assert m == {"camera": CameraSpec("pan-left", block_scales={"up_blocks.1": 0.0})}
    assert _model("cc").off == "off" and _model("lora").off == "none"


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        CM.load_camera_model("mvdream")


@pytest.mark.parametrize("model,n_clips,n_pairs", [("lora", 72, 127), ("cc", 80, 142)])
def test_sweep_subset_full_and_order(model, n_clips, n_pairs):
    sw = _script("camera_sweep")
    cs = CameraSet(CONFIG_DIR / CM.CONFIGS[model][1])
    clips, pairs = sw.subset(cs, 42, 0, 0)
    assert (len(clips), len(pairs)) == (n_clips, n_pairs)
    pos = {c.id: i for i, c in enumerate(sw.topo_order(clips, pairs))}
    assert len(pos) == n_clips and all(pos[p.a.id] < pos[p.b.id] for p in pairs)


def test_sweep_subset_smoke_keeps_anchors():
    sw = _script("camera_sweep")
    cs = CameraSet()
    clips, pairs = sw.subset(cs, 42, 2, 3)
    assert len(clips) == 6
    assert sum(p.kind == "motion" for p in pairs) == 4 and sum(p.kind == "content" for p in pairs) == 3
    # every non-anchor clip is the unit of exactly one camera pair and one content pair
    units = [c for c in clips if c.scene != cs.anchor_scene and c.camera != cs.anchor_camera]
    for u in units:
        assert sorted(p.kind for p in pairs if p.b.id == u.id) == ["content", "motion"]


# --- scripts/camera_switch.py: arm construction and scoring (no model) -----------------------------------

import numpy as np


def test_switch_arms_lora():
    sw = _script("camera_switch")
    metas = sw.build_tracks(_model("lora"), "zoom-in", sw.TAUS, sw.ARMS, sw.BLOCKS["lora"])
    assert len(metas) == 2 + 14 + 6 + 6 and "on_B" not in metas          # no swap arms for a LoRA
    assert metas["off"] == {"camera": "none"}
    f = metas["on_until_3"]["camera"]
    assert [f(s) for s in (0, 2, 3, 24)] == ["zoom-in", "zoom-in", None, None]
    g = metas["on_from_3"]["camera"]
    assert [g(s) for s in (0, 2, 3, 24)] == [None, None, "zoom-in", "zoom-in"]
    assert metas["only_up_blocks.1"]["camera_block_scales"] == {"": 0.0, "up_blocks.1": 1.0}
    assert metas["without_mid_block"]["camera_block_scales"] == {"mid_block": 0.0}


def test_switch_arms_cc_swap():
    sw = _script("camera_switch")
    cm = _model("cc")
    metas = sw.build_tracks(cm, "pan-left", (3,), sw.ARMS, sw.BLOCKS["cc"], [("up_blocks.1", "up_blocks.2")])
    assert metas["on_B"] == {"camera": "pan-right"}
    assert metas["swap_up_blocks.1"]["camera"] == CameraSpec("pan-left", block_trajectories={"up_blocks.1": "pan-right"})
    assert set(metas["swap_all"]["camera"].block_trajectories) == set(sw.BLOCKS["cc"])
    assert "swap_up_blocks.1+up_blocks.2" in metas
    assert "on_B" not in sw.build_tracks(cm, "roll-cw", (3,), sw.ARMS, sw.BLOCKS["cc"])     # no partner defined
    assert sw.default_cameras("cc") == ["zoom-in", "pan-left", "tilt-up", "roll-cw"]
    assert sw.default_cameras("lora")[-1] == "rolling-clockwise"


def test_switch_score():
    sw = _script("camera_switch")
    off, on, b = np.zeros(4), np.array([0.2, 0, 0, 0]), np.array([-0.2, 0, 0, 0])
    sigs = {"on": on, "off": off, "on_B": b, "on_until_3": on / 2, "swap_up_blocks.1": np.zeros(4), "swap_all": b}
    rows = {r["arm"]: r for r in sw.score(sigs, {a: 1.0 for a in sigs})}
    assert rows["on"]["retained"] == 1.0 and rows["off"]["retained"] == 0.0
    assert rows["on_until_3"]["retained"] == pytest.approx(0.5) and rows["on_until_3"]["tau"] == 3
    assert rows["swap_all"]["retained_b"] == pytest.approx(1.0) and rows["on"]["retained_b"] == 0.0
    assert rows["swap_up_blocks.1"]["retained_b"] == pytest.approx(0.5)
    assert rows["swap_up_blocks.1"]["block"] == "up_blocks.1" and rows["swap_up_blocks.1"]["kind"] == "swap"
    # a camera that does nothing gives NaN, not a division by zero
    assert np.isnan(sw.score({"on": off, "off": off}, {"on": 1.0, "off": 1.0})[0]["retained"])


def test_switch_analysis_on_planted_rows(tmp_path):
    sw, an = _script("camera_switch"), _script("camera_switch_analysis")
    import csv
    off, on, b = np.zeros(4), np.array([0.2, 0, 0, 0]), np.array([-0.2, 0, 0, 0])
    for i, scene in enumerate(("lake", "castle", "street")):
        sigs = {"on": on, "off": off, "on_B": b, "on_until_1": on * 0.2, "on_until_3": on * (0.6 + 0.1 * i),
                "on_from_3": on * 0.3, "only_up_blocks.1": on * 0.7, "swap_up_blocks.1": on * 0.4,
                "swap_up_blocks.2": on * 0.9, "swap_all": b}
        rows = sw.score(sigs, {a: 0.9 for a in sigs})
        d = tmp_path / "job0" / f"{scene}__pan-left"
        d.mkdir(parents=True)
        with open(d / "rows.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["model", "scene", "camera", "seed"] + list(rows[0]))
            w.writeheader()
            w.writerows({"model": "cc", "scene": scene, "camera": "pan-left", "seed": 2024, **r} for r in rows)
    s = an.summarise(an.load_rows([tmp_path]), n_boot=200)
    assert s["n_groups"] == 3 and s["arms"]["on"]["retained"]["mean"] == pytest.approx(1.0)
    when = s["verdicts"]["when"]
    assert when["mean"] == pytest.approx(0.7) and when["pass"] and when["lo"] <= 0.7 <= when["hi"]
    assert when["first_tau_until_half"] == 3 and when["last_tau_from_half"] is None
    where = s["verdicts"]["where"]                    # swap_up_blocks.1 keeps 0.4 of A: 0.3 of the way to B
    assert where["single_blocks_ranked"][0] == ("up_blocks.1", 0.3) and not where["pass"]
    an.figures(s, tmp_path)
    assert (tmp_path / "camera_switch.png").exists() and (tmp_path / "camera_blocks.png").exists()


# --- scripts/camera_patch.py -------------------------------------------------------------------------

def test_patch_candidates_and_pairs(tiny_cc_unet):
    from vdprobe.coords import enumerate_coordinates
    pt = _script("camera_patch")
    coords = enumerate_coordinates(tiny_cc_unet)
    cands = pt.candidates(coords)
    assert [c["name"] for c in cands[:3]] == list(pt.NAMED)
    blocks = [c for c in cands if c["name"].startswith("mm_")]
    assert blocks and all(c["coords"] and c["component"] == "all" and c["steps"] == pt.STEPS for c in blocks)
    assert "mm_mid_block" not in [c["name"] for c in blocks]              # the v3 adapter has no mid motion module
    cs = CameraSet(CONFIG_DIR / "cc_set.yaml")
    pairs = pt.draw_pairs(cs, ["zoom-in", "pan-left"], 3)
    assert len(pairs) == 6 and pairs[0] == ("zoom-in", cs.scenes[0], cs.scenes[1]) and pairs[1][0] == "pan-left"
    assert all(x != y for _, x, y in pairs)


def test_patch_score_pair():
    pt = _script("camera_patch")
    A, B, B2 = np.array([0.2, 0, 0, 0]), np.array([0.0, 0.01, 0, 0]), np.array([0.0, 0, 0.01, 0])
    d = A - B
    sigs = {"A": A, "B": B, "B2": B2,
            "B<-A|c": B + 0.5 * d,            # half the camera arrives
            "B2<-A|c": B2 + d,                # all of it, in the other scene
            "A<-B|c": A - 0.25 * d,           # a quarter is removed
            "B<-B2|c": B}                     # the control changes nothing
    dc = {"A": 0.10, "B": 0.10, "B2": -0.10, "B<-A|c": 0.1, "B2<-A|c": -0.06, "A<-B|c": 0.1, "B<-B2|c": 0.08}
    rows = {r["direction"]: r for r in pt.score_pair(sigs, dc, ["c", "absent"])}
    assert set(rows) == {"suff", "transfer", "nec", "ctrl"}
    assert rows["suff"]["eff_cam"] == pytest.approx(0.5) and np.isnan(rows["suff"]["eff_content"])
    assert rows["transfer"]["eff_cam"] == pytest.approx(1.0)
    assert rows["transfer"]["eff_content"] == pytest.approx(0.2)          # (-0.06 + 0.10) / (0.10 + 0.10)
    assert rows["nec"]["eff_cam"] == pytest.approx(0.25)
    assert rows["ctrl"]["eff_cam"] == pytest.approx(0.0) and rows["ctrl"]["eff_content"] == pytest.approx(0.1)
    assert rows["transfer"]["eff_cam_cos"] > 0.9 and abs(rows["ctrl"]["eff_cam_cos"]) < 1e-9
    # a camera with no measurable effect: NaN, not an exception
    flat = pt.score_pair({"A": B, "B": B, "B2": B2, "B<-A|c": B}, {"A": 0.1, "B": 0.1, "B2": -0.1, "B<-A|c": 0.1}, ["c"])
    assert np.isnan(flat[0]["eff_cam"])


def test_patch_verdict_on_planted_rows(tmp_path):
    import pandas as pd
    vd = _script("camera_verdict")
    rows = []
    for i in range(8):
        for cand, s, t, leak, nec, ctrl in (("up1_motion_modules_all", 0.6 + 0.02 * i, 0.5, 0.05, 0.4, 0.1),
                                            ("mm_up_blocks.1", None, 0.45 + 0.01 * i, 0.3, 0.2, None)):
            for direction, eff, content in (("suff", s, np.nan), ("transfer", t, leak), ("nec", nec, np.nan),
                                            ("ctrl", ctrl, 0.0)):
                if eff is not None:
                    rows.append({"model": "lora", "pair": f"p{i}", "camera": "zoom-in", "candidate": cand,
                                 "direction": direction, "eff_cam": eff, "eff_cam_cos": eff, "eff_content": content})
    d = tmp_path / "job0" / "p"
    d.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(d / "rows.csv", index=False)
    with pytest.raises(SystemExit):
        vd.load_rows([tmp_path])                         # no manifest: the pair is not finished
    (d / "manifest.json").write_text("{}")
    df = vd.load_rows([tmp_path])
    s = vd.summarise(df, n_boot=300)
    assert s["verdicts"]["HC2"]["pass"] and s["verdicts"]["HC3"]["pass"]
    assert s["candidates"]["up1_motion_modules_all"]["suff_minus_ctrl"]["mean"] == pytest.approx(0.57)
    assert "suff_minus_ctrl" not in s["candidates"]["mm_up_blocks.1"]
    cc = vd.summarise(df.assign(model="cc"), block="up_blocks.1", n_boot=300)
    assert cc["verdicts"]["HK3"]["transfer"]["lo"] > 0.3 and not cc["verdicts"]["HK3"]["pass"]     # leak 0.3 > 0.1
    assert "up1_motion_modules_all" in vd.markdown(s)
