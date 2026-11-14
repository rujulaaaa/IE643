"""Camera trajectories and Plücker embedding (CameraCtrl input)."""
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

from vdprobe.cameractrl import rays


def emb(name, h=32, w=48, **kw):
    return rays.plucker_embedding(rays.trajectory(name, **kw), h, w)


def test_shape_full_size():
    assert emb("pan-left", 256, 384).shape == (1, 6, 16, 256, 384)


def test_static_is_constant_with_zero_moment():
    e = emb("static")
    assert torch.equal(e[:, :, 1:], e[:, :, :1].expand_as(e[:, :, 1:]))
    assert torch.all(e[:, :3] == 0)


@pytest.mark.parametrize("name", rays.CAMERAS)
def test_frame0_is_identity_pose(name):
    assert torch.equal(emb(name)[:, :, 0], emb("static")[:, :, 0])


def test_directions_are_unit():
    assert torch.allclose(emb("roll-cw")[:, 3:].norm(dim=1), torch.ones(1, 16, 32, 48), atol=1e-5)


@pytest.mark.parametrize("a,b", [("pan-left", "pan-right"), ("tilt-up", "tilt-down"), ("zoom-in", "zoom-out")])
def test_opposite_translations_flip_the_moment_only(a, b):
    ea, eb = emb(a), emb(b)
    assert torch.allclose(ea[:, :3], -eb[:, :3], atol=1e-6) and ea[:, :3].abs().max() > 0.1
    assert torch.equal(ea[:, 3:], eb[:, 3:]) and torch.equal(ea[:, 3:], emb("static")[:, 3:])


def test_roll_changes_directions_not_moment():
    e = emb("roll-cw")
    assert torch.all(e[:, :3] == 0)
    assert not torch.allclose(e[:, 3:, -1], e[:, 3:, 0], atol=1e-3)
    assert not torch.allclose(e, emb("roll-ccw"), atol=1e-3)


def test_all_cameras_are_distinct():
    es = [emb(n) for n in rays.CAMERAS]
    for i in range(len(es)):
        for j in range(i):
            assert (es[i] - es[j]).abs().max() > 1e-2


def test_relative_poses_removes_a_common_transform():
    t = rays.trajectory("pan-left")
    g = torch.eye(4); g[:3, 3] = torch.tensor([3.0, -1.0, 2.0]); g[0, 0] = g[1, 1] = 0.0; g[0, 1], g[1, 0] = -1.0, 1.0
    moved = rays.Trajectory("moved", g[None] @ t.c2w)
    assert torch.allclose(rays.plucker_embedding(moved, 16, 24), rays.plucker_embedding(t, 16, 24), atol=1e-5)


def test_unknown_camera_raises():
    with pytest.raises(KeyError):
        rays.trajectory("dolly-zoom")


def _reference():
    root = os.environ.get("VDPROBE_CAMERACTRL_REF")
    if not root or not (Path(root) / "inference.py").exists():
        pytest.skip("set VDPROBE_CAMERACTRL_REF to a checkout of hehao13/CameraCtrl")
    # only `ray_condition` is needed; cut it out so that the file's heavy imports never run
    src = (Path(root) / "inference.py").read_text()
    body = src[src.index("def custom_meshgrid"):src.index("def get_relative_pose")] + \
        src[src.index("def ray_condition"):src.index("def load_personalized_base_model")]
    ns = {"torch": torch}
    exec("import torch\nfrom packaging import version as pver\n" + body, ns)
    return ns["ray_condition"]


@pytest.mark.parametrize("name", ["pan-left", "zoom-in", "roll-cw", "tilt-down"])
def test_matches_authors_ray_condition(name):
    ray_condition = _reference()
    h, w = 64, 96
    t = rays.trajectory(name)
    f = t.focal * h
    K = torch.tensor([f, f, w / 2, h / 2]).repeat(1, t.frames, 1)
    ref = ray_condition(K, rays.relative_poses(t.c2w)[None], h, w, device="cpu")[0].permute(3, 0, 1, 2)[None]
    assert torch.allclose(rays.plucker_embedding(t, h, w), ref, atol=1e-6)
