"""Edit metrics on synthetic clips with known motion."""

import os

import numpy as np
import pytest
from PIL import Image

from vdprobe import metrics as M


def _clip(dx, dy, color=(255, 255, 255), n=8, size=64):
    frames = []
    for i in range(n):
        a = np.zeros((size, size, 3), np.uint8)
        x, y = 20 + dx * i, 20 + dy * i
        a[y:y + 14, x:x + 14] = color
        a[y + 4:y + 8, x + 4:x + 8] = 60          # texture so flow has something to lock onto
        frames.append(Image.fromarray(a))
    return frames


def test_motion_fidelity_tracks_direction_not_colour():
    right = M.flow_fields(_clip(2, 0))
    right_red = M.flow_fields(_clip(2, 0, color=(255, 60, 60)))
    down = M.flow_fields(_clip(0, 2))
    left = M.flow_fields(_clip(-2, 0))
    same = M.motion_fidelity(right, right_red)
    assert same > 0.8
    assert abs(M.motion_fidelity(right, down)) < 0.3
    assert M.motion_fidelity(right, left) < -0.3      # -0.49: the square ends up on different pixels
    assert M.motion_magnitude(right) > 4 * M.motion_magnitude(M.flow_fields(_clip(0, 0)))


def test_disentanglement_range_and_sign():
    pool_c, pool_m = np.array([0.5, 0.9]), np.array([-0.2, 0.8])
    best = M.disentanglement(0.9, 0.8, 0.5, -0.2, pool_c, pool_m)
    worst = M.disentanglement(0.5, -0.2, 0.9, 0.8, pool_c, pool_m)
    assert np.isclose(best, 1.0) and np.isclose(worst, -1.0)


# ---------------------------------------------------------------------------
# Experiment 2 flow: camera removal and subject-masked fidelity (synthetic flow fields)
# ---------------------------------------------------------------------------

def _flow(obj, cam, n=5, size=32, box=(8, 20)):
    f = np.tile(np.asarray(cam, np.float32), (n, size, size, 1))
    a, b = box
    f[:, a:b, a:b] = np.asarray(obj, np.float32) + np.asarray(cam, np.float32)
    mask = np.zeros((n + 1, size, size), bool)
    mask[:, a:b, a:b] = True
    return f, mask


def test_camera_motion_is_removed():
    f, mask = _flow(obj=(2, 0), cam=(0, 3))
    g = M.remove_camera_motion(f, mask)
    assert np.allclose(g[:, 0, 0], 0)                     # background: only the pan, now removed
    assert np.allclose(g[:, 10, 10], (2, 0))              # subject: its own motion survives


def test_masked_fidelity_ignores_the_background():
    fa, ma = _flow(obj=(2, 0), cam=(0, 0))
    fb, mb = _flow(obj=(2, 0), cam=(0, 0))
    fb[:, :8] = (0, -3)                                   # clip B: a different background moves
    assert M.masked_motion_fidelity(fa, fb, ma, mb) == pytest.approx(1.0)
    assert M.motion_fidelity(fa, fb) < 0.9
    fc, mc = _flow(obj=(-2, 0), cam=(0, 0))               # same place, opposite motion
    assert M.masked_motion_fidelity(fa, fc, ma, mc) == pytest.approx(-1.0)


@pytest.mark.skipif(not os.environ.get("VDPROBE_DOWNLOAD_TESTS"), reason="downloads RAFT weights (~20 MB)")
def test_raft_recovers_a_translation():
    rng = np.random.default_rng(0)
    tex = (rng.random((160, 160, 3)) * 255).astype(np.uint8)
    tex = np.asarray(Image.fromarray(tex).resize((40, 40)).resize((160, 160), Image.BICUBIC))
    frames = [Image.fromarray(np.roll(tex, 2 * i, axis=1)[:128, :128]) for i in range(3)]
    f = M.raft_flow(frames, size=128)
    assert f.shape == (2, 128, 128, 2)
    assert np.median(f[..., 0]) == pytest.approx(2, abs=0.3) and abs(np.median(f[..., 1])) < 0.3


# --- camera signature (novelty phase) ---------------------------------------------------------

def _grid(n=64, w=None):
    w = w or n
    y, x = np.mgrid[0:n, 0:w].astype(np.float32)
    h = n / 2
    return (x - w / 2 + 0.5) / h, (y - h + 0.5) / h, h


def _camera_cases(n=64, w=None):
    x, y, h = _grid(n, w)
    return {0: np.stack([np.full_like(x, 2.0), np.zeros_like(x)], -1),      # pan: tx
            1: np.stack([np.zeros_like(x), np.full_like(x, -3.0)], -1),     # tilt: ty
            2: np.stack([x * h * 0.1, y * h * 0.1], -1),                    # zoom
            3: np.stack([-y * h * 0.1, x * h * 0.1], -1)}                   # roll


@pytest.mark.parametrize("w", [None, 96])
def test_camera_signature_recovers_each_motion(w):
    for k, f in _camera_cases(w=w).items():
        sig = M.camera_signature(np.repeat(f[None], 5, 0))
        assert np.argmax(np.abs(sig)) == k
        assert np.abs(np.delete(sig, k)).max() < 1e-3


def test_camera_signature_values_and_signs():
    c = _camera_cases()
    assert M.camera_signature(c[0][None])[0] == pytest.approx(2.0 / 32)
    assert M.camera_signature(c[1][None])[1] == pytest.approx(-3.0 / 32)
    assert M.camera_signature(c[2][None])[2] == pytest.approx(0.1)
    assert M.camera_signature(c[3][None])[3] == pytest.approx(0.1)
    assert M.camera_signature(-c[3][None])[3] == pytest.approx(-0.1)


def test_camera_signature_roll_is_not_read_as_pan():
    sig = M.camera_signature(np.repeat(_camera_cases()[3][None], 3, 0))
    assert abs(sig[3]) > 50 * max(abs(sig[0]), abs(sig[1]))


def test_camera_signature_is_linear_and_averages_frames():
    c = _camera_cases()
    mix = c[0] + 2 * c[2]
    assert np.allclose(M.camera_signature(mix[None]), [2.0 / 32, 0, 0.2, 0], atol=1e-6)
    two = np.stack([c[0], -c[0] * 0.5])
    assert M.camera_signature(two)[0] == pytest.approx(0.5 * (2.0 / 32 - 1.0 / 32))       # mean over the two frame pairs


def test_camera_signature_ignores_a_moving_object_on_a_static_background():
    f = np.zeros((4, 64, 64, 2), np.float32)
    f[:, 20:30, 20:30, 0] = 6.0                       # a small object moves; mean flow 0.146 px
    sig = M.camera_signature(f)
    assert np.abs(sig).max() < 0.01


def test_camera_signature_static_is_zero_not_nan():
    assert np.array_equal(M.camera_signature(np.zeros((5, 64, 64, 2), np.float32)), np.zeros(4))
    assert np.array_equal(M.camera_signature(np.full((5, 64, 64, 2), 0.01, np.float32)), np.zeros(4))
    assert np.array_equal(M.camera_signature(np.full((2, 8, 8, 2), np.nan)), np.zeros(4))
    assert np.array_equal(M.camera_signature(np.zeros((0, 64, 64, 2))), np.zeros(4))


def test_camera_fidelity_opposites():
    a = np.array([0.1, 0, 0, 0])
    assert M.camera_fidelity(a, -a) == pytest.approx(-1.0)
    assert M.camera_fidelity(a, 3 * a) == pytest.approx(1.0)
    assert M.camera_fidelity(a, np.array([0, 0, 0.2, 0])) == pytest.approx(0.0)
    assert M.camera_fidelity(a, np.zeros(4)) == 0.0


def test_raft_flow_plumbing_non_square(monkeypatch):
    """The path around RAFT (resize, batching, output layout) with the network replaced: no weights needed."""
    import torch
    import torchvision.models.optical_flow as O
    from PIL import Image

    class Fake(torch.nn.Module):
        def forward(self, a, b):
            return [torch.zeros(a.shape[0], 2, a.shape[2], a.shape[3]) + 1.5]

    monkeypatch.setattr(O, "raft_large", lambda weights=None: Fake())
    monkeypatch.setattr(M, "_RAFT", {})
    frames = [Image.new("RGB", (96, 64), (i * 10, 0, 0)) for i in range(7)]
    f = M.raft_flow(frames, size=(128, 192))
    assert f.shape == (6, 128, 192, 2) and np.allclose(f, 1.5)
