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
