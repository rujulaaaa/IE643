"""Edit metrics on synthetic clips with known motion."""

import numpy as np
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
