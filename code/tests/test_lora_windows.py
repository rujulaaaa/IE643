"""Cut-free LoRA windows (D-34): a window never spans a hard cut, and the old recipe's random
stream is untouched when there is only one shot."""

import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from train_motion_lora import FRAMES, pick_shot, usable_shots  # noqa: E402


def _clip_with_cut(n_a=20, n_b=40):
    rng = np.random.default_rng(0)
    a = np.full((n_a, 48, 64, 3), 30, np.uint8) + rng.integers(0, 3, (n_a, 48, 64, 3), dtype=np.uint8)
    b = np.full((n_b, 48, 64, 3), 200, np.uint8) + rng.integers(0, 3, (n_b, 48, 64, 3), dtype=np.uint8)
    return np.concatenate([a, b])


def test_shots_split_at_the_cut():
    clip = _clip_with_cut()
    assert usable_shots(clip) == [(0, 20), (20, 60)]
    short = _clip_with_cut(5, 12)          # no shot reaches FRAMES: the longest one alone
    assert usable_shots(short) == [(5, 17)] and 12 < FRAMES


def test_picked_window_stays_in_one_shot():
    clip = _clip_with_cut()
    shots = usable_shots(clip)
    rng = random.Random(0)
    for _ in range(50):
        seg = pick_shot(clip, shots, rng)
        assert len(np.unique(seg.mean(axis=(1, 2, 3)) > 100)) == 1
    assert len(pick_shot(clip, shots, None)) == 40        # validation: the longest shot


def test_single_shot_draws_nothing():
    clip = _clip_with_cut()
    r1, r2 = random.Random(5), random.Random(5)
    pick_shot(clip, [(0, len(clip))], r1)
    assert r1.random() == r2.random()
