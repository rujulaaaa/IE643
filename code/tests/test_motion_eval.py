"""Motion Fidelity Score on synthetic tracklets (no CoTracker download)."""

import torch

from vdprobe.motion_eval import motion_fidelity_score, track_similarity


def _tracks(direction, n=6, f=8):
    t = torch.arange(f).float()[None, :, None] * torch.tensor(direction).float()
    return t.expand(n, f, 2) + torch.randn(n, 1, 2)            # same motion, scattered start points


def test_same_motion_anywhere_scores_one():
    a, b = _tracks((1, 0)), _tracks((1, 0)) + 50                # different layout, same displacement
    assert abs(motion_fidelity_score(a, b)["mfs_symmetric"] - 1) < 1e-4


def test_opposite_and_still():
    a = _tracks((1, 0))
    assert abs(motion_fidelity_score(a, _tracks((-1, 0)))["mfs_official"] + 1) < 1e-4
    still = torch.zeros(4, 8, 2)
    s = track_similarity(a, still)
    assert torch.isfinite(s).all() and s.abs().max() < 1e-3      # no NaN for a point that never moves


def test_official_direction_is_generated_to_reference():
    gen = _tracks((1, 0))
    ref = torch.cat([_tracks((1, 0), n=1), _tracks((0, 1), n=9)])  # one matching track among others
    r = motion_fidelity_score(gen, ref)
    assert r["mfs_official"] > 0.99                             # every generated track finds its match
    assert r["mfs_symmetric"] < 0.6                             # most reference tracks find none
