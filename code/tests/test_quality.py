"""Quality-gate formulas on synthetic inputs (no model downloads)."""

import numpy as np
import torch

from vdprobe import quality as Q


def test_subject_consistency_formula():
    same = torch.nn.functional.normalize(torch.ones(5, 8), dim=-1)
    assert abs(Q.subject_consistency(same) - 1.0) < 1e-6
    alt = torch.stack([torch.eye(8)[i % 2] for i in range(5)])      # alternates between two orthogonal features
    # t=1: prev 0, first 0; t=2: prev 0, first 1; ... -> mean of (0, .5, 0, .5)
    assert abs(Q.subject_consistency(alt) - 0.25) < 1e-6


def test_dynamic_degree_threshold():
    still = np.zeros((15, 64, 64, 2), np.float32)
    assert Q.dynamic_degree(still) == (False, 0.0)
    moving = still.copy()
    moving[:, :8, :8, 0] = 10.0                     # top 5% of 4096 = 205 values; 64 move by 10 px
    dyn, top = Q.dynamic_degree(moving)
    assert abs(top - 10.0 * 64 / 205) < 1e-4
    assert dyn is False                             # 3.1 px average in the top 5%: below 6 px
    moving[:, :16, :16, 0] = 10.0                   # 256 moving values: the whole top 5% at 10 px
    assert Q.dynamic_degree(moving)[0] is True
