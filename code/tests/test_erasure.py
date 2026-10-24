"""LEACE: the concept becomes linearly unpredictable; a same-rank random eraser does not do that."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from vdprobe.erasure import EraseProbe, fit_leace, random_eraser


def _data(seed=0, n=2000, C=16, K=4):
    rng = np.random.default_rng(seed)
    z = rng.integers(0, K, n)
    Z = np.eye(K)[z]
    X = rng.normal(size=(n, C)) @ rng.normal(size=(C, C))      # correlated features
    X += Z @ rng.normal(0, 2, (K, C))                           # planted concept
    return X, Z, z


def _acc(X, z):
    m = LogisticRegression(max_iter=2000).fit(X[:1000], z[:1000])
    return m.score(X[1000:], z[1000:])


def test_leace_removes_linear_information_minimally():
    X, Z, z = _data()
    e = fit_leace(X[:1000], Z[:1000])
    assert e.rank == 3                                           # K - 1 after centring
    Xe = e(torch.from_numpy(X).float()).numpy()
    assert _acc(X, z) > 0.9
    assert _acc(Xe, z) < 0.35                                    # chance 0.25
    r = random_eraser(X[:1000], e.rank, seed=1)
    assert _acc(r(torch.from_numpy(X).float()).numpy(), z) > 0.8
    # the eraser removes rank-3 structure only: the change is small relative to the data
    assert np.linalg.norm(Xe - X) / np.linalg.norm(X - X.mean(0)) < 0.7


def test_erase_probe_keeps_the_static_part(tiny_unet, tiny_run):
    from vdprobe.coords import enumerate_coordinates
    from vdprobe.harness import Harness
    from vdprobe.patching import TensorCache

    coords = enumerate_coordinates(tiny_unet)
    c = next(x for x in coords if x.kind == "temporal_module")
    C = dict(tiny_unet.named_modules())[c.name].proj_in.in_features
    rng = np.random.default_rng(0)
    e = random_eraser(rng.normal(size=(500, C)), rank=2)
    before = TensorCache([c.name], [0], mode="cond")
    after = TensorCache([c.name], [0], mode="cond")
    probe = EraseProbe(c, e, steps=[0])
    run = tiny_run()
    with Harness(tiny_unet, coords, [before, probe, after]):
        run(steps=1)
    assert probe.applied == 1
    xb, xa = before.store[(c.name, 0)].float(), after.store[(c.name, 0)].float()
    assert torch.allclose(xb.mean(0), xa.mean(0), atol=1e-4)        # frame mean (x_bar) untouched
    assert not torch.allclose(xb, xa)
