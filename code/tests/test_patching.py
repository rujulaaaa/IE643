"""Patching framework tests on a tiny CPU UNetMotionModel (fp32, so 'identical' means bit-identical)."""

import pytest
import torch

from vdprobe.coords import branch_rows, coarse_coordinates, enumerate_coordinates
from vdprobe.harness import Harness
from vdprobe.patching import PairDistance, Patcher, PatchSpec, TensorCache, window

STEPS = (0, 1, 2)


def _run(unet, run, probes):
    coords = enumerate_coordinates(unet)
    with Harness(unet, coords, probes):
        return run(steps=3)


def test_windows():
    assert window("all") == tuple(range(25))
    assert window("first_third") == tuple(range(9))
    assert window("five", center=1) == (0, 1, 2, 3)
    assert window("one", center=12) == (12,)


def test_null_patch_is_exact_and_misindexed_patch_is_not(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    names = tuple(c.name for c in coarse_coordinates(coords))
    run = tiny_run(0)
    cache = TensorCache(names, STEPS, mode="raw")
    ref = _run(tiny_unet, run, [cache])
    spec = PatchSpec(names, STEPS)
    patcher = Patcher(cache, spec)
    out = _run(tiny_unet, run, [patcher])
    assert len(patcher.applied) == len(names) * len(STEPS)
    assert torch.equal(out, ref)
    # negative control: reading the cache one step late must visibly change the output
    bad = Patcher(cache, PatchSpec(names, (1, 2)), step_offset=-1)
    assert not torch.allclose(_run(tiny_unet, run, [bad]), ref, atol=1e-4)


def test_alpha_zero_is_a_noop_and_alpha_one_transplants(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    target = next(c for c in coords if c.kind == "temporal_module" and c.block == "up_blocks.1")
    src_cache = TensorCache([target], STEPS, mode="raw")
    _run(tiny_unet, tiny_run(1), [src_cache])            # source clip: a different input
    plain = _run(tiny_unet, tiny_run(0), [])
    noop = _run(tiny_unet, tiny_run(0), [Patcher(src_cache, PatchSpec((target.name,), STEPS, alpha=0.0))])
    assert torch.equal(noop, plain)
    seen = TensorCache([target], STEPS, mode="raw")     # records what flowed on after patching
    _run(tiny_unet, tiny_run(0), [Patcher(src_cache, PatchSpec((target.name,), STEPS)), seen])
    for s in STEPS:
        assert torch.equal(seen.store[(target.name, s)], src_cache.store[(target.name, s)])


def test_cond_only_and_channel_subset(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    target = next(c for c in coords if c.kind == "spatial_attn2")
    src = TensorCache([target], STEPS, mode="raw")
    _run(tiny_unet, tiny_run(1), [src])
    base = TensorCache([target], STEPS, mode="raw")
    _run(tiny_unet, tiny_run(0), [base])

    seen = TensorCache([target], (0,), mode="raw")
    _run(tiny_unet, tiny_run(0), [Patcher(src, PatchSpec((target.name,), (0,), branches="cond")), seen])
    got, s, b = seen.store[(target.name, 0)], src.store[(target.name, 0)], base.store[(target.name, 0)]
    uncond, cond = branch_rows(got.shape[0], 2, 0), branch_rows(got.shape[0], 2, 1)
    assert torch.equal(got[uncond], b[uncond]) and torch.equal(got[cond], s[cond])

    C = got.shape[-1]
    mask = torch.zeros(C)
    mask[: C // 4] = 1
    seen = TensorCache([target], (0,), mode="raw")
    _run(tiny_unet, tiny_run(0), [Patcher(src, PatchSpec((target.name,), (0,), channels={target.name: mask})), seen])
    got = seen.store[(target.name, 0)]
    assert torch.equal(got[..., : C // 4], s[..., : C // 4])
    assert torch.equal(got[..., C // 4:], b[..., C // 4:])


def test_patch_shape_mismatch_raises(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    a, b = [c for c in coords if c.kind == "temporal_module"][:2]
    cache = TensorCache([a], (0,), mode="raw")
    _run(tiny_unet, tiny_run(0), [cache])
    cache.store[(b.name, 0)] = torch.zeros(1)
    with pytest.raises(ValueError):
        _run(tiny_unet, tiny_run(0), [Patcher(cache, PatchSpec((b.name,), (0,)))])


def test_pair_distance_self_is_zero_and_other_is_positive(tiny_unet, tiny_run):
    coords = coarse_coordinates(enumerate_coordinates(tiny_unet))
    ref = TensorCache(coords, STEPS, mode="cond")
    _run(tiny_unet, tiny_run(0), [ref])
    same = PairDistance(ref, per_channel=True)
    _run(tiny_unet, tiny_run(0), [same])
    assert len(same.rows) == len(coords) * len(STEPS)
    assert max(abs(r["d_static"]) + abs(r["d_dynamic"]) for r in same.rows) < 1e-5
    other = PairDistance(ref)
    _run(tiny_unet, tiny_run(1), [other])
    assert min(r["d_raw"] for r in other.rows) > 0
