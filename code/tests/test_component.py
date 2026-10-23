"""Component patches (D-29): transplanting only x̃ or only x̄ of the source, on the tiny CPU UNet."""

import pytest
import torch

from vdprobe.coords import canonical, enumerate_coordinates, frame_mean_raw
from vdprobe.harness import Harness
from vdprobe.patching import Patcher, PatchSpec, TensorCache
from vdprobe.summaries import split_static_dynamic

STEPS = (0, 1, 2)
NB, F = 2, 4


def _run(unet, run, probes):
    with Harness(unet, enumerate_coordinates(unet), probes):
        return run(steps=3)


@pytest.mark.parametrize("kind,shape", [("spatial_attn1", (NB * F, 64, 32)),
                                        ("temporal_attn1", (NB * 64, F, 32)),
                                        ("temporal_module", (NB * F, 32, 8, 8))])
def test_frame_mean_raw_matches_canonical_split(kind, shape):
    t = torch.randn(shape)
    fm = frame_mean_raw(t, kind, NB, F)
    assert fm.shape == t.shape
    for b in range(NB):
        x_bar, _ = split_static_dynamic(canonical(t, kind, NB, F)[b])
        got = canonical(fm.contiguous(), kind, NB, F)[b]
        assert torch.allclose(got, x_bar.unsqueeze(0).expand_as(got), atol=1e-6)


@pytest.mark.parametrize("kind", ["temporal_module", "temporal_attn1", "spatial_attn1"])
@pytest.mark.parametrize("component", ["dynamic", "static"])
def test_component_patch_takes_one_part_and_keeps_the_other(tiny_unet, tiny_run, kind, component):
    coords = enumerate_coordinates(tiny_unet)
    c = next(x for x in coords if x.kind == kind and x.block == "up_blocks.1")
    src = TensorCache([c], STEPS, mode="cond")
    _run(tiny_unet, tiny_run(1), [src])                     # clip A
    plain = TensorCache([c], STEPS, mode="cond")
    _run(tiny_unet, tiny_run(0), [plain])                   # clip B, unpatched
    raw_src = TensorCache([c], STEPS, mode="raw")
    _run(tiny_unet, tiny_run(1), [raw_src])
    seen = TensorCache([c], STEPS, mode="cond")             # B's output after the patch
    _run(tiny_unet, tiny_run(0), [Patcher(raw_src, PatchSpec((c.name,), (0,), component=component)), seen])
    a_bar, a_til = split_static_dynamic(src.store[(c.name, 0)])
    b_bar, b_til = split_static_dynamic(plain.store[(c.name, 0)])
    o_bar, o_til = split_static_dynamic(seen.store[(c.name, 0)])
    if component == "dynamic":
        assert torch.allclose(o_bar, b_bar, atol=1e-5) and torch.allclose(o_til, a_til, atol=1e-5)
        assert not torch.allclose(o_til, b_til, atol=1e-3)
    else:
        assert torch.allclose(o_bar, a_bar, atol=1e-5) and torch.allclose(o_til, b_til, atol=1e-5)
        assert not torch.allclose(o_bar, b_bar, atol=1e-3)


def test_component_self_patch_is_null(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    names = tuple(c.name for c in coords if c.kind == "temporal_module")
    run = tiny_run(0)
    cache = TensorCache(names, STEPS, mode="raw")
    ref = _run(tiny_unet, run, [cache])
    for comp in ("dynamic", "static"):
        out = _run(tiny_unet, run, [Patcher(cache, PatchSpec(names, STEPS, component=comp))])
        assert torch.equal(out, ref)        # δ = 0 exactly, so no rounding (D-11 on the real model)


def test_bad_component_rejected():
    with pytest.raises(ValueError):
        PatchSpec(("x",), (0,), component="motion")
