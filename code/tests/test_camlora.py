"""Per-track camera MotionLoRA switching (two random PEFT adapters on the tiny UNet stand in for the camera LoRAs)."""
import copy
from types import SimpleNamespace

import pytest
import torch

from conftest import TinyRun
from vdprobe import camlora, lora
from vdprobe.lockstep import Track


@pytest.fixture(scope="module")
def rig(tiny_unet):
    unet = copy.deepcopy(tiny_unet)
    run = TinyRun(unet)
    plain = run()
    g = torch.Generator().manual_seed(5)
    for name in ("a", "b"):
        lora.add_motion_lora(unet, "full", rank=4, alpha=4, name=name)
    for n, p in unet.named_parameters():
        if "lora_B" in n:                      # PEFT initialises B to zero: make the adapters do something
            p.data = torch.randn(p.shape, generator=g) * 0.3
        p.requires_grad_(False)
    unet.disable_adapters()
    return SimpleNamespace(unet=unet, run=run, plain=plain)


def under(rig, camera, **kw):
    camlora.use_camera(rig.unet, camera, **kw)
    return rig.run()


def test_none_track_equals_no_adapter(rig):                 # state leak between tracks
    under(rig, "a")
    for off in (None, "none", "off"):
        assert torch.equal(under(rig, off), rig.plain)


def test_switch_is_order_independent(rig):
    first = under(rig, "a")
    under(rig, "b"); under(rig, None)
    assert torch.equal(under(rig, "a"), first)
    under(rig, "b", scale=0.3, block_scales={"": 0.0})
    assert torch.equal(under(rig, "a"), first)


def test_adapter_changes_output(rig):
    a, b = under(rig, "a"), under(rig, "b")
    assert not torch.allclose(a, rig.plain, atol=1e-4) and not torch.allclose(a, b, atol=1e-4)
    assert not torch.allclose(under(rig, "a", scale=0.5), a, atol=1e-4)


def test_block_scales_zero_everything_equals_none(rig):
    assert torch.allclose(under(rig, "a", block_scales={"": 0.0}), rig.plain, atol=1e-6)


def test_block_scales_restore(rig):
    a = under(rig, "a")
    under(rig, "a", block_scales={"": 0.0})
    assert torch.equal(under(rig, "a"), a)
    assert torch.equal(under(rig, "a", block_scales={}), a)


def test_block_scales_single_block_and_longest_prefix(rig):
    a = under(rig, "a")
    only = under(rig, "a", block_scales={"": 0.0, "up_blocks.1": 1.0})
    assert not torch.allclose(only, a, atol=1e-5) and not torch.allclose(only, rig.plain, atol=1e-5)
    assert torch.equal(under(rig, "a", block_scales={"up_blocks": 0.0, "down_blocks": 0.0, "mid_block": 0.0, "up_blocks.1": 1.0}), only)
    assert torch.equal(under(rig, "a", block_scales={"": 0.5}), under(rig, "a", scale=0.5))
    camlora.use_camera(rig.unet, "a")
    assert camlora.set_block_scales(rig.unet, "a", {"no_such_block": 0.0}) == 0


def test_unknown_camera_raises(rig):
    with pytest.raises(KeyError):
        camlora.use_camera(rig.unet, "zoom-in")


def test_switcher_accepts_name_none_and_step_function(rig):
    before = camlora.camera_switcher(rig.unet)
    a, b = under(rig, "a"), under(rig, "b", scale=0.5)
    before(0, Track("t", "p", meta={"camera": "b", "camera_scale": 0.5})); assert torch.equal(rig.run(), b)
    before(0, Track("t", "p")); assert torch.equal(rig.run(), rig.plain)              # no entry: off
    tr = Track("t", "p", meta={"camera": lambda s: "a" if s < 2 else None})
    before(0, tr); assert torch.equal(rig.run(), a)
    before(1, tr); assert torch.equal(rig.run(), a)
    before(2, tr); assert torch.equal(rig.run(), rig.plain)
    tr = Track("t", "p", meta={"camera": "a", "camera_block_scales": {"": 0.0}})
    before(0, tr); assert torch.allclose(rig.run(), rig.plain, atol=1e-6)


def test_load_camera_loras_names_and_leaves_adapters_off(rig):
    calls = []
    pipe = SimpleNamespace(unet=rig.unet, load_lora_weights=lambda repo, adapter_name: calls.append((repo, adapter_name)))
    camlora.use_camera(rig.unet, "a")
    names = camlora.load_camera_loras(pipe, ["none", "zoom-in", "pan-left"])
    assert names == ["zoom-in", "pan-left"]
    assert calls == [("guoyww/animatediff-motion-lora-zoom-in", "zoom-in"),
                     ("guoyww/animatediff-motion-lora-pan-left", "pan-left")]
    assert torch.equal(rig.run(), rig.plain)


def test_run_arms_source_lora_and_track_meta(rig, monkeypatch):
    from test_kv import _FakePipe
    from vdprobe.coords import enumerate_coordinates
    from vdprobe.edit import Arm, EditSpec, run_arms
    from vdprobe.pipeline import Loaded

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    cfg = SimpleNamespace(frames=4, steps=3, guidance_scale=7.5, negative_prompt="")
    loaded = Loaded(_FakePipe(rig.unet), cfg, "test", "tiny", {}, 0.0, device="cpu")
    coords = enumerate_coordinates(rig.unet)
    noise = torch.randn(1, 4, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    arms = lambda: [Arm("plain", "p", EditSpec()), Arm("with_a", "p", EditSpec(), lora="a")]
    go = lambda **kw: run_arms(loaded, coords, "p", arms(), noise, decode_frames=False, cache_device="cpu", **kw)

    base = go()
    assert torch.equal(base.source_latents, base.latents["plain"])                    # same prompt, no LoRA
    assert not torch.allclose(base.latents["with_a"], base.latents["plain"], atol=1e-5)

    r = go(source_lora=("a", 1.0))                                                    # the source runs with the LoRA
    assert torch.equal(r.source_latents, base.latents["with_a"])
    assert torch.equal(r.latents["plain"], base.latents["plain"])                     # and it does not leak into the arms
    assert torch.equal(go(source_lora=("b", 1.0)).latents["with_a"], base.latents["with_a"])

    seen = []
    go(track_meta={"source": {"camera": "x"}, "plain": {"camera": "y"}},
       before_track=lambda step, tr: seen.append((step, tr.id, tr.meta.get("camera"))))
    assert seen[:3] == [(0, "source", "x"), (0, "plain", "y"), (0, "with_a", None)] and len(seen) == 9

    rig.unet.disable_adapters()
