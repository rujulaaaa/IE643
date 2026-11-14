"""CameraCtrl graft: pose encoder, hooks, per-block camera state."""
import json
from pathlib import Path

import pytest
import torch

from conftest import TinyRun, make_tiny_adapter
from vdprobe.cameractrl.adapter import CameraAdapter, CameraSpec, merge_key
from vdprobe.cameractrl.pose_encoder import CameraPoseEncoder
from vdprobe.cameractrl.rays import plucker_embedding, trajectory

KEYS = json.loads((Path(__file__).parent / "data" / "cameractrl_keys.json").read_text())
HW, F = 16, TinyRun.F


@pytest.fixture
def rig(tiny_cc_unet):
    ad = make_tiny_adapter(tiny_cc_unet).attach()
    run = TinyRun(tiny_cc_unet)
    yield ad, run
    ad.detach()


def on(ad, traj="pan-left", **kw):
    ad.set(CameraSpec(traj, **kw), HW, HW, F)


def test_encoder_keys_and_shapes_equal_the_checkpoint():
    sd = {k: list(v.shape) for k, v in CameraPoseEncoder().state_dict().items()}
    assert sd == KEYS["pose_encoder_state_dict"]


def test_encoder_output_shapes():
    enc = CameraPoseEncoder(channels=(32, 64), cin=24, downscale_factor=2, heads=2, max_len=8)
    out = enc(plucker_embedding(trajectory("zoom-in", F), HW, HW))
    assert [tuple(o.shape) for o in out] == [(1, 32, F, 8, 8), (1, 64, F, 4, 4)]


def test_encoder_rejects_wrong_cin():
    with pytest.raises(ValueError):
        CameraPoseEncoder(cin=100)


def test_sites_are_the_first_temporal_attention_of_every_motion_module(tiny_cc_unet):
    ad = make_tiny_adapter(tiny_cc_unet)
    n_mm = sum(len(b.motion_modules) for b in list(tiny_cc_unet.down_blocks) + list(tiny_cc_unet.up_blocks))
    assert len(ad.sites) == n_mm == len(ad.merges)
    assert all(s.endswith(".transformer_blocks.0.attn1") and ".motion_modules." in s for s in ad.sites)
    assert ad.level["down_blocks.0.motion_modules.0.transformer_blocks.0.attn1"] == 0
    assert ad.level["up_blocks.0.motion_modules.0.transformer_blocks.0.attn1"] == 1     # deepest feature first


def test_off_is_bit_identical(tiny_cc_unet):
    run = TinyRun(tiny_cc_unet)
    plain = run()
    ad = make_tiny_adapter(tiny_cc_unet).attach()
    try:
        assert torch.equal(run(), plain)                     # attached, never set
        on(ad); ad.set(None)
        assert torch.equal(run(), plain)                     # set, then switched off
        on(ad, scale=0.0)
        assert torch.equal(run(), plain) and ad.calls == 0   # scale 0 never touches a tensor
    finally:
        ad.detach()
    assert torch.equal(run(), plain)


def test_zero_initialised_merges_change_nothing(tiny_cc_unet):
    run = TinyRun(tiny_cc_unet)
    plain = run()
    ad = make_tiny_adapter(tiny_cc_unet, randomise=False).attach()
    try:
        on(ad)
        assert torch.allclose(run(), plain, atol=1e-6) and ad.calls > 0
    finally:
        ad.detach()


def test_camera_changes_output_and_cameras_differ(rig):
    ad, run = rig
    ad.set(None); off = run()
    outs = {}
    for name in ("static", "pan-left", "pan-right", "roll-cw"):
        on(ad, name); outs[name] = run()
    assert all(not torch.allclose(o, off, atol=1e-4) for o in outs.values())      # `static` is not `off`
    names = list(outs)
    for i in range(len(names)):
        for j in range(i):
            assert not torch.allclose(outs[names[i]], outs[names[j]], atol=1e-5)


def test_hook_arithmetic_and_token_layout(rig, tiny_cc_unet):
    ad, run = rig
    site = "down_blocks.1.motion_modules.0.transformer_blocks.0.attn1"
    mod = dict(tiny_cc_unet.named_modules())[site]
    seen = {}
    # registered after the adapter's hook, so with the camera on it sees the replaced tensor
    h = mod.register_forward_pre_hook(lambda m, a: seen.__setitem__("x", a[0].clone()))
    try:
        ad.set(None); run(steps=1); before = seen["x"]
        # camera in this layer only: everything upstream is as in the off run, so `before` is the hook's input
        on(ad, "roll-cw", scale=0.7, block_scales={"": 0.0, site: 1.0}); run(steps=1); after = seen["x"]
    finally:
        h.remove()
    feat = ad.encoder(plucker_embedding(trajectory("roll-cw", F), HW, HW))[1]       # (1, 64, F, 4, 4)
    b = TinyRun.B
    assert before.shape == (b * 16, F, 64)
    pose = torch.empty(b, 4, 4, F, 64)
    for y in range(4):
        for x in range(4):
            pose[:, y, x] = feat[0, :, :, y, x].T                                    # token (batch, y, x) <- pixel (y, x)
    pose = pose.reshape(b * 16, F, 64)
    expect = before + 0.7 * ad.merges[merge_key(site)](before + pose)
    assert torch.allclose(after, expect, atol=1e-5)
    assert not torch.allclose(after, before, atol=1e-3)


def test_block_scales(rig):
    ad, run = rig
    ad.set(None); off = run()
    on(ad); full = run()
    on(ad, block_scales={"": 0.0}); assert torch.equal(run(), off)
    on(ad, block_scales={"": 0.0, "up_blocks.1": 1.0}); only = run()
    assert not torch.allclose(only, off, atol=1e-5) and not torch.allclose(only, full, atol=1e-5)
    on(ad, block_scales={"down_blocks": 1.0, "up_blocks": 1.0}); assert torch.equal(run(), full)
    on(ad, block_scales={"up_blocks": 0.0, "up_blocks.1": 1.0, "down_blocks": 0.0})      # longest prefix wins
    assert torch.equal(run(), only)
    on(ad, scale=0.5); half = run()
    on(ad, block_scales={"": 0.5}); assert torch.equal(run(), half)


def test_block_trajectories(rig):
    ad, run = rig
    on(ad, "pan-left"); a = run()
    on(ad, "zoom-in"); b = run()
    on(ad, "pan-left", block_trajectories={"up_blocks.0": "zoom-in"}); mixed = run()
    assert not torch.allclose(mixed, a, atol=1e-5) and not torch.allclose(mixed, b, atol=1e-5)
    on(ad, "pan-left", block_trajectories={"": "zoom-in"}); assert torch.equal(run(), b)


def test_wrong_size_raises_instead_of_mixing_tokens(rig):
    ad, run = rig
    ad.set(CameraSpec("pan-left"), 32, 32, F)        # features for a 16x16 latent, clip has 8x8
    with pytest.raises(ValueError, match="do not fit"):
        run(steps=1)
    with pytest.raises(ValueError, match="frames"):
        ad.set(CameraSpec(trajectory("pan-left", F + 1)), HW, HW, F)


def test_features_are_cached_and_cache_clears(rig):
    ad, _ = rig
    f1 = ad.features("pan-left", HW, HW, F)
    assert ad.features("pan-left", HW, HW, F) is f1
    assert ad.features("pan-right", HW, HW, F) is not f1
    ad.clear_cache()
    assert ad.features("pan-left", HW, HW, F) is not f1


def test_state_dict_holds_only_encoder_and_merges(rig, tiny_cc_unet):
    ad, _ = rig
    assert all(k.startswith(("encoder.", "merges.")) for k in ad.state_dict())
    assert not any(p is q for p in ad.parameters() for q in tiny_cc_unet.parameters())


def test_gradients_reach_the_merge_layers(rig):
    ad, run = rig
    on(ad)
    out = run.unet(run.sample, torch.tensor(500), encoder_hidden_states=run.text).sample
    out.pow(2).sum().backward()
    assert all(lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0 for lin in ad.merges.values())
    ad.zero_grad()
