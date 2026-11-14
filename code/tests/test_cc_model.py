"""CameraCtrl in the lockstep loop: per-track camera, switching during denoising, no leak between tracks."""
from types import SimpleNamespace

import pytest
import torch
from diffusers import DDIMScheduler

from conftest import make_tiny_adapter
from vdprobe.cameractrl import model as M
from vdprobe.cameractrl.adapter import CameraSpec
from vdprobe.cameractrl.rays import CAMERAS, trajectory
from vdprobe.config import GenerationConfig
from vdprobe.lockstep import Track, run_lockstep

F_, STEPS = 4, 4


@pytest.fixture
def rig(tiny_cc_unet, monkeypatch):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)      # CPU-only test
    cfg = SimpleNamespace(steps=STEPS, guidance_scale=7.5, frames=F_, height=16, width=16,
                          extra={"trajectory_step": 0.1, "trajectory_roll_deg": 2.0})
    loaded = SimpleNamespace(pipe=SimpleNamespace(unet=tiny_cc_unet, scheduler=DDIMScheduler()), unet=tiny_cc_unet,
                             cfg=cfg, device="cpu", extras={})
    g = torch.Generator().manual_seed(0)
    emb = torch.randn(2, 6, 16, generator=g).repeat_interleave(F_, dim=0)
    noise = torch.randn(1, 4, F_, 8, 8, generator=g)

    def run(cameras, switch=True):
        tracks = [Track(f"t{i}", "p", embeds=emb.clone(), meta={} if c == "absent" else {"camera": c})
                  for i, c in enumerate(cameras)]
        run_lockstep(loaded, [], tracks, noise, before_track=M.camera_switcher(loaded) if switch else None)
        return [t.latents for t in tracks]

    plain = run(["absent"], switch=False)[0]            # before any adapter exists
    ad = make_tiny_adapter(tiny_cc_unet)
    ad.attach()
    loaded.extras["camera"] = ad
    yield SimpleNamespace(run=run, plain=plain, adapter=ad, loaded=loaded, cfg=cfg)
    ad.detach()


def test_off_track_equals_run_without_adapter(rig):
    for off in (None, "off", "none", "absent"):
        assert torch.equal(rig.run([off])[0], rig.plain)


def test_no_leak_from_a_camera_track_into_an_off_track(rig):
    pan = rig.run(["pan-left"])[0]
    assert not torch.allclose(pan, rig.plain, atol=1e-4)
    for order in (["pan-left", None], [None, "pan-left"], ["pan-left", "absent", "zoom-in"]):
        out = dict(zip(order, rig.run(order)))
        assert torch.equal(out["pan-left"], pan)
        off = out[None] if None in out else out["absent"]
        assert torch.equal(off, rig.plain)


def test_spec_name_and_trajectory_are_equivalent(rig):
    a, b, c = rig.run(["roll-cw", CameraSpec("roll-cw"), M.named_trajectory(rig.cfg, "roll-cw")])
    assert torch.equal(a, b) and torch.equal(a, c)


def test_config_magnitudes_are_used(rig):
    a = rig.run(["pan-left"])[0]
    rig.cfg.extra["trajectory_step"] = 0.3
    assert not torch.allclose(rig.run(["pan-left"])[0], a, atol=1e-5)
    assert torch.equal(M.named_trajectory(rig.cfg, "pan-left").c2w, trajectory("pan-left", F_, step=0.3).c2w)


def test_switching_during_denoising(rig):
    full = rig.run(["pan-left"])[0]
    early, late, never, always = rig.run([M.on_until("pan-left", 2), M.on_from("pan-left", 2),
                                          M.on_until("pan-left", 0), M.on_until("pan-left", STEPS)])
    assert torch.equal(never, rig.plain) and torch.equal(always, full)
    for x in (early, late):
        assert not torch.allclose(x, full, atol=1e-5) and not torch.allclose(x, rig.plain, atol=1e-5)
    assert not torch.allclose(early, late, atol=1e-5)


def test_step_function_may_change_the_trajectory(rig):
    a, b = rig.run(["pan-left", "zoom-in"])
    mixed = rig.run([lambda s: "pan-left" if s < 2 else "zoom-in"])[0]
    assert not torch.allclose(mixed, a, atol=1e-5) and not torch.allclose(mixed, b, atol=1e-5)


def test_resolve_rejects_garbage(rig):
    with pytest.raises(TypeError):
        M.resolve(rig.cfg, 3)
    with pytest.raises(KeyError):
        rig.run(["dolly-zoom"])


def test_attach_adapter_registers_and_matches_dtype(tiny_cc_unet):
    from vdprobe.cameractrl.pose_encoder import CameraPoseEncoder
    loaded = SimpleNamespace(unet=tiny_cc_unet, extras={})
    enc = CameraPoseEncoder(channels=(32, 64), cin=24, downscale_factor=2, heads=2, max_len=8)
    ad = M.attach_adapter(loaded, encoder=enc)
    try:
        assert loaded.extras["camera"] is ad and len(ad._handles) == len(ad.sites)
        assert not any(p.requires_grad for p in ad.parameters())
    finally:
        ad.detach()


def test_config_file():
    cfg = M.cameractrl_config()
    assert (cfg.frames, cfg.height, cfg.width, cfg.steps, cfg.guidance_scale) == (16, 256, 384, 25, 14.0)
    assert cfg.adapter_id.endswith("v1-5-3") and cfg.latent_shape == (1, 4, 16, 32, 48)
    assert cfg.extra["cameractrl_file"] == "CameraCtrl.ckpt" and cfg.extra["domain_lora_file"] == "v3_sd15_adapter.ckpt"
    assert cfg.prompt_template.format(scene="a lake") == "a lake, high quality, detailed"
    assert GenerationConfig.from_yaml().height == 512            # the frozen config is untouched
    assert all(M.named_trajectory(cfg, c).frames == 16 for c in CAMERAS)
