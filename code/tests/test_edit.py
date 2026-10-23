"""Granularity 3 (temporal attention-weight injection) on the tiny CPU UNetMotionModel."""

import copy

import pytest
import torch

from vdprobe.coords import enumerate_coordinates
from vdprobe.edit import AttnController, AttnSpec, attention_injection, channel_mask, temporal_attention_modules


@pytest.fixture(scope="module")
def live_unet(tiny_unet):
    """diffusers zero-initialises each motion module's proj_out, so in a random UNetMotionModel the
    temporal path contributes nothing; give those projections random weights so injection can show."""
    u = copy.deepcopy(tiny_unet)
    g = torch.Generator().manual_seed(1)
    for name, m in u.named_modules():
        if name.endswith("proj_out") and "motion_modules" in name:
            m.weight.data = torch.randn(m.weight.shape, generator=g) * 0.5
    return u


def _forward(unet, run, ctl, mode, step=0):
    ctl.mode, ctl.step = mode, step
    with torch.no_grad():
        return unet(run.sample, torch.tensor(999), encoder_hidden_states=run.text).sample


def test_recording_reproduces_stock_attention(tiny_unet, tiny_run):
    run = tiny_run(0)
    mods = temporal_attention_modules(enumerate_coordinates(tiny_unet))
    assert len(mods) > 0
    with torch.no_grad():
        stock = tiny_unet(run.sample, torch.tensor(999), encoder_hidden_states=run.text).sample
    ctl = AttnController(AttnSpec(mods, steps=(0,)))
    with attention_injection(tiny_unet, ctl):
        rec = _forward(tiny_unet, run, ctl, "record")
    assert len(ctl.store) == len(mods)
    assert torch.equal(rec, stock)          # recording is read-only (returns the stock output)
    # processors are restored on exit
    with torch.no_grad():
        again = tiny_unet(run.sample, torch.tensor(999), encoder_hidden_states=run.text).sample
    assert torch.equal(again, stock)


def test_self_injection_is_null_and_alpha_zero_is_noop(live_unet, tiny_run):
    tiny_unet = live_unet
    mods = temporal_attention_modules(enumerate_coordinates(tiny_unet))
    a, b = tiny_run(0), tiny_run(1)
    a.unet = b.unet = tiny_unet
    with torch.no_grad():                                          # stock processors, no injection
        b_plain = tiny_unet(b.sample, torch.tensor(999), encoder_hidden_states=b.text).sample
    ctl = AttnController(AttnSpec(mods, steps=(0,)))
    with attention_injection(tiny_unet, ctl):
        rec_a = _forward(tiny_unet, a, ctl, "record")
        self_inj = _forward(tiny_unet, a, ctl, "inject")
        assert torch.equal(self_inj, rec_a)                        # own weights: adds exactly 0
        _forward(tiny_unet, a, ctl, "record")
        b_inj = _forward(tiny_unet, b, ctl, "inject")              # A's weights drive B's values
        assert not torch.allclose(b_inj, b_plain, atol=1e-4)
    ctl0 = AttnController(AttnSpec(mods, steps=(0,), alpha=0.0))
    with attention_injection(tiny_unet, ctl0):
        _forward(tiny_unet, a, ctl0, "record")
        b_alpha0 = _forward(tiny_unet, b, ctl0, "inject")
    assert torch.equal(b_alpha0, b_plain)


def test_inactive_step_runs_stock(tiny_unet, tiny_run):
    mods = temporal_attention_modules(enumerate_coordinates(tiny_unet))
    run = tiny_run(0)
    with torch.no_grad():
        stock = tiny_unet(run.sample, torch.tensor(999), encoder_hidden_states=run.text).sample
    ctl = AttnController(AttnSpec(mods, steps=(5,)))
    with attention_injection(tiny_unet, ctl):
        out = _forward(tiny_unet, run, ctl, "inject", step=0)     # step 0 not in window: stock path
    assert torch.equal(out, stock) and ctl.applied == 0


def test_channel_mask():
    s = torch.tensor([0.5, -0.9, 0.1, -0.2, -0.8, 0.0, 0.3, -0.1, 0.2, 0.4])
    m = channel_mask(s, 0.2)
    assert m.tolist() == [0, 1, 0, 0, 1, 0, 0, 0, 0, 0]
