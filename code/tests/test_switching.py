"""D-38: probe-timed prompt switching and its subject-masked variant, on a tiny CPU UNet."""

from types import SimpleNamespace

import pytest
import torch
from diffusers import DDIMScheduler

from vdprobe import switching
from vdprobe.coords import enumerate_coordinates
from vdprobe.lockstep import Track, run_lockstep
from vdprobe.switching import SwitchSpec, run_switch, switch_arms

F_, T_ = 4, 6


@pytest.fixture
def fake(tiny_unet, monkeypatch):
    texts = {}

    def encode(loaded, prompt):          # one fixed random embedding per prompt, [neg, pos] per frame
        if prompt not in texts:
            g = torch.Generator().manual_seed(len(texts) + 1)
            texts[prompt] = torch.randn(2, T_, 16, generator=g).repeat_interleave(F_, dim=0)
        return texts[prompt]

    monkeypatch.setattr(switching, "encode", encode)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)      # CPU-only test
    monkeypatch.setattr(switching, "phrase_token_positions", lambda tok, prompt, phrases: {"subject": [1, 2]})
    cfg = SimpleNamespace(steps=4, guidance_scale=7.5, frames=F_)
    loaded = SimpleNamespace(pipe=SimpleNamespace(unet=tiny_unet, scheduler=DDIMScheduler(), tokenizer=None),
                             unet=tiny_unet, cfg=cfg, device="cpu")
    noise = torch.randn(1, 4, F_, 8, 8, generator=torch.Generator().manual_seed(7))
    return loaded, enumerate_coordinates(tiny_unet), noise, encode


def _solo(loaded, noise, emb):
    tr = Track("x", "p", embeds=emb)
    run_lockstep(loaded, [], [tr], noise)
    return tr.latents


def test_switch_limits_and_source_untouched(fake):
    loaded, coords, noise, encode = fake
    arms = {"motion.switch.t0": SwitchSpec("A", "M", 0), "motion.switch.t9": SwitchSpec("A", "M", 9),
            "motion.switch.t2": SwitchSpec("A", "M", 2)}
    r = run_switch(loaded, coords, "A", "dog", arms, noise)
    a, m = _solo(loaded, noise, encode(loaded, "A")), _solo(loaded, noise, encode(loaded, "M"))
    assert torch.equal(r.source_latents, a)                         # recording A is read-only
    assert torch.equal(r.latents["motion.switch.t0"], m)            # switch at 0 = the late prompt alone
    assert torch.equal(r.latents["motion.switch.t9"], a)            # never switching = A
    mid = r.latents["motion.switch.t2"]
    assert not torch.equal(mid, a) and not torch.equal(mid, m)
    assert r.mask is None


def test_masked_switch_keeps_a_outside_the_mask(fake, monkeypatch):
    loaded, coords, noise, encode = fake
    half = torch.zeros(1, 1, F_, 8, 8)
    half[..., :, 4:] = 1                       # the toy UNet's attention is near uniform, so fix the mask
    seen = {}

    def fixed_mask(rec, phrase, hw, tau, dilate):
        seen["total"] = rec.total[phrase]
        return half

    monkeypatch.setattr(switching, "subject_mask", fixed_mask)
    arms = {"motion.switch.t1": SwitchSpec("A", "M", 1), "motion.switchmask.t1": SwitchSpec("A", "M", 1, masked=True)}
    r = run_switch(loaded, coords, "A", "dog", arms, noise)
    m = r.mask
    assert seen["total"].shape[0] == F_                             # A's DAAM map was recorded
    out, a = r.latents["motion.switchmask.t1"], r.source_latents
    outside = (m.expand_as(out) == 0)
    assert torch.equal(out[outside], a[outside])                    # background and camera are A's
    assert not torch.equal(out, a)                                  # the subject region changed


def test_switch_arm_set():
    p = {"A": "a dog running", "motion": "a horse running", "content": "a dog jumping"}
    arms = switch_arms(p)
    assert arms["motion.switch.t2"] == SwitchSpec(p["A"], p["motion"], 2)
    assert arms["content.switch.t2"] == SwitchSpec(p["content"], p["A"], 2)
    assert arms["motion.switchmask.t2"].masked and set(arms) >= {"motion.switchmask.t3", "content.switch.t8"}
    assert all(k.split(".")[0] in ("motion", "content") for k in arms)


def test_subject_mask_dilates_and_resizes():
    rec = SimpleNamespace(mask=lambda phrase, tau: torch.zeros(2, 16, 16, dtype=torch.bool).index_put_(
        (torch.tensor([0]), torch.tensor([8]), torch.tensor([8])), torch.tensor(True)))
    m = switching.subject_mask(rec, "subject", (16, 16), dilate=3)
    assert m.shape == (1, 1, 2, 16, 16) and m[0, 0, 0].sum() == 9 and m[0, 0, 1].sum() == 0
    assert switching.subject_mask(rec, "subject", (8, 8), dilate=1).shape == (1, 1, 2, 8, 8)
