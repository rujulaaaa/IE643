"""Stage 6b (D-37): spatial K/V injection (MasaCtrl) and injection from a second source track."""

import pytest
import torch

from vdprobe.config import GenerationConfig
from vdprobe.coords import enumerate_coordinates
from vdprobe.edit import (AttnController, AttnSpec, Arm, EditSpec, KVInjectingProcessor, KVSpec, _recording_specs,
                          attention_injection, temporal_attention_modules)
from vdprobe.edit_plan import (KV_BLOCKS, literature_arms, literature_specs, literature_test_arms, load_pairs,
                               spatial_self_attention)


def _spatial_self(unet):
    return spatial_self_attention(enumerate_coordinates(unet))


def _fwd(unet, run, ctl, mode, track, step=0):
    ctl.mode, ctl.step, ctl.track = mode, step, track
    with torch.no_grad():
        return unet(run.sample, torch.tensor(999), encoder_hidden_states=run.text).sample


def _stock(unet, run):
    with torch.no_grad():
        return unet(run.sample, torch.tensor(999), encoder_hidden_states=run.text).sample


def test_kv_record_is_read_only_and_self_injection_is_exact(tiny_unet, tiny_run):
    mods = _spatial_self(tiny_unet)
    assert len(mods) > 0
    run = tiny_run(0)
    stock = _stock(tiny_unet, run)
    ctl = AttnController(KVSpec(mods, steps=(0,)))
    ctl.frames = run.F
    with attention_injection(tiny_unet, ctl, KVInjectingProcessor):
        assert torch.equal(_fwd(tiny_unet, run, ctl, "record", "source"), stock)
        assert len(ctl.store) == len(mods)
        assert torch.equal(_fwd(tiny_unet, run, ctl, "inject", "target"), stock)   # own K/V: stock output
        assert ctl.applied == len(mods)
    assert torch.equal(_stock(tiny_unet, run), stock)                                 # processors restored


def test_kv_injection_moves_the_target(tiny_unet, tiny_run):
    mods = _spatial_self(tiny_unet)
    a, b = tiny_run(0), tiny_run(1)
    b_plain = _stock(tiny_unet, b)
    outs = {}
    for frames in ("framewise", "first"):
        ctl = AttnController(KVSpec(mods, steps=(0,), frames=frames))
        ctl.frames = a.F
        with attention_injection(tiny_unet, ctl, KVInjectingProcessor):
            _fwd(tiny_unet, a, ctl, "record", "source")
            outs[frames] = _fwd(tiny_unet, b, ctl, "inject", "target")
            assert torch.equal(_fwd(tiny_unet, b, ctl, "inject", "target", step=3), b_plain)  # inactive step
    assert not torch.allclose(outs["framewise"], b_plain, atol=1e-4)
    assert not torch.allclose(outs["first"], outs["framewise"], atol=1e-4)
    with pytest.raises(ValueError):
        KVSpec(mods, (0,), frames="last")


def test_temporal_injection_reads_the_named_source(tiny_unet, tiny_run):
    mods = temporal_attention_modules(enumerate_coordinates(tiny_unet))
    a, b = tiny_run(0), tiny_run(1)
    res = {}
    for src in ("source", "B_ref"):
        ctl = AttnController(AttnSpec(mods, steps=(0,), source=src))
        with attention_injection(tiny_unet, ctl):
            _fwd(tiny_unet, a, ctl, "record", src)
            _fwd(tiny_unet, b, ctl, "record", "other")       # a recording under another id is not read
            ctl.spec = AttnSpec(mods, steps=(0,), source=src)
            res[src] = _fwd(tiny_unet, b, ctl, "inject", "target")
    assert torch.equal(res["source"], res["B_ref"])


def test_recording_specs_check_the_sources():
    kv = KVSpec(("m",), (0,), source="B_ref")
    ok = [Arm("B_ref", "p", EditSpec()), Arm("x", "p", EditSpec(kv=kv)), Arm("y", "p", EditSpec(kv=KVSpec(("n",), (1,))))]
    rec = _recording_specs(ok, "kv")
    assert set(rec) == {"B_ref", "source"} and rec["B_ref"].modules == ("m",) and rec["source"].steps == (1,)
    assert _recording_specs(ok, "attn") == {}
    with pytest.raises(ValueError):                         # source runs after the arm reading it
        _recording_specs([Arm("x", "p", EditSpec(kv=kv)), Arm("B_ref", "p", EditSpec())], "kv")
    with pytest.raises(ValueError):                         # a source that is itself edited
        _recording_specs([Arm("B_ref", "p", EditSpec(kv=KVSpec(("m",), (0,)))), Arm("x", "p", EditSpec(kv=kv))], "kv")


@pytest.fixture(scope="module")
def sd_layout_coords():
    """SD1.5's block layout (cross-attention in up_blocks.1-3) at toy width."""
    from diffusers import UNetMotionModel

    u = UNetMotionModel(block_out_channels=(32, 32, 32, 32), layers_per_block=1,
                        down_block_types=("CrossAttnDownBlockMotion",) * 3 + ("DownBlockMotion",),
                        up_block_types=("UpBlockMotion",) + ("CrossAttnUpBlockMotion",) * 3,
                        cross_attention_dim=16, num_attention_heads=2, motion_num_attention_heads=2,
                        motion_max_seq_length=8, sample_size=16)
    return enumerate_coordinates(u)


def test_literature_arms(sd_layout_coords):
    coords = sd_layout_coords
    cfg = GenerationConfig.from_yaml()
    pair = load_pairs("dev")[0]
    p = pair.prompts(cfg)
    specs = literature_specs(coords)
    arms = literature_arms(pair, cfg, coords)
    ids = [a.id for a in arms]
    assert ids[:3] == ["B_ref", "motion.base", "content.base"] and len(ids) == len(set(ids)) == 3 + len(specs)
    for a in arms[3:]:
        d = a.id.split(".")[0]
        assert a.prompt == p[d] and a.spec.label in ("single", "dual")
        srcs = {s.source for s in (a.spec.attn, a.spec.kv) if s is not None}
        assert srcs == ({"source", "B_ref"} if a.spec.label == "dual" else {"source"})
        if a.spec.kv is not None:
            assert a.spec.kv.modules and all(".attn1" in m and "motion_modules" not in m for m in a.spec.kv.modules)
    up23 = spatial_self_attention(coords, KV_BLOCKS["up23"])
    assert specs["content.kv.up23.from2"].kv.modules == up23 and min(specs["content.kv.up23.from2"].kv.steps) == 2
    pr = specs["motion.attn.probe.early"].attn
    assert pr.steps == (0, 1, 2) and pr.modules and all(m.startswith(("up_blocks.0.", "mid_block.")) for m in pr.modules)
    assert all(m.startswith("down_blocks.2.") for m in specs["content.kv.probe.from2"].kv.modules)
    # the content-direction dual arm takes A's appearance and B_ref's motion
    s = specs["content.dual.all"]
    assert s.kv.source == "source" and s.attn.source == "B_ref" and len(s.attn.steps) == cfg.steps
    _recording_specs(arms, "kv"), _recording_specs(arms, "attn")          # valid source order

    sel = {"motion": {"blocks": ["up_blocks.3"], "window": "ft"}, "content": {"blocks": ["down_blocks.3"], "window": "ft"}}
    lit = {"motion": {"single": "motion.attn.all.all", "dual": "motion.dual.up23.from2"},
           "content": {"single": "content.kv.up23.from2", "dual": "content.dual.all"}, "dev_median_dD": {}}
    t = [a.id for a in literature_test_arms(pair, cfg, coords, sel, lit)]
    assert set(t) == {"B_ref", "motion.base", "content.base", "motion.attn.all.all", "motion.dual.up23.from2",
                      "content.kv.up23.from2", "content.dual.all", "motion.sel", "content.sel", "motion.attn.all.ft"}


class _FakePipe:
    """Just enough of AnimateDiffPipeline for run_lockstep: the tiny UNet, DDIM, a prompt -> embedding table."""

    def __init__(self, unet):
        from diffusers import DDIMScheduler

        self.unet, self.scheduler = unet, DDIMScheduler()

    def encode_prompt(self, prompt, device, n, cfg, negative):
        emb = lambda s: torch.randn(1, 6, 16, generator=torch.Generator().manual_seed(sum(map(ord, s))))
        return emb(prompt), emb(negative or "")


def test_run_arms_routes_each_operator_to_its_source(live_unet_kv, monkeypatch):
    from types import SimpleNamespace

    from vdprobe.edit import run_arms
    from vdprobe.pipeline import Loaded

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: None)
    unet = live_unet_kv
    cfg = SimpleNamespace(frames=4, steps=3, guidance_scale=7.5, negative_prompt="")
    loaded = Loaded(_FakePipe(unet), cfg, "test", "tiny", {}, 0.0, device="cpu")
    coords = enumerate_coordinates(unet)
    kvm, tam = _spatial_self(unet), temporal_attention_modules(coords)
    steps = (0, 1, 2)
    arms = [Arm("B_ref", "a fox jumping", EditSpec()),
            Arm("self_kv", "a dog running", EditSpec(kv=KVSpec(kvm, steps))),                 # A's prompt + A's own K/V
            Arm("self_attn", "a fox jumping", EditSpec(attn=AttnSpec(tam, steps, source="B_ref"))),  # B's prompt + B's attn
            Arm("dual", "a dog jumping", EditSpec(kv=KVSpec(kvm, steps), attn=AttnSpec(tam, steps, source="B_ref"))),
            Arm("plain", "a dog jumping", EditSpec())]
    noise = torch.randn(1, 4, 4, 8, 8, generator=torch.Generator().manual_seed(0))
    r = run_arms(loaded, coords, "a dog running", arms, noise, decode_frames=False, cache_device="cpu")
    assert torch.equal(r.latents["self_kv"], r.source_latents)       # K/V read from the source track
    assert torch.equal(r.latents["self_attn"], r.latents["B_ref"])   # temporal attention read from B_ref
    assert not torch.allclose(r.latents["dual"], r.latents["plain"], atol=1e-5)
    assert r.n_kv_injected["dual"] == len(kvm) * len(steps) and r.n_kv_injected["B_ref"] == 0
    assert r.n_attn_injected["dual"] == len(tam) * len(steps) and r.n_attn_injected["self_kv"] == 0


@pytest.fixture(scope="module")
def live_unet_kv(tiny_unet):
    import copy

    u = copy.deepcopy(tiny_unet)
    g = torch.Generator().manual_seed(1)
    for name, m in u.named_modules():
        if name.endswith("proj_out") and "motion_modules" in name:
            m.weight.data = torch.randn(m.weight.shape, generator=g) * 0.5
    return u


def test_demo_lit_methods(sd_layout_coords):
    from vdprobe.demo import EditRequest, build_arms, lit_selection

    cfg = GenerationConfig.from_yaml()
    for d in ("motion", "content"):
        for m in ("lit_single", "lit_dual"):
            src, arms = build_arms(EditRequest("fox", "running", "wolf", "jumping", "in a forest", 1, d, m), cfg,
                                   sd_layout_coords, sel={})
            assert [a.id for a in arms] == ["B_ref", "base", "edit"]
            assert lit_selection()[d][m[4:]] in arms[-1].spec.label
            _recording_specs(arms, "kv"), _recording_specs(arms, "attn")
    with pytest.raises(ValueError):
        build_arms(EditRequest("fox", "running", "wolf", "jumping", "in a forest", 1, "motion", "lit_dual",
                               with_reference=False), cfg, sd_layout_coords, sel={})
