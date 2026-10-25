"""Harness tests on a tiny CPU UNetMotionModel: coordinates, layouts, summaries, attention maps."""

import torch

from vdprobe import attention as A
from vdprobe.coords import (canonical, coarse_coordinates, count_by_kind, enumerate_coordinates,
                            spatial_hw)
from vdprobe.harness import Harness, ShapeAccountant, SummaryRecorder
from vdprobe.patching import TensorCache
from vdprobe.summaries import channel_stats

TOKENS = {"subject": [1, 2], "_all": [1, 2, 3, 4]}


def test_coordinates_match_architecture(tiny_unet):
    coords = enumerate_coordinates(tiny_unet)
    # 4 spatial transformers (down0, mid, 2x up1) and 7 motion modules (1+1+1+2+2)
    assert count_by_kind(coords) == {"spatial_attn1": 4, "spatial_attn2": 4, "temporal_attn1": 7,
                                     "temporal_attn2": 7, "temporal_module": 7}
    levels = {c.block: c.level for c in coords}
    assert levels == {"down_blocks.0": 0, "down_blocks.1": 1, "mid_block": 1, "up_blocks.0": 1, "up_blocks.1": 0}
    coarse = coarse_coordinates(coords)
    assert sum(c.kind == "temporal_module" for c in coarse) == 5      # one per block
    assert sum(c.kind == "spatial_attn2" for c in coarse) == 3        # one per block with attention
    assert "up_blocks.1.motion_modules.1" in {c.name for c in coarse}  # the last one in the block


def test_step_counter_and_branches(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    acct = ShapeAccountant()
    with Harness(tiny_unet, coords, [acct]) as ctx:
        tiny_run()(steps=3)
    assert ctx.step == 2 and ctx.n_branches == 2 and ctx.frames == 4 and ctx.cond_branch == 1
    assert len(acct.records) == 3 * len(coords)
    full, cond = sum(acct.bytes_by_kind().values()), sum(acct.bytes_by_kind(cond_only=True).values())
    assert full == 2 * cond


def test_canonical_layouts_agree(tiny_unet, tiny_run):
    """The same (frame, position) must land in the same canonical cell whichever layout it came from:
    a motion module's input equals the spatial path's output just before it, so check the
    temporal attention layout against the motion module layout via the module's own reshape."""
    F, B = 4, 2
    x = torch.arange(B * F * 3 * 2 * 2, dtype=torch.float32).reshape(B * F, 3, 2, 2)  # (B*F, C, H, W)
    as_module = canonical(x, "temporal_module", B, F)
    # AnimateDiffTransformer3D: (B*F,C,H,W) -> (B*H*W, F, C)
    t = x.reshape(B, F, 3, 2, 2).permute(0, 3, 4, 1, 2).reshape(B * 4, F, 3)
    assert torch.equal(canonical(t, "temporal_attn1", B, F), as_module)
    # Transformer2DModel: (B*F,C,H,W) -> (B*F, H*W, C)
    s = x.permute(0, 2, 3, 1).reshape(B * F, 4, 3)
    assert torch.equal(canonical(s, "spatial_attn1", B, F), as_module)


def test_summary_store_shapes_and_values(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    rec = SummaryRecorder(grid=4)
    probe_coord = next(c for c in coords if c.kind == "temporal_module")
    cache = TensorCache([probe_coord], steps=[1], mode="cond")
    with Harness(tiny_unet, coords, [rec, cache], token_index=TOKENS):
        tiny_run()(steps=3)
    t = rec.tensors()
    n_attn = sum(c.is_attention for c in coords)
    assert len(t) == 3 * len(coords) + n_attn
    for c in coords:
        assert t[f"{c.name}::resnorm"].shape == (3, 4)
        assert t[f"{c.name}::mean"].shape[0] == 3
        if c.is_attention:
            assert t[f"{c.name}::attn"].shape == (3, 4, 4, 4)
    ref = channel_stats(cache.store[(probe_coord.name, 1)])
    for k in ("mean", "var", "resnorm"):
        assert torch.allclose(t[f"{probe_coord.name}::{k}"][1], ref[k])


def test_attention_maps_are_valid(tiny_unet, tiny_run):
    coords = enumerate_coordinates(tiny_unet)
    rec = SummaryRecorder(grid=4)
    with Harness(tiny_unet, coords, [rec], token_index=TOKENS):
        tiny_run()(steps=1)
    t = rec.tensors()
    for c in coords:
        if not c.is_attention:
            continue
        m = t[f"{c.name}::attn"]
        assert torch.isfinite(m).all() and (m >= -1e-6).all() and (m <= 1 + 1e-6).all(), c.name
        if c.kind == "spatial_attn1":  # attention received, averaged: mean over positions is 1/S
            h, w = spatial_hw(c.level, (8, 8))
            assert torch.allclose(m.mean(dim=(-1, -2)), torch.full_like(m.mean(dim=(-1, -2)), 1 / (h * w)), atol=1e-5)


def _capture(module):
    store = {}

    def hook(mod, args, kwargs, out):
        store.update(hs=args[0] if args else kwargs["hidden_states"],
                     ehs=kwargs.get("encoder_hidden_states"), out=out)
    return store, module.register_forward_hook(hook, with_kwargs=True)


def test_reconstructed_probabilities_reproduce_module_output(tiny_unet, tiny_run):
    """probs @ v through to_out must equal what the attention module actually returned."""
    coords = enumerate_coordinates(tiny_unet)
    modules = dict(tiny_unet.named_modules())
    for kind in ("spatial_attn1", "spatial_attn2", "temporal_attn1", "temporal_attn2"):
        c = next(x for x in coords if x.kind == kind)
        attn = modules[c.name]
        store, h = _capture(attn)
        tiny_run()(steps=1)
        h.remove()
        p = A.attention_probs(attn, store["hs"], store["ehs"])
        ctx = store["hs"] if store["ehs"] is None else store["ehs"]
        v = attn.to_v(ctx)
        v = v.view(v.shape[0], -1, attn.heads, v.shape[-1] // attn.heads).transpose(1, 2)
        o = (p @ v).transpose(1, 2).reshape(store["hs"].shape[0], -1, attn.heads * v.shape[-1])
        assert torch.allclose(attn.to_out[0](o), store["out"], atol=1e-5), kind


def test_cross_attention_maps(tiny_unet, tiny_run):
    from vdprobe.harness import CrossAttentionMaps

    coords = enumerate_coordinates(tiny_unet)
    n_cross = sum(c.kind == "spatial_attn2" for c in coords)
    rec = CrossAttentionMaps(["subject", "_all"], size=16, grid=4)
    run = tiny_run()
    with Harness(tiny_unet, coords, [rec], token_index=TOKENS):
        run(steps=2)
    sub, everything = rec.total["subject"], rec.total["_all"]
    assert sub.shape == (run.F, 16, 16) and (sub >= 0).all()
    assert (sub <= everything + 1e-5).all()                         # a token subset gets at most the total
    # the 6 tokens of TinyRun hold all the mass, and "_all" here is 4 of them: < 1 per position per call
    assert everything.max() <= 2 * n_cross + 1e-3
    assert len(rec.cells) == 2 * n_cross
    h = rec.heatmap("subject")
    assert h.max() == 1 and rec.mask("subject").any()
    assert 0.1 <= rec.concentration(h[0]) <= 1.0


def test_temporal_head_stats_bounds(tiny_unet, tiny_run):
    import math

    coords = [c for c in enumerate_coordinates(tiny_unet) if c.kind == "temporal_attn1"]
    module = dict(tiny_unet.named_modules())[coords[0].name]
    store, h = _capture(module)
    tiny_run()(steps=1)
    h.remove()
    hs = store["hs"][: store["hs"].shape[0] // 2]                  # one CFG branch, (S, F, C)
    st = A.temporal_head_stats(module, hs, row_chunk=7)
    F_ = hs.shape[1]
    assert st.shape == (module.heads, 3)
    assert (st[:, 0] >= 0).all() and (st[:, 0] <= math.log(F_) + 1e-5).all()
    assert (st[:, 1] >= 0).all() and (st[:, 1] <= 1).all()
    assert (st[:, 2] >= 0).all() and (st[:, 2] <= F_ - 1).all()
    p = A.attention_probs(module, hs).float()                    # direct check of the mean entropy
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1).mean(dim=(0, 2))
    assert torch.allclose(st[:, 0], ent, atol=1e-5)


def test_resnet_coordinates_and_exp2_probes(tiny_unet, tiny_run):
    from vdprobe.coords import RESNET, in_blocks
    from vdprobe.lockstep import ChannelStats, HeadStats, Sketcher

    coords = enumerate_coordinates(tiny_unet, resnets=True)
    rn = [c for c in coords if c.kind == RESNET]
    assert len(rn) == len([n for n, _ in tiny_unet.named_modules() if n.split(".")[-2:-1] == ["resnets"]])
    assert len(coords) == len(enumerate_coordinates(tiny_unet)) + len(rn)
    refined = in_blocks(coords, ["up_blocks.1"])
    assert refined and all(c.block == "up_blocks.1" for c in refined) and any(c.kind == RESNET for c in refined)
    chan, head = ChannelStats(refined), HeadStats(coords)
    sk = Sketcher(refined, steps=range(2), dim=16, parts=("tilde",))
    with Harness(tiny_unet, coords, [chan, head, sk]):
        tiny_run()(steps=2)
    for c in refined:
        st = chan.out[(c.name, 1)]
        assert st.shape[0] == 2 and (st[1] >= 0).all()                 # dynamic energy is a mean square
        assert sk.out[(c.name, 0)].shape == (1, 16)
    t_attn = [c for c in coords if c.kind.startswith("temporal_attn")]
    assert {k[0] for k in head.out} == {c.name for c in t_attn}
