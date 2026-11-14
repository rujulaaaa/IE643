"""CameraCtrl checkpoint conversion and the domain-adapter LoRA merge."""
import json
from pathlib import Path

import pytest
import torch

from conftest import TinyRun, make_tiny_adapter
from vdprobe.cameractrl import convert
from vdprobe.cameractrl.adapter import merge_key

KEYS = json.loads((Path(__file__).parent / "data" / "cameractrl_keys.json").read_text())


def authors_key(site, leaf):
    mm = site[:site.index(".transformer_blocks")]
    return f"{mm}.temporal_transformer.transformer_blocks.0.attention_blocks.0.processor.qkv_merge.{leaf}"


def test_every_real_processor_key_maps_to_a_v3_site():
    sites = {convert.site_of(k)[0] for k in KEYS["attention_processor_state_dict"]}
    expect = {f"down_blocks.{i}.motion_modules.{j}.transformer_blocks.0.attn1" for i in range(4) for j in range(2)} | \
             {f"up_blocks.{i}.motion_modules.{j}.transformer_blocks.0.attn1" for i in range(4) for j in range(3)}
    assert sites == expect and len(KEYS["attention_processor_state_dict"]) == 40


def test_real_merge_widths_follow_the_feature_levels():
    widths = {"down_blocks.0": 320, "down_blocks.1": 640, "down_blocks.2": 1280, "down_blocks.3": 1280,
              "up_blocks.0": 1280, "up_blocks.1": 1280, "up_blocks.2": 640, "up_blocks.3": 320}
    for k, shape in KEYS["attention_processor_state_dict"].items():
        site, leaf = convert.site_of(k)
        w = widths[site[:site.index(".motion_modules")]]
        assert shape == ([w, w] if leaf == "weight" else [w])


def test_stray_key_raises():
    with pytest.raises(KeyError):
        convert.site_of("down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor.qkv_merge.weight")
    with pytest.raises(KeyError):
        convert.site_of("down_blocks.0.motion_modules.0.temporal_transformer.transformer_blocks.0.attention_blocks.1"
                        ".processor.qkv_merge.weight")


def test_checkpoint_round_trip(tiny_cc_unet):
    src = make_tiny_adapter(tiny_cc_unet, seed=3)
    ckpt = {"pose_encoder_state_dict": src.encoder.state_dict(),
            "attention_processor_state_dict": {authors_key(s, leaf): getattr(src.merges[merge_key(s)], leaf).detach()
                                               for s in src.sites for leaf in ("weight", "bias")},
            "epoch": 12, "global_step": 25000, "optimizer_state_dict": {}}
    dst = make_tiny_adapter(tiny_cc_unet, randomise=False, seed=4)
    dst.features("pan-left", 16, 16, TinyRun.F)
    convert.load_into(dst, ckpt)
    assert dst._cache == {}                                         # features of the old weights are dropped
    a, b = src.state_dict(), dst.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_load_is_strict(tiny_cc_unet):
    ad = make_tiny_adapter(tiny_cc_unet)
    ckpt = {"pose_encoder_state_dict": ad.encoder.state_dict(), "attention_processor_state_dict": {}}
    with pytest.raises(RuntimeError):
        convert.load_into(ad, ckpt)


def _lora(unet, rank=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    sd = {}
    for name, p in unet.named_parameters():
        if "motion_modules" in name or ".attentions." not in name or not name.endswith(".weight"):
            continue
        for proj, leaf in (("to_q", "to_q"), ("to_k", "to_k"), ("to_v", "to_v"), ("to_out", "to_out.0")):
            if name.endswith(f".{leaf}.weight") and (".attn1." in name or ".attn2." in name):
                path = name[:-len(f".{leaf}.weight")]
                sd[f"{path}.processor.{proj}_lora.down.weight"] = torch.randn(rank, p.shape[1], generator=g)
                sd[f"{path}.processor.{proj}_lora.up.weight"] = torch.randn(p.shape[0], rank, generator=g)
    return sd


def test_real_domain_lora_keys_parse():
    assert len(KEYS["domain_lora"]) == 256
    assert all(convert._LORA.match(k) for k in KEYS["domain_lora"])


def test_merge_domain_lora(tiny_cc_unet):
    import copy
    unet = copy.deepcopy(tiny_cc_unet)
    before = {k: v.clone() for k, v in unet.state_dict().items()}
    sd = _lora(unet)
    n = convert.merge_domain_lora(unet, sd, scale=0.5)
    assert n == len(sd) // 2 > 0
    after = unet.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert len(changed) == n and not any("motion_modules" in k for k in changed)
    k = next(iter(sorted(changed)))
    path, leaf = k[:-len(".weight")].rsplit(".to_", 1)
    proj = "to_" + leaf.split(".")[0]
    delta = sd[f"{path}.processor.{proj}_lora.up.weight"] @ sd[f"{path}.processor.{proj}_lora.down.weight"]
    assert torch.allclose(after[k], before[k] + 0.5 * delta, atol=1e-6)


def test_merge_domain_lora_rejects_unknown_targets(tiny_cc_unet):
    import copy
    unet = copy.deepcopy(tiny_cc_unet)
    sd = _lora(unet)
    k = next(iter(sd))
    bad = dict(sd)
    bad["up_blocks.9." + k] = bad.pop(k)
    with pytest.raises(KeyError):
        convert.merge_domain_lora(unet, bad)
    with pytest.raises(KeyError):
        convert.merge_domain_lora(unet, {"down_blocks.0.motion_modules.0.transformer_blocks.0.attn1.processor.to_q_lora.down.weight": torch.zeros(1, 1)})
    with pytest.raises(KeyError):
        convert.merge_domain_lora(unet, {"something.else": torch.zeros(1)})
