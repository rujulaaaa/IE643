"""Motion LoRA targeting, save/load round trip, ‖ΔW‖, and per-call enable/disable (tiny CPU UNet)."""

import copy

import torch

from vdprobe.lora import add_motion_lora, delta_w_norms, load_motion_lora, save_motion_lora


def _out(unet, run):
    with torch.no_grad():
        return unet(run.sample, torch.tensor(500), encoder_hidden_states=run.text).sample


def _perturb(unet):
    g = torch.Generator().manual_seed(3)
    for n, p in unet.named_parameters():
        if "lora_B" in n:
            p.data = torch.randn(p.shape, generator=g) * 0.1


def test_targets_are_temporal_projections_only(tiny_unet):
    u = copy.deepcopy(tiny_unet)
    add_motion_lora(u, "full", rank=4, alpha=4)
    adapted = {n.split(".lora_A")[0] for n, _ in u.named_parameters() if ".lora_A." in n}
    assert adapted and all("motion_modules" in n and n.split(".")[-1] in ("to_q", "to_k", "to_v", "0") for n in adapted)
    u2 = copy.deepcopy(tiny_unet)
    add_motion_lora(u2, "probe", rank=4, alpha=4)
    probe = {n.split(".lora_A")[0] for n, _ in u2.named_parameters() if ".lora_A." in n}
    assert probe and probe < adapted and all(n.startswith("up_blocks.0.") for n in probe)


def test_roundtrip_and_toggle(tiny_unet, tiny_run, tmp_path):
    run = tiny_run(0)
    base = _out(tiny_unet, run)
    u = copy.deepcopy(tiny_unet)
    add_motion_lora(u, "full", rank=4, alpha=4)
    _perturb(u)
    u = u.float()
    trained = _out(u, run)
    assert not torch.allclose(trained, base, atol=1e-4)
    path = tmp_path / "m.safetensors"
    save_motion_lora(u, path, {"variant": "full", "rank": 4, "alpha": 4})

    v = copy.deepcopy(tiny_unet)
    load_motion_lora(v, path)
    assert torch.allclose(_out(v, run), trained, atol=1e-2)       # weights saved in fp16
    v.disable_adapters()
    assert torch.allclose(_out(v, run), base, atol=1e-6)
    v.enable_adapters()
    v.set_adapters(["motion"], weights=[0.0])
    assert torch.allclose(_out(v, run), base, atol=1e-6)

    from safetensors.torch import load_file
    dw = delta_w_norms(load_file(str(path)), alpha=4, rank=4)
    assert dw and all(v > 0 for v in dw.values())
