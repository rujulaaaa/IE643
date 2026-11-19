import copy

import pytest
import torch

from vdprobe.attribution import GradCache, atp_scores, atp_step


def _toy(nonlinear=False):
    torch.manual_seed(0)
    layers = [torch.nn.Linear(6, 8), torch.nn.Linear(8, 8), torch.nn.Linear(8, 5)]
    if nonlinear:
        layers = [layers[0], torch.nn.Tanh(), layers[1], torch.nn.Tanh(), layers[2]]
    return torch.nn.Sequential(*layers)


def _patched(model, site, x_b, act_a):
    h = dict(model.named_modules())[site].register_forward_hook(lambda m, a, o: act_a)
    try:
        with torch.no_grad():
            return model(x_b)
    finally:
        h.remove()


def test_linear_network_attribution_is_exact():
    model = _toy()
    xa, xb = torch.randn(3, 6), torch.randn(3, 6)
    sites = ["0", "1"]
    res = atp_step(model, sites, lambda: model(xa), lambda: model(xb))
    with torch.no_grad():
        eps_a, eps_b = model(xa), model(xb)
    d = eps_a - eps_b
    for site in sites:
        with torch.no_grad(), GradCache(model, [site]) as c:
            model(xa)
        exact = float((_patched(model, site, xb, c.acts[site]) * d).sum() - (eps_b * d).sum())
        assert res["scores"][site] == pytest.approx(exact, rel=1e-4, abs=1e-5)
    # patching the last hidden layer moves the prediction all the way to A: the fraction is 1
    assert res["frac"]["1"] == pytest.approx(1.0, rel=1e-4)
    assert res["gap"] == pytest.approx(float((d * d).sum()), rel=1e-6)


def test_nonlinear_first_order_and_same_input_is_zero():
    model = _toy(nonlinear=True)
    xb = torch.randn(3, 6)
    xa = xb + 0.01 * torch.randn(3, 6)                     # small change: first order is close
    res = atp_step(model, ["0", "2"], lambda: model(xa), lambda: model(xb))
    with torch.no_grad(), GradCache(model, ["2"]) as c:
        model(xa)
    d = (model(xa) - model(xb)).detach()
    exact = float((_patched(model, "2", xb, c.acts["2"]) * d).sum() - (model(xb) * d).sum())
    assert res["scores"]["2"] == pytest.approx(exact, rel=0.05)
    same = atp_step(model, ["0", "2"], lambda: model(xb), lambda: model(xb))
    assert all(v == 0.0 for v in same["scores"].values()) and same["gap"] == 0.0
    assert all(p.grad is None for p in model.parameters())          # gradients cleaned up


def test_shape_mismatch_and_unknown_site():
    with pytest.raises(ValueError):
        atp_scores({"a": torch.zeros(2)}, {"a": torch.zeros(3)}, {"a": torch.zeros(3)})
    with pytest.raises(KeyError):
        with GradCache(_toy(), ["nope"]):
            pass


def test_on_the_tiny_cc_unet(tiny_cc_unet):
    from vdprobe.coords import enumerate_coordinates
    unet = copy.deepcopy(tiny_cc_unet).eval()
    names = [c.name for c in enumerate_coordinates(unet)]
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, unet.config.in_channels, 4, 8, 8, generator=g)
    ea, eb = (torch.randn(4, 3, unet.config.cross_attention_dim, generator=g) for _ in range(2))
    call = lambda e: (lambda: unet(x, torch.tensor(500), encoder_hidden_states=e).sample)
    res = atp_step(unet, names, call(ea), call(eb))
    vals = torch.tensor(list(res["scores"].values()))
    assert len(res["scores"]) == len(names) and torch.isfinite(vals).all() and (vals.abs() > 0).any()
    same = atp_step(unet, names, call(eb), call(eb))
    assert all(v == 0.0 for v in same["scores"].values())


def test_frozen_weights_with_input_requiring_grad_give_the_same_scores(tiny_cc_unet):
    from vdprobe.coords import enumerate_coordinates
    unet = copy.deepcopy(tiny_cc_unet).eval()
    names = [c.name for c in enumerate_coordinates(unet)][:6]
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1, unet.config.in_channels, 4, 8, 8, generator=g)
    ea, eb = (torch.randn(4, 3, unet.config.cross_attention_dim, generator=g) for _ in range(2))
    call = lambda e, grad=False: (lambda: unet(x.clone().requires_grad_(grad), torch.tensor(500), encoder_hidden_states=e).sample)
    ref = atp_step(unet, names, call(ea), call(eb))
    unet.requires_grad_(False)
    frozen = atp_step(unet, names, call(ea), call(eb, grad=True))
    assert set(frozen["scores"]) == set(ref["scores"])
    for k in ref["scores"]:
        assert frozen["scores"][k] == pytest.approx(ref["scores"][k], rel=1e-5, abs=1e-7)
    assert all(p.grad is None for p in unet.parameters())


def test_block_sums_and_compare(tmp_path):
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("atp", Path(__file__).resolve().parents[1] / "scripts" / "attribution_screen.py")
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    rows = []
    for pair in ("p1", "p2"):
        for step in (0, 2, 16):
            for b, v in (("up_blocks.1", 0.3), ("down_blocks.0", 0.1), ("up_blocks.2", 0.2)):
                for j in range(2):               # two motion modules per block
                    rows.append({"pair": pair, "step": step, "kind": "temporal_module", "block": b, "frac": v / 2})
            rows.append({"pair": pair, "step": step, "kind": "spatial_attn2", "block": "up_blocks.1", "frac": 9.0})
    bs = sc.block_sums(rows)
    assert bs == pytest.approx({"up_blocks.1": 0.3, "down_blocks.0": 0.1, "up_blocks.2": 0.2})   # step 16 and attn2 left out
    import pandas as pd
    d = tmp_path / "pair0"
    d.mkdir()
    pd.DataFrame([{"model": "cc", "direction": "suff", "candidate": f"mm_{b}", "eff_cam": e}
                  for b, e in (("up_blocks.1", 0.9), ("down_blocks.0", 0.1), ("up_blocks.2", 0.5))]
                 + [{"model": "cc", "direction": "nec", "candidate": "mm_up_blocks.1", "eff_cam": -5.0}]).to_csv(d / "rows.csv", index=False)
    out = sc.compare(rows, tmp_path, "cc")
    assert out["n_blocks"] == 3 and out["spearman"] == pytest.approx(1.0) and out["direction"] == "suff"
    # camera_patch runs the mm_<block> candidates without B<-A: fall back to necessity
    pd.DataFrame([{"model": "cc", "direction": "nec", "candidate": f"mm_{b}", "eff_cam": e}
                  for b, e in (("up_blocks.1", 0.2), ("down_blocks.0", 0.9), ("up_blocks.2", 0.5))]).to_csv(d / "rows.csv", index=False)
    out = sc.compare(rows, tmp_path, "cc")
    assert out["direction"] == "nec" and out["n_blocks"] == 3 and out["spearman"] == pytest.approx(-1.0)
