import copy

import torch

from vdprobe.cameractrl.groupnorm import PerFrameGroupNorm, authors_groupnorm


def _call(unet, frames=4, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, unet.config.in_channels, frames, 8, 8, generator=g)
    ehs = torch.randn(batch * frames, 4, unet.config.cross_attention_dim, generator=g)
    with torch.no_grad():
        return unet(x, torch.tensor(500), encoder_hidden_states=ehs).sample


def test_per_frame_layer_equals_groupnorm_on_each_frame():
    gn = torch.nn.GroupNorm(4, 8)
    torch.nn.init.normal_(gn.weight)
    torch.nn.init.normal_(gn.bias)
    x = torch.randn(2, 8, 5, 6, 7)                                    # (batch, C, frames, h, w)
    want = torch.stack([gn(x[:, :, f]) for f in range(5)], dim=2)
    stock = gn(x)
    gn.__class__, gn.clip = PerFrameGroupNorm, {}
    assert torch.allclose(gn(x), want, atol=1e-6) and not torch.allclose(gn(x), stock, atol=1e-3)
    gn.clip["authors"] = False
    assert torch.equal(gn(x), stock)                                  # switched off: the ordinary layer


def test_switches_the_motion_module_norms_only(tiny_cc_unet):
    unet = copy.deepcopy(tiny_cc_unet).eval()
    keys = list(unet.state_dict())
    n = authors_groupnorm(unet)
    switched = [k for k, m in unet.named_modules() if isinstance(m, PerFrameGroupNorm)]
    n_mm = sum(1 for k, _ in unet.named_modules() if k.split(".")[-2:-1] == ["motion_modules"])
    assert n == len(switched) == n_mm > 0 and all("motion_modules" in k for k in switched)
    assert list(unet.state_dict()) == keys                            # same parameters, same names
    assert authors_groupnorm(unet) == n                               # idempotent


def test_changes_a_clip_but_not_a_single_frame(tiny_cc_unet):
    plain, fixed = copy.deepcopy(tiny_cc_unet).eval(), copy.deepcopy(tiny_cc_unet).eval()
    authors_groupnorm(fixed)
    assert not torch.allclose(_call(plain), _call(fixed), atol=1e-4)
    assert torch.allclose(_call(plain, frames=1), _call(fixed, frames=1), atol=1e-6)
    fixed._vd_clip["authors"] = False
    assert torch.equal(_call(plain), _call(fixed))                    # the switch restores diffusers exactly
