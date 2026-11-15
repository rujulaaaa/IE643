import copy

import torch
import torch.nn.functional as F

from vdprobe.cameractrl.groupnorm import CrossFrameGroupNorm, PerFrameGroupNorm, authors_groupnorm


def _call(unet, frames=4, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, unet.config.in_channels, frames, 8, 8, generator=g)
    ehs = torch.randn(batch * frames, 4, unet.config.cross_attention_dim, generator=g)
    with torch.no_grad():
        return unet(x, torch.tensor(500), encoder_hidden_states=ehs).sample


def _gn():
    gn = torch.nn.GroupNorm(4, 8)
    torch.nn.init.normal_(gn.weight)
    torch.nn.init.normal_(gn.bias)
    return gn


def test_cross_frame_layer_equals_groupnorm_on_the_5d_tensor():
    gn = _gn()
    x = torch.randn(2 * 5, 8, 6, 7)                                   # (batch * frames, C, h, w), batch-major
    want = F.group_norm(x.reshape(2, 5, 8, 6, 7).permute(0, 2, 1, 3, 4), 4, gn.weight, gn.bias, gn.eps)
    want = want.permute(0, 2, 1, 3, 4).reshape(10, 8, 6, 7)
    stock = gn(x)
    gn.__class__, gn.clip = CrossFrameGroupNorm, {"frames": 5}
    assert torch.allclose(gn(x), want, atol=1e-6) and not torch.allclose(gn(x), stock, atol=1e-3)
    gn.clip["frames"] = 1
    assert torch.equal(gn(x), stock)                                  # one frame: the ordinary layer
    gn.clip.update(frames=5, authors=False)
    assert torch.equal(gn(x), stock)                                  # switched off


def test_per_frame_layer_equals_groupnorm_on_each_frame():
    gn = _gn()
    x = torch.randn(2, 8, 5, 6, 7)                                    # (batch, C, frames, h, w)
    want = torch.stack([gn(x[:, :, f]) for f in range(5)], dim=2)
    stock = gn(x)
    gn.__class__, gn.clip = PerFrameGroupNorm, {"frames": 5}
    assert torch.allclose(gn(x), want, atol=1e-6) and not torch.allclose(gn(x), stock, atol=1e-3)
    gn.clip["authors"] = False
    assert torch.equal(gn(x), stock)


def test_switches_motion_module_and_output_norms_only(tiny_cc_unet):
    unet = copy.deepcopy(tiny_cc_unet).eval()
    keys = list(unet.state_dict())
    n = authors_groupnorm(unet)
    per_frame = [k for k, m in unet.named_modules() if isinstance(m, PerFrameGroupNorm)]
    joint = [k for k, m in unet.named_modules() if isinstance(m, CrossFrameGroupNorm)]
    n_mm = sum(1 for k, _ in unet.named_modules() if k.split(".")[-2:-1] == ["motion_modules"])
    assert joint == ["conv_norm_out"] and n == {"motion_module_norm": n_mm, "output_norm": 1}
    assert len(per_frame) == n_mm > 0 and all("motion_modules" in k for k in per_frame)
    assert not any(".resnets." in k or ".attentions." in k for k in per_frame + joint)
    assert list(unet.state_dict()) == keys                            # same parameters, same names
    assert authors_groupnorm(unet) == n and len(unet._forward_pre_hooks) == 1          # idempotent


def test_changes_a_clip_but_not_a_single_frame(tiny_cc_unet):
    plain, fixed = copy.deepcopy(tiny_cc_unet).eval(), copy.deepcopy(tiny_cc_unet).eval()
    authors_groupnorm(fixed)
    assert not torch.allclose(_call(plain), _call(fixed), atol=1e-4)
    assert torch.allclose(_call(plain, frames=1), _call(fixed, frames=1), atol=1e-6)
    fixed._vd_clip["authors"] = False
    assert torch.equal(_call(plain), _call(fixed))                    # the switch restores diffusers exactly
    fixed._vd_clip["authors"] = True
    # samples of a batch stay independent: the CFG pair is two clips, not one
    both = _call(fixed, batch=2)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, fixed.config.in_channels, 4, 8, 8, generator=g)
    ehs = torch.randn(8, 4, fixed.config.cross_attention_dim, generator=g)
    with torch.no_grad():
        first = fixed(x[:1], torch.tensor(500), encoder_hidden_states=ehs[:4]).sample
    assert torch.allclose(both[:1], first, atol=1e-5)
