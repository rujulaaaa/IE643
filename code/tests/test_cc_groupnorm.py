import torch
import torch.nn.functional as F

from vdprobe.cameractrl.groupnorm import CrossFrameGroupNorm, cross_frame_groupnorm


def _fresh(tiny_cc_unet):
    import copy
    return copy.deepcopy(tiny_cc_unet).eval()


def _call(unet, frames=4, batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    cfgd = unet.config
    x = torch.randn(batch, cfgd.in_channels, frames, 8, 8, generator=g)
    ehs = torch.randn(batch * frames, 4, cfgd.cross_attention_dim, generator=g)
    with torch.no_grad():
        return unet(x, torch.tensor(500), encoder_hidden_states=ehs).sample


def test_layer_equals_groupnorm_on_the_5d_tensor():
    gn = torch.nn.GroupNorm(4, 8)
    torch.nn.init.normal_(gn.weight)
    torch.nn.init.normal_(gn.bias)
    x = torch.randn(2 * 5, 8, 6, 7)                                   # (batch * frames, C, h, w), batch-major
    want = F.group_norm(x.reshape(2, 5, 8, 6, 7).permute(0, 2, 1, 3, 4), 4, gn.weight, gn.bias, gn.eps)
    want = want.permute(0, 2, 1, 3, 4).reshape(10, 8, 6, 7)
    per_frame = gn(x)
    gn.__class__, gn.clip = CrossFrameGroupNorm, {"frames": 5}
    assert torch.allclose(gn(x), want, atol=1e-6) and not torch.allclose(gn(x), per_frame, atol=1e-3)
    gn.clip["frames"] = 1
    assert torch.equal(gn(x), per_frame)                              # one frame: the ordinary layer


def test_switches_resnet_and_output_norms_only(tiny_cc_unet):
    unet = _fresh(tiny_cc_unet)
    keys = list(unet.state_dict())
    n = cross_frame_groupnorm(unet)
    switched = [k for k, m in unet.named_modules() if isinstance(m, CrossFrameGroupNorm)]
    assert n == len(switched) > 0 and "conv_norm_out" in switched
    assert all(".resnets." in k or k == "conv_norm_out" for k in switched)
    assert not any("motion_modules" in k or "attentions" in k for k in switched)
    assert list(unet.state_dict()) == keys                            # same parameters, same names
    assert cross_frame_groupnorm(unet) == n and len(unet._forward_pre_hooks) == 1     # idempotent


def test_changes_a_clip_but_not_a_single_frame(tiny_cc_unet):
    plain, joint = _fresh(tiny_cc_unet), _fresh(tiny_cc_unet)
    cross_frame_groupnorm(joint)
    assert not torch.allclose(_call(plain), _call(joint), atol=1e-4)
    assert torch.allclose(_call(plain, frames=1), _call(joint, frames=1), atol=1e-6)
    # samples of a batch stay independent: the CFG pair is two clips, not one
    both = _call(joint, batch=2)
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, joint.config.in_channels, 4, 8, 8, generator=g)
    ehs = torch.randn(8, 4, joint.config.cross_attention_dim, generator=g)
    with torch.no_grad():
        first = joint(x[:1], torch.tensor(500), encoder_hidden_states=ehs[:4]).sample
    assert torch.allclose(both[:1], first, atol=1e-5)
