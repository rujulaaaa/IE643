"""Steering probes on the tiny UNet."""
import pytest
import torch

from vdprobe.coords import by_name, enumerate_coordinates
from vdprobe.harness import Harness
from vdprobe.sae import Normalizer, TopKSAE
from vdprobe.steer import AddDirection, FeatureAblate

# not the last motion module: in the tiny model its 32 channels are 32 GroupNorm groups in conv_norm_out,
# which removes any per-channel constant, so AddDirection would be invisible there
SITE = "down_blocks.0.motion_modules.0"
C = 32


def run(tiny_unet, tiny_run, probes, steps=2):
    coords = enumerate_coordinates(tiny_unet)
    with Harness(tiny_unet, coords, probes):
        return tiny_run()(steps=steps)


def site_io(tiny_unet, tiny_run, probes):
    """(input, output seen downstream) of the site on the first step, with `probes` active."""
    mod = dict(tiny_unet.named_modules())[SITE]
    seen = {}

    def record(m, a, out):
        seen.setdefault("io", (a[0].clone(), out.clone()))

    coords = enumerate_coordinates(tiny_unet)
    with Harness(tiny_unet, coords, probes):          # harness hook first, so `record` sees the replaced output
        h = mod.register_forward_hook(record)
        try:
            tiny_run()(steps=1)
        finally:
            h.remove()
    return seen["io"]


def test_add_direction_strength_zero_is_bit_identical(tiny_unet, tiny_run):
    plain = run(tiny_unet, tiny_run, [])
    p = AddDirection(SITE, None, torch.randn(C), 0.0)
    assert torch.equal(run(tiny_unet, tiny_run, [p]), plain) and p.applied == 0


def test_add_direction_adds_exactly_the_direction(tiny_unet, tiny_run):
    d = torch.randn(C, generator=torch.Generator().manual_seed(0))
    _, base = site_io(tiny_unet, tiny_run, [])
    _, steered = site_io(tiny_unet, tiny_run, [AddDirection(SITE, None, d, 2.0)])
    assert steered.shape == base.shape and by_name(enumerate_coordinates(tiny_unet))[SITE].channel_dim == 1
    assert torch.allclose(steered - base, (2.0 * d).reshape(1, C, 1, 1).expand_as(base), atol=1e-5)


def test_add_direction_respects_steps_and_changes_output(tiny_unet, tiny_run):
    d = torch.randn(C, generator=torch.Generator().manual_seed(0))
    plain = run(tiny_unet, tiny_run, [])
    late = AddDirection(SITE, (5,), d, 3.0)
    assert torch.equal(run(tiny_unet, tiny_run, [late]), plain) and late.applied == 0
    now = AddDirection(SITE, (0,), d, 3.0)
    assert not torch.allclose(run(tiny_unet, tiny_run, [now]), plain, atol=1e-4) and now.applied == 1


def test_add_direction_rejects_wrong_width(tiny_unet, tiny_run):
    with pytest.raises(ValueError, match="channels"):
        run(tiny_unet, tiny_run, [AddDirection(SITE, None, torch.randn(C + 1), 1.0)])


@pytest.fixture(scope="module")
def sae():
    torch.manual_seed(0)
    s = TopKSAE(C, 96, 8).eval()
    with torch.no_grad():
        s.encoder.bias.fill_(0.5)                    # make sure features fire on any input
    return s, Normalizer(torch.zeros(C), 0.5)


def test_feature_ablate_empty_list_changes_nothing(tiny_unet, tiny_run, sae):
    plain = run(tiny_unet, tiny_run, [])
    p = FeatureAblate(SITE, None, *sae, features=[])
    assert torch.allclose(run(tiny_unet, tiny_run, [p]), plain, atol=1e-4) and p.applied == 2
    assert p.removed_norm < 1e-3


def test_feature_ablate_removes_only_the_listed_features(tiny_unet, tiny_run, sae):
    s, norm = sae
    inp, base = site_io(tiny_unet, tiny_run, [])
    upd = (base - inp).movedim(1, -1).reshape(-1, C)                        # (tokens, C), as the probe sees it
    z = s.encode(norm(upd))
    feats = z.sum(0).topk(3).indices.tolist()
    p = FeatureAblate(SITE, None, s, norm, feats)
    _, out = site_io(tiny_unet, tiny_run, [p])
    removed = (base - out).movedim(1, -1).reshape(-1, C)
    expect = (z[:, feats] @ s.decoder.weight[:, feats].T) * norm.scale       # decoder contribution, back in activation units
    assert torch.allclose(removed, expect, atol=1e-4) and expect.abs().max() > 1e-3
    assert p.removed_norm > 0

    never = [int(z.sum(0).argmin())] if z.sum(0).min().item() == 0 else []
    if never:                                                                # a feature that never fires removes nothing
        q = FeatureAblate(SITE, None, s, norm, never)
        _, same = site_io(tiny_unet, tiny_run, [q])
        assert torch.allclose(same, base, atol=1e-5)


def test_feature_ablate_needs_same_shape_io(tiny_unet, tiny_run, sae):
    coords = enumerate_coordinates(tiny_unet)
    p = FeatureAblate(SITE, None, *sae, features=[0])
    with pytest.raises(ValueError, match="input's shape"):
        p(Harness(tiny_unet, coords, []).ctx, by_name(coords)[SITE], None, torch.zeros(2, C, 4, 4), (torch.zeros(2, C, 2, 2),), {})
