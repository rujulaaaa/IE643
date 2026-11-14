"""TopK sparse autoencoder on planted data."""
import pytest
import torch
import torch.nn.functional as F

from vdprobe import sae as S
from vdprobe.sae import Normalizer, TopKSAE, splice, train_sae


def _planted(n=20000, d=64, m=128, k=4, seed=0):
    """Each sample is the sum of k random atoms of a random unit dictionary, coefficients in [1, 2]."""
    g = torch.Generator().manual_seed(seed)
    D = F.normalize(torch.randn(m, d, generator=g), dim=1)
    idx = torch.stack([torch.randperm(m, generator=g)[:k] for _ in range(n)])
    coef = 1 + torch.rand(n, k, generator=g)
    return torch.einsum("nk,nkd->nd", coef, D[idx]), D


@pytest.fixture(scope="module")
def trained():
    x, D = _planted()
    sae, log = train_sae(x, n_features=256, k=4, epochs=30, batch=1024, lr=1e-3)
    return x, D, sae, log


def test_recovers_planted_dictionary(trained):
    x, D, sae, log = trained
    cos = (F.normalize(sae.decoder.weight.T, dim=1) @ D.T).abs().max(0).values     # best match per true atom
    assert (cos > 0.9).float().mean() > 0.8
    assert log["explained_variance"] > 0.9 and log["dead_fraction"] < 0.5
    assert len(log["loss_curve"]) == 30 and log["loss_curve"][-1] < 0.5 * log["loss_curve"][0]


def test_decoder_unit_norm_after_training(trained):
    sae = trained[2]
    assert torch.allclose(sae.decoder.weight.norm(dim=0), torch.ones(256), atol=1e-4)


def test_topk_exact():
    z = TopKSAE(16, 64, 5).encode(torch.randn(10, 16))
    assert ((z > 0).sum(1) <= 5).all() and (z >= 0).all()
    assert z.shape == (10, 64)
    with pytest.raises(ValueError):
        TopKSAE(16, 8, 9)


def test_encoder_starts_as_decoder_transpose():
    s = TopKSAE(16, 64, 5)
    assert torch.equal(s.encoder.weight, s.decoder.weight.T)
    assert torch.allclose(s.decoder.weight.norm(dim=0), torch.ones(64), atol=1e-6)


def test_normalizer_inverse_and_scale():
    x = torch.randn(500, 12) * 7 + 3
    n = Normalizer.fit(x)
    xn = n(x)
    assert torch.allclose(n.inverse(xn), x, atol=1e-4)
    assert torch.allclose(xn.mean(0), torch.zeros(12), atol=1e-4)
    assert xn.pow(2).sum(1).mean().item() == pytest.approx(12, rel=1e-4)        # mean squared norm = d_in
    n0 = Normalizer.fit(torch.ones(10, 4))                                       # constant input: no division by zero
    assert torch.isfinite(n0(torch.ones(10, 4))).all()


def test_unnormalised_input_does_not_give_a_trivially_low_loss():
    """The loss is relative to scale: tiny inputs must not look well explained."""
    x, _ = _planted(n=2000)
    _, small = train_sae(x * 1e-3, n_features=64, k=1, epochs=1, batch=500, lr=1e-9)
    assert small["explained_variance"] < 0.5


def test_splice_identity_edit_is_exact(trained):
    x, _, sae, _ = trained
    norm = Normalizer.fit(x)
    xs = x[:200] * 3 + 1
    assert torch.allclose(splice(sae, norm, xs, lambda z: z), xs, atol=1e-5)


def test_splice_zeroing_a_feature_removes_its_direction(trained):
    x, _, sae, _ = trained
    norm = Normalizer(torch.zeros(64), 1.0)
    z = sae.encode(x[:500])
    j = int(z.sum(0).argmax())
    rows = z[:, j] > 0

    def kill(z):
        z = z.clone(); z[:, j] = 0
        return z

    out = splice(sae, norm, x[:500], kill)
    delta = x[:500] - out
    assert torch.allclose(delta[~rows], torch.zeros_like(delta[~rows]), atol=1e-5)
    expect = z[rows, j][:, None] * sae.decoder.weight[:, j][None]
    assert torch.allclose(delta[rows], expect, atol=1e-4)


def test_save_load_roundtrip(tmp_path, trained):
    x, _, sae, _ = trained
    norm = Normalizer.fit(x)
    S.save(tmp_path / "sae.safetensors", sae, norm, {"site": "up_blocks.1.motion_modules.2", "steps": [0, 1, 2]})
    sae2, norm2, meta = S.load(tmp_path / "sae.safetensors")
    assert meta == {"site": "up_blocks.1.motion_modules.2", "steps": [0, 1, 2]}
    assert (sae2.d_in, sae2.n_features, sae2.k) == (64, 256, 4) and norm2.scale == pytest.approx(norm.scale)
    assert torch.equal(norm2.mean, norm.mean)
    assert torch.equal(sae2(norm2(x[:50]))[0], sae(norm(x[:50]))[0])


def test_dead_features_are_resampled():
    x, _ = _planted(n=4000)
    _, with_re = train_sae(x, n_features=512, k=2, epochs=6, batch=500, lr=1e-3, seed=1)
    _, without = train_sae(x, n_features=512, k=2, epochs=6, batch=500, lr=1e-3, seed=1, resample_dead=False)
    assert sum(with_re["resampled"]) > 0 and sum(without["resampled"]) == 0
    assert with_re["dead_fraction"] <= without["dead_fraction"]
