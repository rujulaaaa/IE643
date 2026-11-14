"""TopK sparse autoencoder (Gao et al., "Scaling and evaluating sparse autoencoders", arXiv:2406.04093)
for per-token activations of one U-Net site.

    x  ->  z = TopK(ReLU(W_enc (x - b_dec) + b_enc))  ->  x_hat = W_dec z + b_dec

Decoder columns have unit norm (renormalised after every optimiser step), the encoder starts as
the decoder's transpose, and the loss is the mean squared error; sparsity is exact (at most k
active features per token), so there is no sparsity penalty to tune. Features that never fired in
an epoch are re-initialised from the tokens the model reconstructs worst.

Inputs should be normalised first (`Normalizer`): centred, and scaled so the mean squared norm
equals the dimension. `splice` keeps the SAE's error term, so an identity edit returns the input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKSAE(nn.Module):
    def __init__(self, d_in: int, n_features: int, k: int):
        super().__init__()
        if not 0 < k <= n_features:
            raise ValueError("k must be in 1..n_features")
        self.d_in, self.n_features, self.k = d_in, n_features, k
        self.decoder = nn.Linear(n_features, d_in, bias=False)          # weight (d_in, n_features): one column per feature
        self.encoder = nn.Linear(d_in, n_features)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        with torch.no_grad():
            self.decoder.weight.copy_(F.normalize(torch.randn(d_in, n_features), dim=0))
            self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()

    def pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.encoder(x - self.b_dec))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        a = self.pre_activations(x)
        top = a.topk(self.k, dim=-1)
        return torch.zeros_like(a).scatter_(-1, top.indices, top.values)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def renormalise(self) -> None:
        self.decoder.weight.copy_(F.normalize(self.decoder.weight, dim=0))


@dataclass
class Normalizer:
    mean: torch.Tensor          # (d_in,)
    scale: float

    @classmethod
    def fit(cls, x: torch.Tensor) -> "Normalizer":
        x = x.float()
        mean = x.mean(0)
        msn = (x - mean).pow(2).sum(1).mean()
        return cls(mean, float((msn / x.shape[1]).sqrt().clamp_min(1e-12)))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean.to(x.device)) / self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale + self.mean.to(x.device)


def train_sae(x: torch.Tensor, n_features: int = 5120, k: int = 32, epochs: int = 8, batch: int = 4096,
              lr: float = 4e-4, seed: int = 0, device: str = "cpu", resample_dead: bool = True):
    """Train on tokens `x` (N, d_in), already normalised. Returns (sae on the CPU, log).

    log: `explained_variance` (1 - MSE / variance, over all of x after training), `dead_fraction`
    (features that never fired in the last epoch), `loss_curve` (mean loss per epoch), `resampled`
    (features re-initialised after each epoch).
    """
    torch.manual_seed(seed)
    x = x.float()
    n, d = x.shape
    sae = TopKSAE(d, n_features, k).to(device)
    with torch.no_grad():
        sae.b_dec.copy_(x.mean(0).to(device))
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    log = {"loss_curve": [], "resampled": []}
    fired = torch.zeros(n_features, dtype=torch.bool, device=device)
    for epoch in range(epochs):
        fired.zero_()
        total, worst_err, worst_x = 0.0, None, None
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            xb = x[perm[i:i + batch]].to(device)
            x_hat, z = sae(xb)
            err = (x_hat - xb).pow(2).sum(1)
            loss = err.mean() / d
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sae.renormalise()
            fired |= (z > 0).any(0)
            total += loss.item() * len(xb)
            if worst_err is None or err.max().item() > worst_err.max().item():
                worst_err, worst_x = err.detach(), xb
        log["loss_curve"].append(total / n)
        dead = (~fired).nonzero().flatten()
        n_re = 0
        if resample_dead and epoch < epochs - 1 and len(dead):
            # point each dead feature at one of the worst-reconstructed tokens of the hardest batch
            order = worst_err.argsort(descending=True)[:len(dead)]
            dead = dead[:len(order)]
            with torch.no_grad():
                dirs = F.normalize(worst_x[order] - sae.b_dec, dim=1)
                sae.decoder.weight[:, dead] = dirs.T
                sae.encoder.weight[dead] = dirs
                sae.encoder.bias[dead] = 0.0
            n_re = len(dead)
        log["resampled"].append(n_re)
    log["dead_fraction"] = float((~fired).float().mean())
    sae = sae.cpu().eval()
    with torch.no_grad():
        sq = sum(float((sae(x[i:i + batch])[0] - x[i:i + batch]).pow(2).sum()) for i in range(0, n, batch))
    log["explained_variance"] = 1.0 - sq / float((x - x.mean(0)).pow(2).sum())
    return sae, log


def save(path: str | Path, sae: TopKSAE, normalizer: Normalizer, meta: dict | None = None) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {**{k: v.contiguous() for k, v in sae.state_dict().items()}, "normalizer.mean": normalizer.mean.contiguous()}
    header = {"d_in": sae.d_in, "n_features": sae.n_features, "k": sae.k, "scale": normalizer.scale, "meta": meta or {}}
    save_file(tensors, str(path), metadata={"vdprobe_sae": json.dumps(header)})


def load(path: str | Path):
    """-> (sae, normalizer, meta)"""
    from safetensors import safe_open
    from safetensors.torch import load_file

    with safe_open(str(path), "pt") as f:
        header = json.loads(f.metadata()["vdprobe_sae"])
    tensors = load_file(str(path))
    sae = TopKSAE(header["d_in"], header["n_features"], header["k"])
    mean = tensors.pop("normalizer.mean")
    sae.load_state_dict(tensors, strict=True)
    return sae.eval(), Normalizer(mean, header["scale"]), header["meta"]


def splice(sae: TopKSAE, normalizer: Normalizer, x: torch.Tensor, edit) -> torch.Tensor:
    """Replace the SAE-explained part of `x` (original scale) by `decode(edit(z))`, keeping the
    reconstruction error: x - x_hat + decode(edit(z)). `edit` maps features (N, n_features) to features."""
    xn = normalizer(x)
    x_hat, z = sae(xn)
    return normalizer.inverse(xn - x_hat + sae.decode(edit(z))).to(x.dtype)
