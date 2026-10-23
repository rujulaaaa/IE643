"""Attention probabilities, reconstructed from a module's inputs, and the per-kind
reductions to one (F, S) map per coordinate and step (D-24).

SDPA never materialises the probability matrix, so we recompute q and k from the
same inputs the processor saw and apply the same scaling (head_dim ** -0.5, which is
what F.scaled_dot_product_attention uses). tests/test_harness.py checks that
probs @ v, projected through to_out, reproduces the module's real output.

Reductions, all head-averaged:
  spatial_attn1   attention *received* by each position (mean over queries)
  spatial_attn2   mass on a token set, e.g. the subject phrase (DAAM-style)
  temporal_attn*  cross-frame mass 1 - P[f, f]: how much frame f reads from other frames
"""

from __future__ import annotations

import torch


def _project_qk(attn, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor | None):
    if getattr(attn, "spatial_norm", None) is not None or getattr(attn, "group_norm", None) is not None:
        raise NotImplementedError("attention modules with spatial/group norm are not used by AnimateDiff")
    q = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    k = attn.to_k(encoder_hidden_states)
    heads = attn.heads
    d = k.shape[-1] // heads
    q = q.view(q.shape[0], -1, heads, d).transpose(1, 2)
    k = k.view(k.shape[0], -1, heads, d).transpose(1, 2)
    if getattr(attn, "norm_q", None) is not None:
        q = attn.norm_q(q)
    if getattr(attn, "norm_k", None) is not None:
        k = attn.norm_k(k)
    return q, k, d ** -0.5


def _softmax_scores(q: torch.Tensor, k: torch.Tensor, scale: float) -> torch.Tensor:
    return (torch.matmul(q, k.transpose(-1, -2)).float() * scale).softmax(dim=-1)


@torch.no_grad()
def attention_probs(attn, hidden_states, encoder_hidden_states=None) -> torch.Tensor:
    """Full (N, heads, Tq, Tk) probabilities. For tests and small inputs only."""
    q, k, scale = _project_qk(attn, hidden_states, encoder_hidden_states)
    return _softmax_scores(q, k, scale)


@torch.no_grad()
def spatial_received(attn, hs: torch.Tensor, query_chunk: int = 1024) -> torch.Tensor:
    """hs: (F, S, C) one CFG branch. Returns (F, S): mean over heads and queries of P[q, s]."""
    out = []
    for f in range(hs.shape[0]):
        q, k, scale = _project_qk(attn, hs[f:f + 1], None)  # (1, h, S, d)
        acc = torch.zeros(k.shape[2], device=hs.device, dtype=torch.float32)
        for i in range(0, q.shape[2], query_chunk):
            acc += _softmax_scores(q[:, :, i:i + query_chunk], k, scale).sum(dim=(0, 1, 2))
        out.append(acc / (q.shape[1] * q.shape[2]))
    return torch.stack(out)


@torch.no_grad()
def cross_token_mass(attn, hs: torch.Tensor, ehs: torch.Tensor, token_idx: list[int],
                     frame_chunk: int = 4) -> torch.Tensor:
    """hs: (F, S, C), ehs: (F, T, D) one CFG branch. Returns (F, S): attention mass on token_idx."""
    idx = torch.as_tensor(token_idx, device=hs.device)
    out = []
    for f in range(0, hs.shape[0], frame_chunk):
        q, k, scale = _project_qk(attn, hs[f:f + frame_chunk], ehs[f:f + frame_chunk])
        p = _softmax_scores(q, k, scale)                  # (f, h, S, T)
        out.append(p.index_select(-1, idx).sum(-1).mean(1))
    return torch.cat(out)


@torch.no_grad()
def temporal_cross_frame_mass(attn, hs: torch.Tensor, row_chunk: int = 4096) -> torch.Tensor:
    """hs: (S, F, C) one CFG branch. Returns (F, S): 1 - P[f, f], head-averaged."""
    out = []
    for i in range(0, hs.shape[0], row_chunk):
        q, k, scale = _project_qk(attn, hs[i:i + row_chunk], None)
        p = _softmax_scores(q, k, scale)                  # (s, h, F, F)
        out.append(1.0 - p.diagonal(dim1=-2, dim2=-1).mean(1))   # (s, F)
    return torch.cat(out).transpose(0, 1)
