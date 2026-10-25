"""Probe-timed prompt switching, optionally gated by the subject's cross-attention mask (D-38).

The probing sweep found that motion selectivity lives only in the first denoising steps (H2:
changepoint tau = 2 of 25, 95% CI [1, 3]). If the model fixes the motion there and the
appearance afterwards, an edit needs no activation surgery: denoise with the prompt that
carries the wanted motion for steps < tau, then with the prompt that carries the wanted
content.

    motion direction   (keep A's motion, take B's content)  early = A's prompt,          late = (B.subject, A.verb)
    content direction  (keep A's content, take B's motion)  early = (A.subject, B.verb), late = A's prompt

Every track starts from the pair's shared noise, so before tau a motion-direction track is clip A
itself. The masked variant follows EIDT-V's per-region switch (Jagpal, Chen & Namboodiri, CVPR
2025): from tau on, the latent is the switched track inside A's subject mask and A's own
trajectory outside it,

    z_t <- M * z_t + (1 - M) * z_t^A,     M = dilate(DAAM mask of A's subject phrase)

so the background and the camera stay A's. EIDT-V tunes its switch-time map by hand for image
models; here the switch time comes from the probing map (H2) and the region from DAAM maps of
the video model (harness.CrossAttentionMaps).

Pass 1 runs A (recording its trajectory and subject map) and the unmasked arms in lockstep;
pass 2 runs the masked arms against A's recorded trajectory, since the mask needs A's
completed run.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .coords import Coordinate
from .harness import CrossAttentionMaps
from .lockstep import Track, encode, run_lockstep
from .pipeline import Loaded, phrase_token_positions


@dataclass(frozen=True)
class SwitchSpec:
    early: str              # prompt for steps < tau
    late: str               # prompt from step tau on
    tau: int
    masked: bool = False    # blend with A's trajectory outside A's subject mask from step tau on


def describe_switch(arm_id: str, s: SwitchSpec) -> dict:
    return {"id": arm_id, "prompt": s.late, "label": "switch (masked)" if s.masked else "switch",
            "switch": asdict(s)}


def subject_mask(rec: CrossAttentionMaps, phrase: str, hw: tuple[int, int], tau: float = 0.4,
                 dilate: int = 7) -> torch.Tensor:
    """(1, 1, F, h, w) float mask: DAAM's per-frame tau mask, max-pool dilated by `dilate` latent px."""
    m = rec.mask(phrase, tau).float()[:, None]                         # (F, 1, s, s)
    if m.shape[-2:] != hw:
        m = F.interpolate(m, size=hw, mode="nearest")
    if dilate > 1:
        m = F.max_pool2d(m, dilate, stride=1, padding=dilate // 2)
    return m[:, 0][None, None]


@dataclass
class SwitchResult:
    latents: dict
    source_latents: torch.Tensor
    mask: torch.Tensor | None
    seconds: float


@torch.no_grad()
def run_switch(loaded: Loaded, coords: list[Coordinate], source_prompt: str, subject: str,
               arms: dict[str, SwitchSpec], noise: torch.Tensor, mask_tau: float = 0.4, dilate: int = 7,
               log_every: int = 0) -> SwitchResult:
    xattn = [c for c in coords if c.kind == "spatial_attn2"]
    tokens = phrase_token_positions(loaded.pipe.tokenizer, source_prompt, {"subject": subject})
    rec = CrossAttentionMaps(["subject"], coords=xattn, token_index=tokens)
    emb = {p: encode(loaded, p) for p in {source_prompt} | {s.early for s in arms.values()} | {s.late for s in arms.values()}}
    traj: list[torch.Tensor] = []
    mask: dict = {}

    def before_track(step, tr):
        s = arms.get(tr.id)
        if s is not None:
            tr.embeds = emb[s.late] if step >= s.tau else emb[s.early]

    def after_track(step, tr):
        if tr.id == "source":
            traj.append(tr.latents.detach().clone())
        elif arms[tr.id].masked and step >= arms[tr.id].tau:
            m = mask["m"].to(tr.latents.device, tr.latents.dtype)
            tr.latents = m * tr.latents + (1 - m) * traj[step].to(tr.latents.device)

    t0 = time.perf_counter()
    plain = [Track(k, s.early, embeds=emb[s.early]) for k, s in arms.items() if not s.masked]
    tracks = [Track("source", source_prompt, [rec], embeds=emb[source_prompt])] + plain
    run_lockstep(loaded, xattn, tracks, noise, before_track=before_track, after_track=after_track,
                 log_every=log_every)
    out = {tr.id: tr.latents for tr in tracks[1:]}
    m = None
    masked = [Track(k, s.early, embeds=emb[s.early]) for k, s in arms.items() if s.masked]
    if masked:
        m = mask["m"] = subject_mask(rec, "subject", tuple(noise.shape[-2:]), mask_tau, dilate)
        run_lockstep(loaded, [], masked, noise, before_track=before_track, after_track=after_track,
                     log_every=log_every)
        out.update({tr.id: tr.latents for tr in masked})
    return SwitchResult(out, tracks[0].latents, m, time.perf_counter() - t0)


def switch_arms(prompts: dict, taus=(1, 2, 3, 5, 8), masked_taus=(2, 3)) -> dict[str, SwitchSpec]:
    """D-38 arm set for one pair (prompts = EditPair.prompts): both directions at every tau, plus
    the masked motion-direction arm at `masked_taus`. The pre-registered arms are tau = 2."""
    arms = {}
    for t in taus:
        arms[f"motion.switch.t{t}"] = SwitchSpec(prompts["A"], prompts["motion"], t)
        arms[f"content.switch.t{t}"] = SwitchSpec(prompts["content"], prompts["A"], t)
    for t in masked_taus:
        arms[f"motion.switchmask.t{t}"] = SwitchSpec(prompts["A"], prompts["motion"], t, masked=True)
    return arms
