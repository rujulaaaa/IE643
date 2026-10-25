"""The interface's engine: live edits, edits driven by an uploaded video, cached results.

Everything the Gradio app (scripts/app.py) runs goes through here, so it can be tested without
a browser. Three entry points:

  Engine.edit(EditRequest)       two prompts in, source A + prompt-only baseline + the chosen edit
                                 (+ B's own clip as the reference), all from one noise tensor,
                                 scored with s_c, s_m and D exactly as score_edits.py does
  Engine.upload_edit(...)        a real video as clip A: DDIM inversion (inversion.py), then the
                                 motion-direction edit onto a new subject
  cached_runs() / load_cached()  the Kaggle dev/test sweeps, for instant browsing during a viva

D needs a min-max pool (D-14). A single live edit has no pool of its own, so it is normalised
with the pool of the scored test run (dev run until the test run is scored); values outside
that pool fall outside [-1, 1]. The app says which pool is in use.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import torch
import yaml

from .config import CONFIG_DIR, REPO_ROOT
from .edit import Arm, AttnSpec, EditSpec, run_arms
from .edit_plan import WINDOWS, EditPair, literature_specs, motion_modules, temporal_attention, temporal_blocks
from .metrics import (content_fidelity, disentanglement, flow_fields, motion_fidelity, motion_magnitude,
                      subject_consistency)
from .harness import Probe
from .patching import PatchSpec

RESULTS = REPO_ROOT / "results"
LORA_DIR = RESULTS / "local" / "motion_lora"
SELECTION = CONFIG_DIR / "edit_selection.json"
SCORED_RUNS = {"stage 6b test": RESULTS / "kaggle" / "vdprobe-lit-test" / "out",   # attention-level edits (D-37)
               "test": RESULTS / "kaggle" / "vdprobe-test" / "out",
               "stage 6b dev": RESULTS / "kaggle" / "vdprobe-lit-dev" / "out",
               "dev": RESULTS / "kaggle" / "vdprobe-dev" / "out",
               "local smoke": RESULTS / "local" / "edit_smoke"}
POOL_SNAPSHOT = CONFIG_DIR / "d_pool.json"     # frozen copy of a scored run's pool, for machines without it
XATTN_RUN = RESULTS / "kaggle" / "vdprobe-xattn" / "out"     # model variants x cross-attention (session 5)
VARIANTS = {"sd15_mm2": CONFIG_DIR / "generation.yaml",
            **{p.stem: p for p in sorted((CONFIG_DIR / "variants").glob("*.yaml"))}}
METHODS = {
    "selected": "component patch at the Stage 5 selection",
    "custom": "component patch, block and window chosen below",
    "attn": "temporal attention injection (training-free baseline)",
    "attn_all": "temporal attention injection at all 25 steps",
    "tensor": "whole-tensor patch at the selected place",
    "lora_probe": "prompt + probe-guided motion LoRA (up_blocks.0)",
    "lora_full": "prompt + full motion LoRA (all motion modules)",
    "lora_anti": "prompt + wrong-place LoRA control (down_blocks.2-3)",
    "lit_single": "attention-level edit selected on dev (D-37), reads clip A only",
    "lit_dual": "attention-level edit selected on dev (D-37), reads A and B (appearance from one, motion from the other)",
}
LIT_SELECTION = CONFIG_DIR / "lit_selection.json"      # written from the Stage 6b dev run (D-37)
# used until that file exists: the papers' defaults (UniEdit SA-T; MasaCtrl K/V at up_blocks.2-3 from step 2)
LIT_DEFAULT = {"motion": {"single": "motion.attn.all.all", "dual": "motion.dual.up23.from2"},
               "content": {"single": "content.kv.up23.from2", "dual": "content.dual.all"}}


def lit_selection() -> dict:
    return json.loads(LIT_SELECTION.read_text()) if LIT_SELECTION.exists() else LIT_DEFAULT
SEC_PER_CLIP = 65          # laptop RTX 4060, CPU-side cache (patch controls, §16 of the log)


def link_assets(root: Path) -> int:
    """Symlink every file under `root` (laid out like the repository: results/..., data/...) into the
    repository where it is missing. For machines that get the code and its assets separately (a
    Kaggle dataset, a fresh clone). Returns the number of links made."""
    root, n = Path(root), 0
    for f in root.rglob("*"):
        if f.is_file():
            dst = REPO_ROOT / f.relative_to(root)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                try:
                    dst.symlink_to(f.resolve())
                    n += 1
                except FileExistsError:       # another process linked it first (parallel Kaggle jobs)
                    pass
    return n


# --- presentation helpers (plain-language names, playable clips, progress) -----------------

WINDOW_NAMES = {"ft": "steps 0-8", "all": "all 25 steps", "one0": "step 0", "five1": "steps 0-3",
                "from2": "steps 2-24", "from5": "steps 5-24", "early": "steps 0-2", "first": "frame 0 only"}
SCORE_NAMES = {"clip": "Clip", "sc_A": "■ Looks like A", "sc_B": "■ Looks like B", "sm_A": "▲ Moves like A",
               "sm_B": "▲ Moves like B", "D": "Disentanglement D", "consistency_ratio": "Identity stability",
               "motion_magnitude": "Motion amount (px)", "sm_video": "▲ Moves like the video",
               "sc_video": "■ Looks like the video", "clip_text_target": "Matches target prompt"}
CLIP_NAMES = {"source": "A · source", "B_ref": "B · as generated", "base": "Prompt only", "edit": "Edit",
              "video": "Your video", "motion.base": "Prompt only (keep motion)",
              "content.base": "Prompt only (keep content)"}


def arm_label(arm_id: str) -> str:
    """'motion.dynamic.up_blocks.0.ft' -> 'Keep motion · frame-varying patch · up_blocks.0 · steps 0-8'."""
    import re

    if arm_id in CLIP_NAMES:
        return CLIP_NAMES[arm_id]
    m = re.search(r"\.a(\d+(?:\.\d+)?)$", arm_id)
    strength = f"strength {m.group(1)}" if m else None
    parts = (arm_id[:m.start()] if m else arm_id).split(".")
    head = {"motion": "Keep motion", "content": "Keep content"}.get(parts[0], parts[0])
    what = {"dynamic": "frame-varying patch", "static": "frame-mean patch", "tensor": "whole-tensor patch",
            "attn": "attention injection", "sel": "selected patch", "kv": "key/value injection (MasaCtrl)",
            "dual": "A + B dual source (UniEdit)"}
    rest, out = parts[1:], [head]
    if rest and rest[0] in what:
        out.append(what[rest.pop(0)])
    if rest and rest[-1] in WINDOW_NAMES:
        win = WINDOW_NAMES[rest.pop()]
    else:
        win = None
    loras = {"lora_probe": "probe-guided motion LoRA", "lora_full": "full motion LoRA",
             "lora_anti": "wrong-place LoRA (control)"}
    for r in list(rest):
        if r in loras:
            out.append("prompt + " + loras[r])
            rest.remove(r)
    if rest:
        place = ".".join(rest)
        out.append({"allmm": "all motion modules", "all": "all temporal attention", "up23": "up_blocks.2-3",
                    "dec": "all decoder blocks", "probe": "probe-placed", "all.all": "all temporal attention",
                    "up0.all": "up_blocks.0", "up1.all": "up_blocks.1"}.get(place, place))
    if win:
        out.append(win)
    if strength:
        out.append(strength)
    return " · ".join(out)


def pretty_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Scores table with plain-language headers and clip names (■ = content measure, ▲ = motion)."""
    out = df.copy()
    if "clip" in out.columns:
        out["clip"] = out["clip"].map(arm_label)
    if "arm" in out.columns:
        out["arm"] = out["arm"].map(arm_label)
    return out.rename(columns={**SCORE_NAMES, "arm": "Clip"}).round(3)


def playable(path: str | None) -> str | None:
    """An H.264 MP4 next to a GIF (made once with ffmpeg), so clips get play / pause controls instead
    of looping on their own. Falls back to the GIF when ffmpeg is missing or fails."""
    import shutil
    import subprocess

    if not path or not str(path).endswith(".gif") or shutil.which("ffmpeg") is None:
        return path
    mp4 = Path(str(path)[:-4] + ".mp4")
    if not mp4.exists():
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-movflags", "+faststart",
                            "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(mp4)],
                           check=True, timeout=60)
        except Exception:
            return path
    return str(mp4)


class StepProgress(Probe):
    """Reports denoising progress once per step: hooked on one coordinate of one track."""

    def __init__(self, coord: str, n_steps: int, callback, label: str):
        super().__init__([coord], None)
        self.n, self.cb, self.label, self.last = n_steps, callback, label, -1

    def __call__(self, ctx, coord, module, output, args, kwargs):
        if ctx.step != self.last:
            self.last = ctx.step
            self.cb((ctx.step + 1) / self.n, f"{self.label} · step {ctx.step + 1} of {self.n}")
        return None


def vocabulary() -> dict:
    v = yaml.safe_load((CONFIG_DIR / "probing_set.yaml").read_text())
    subjects = [s for f in v["families"].values() for s in f["subjects"]]
    return {"subjects": subjects, "verbs": v["verbs"], "scenes": v["scenes"], "seeds": v["seeds"]}


def selection() -> dict:
    return json.loads(SELECTION.read_text())


def reference_pool() -> tuple[str, dict]:
    """The D pool: the first scored run found (test, dev, local smoke), else the frozen snapshot
    configs/d_pool.json. FileNotFoundError if none exists (the app then shows s_c / s_m without D)."""
    for name, root in SCORED_RUNS.items():
        f = root / "score_summary.json"
        if f.exists():
            return name, json.loads(f.read_text())["pool"]
    if POOL_SNAPSHOT.exists():
        snap = json.loads(POOL_SNAPSHOT.read_text())
        return f"{snap['run']} (snapshot)", snap["pool"]
    raise FileNotFoundError("no scored edit run and no configs/d_pool.json: run scripts/score_edits.py first")


def upload_start(z_inv: torch.Tensor, gauss: torch.Tensor, start: str) -> torch.Tensor:
    """Starting latent of the new clips in the upload path (see Engine.upload_edit), fp16."""
    if start == "gaussian":
        return gauss.half()
    if start == "inverted":
        return z_inv.half()
    if start != "blend":
        raise ValueError(f"unknown start {start!r}: blend, inverted or gaussian")
    z = z_inv.float() / z_inv.float().std() + gauss.float().to(z_inv.device)
    return (z / z.std()).half()


@dataclass
class EditRequest:
    subject_a: str
    verb_a: str
    subject_b: str
    verb_b: str
    scene: str
    seed: int
    direction: str = "motion"          # "motion": keep A's motion, take B's subject; "content": the reverse
    method: str = "selected"
    block: str = "up_blocks.3"         # method "custom" only; "all" = all 21 motion modules
    window: str = "ft"                 # method "custom" only; key of edit_plan.WINDOWS
    alpha: float = 1.0
    with_reference: bool = True        # also generate B's own clip (needed for D)

    @property
    def pair(self) -> EditPair:
        return EditPair(self.subject_a.strip(), self.verb_a.strip(), self.subject_b.strip(),
                        self.verb_b.strip(), self.scene.strip(), int(self.seed))


def _component(direction: str) -> str:
    return "dynamic" if direction == "motion" else "static"


def edit_arm(arm_id: str, prompt: str, direction: str, method: str, coords, sel: dict,
             block: str = "", window: str = "", alpha: float = 1.0) -> Arm:
    """One edited target track. `sel` is the Stage 5 selection (configs/edit_selection.json)."""
    if method.startswith("lora_"):
        if direction != "motion":
            raise ValueError("the motion LoRAs only apply to the motion direction")
        return Arm(arm_id, prompt, EditSpec(label=METHODS[method]), lora=method)
    if method.startswith("lit_"):
        name = lit_selection()[direction][method[4:]]
        spec = literature_specs(coords)[name]
        return Arm(arm_id, prompt, EditSpec(spec.patch, spec.attn, label=f"{METHODS[method]}: {name}", kv=spec.kv))
    if method == "custom":
        blocks, win = ("all" if block == "all" else [block]), window
    else:
        blocks, win = sel[direction]["blocks"], sel[direction]["window"]
    if win not in WINDOWS:
        raise ValueError(f"unknown window {win!r}; one of {sorted(WINDOWS)}")
    steps = WINDOWS["all" if method == "attn_all" else win]
    if method in ("attn", "attn_all"):
        if direction != "motion":
            raise ValueError("attention injection transfers motion: use the motion direction")
        return Arm(arm_id, prompt, EditSpec(attn=AttnSpec(temporal_attention(coords), steps, alpha=alpha),
                                            label=METHODS[method]))
    if blocks != "all" and blocks[0] not in temporal_blocks(coords):
        raise ValueError(f"unknown block {blocks[0]!r}")
    mods = motion_modules(coords) if blocks == "all" else tuple(m for b in blocks for m in motion_modules(coords, b))
    comp = "all" if method == "tensor" else _component(direction)
    return Arm(arm_id, prompt, EditSpec(PatchSpec(mods, steps, alpha=alpha, component=comp), label=METHODS[method]))


def build_arms(req: EditRequest, cfg, coords, sel: dict) -> tuple[str, list[Arm]]:
    """Source prompt (A) and the target arms: [B_ref,] prompt-only baseline, the edit."""
    p = req.pair.prompts(cfg)
    d = req.direction
    if d not in ("motion", "content"):
        raise ValueError(d)
    if req.method == "lit_dual" and not req.with_reference:
        raise ValueError("the dual-source edit reads clip B: tick 'Also generate B'")
    arms = [Arm("B_ref", p["B"], EditSpec(label="B, unpatched"))] if req.with_reference else []
    arms.append(Arm("base", p[d], EditSpec(label="prompt only")))
    arms.append(edit_arm("edit", p[d], d, req.method, coords, sel, req.block, req.window, req.alpha))
    return p["A"], arms


def score_edit(feats: dict, direction: str, pool: dict | None) -> pd.DataFrame:
    """feats: clip id -> {"emb", "flow"}; ids "source" (A), optional "B_ref", then arms.

    Same quantities as score_edits.py: s_c = per-frame CLIP cosine, s_m = Farneback flow cosine,
    D with the fixed reference pool. Without B_ref only the A-side numbers exist."""
    A, B = feats["source"], feats.get("B_ref")
    cons_src = subject_consistency(A["emb"])
    rows = []
    for k, f in feats.items():
        r = {"clip": k, "sc_A": content_fidelity(f["emb"], A["emb"]), "sm_A": motion_fidelity(f["flow"], A["flow"]),
             "motion_magnitude": motion_magnitude(f["flow"]),
             "consistency_ratio": subject_consistency(f["emb"]) / cons_src}
        if B is not None:
            r["sc_B"] = content_fidelity(f["emb"], B["emb"])
            r["sm_B"] = motion_fidelity(f["flow"], B["flow"])
        rows.append(r)
    df = pd.DataFrame(rows)
    if B is not None and pool is not None:
        if direction == "motion":     # output should have B's content and A's motion
            d = disentanglement(df.sc_B, df.sm_A, df.sc_A, df.sm_B, pool["sc"], pool["sm"])
        else:
            d = disentanglement(df.sc_A, df.sm_B, df.sc_B, df.sm_A, pool["sc"], pool["sm"])
        df["D"] = d
        mask = df["clip"].isin(["source", "B_ref"])   # not df.clip: that is DataFrame.clip
        df.loc[mask, "D"] = float("nan")       # D is defined for edit outputs only
    return df


class Engine:
    """Loads AnimateDiff, CLIP and the three motion LoRAs once, on first use. `variant` picks the
    model cell (VARIANTS: the frozen SD1.5 + v1-5-2, or an Experiment 2 variant config)."""

    def __init__(self, profile: str | None = None, cache_device: str = "cpu", out: Path | None = None,
                 variant: str = "sd15_mm2"):
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant!r}; one of {sorted(VARIANTS)}")
        self.profile, self.cache_device, self.variant = profile, cache_device, variant
        self.out = out or RESULTS / "interface"
        self.loaded = self.coords = self.clip = None
        self.loras: list[str] = []
        self.lora_skipped: dict[str, str] = {}

    def ensure(self):
        if self.loaded is not None:
            return
        from .coords import enumerate_coordinates
        from .lora import load_motion_lora
        from .metrics import ClipScorer
        from .pipeline import build_pipeline
        from .config import GenerationConfig
        from .runio import default_profile

        cfg = GenerationConfig.from_yaml(VARIANTS[self.variant])
        self.loaded = build_pipeline(cfg, profile=self.profile or default_profile())
        for v in ("probe", "full", "anti"):
            f = LORA_DIR / v / "motion_lora.safetensors"
            if f.exists():
                try:     # trained on SD1.5 + v1-5-2: v1-5-3 lacks the mid-block module the full LoRA targets
                    load_motion_lora(self.loaded.unet, f, name=f"lora_{v}")
                    self.loras.append(f"lora_{v}")
                except Exception as e:
                    self.lora_skipped[f"lora_{v}"] = f"{type(e).__name__}: {e}"[:200]
                    try:
                        self.loaded.unet.delete_adapters(f"lora_{v}")
                    except Exception:
                        pass
        if self.loras:
            self.loaded.unet.disable_adapters()
        self.coords = enumerate_coordinates(self.loaded.unet)
        self.clip = ClipScorer()

    def _features(self, frames: dict) -> dict:
        return {k: {"emb": self.clip.image(fr).cpu(), "flow": flow_fields(fr)} for k, fr in frames.items()}

    def _save(self, tag: str, frames: dict, table: pd.DataFrame, record: dict) -> dict:
        from .runio import save_gif

        d = self.out / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
        gifs = {}
        for k, fr in frames.items():
            gifs[k] = d / f"{k}.gif"
            save_gif(fr, gifs[k])
        table.to_csv(d / "scores.csv", index=False)
        (d / "request.json").write_text(json.dumps(record, indent=1, default=str))
        return {k: str(v) for k, v in gifs.items()}

    def _progress(self, cb, label: str, n_clips: int):
        """A StepProgress for the last track of a lockstep group (None without a callback)."""
        if cb is None:
            return None
        return StepProgress(self.coords[0].name, self.loaded.cfg.steps, cb, f"{label} ({n_clips} clips together)")

    def edit(self, req: EditRequest, progress=None) -> dict:
        from .pipeline import make_noise

        self.ensure()
        if req.method.startswith("lora_") and req.method not in self.loras:
            raise ValueError(f"{req.method} is not available (no file in {LORA_DIR})")
        src, arms = build_arms(req, self.loaded.cfg, self.coords, selection())
        t0 = time.perf_counter()
        sp = self._progress(progress, "Generating", len(arms) + 1)
        r = run_arms(self.loaded, self.coords, src, arms, make_noise(self.loaded.cfg, req.seed),
                     cache_device=self.cache_device, extra_probes={arms[-1].id: [sp]} if sp else None)
        if progress:
            progress(1.0, "Scoring the clips (CLIP and optical flow)")
        frames = {"source": r.frames["source"], **{a.id: r.frames[a.id] for a in arms}}
        try:
            pool_name, pool = reference_pool()
        except FileNotFoundError:
            pool_name, pool = "none (D not computed)", None
        table = score_edit(self._features(frames), req.direction, pool)
        prompts = {"source": src, **{a.id: a.prompt for a in arms}}
        rec = {"request": asdict(req), "prompts": prompts, "arms": [a.id for a in arms],
               "seconds": time.perf_counter() - t0, "pool": pool_name, "n_patched": r.n_patched,
               "n_attn_injected": r.n_attn_injected}
        gifs = self._save(req.pair.id + f"__{req.direction}__{req.method}", frames, table, rec)
        return {"gifs": gifs, "table": table, "prompts": prompts, "pool": pool_name, "seconds": rec["seconds"]}

    def upload_edit(self, video: str, source_prompt: str, target_prompt: str, method: str = "attn_all",
                    alpha: float = 1.0, seed: int = 42, start: str = "blend", progress=None) -> dict:
        """Clip A = an uploaded video, inverted onto the sampler's trajectory; motion direction only.

        `start` is the latent the new clips start from (§22 of the log, checked 2026-10-22):
          "blend"     z_t0 and the seed's Gaussian noise mixed at equal variance (default)
          "inverted"  the inverted noise z_t0 itself (layout of the video; grainier)
          "gaussian"  the seed's noise only (layout unrelated to the video)
        A layout-aligned start is what makes the transfer work: with "gaussian", attention
        injection gives a near-static clip, and the component patch runs away (latent std 26)
        under "gaussian" and "blend", because the video's frame-varying activations do not line
        up with the new clip's layout. Default: attention injection at all steps + "blend".
        """
        from .inversion import invert, read_video
        from .pipeline import make_noise

        self.ensure()
        cfg = self.loaded.cfg
        frames_in = read_video(video, frames=cfg.frames, size=cfg.height)
        t0 = time.perf_counter()
        if progress:
            progress(0.0, "Inverting your video onto the 25 denoising steps")
        traj = invert(self.loaded, frames_in, source_prompt)
        t_inv = time.perf_counter() - t0
        arms = [Arm("base", target_prompt, EditSpec(label="prompt only")),
                edit_arm("edit", target_prompt, "motion", method, self.coords, selection(), alpha=alpha)]
        sp = self._progress(progress, "Generating", len(arms) + 1)
        r = run_arms(self.loaded, self.coords, source_prompt, arms, upload_start(traj[0], make_noise(cfg, seed), start),
                     cache_device=self.cache_device, source_trajectory=traj,
                     extra_probes={arms[-1].id: [sp]} if sp else None)
        if progress:
            progress(1.0, "Scoring the clips")
        frames = {"video": frames_in, "source": r.frames["source"], **{a.id: r.frames[a.id] for a in arms}}
        feats = self._features(frames)
        txt = self.clip_text(target_prompt)
        rows = []
        for k, f in feats.items():
            rows.append({"clip": k, "sm_video": motion_fidelity(f["flow"], feats["video"]["flow"]),
                         "sc_video": content_fidelity(f["emb"], feats["video"]["emb"]),
                         "clip_text_target": float((f["emb"] @ txt.T).mean()),
                         "motion_magnitude": motion_magnitude(f["flow"])})
        table = pd.DataFrame(rows)
        rec = {"video": video, "source_prompt": source_prompt, "target_prompt": target_prompt, "method": method,
               "alpha": alpha, "seed": seed, "start": start, "inversion_seconds": t_inv, "seconds": time.perf_counter() - t0}
        gifs = self._save("upload__" + method, frames, table, rec)
        return {"gifs": gifs, "table": table, "seconds": rec["seconds"]}

    def cross_attention(self, subject: str, verb: str, scene: str, seed: int = 42, tau: float = 0.4,
                        progress=None) -> dict:
        """One clip with DAAM-style per-frame cross-attention maps for its subject, verb and scene
        (harness.CrossAttentionMaps); overlay GIFs and the layer x step-window grid per phrase."""
        import importlib.util

        from .harness import CrossAttentionMaps, Harness
        from .pipeline import generate, make_noise, phrase_token_positions
        from .probing_set import Clip

        self.ensure()
        spec = importlib.util.spec_from_file_location("xattn_script", REPO_ROOT / "scripts" / "cross_attention_maps.py")
        xs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(xs)
        cfg = self.loaded.cfg
        clip = Clip(subject.strip(), verb.strip(), scene.strip(), int(seed))
        prompt = clip.prompt(cfg)
        tokens = phrase_token_positions(self.loaded.pipe.tokenizer, prompt, clip.phrases)
        rec = CrossAttentionMaps(list(clip.phrases), coords=[c for c in self.coords if c.kind == "spatial_attn2"])
        t0 = time.perf_counter()
        probes = [rec] + ([StepProgress(self.coords[0].name, cfg.steps, progress, "Generating")] if progress else [])
        with Harness(self.loaded.unet, self.coords, probes, token_index=tokens):
            gen = generate(self.loaded, prompt, make_noise(cfg, clip.seed))
        if progress:
            progress(1.0, "Drawing the attention maps")
        d = self.out / f"{time.strftime('%Y%m%d-%H%M%S')}_xattn__{self.variant}__{clip.id}"
        d.mkdir(parents=True, exist_ok=True)
        from .runio import save_gif
        save_gif(gen.frames, d / "clip.gif")
        out = {"clip": str(d / "clip.gif"), "overlay": {}, "grid": {}, "mask_area": {}}
        for ph in clip.phrases:
            fr = xs.overlay(gen.frames, rec.heatmap(ph))
            fr[0].save(d / f"overlay_{ph}.gif", save_all=True, append_images=fr[1:], duration=125, loop=0)
            xs.grid_figure(rec, self.coords, ph, d / f"grid_{ph}.png", cfg.steps)
            out["overlay"][ph], out["grid"][ph] = str(d / f"overlay_{ph}.gif"), str(d / f"grid_{ph}.png")
            out["mask_area"][ph] = float(rec.mask(ph, tau).float().mean())
        out["seconds"], out["prompt"] = time.perf_counter() - t0, prompt
        return out

    @torch.no_grad()
    def clip_text(self, text: str) -> torch.Tensor:
        m = self.clip.model
        inp = self.clip.proc(text=[text], return_tensors="pt", padding=True).to(self.clip.device)
        e = self.clip._to_embedding(m.get_text_features(**inp), m.text_projection)
        return torch.nn.functional.normalize(e.float(), dim=-1).cpu()


# --- cached sweep results ------------------------------------------------------------------

def cached_runs() -> dict[str, list[Path]]:
    """run name -> pair directories that finished (manifest.json written last)."""
    out = {}
    for name, root in SCORED_RUNS.items():
        dirs = sorted(p.parent for p in root.rglob("manifest.json")) if root.exists() else []
        if dirs:
            out[name] = dirs
    return out


def load_cached(pair_dir: Path) -> dict:
    """GIF paths, prompts and (if the run is scored) the per-arm scores of one sweep pair."""
    arms = json.loads((pair_dir / "arms.json").read_text())
    gifs = {p.stem: str(p) for p in sorted((pair_dir / "gifs").glob("*.gif"))}
    run_root = next((r for r in SCORED_RUNS.values() if r in pair_dir.parents), None)
    scores = None
    if run_root is not None and (run_root / "scores_D.csv").exists():
        df = pd.read_csv(run_root / "scores_D.csv")
        scores = df[df.pair == pair_dir.name].drop(columns=["pair", "family_seed"])
    prompts = {"source": arms["prompts"]["A"], **{a["id"]: a["prompt"] for a in arms["arms"]}}
    return {"pair": arms["pair"], "gifs": gifs, "prompts": prompts, "scores": scores,
            "labels": {a["id"]: a.get("label", "") for a in arms["arms"]}}


def run_summary(name: str) -> pd.DataFrame | None:
    f = SCORED_RUNS[name] / "summary.csv"
    return pd.read_csv(f) if f.exists() else None


def cached_xattn() -> dict[str, dict]:
    """Model variant -> the cached cross-attention run of the probing prompt (clip, overlays, grids)."""
    out = {}
    if not XATTN_RUN.exists():
        return out
    for v in sorted(p for p in XATTN_RUN.iterdir() if p.is_dir()):
        for d in sorted(x for x in v.iterdir() if (x / "cross_attention.json").exists()):
            info = json.loads((d / "cross_attention.json").read_text())
            out[v.name] = {"dir": str(d), "prompt": info["prompt"], "clip": str(d / "clip.gif"),
                           "overlay": {ph: str(d / f"overlay_{ph}.gif") for ph in info["phrases"]},
                           "grid": {ph: str(d / f"grid_{ph}.png") for ph in info["phrases"]},
                           "stats": {ph: {k: v2 for k, v2 in st.items() if k != "concentration"}
                                     for ph, st in info["phrases"].items()}}
    return out
