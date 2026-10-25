"""The project interface (Gradio). Five tabs:

  Edit                  two prompts (A, B) and what to keep from A -> A, B, the prompt-only baseline and the
                        edit, generated from one noise tensor, with the content / motion scores and D
  Edit from your video  a real clip as the motion source (DDIM inversion), moved onto a new subject
  Results               the saved edit sweeps, every arm of every pair (no GPU needed)
  Where motion lives    the Stage 4 selectivity maps, per coordinate, and the H1-H3 verdicts
  Cross-attention       per-frame maps of where each prompt word looks (saved for the four model cells, or
                        live), and Experiment 2 results when present

A live edit costs 70-105 s per generated clip on the laptop GPU (4 clips with the reference),
so the results browser is the fast path during the viva.

Design (docs/experiment2.md, session 5 of the log): colour means a channel and nothing else —
blue ■ = content, orange ▲ = motion, always with the shape too — and the controls stay neutral ink.
Clips play as videos with controls instead of looping on their own; long runs show determinate progress.

    python scripts/app.py                 # http://127.0.0.1:7860
    python scripts/app.py --share         # public Gradio link
    python scripts/app.py --no-gpu        # results, maps and saved attention only
    python scripts/app.py --variant epicrealism_mm3   # an Experiment 2 model cell
"""

import argparse
import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vdprobe.demo import (RESULTS, SEC_PER_CLIP, VARIANTS, XATTN_RUN, EditRequest, Engine, arm_label, cached_runs,
                          cached_xattn, link_assets, load_cached, playable, pretty_scores, reference_pool,
                          run_summary, selection, vocabulary)

# --- design tokens (contrast checked: text >= 5.4:1, data graphics >= 3.3:1, light and dark) ---------------
INK, INK2, CARD, LINE = "#1B2127", "#4F5B66", "#F4F6F8", "#D5DCE2"
CONTENT, MOTION = "#2A78D6", "#E0602C"
ANALYSIS = RESULTS / "analysis"
FIGS = RESULTS / "figures"
BLOCKS = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3", "mid_block",
          "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3", "all"]
WINDOW_LABELS = {"Steps 0-8 (first third)": "ft", "All 25 steps": "all", "Step 0 only": "one0", "Steps 0-3": "five1"}
METHOD_LABELS = {
    "lit_dual": "Attention edit, appearance from one clip + motion from the other (needs B; passed on test)",
    "lit_single": "Attention edit, from A only (MasaCtrl / UniEdit, chosen on dev)",
    "selected": "Selected patch (chosen on the dev pairs)",
    "custom": "Patch a block and window of your choice",
    "attn": "Temporal attention injection, steps 0-8",
    "attn_all": "Temporal attention injection, all steps",
    "tensor": "Whole-tensor patch (control)",
    "lora_probe": "Prompt + probe-guided motion LoRA",
    "lora_full": "Prompt + full motion LoRA",
    "lora_anti": "Prompt + wrong-place LoRA (control)",
}
SYSTEM_FONT = ["-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]
MONO_FONT = ["ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace"]

THEME = gr.themes.Default(
    primary_hue=gr.themes.colors.slate, neutral_hue=gr.themes.colors.slate, font=SYSTEM_FONT, font_mono=MONO_FONT,
    text_size=gr.themes.Size(xxs="11px", xs="12px", sm="13px", md="15px", lg="17px", xl="22px", xxl="28px"),
    radius_size=gr.themes.sizes.radius_md,
).set(
    body_background_fill="#FFFFFF", body_background_fill_dark="#101418",
    background_fill_primary="#FFFFFF", background_fill_primary_dark="#101418",
    background_fill_secondary=CARD, background_fill_secondary_dark="#171C21",
    block_background_fill="#FFFFFF", block_background_fill_dark="#171C21",
    body_text_color=INK, body_text_color_dark="#E8ECF0",
    body_text_color_subdued=INK2, body_text_color_subdued_dark="#A7B1BB",
    block_label_text_color=INK2, block_label_text_color_dark="#A7B1BB",
    block_title_text_color=INK, block_title_text_color_dark="#E8ECF0",
    border_color_primary=LINE, border_color_primary_dark="#323A42",
    button_primary_background_fill=INK, button_primary_background_fill_dark="#E8ECF0",
    button_primary_background_fill_hover="#2E3740", button_primary_background_fill_hover_dark="#FFFFFF",
    button_primary_text_color="#FFFFFF", button_primary_text_color_dark="#101418",
    link_text_color="#1D5FAF", link_text_color_dark="#7FB2F2",
)

CSS = """
:root { --c-content: #2A78D6; --c-motion: #E0602C; --t-content: #1D5FAF; --t-motion: #AE4318; }
.dark { --c-content: #5B9BEB; --c-motion: #F08A5D; --t-content: #7FB2F2; --t-motion: #F39A70; }
.hero h1 { font-size: 28px; font-weight: 650; letter-spacing: -0.01em; margin: 4px 0 6px; }
.hero .thesis { font-size: 17px; margin: 0 0 12px; max-width: 62ch; }
.hero .thesis .c { color: var(--t-content); font-weight: 600; }
.hero .thesis .m { color: var(--t-motion); font-weight: 600; }
.legend { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 4px; }
.chip { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px; border-radius: 999px;
        border: 1px solid var(--border-color-primary); font-size: 14px; }
.chip .dot-c { width: 11px; height: 11px; background: var(--c-content); display: inline-block; }
.chip .dot-m { width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent;
               border-bottom: 11px solid var(--c-motion); display: inline-block; }
.lede { font-size: 15px; max-width: 75ch; }
table, table * { font-family: var(--font) !important; font-variant-numeric: tabular-nums; }
.group-title { padding: 8px 12px 0; }
button[role="tab"][aria-selected="true"] { color: var(--body-text-color) !important; font-weight: 600;
                                          border-bottom-color: var(--body-text-color) !important; }
footer { display: none !important; }
"""

HEADER = """
<div class="hero">
  <h1>Content vs. motion in a video diffusion model</h1>
  <p class="thesis">Where AnimateDiff decides <span class="c">what is shown</span> and <span class="m">how it
  moves</span>, and edits that change one while keeping the other.</p>
  <div class="legend">
    <span class="chip"><span class="dot-c"></span>Content: what is depicted</span>
    <span class="chip"><span class="dot-m"></span>Motion: how it moves</span>
  </div>
</div>
"""

SCORES_HELP = """
- **■ Looks like A / B**: CLIP image similarity, frame by frame (1 = looks the same).
- **▲ Moves like A / B**: cosine of the optical-flow fields (1 = every pixel moves the same way).
- **Disentanglement D**: for *keep A's motion*, ½(looks like B + moves like A) − ½(looks like A + moves like B),
  each rescaled over a reference pool of edits; the other direction mirrors it. Higher is better. Compare an edit
  with the *Prompt only* row: that is the same prompt without any intervention.
- **Identity stability**: how steady the subject's look stays over the clip, relative to clip A (1 = as steady).
- **Motion amount**: mean optical-flow speed, in pixels per frame at 128 px.

IE 643 course project, team Claudn't. Model: Stable Diffusion 1.5 with the AnimateDiff motion adapter
(16 frames, 512 × 512, 25 DDIM steps).
"""


EDIT_COLS = ["Clip", "■ Looks like A", "■ Looks like B", "▲ Moves like A", "▲ Moves like B", "Disentanglement D",
             "Identity stability", "Motion amount (px)"]
UPLOAD_COLS = ["Clip", "▲ Moves like the video", "■ Looks like the video", "Matches target prompt", "Motion amount (px)"]
CELL_NAMES = {"sd15_mm2": "SD 1.5 + adapter v1-5-2 (the frozen model)", "epicrealism_mm2": "epiCRealism + adapter v1-5-2",
              "sd15_mm3": "SD 1.5 + adapter v1-5-3", "epicrealism_mm3": "epiCRealism + adapter v1-5-3"}


def video_outputs(labels, height=256):
    return [gr.Video(label=l, height=height, autoplay=False, loop=True, interactive=False) for l in labels]


def no_gpu_note(engine):
    return ("" if engine else "Generating needs a GPU, and this session runs without one (`--no-gpu`). "
            "**Results**, **Where motion lives** and the saved cross-attention maps work without it.")


# --- Edit -----------------------------------------------------------------------------------------

def edit_tab(engine: Engine | None, vocab: dict):
    sel = selection()
    with gr.Tab("Edit"):
        gr.Markdown("Generate clip **A**, then a clip that keeps one of A's channels and takes the other from **B**. "
                    "Every clip starts from the same noise, so any difference comes from the edit. Subjects and "
                    "verbs can be typed freely; the lists hold the probing vocabulary.", elem_classes="lede")
        with gr.Row(equal_height=True):
            with gr.Group():
                gr.Markdown("**Clip A** · the source", elem_classes="group-title")
                sa = gr.Dropdown(vocab["subjects"], value="golden retriever", label="Subject", allow_custom_value=True)
                va = gr.Dropdown(vocab["verbs"], value="running", label="Verb", allow_custom_value=True)
            with gr.Group():
                gr.Markdown("**Clip B** · the target", elem_classes="group-title")
                sb = gr.Dropdown(vocab["subjects"], value="horse", label="Subject", allow_custom_value=True)
                vb = gr.Dropdown(vocab["verbs"], value="walking", label="Verb", allow_custom_value=True)
            with gr.Group():
                gr.Markdown("**Shared**", elem_classes="group-title")
                scene = gr.Dropdown(vocab["scenes"], value="on a beach", label="Scene", allow_custom_value=True)
                seed = gr.Number(42, precision=0, label="Seed (fixes the starting noise)")
        with gr.Row():
            direction = gr.Radio([("▲ Keep A's motion, give it B's subject", "motion"),
                                  ("■ Keep A's subject, give it B's motion", "content")],
                                 value="motion", label="What to keep from A")
            method = gr.Dropdown([(v, k) for k, v in METHOD_LABELS.items()], value="lit_dual", label="Method")
        with gr.Accordion("Advanced settings", open=False):
            gr.Markdown(f"The selected patch was fixed on the dev pairs before the test run (rule D-31): "
                        f"keep motion = `{arm_label(sel['motion']['arm'])}`, keep content = "
                        f"`{arm_label(sel['content']['arm'])}`. Block and window apply to *Patch a block and "
                        f"window of your choice*.")
            with gr.Row():
                block = gr.Dropdown(BLOCKS, value="up_blocks.3", label="Block")
                window = gr.Dropdown(list(WINDOW_LABELS), value="Steps 0-8 (first third)", label="Steps")
                alpha = gr.Slider(0, 1, value=1.0, step=0.05, label="Strength (1 = full patch or injection)")
                with_ref = gr.Checkbox(True, label="Also generate B (needed for D; one more clip)")
        go = gr.Button("Generate edit", variant="primary", interactive=engine is not None)
        status = gr.Markdown(no_gpu_note(engine))
        with gr.Row():
            outs = video_outputs(("A · source", "B · as generated", "Prompt only", "Edit"))
        table = gr.Dataframe(headers=EDIT_COLS, label="Scores", interactive=False, wrap=True)

        def run(sa, va, sb, vb, scene, seed, direction, method, block, window, alpha, with_ref,
                progress=gr.Progress()):
            req = EditRequest(sa, va, sb, vb, scene, int(seed), direction, method, block,
                              WINDOW_LABELS[window], float(alpha), bool(with_ref))
            try:
                res = engine.edit(req, progress=progress)
            except ValueError as e:
                raise gr.Error(str(e))
            g, p = res["gifs"], res["prompts"]
            text = (f"Done in {res['seconds'] / 60:.1f} min. D uses the **{res['pool']}** reference pool.  \n"
                    + "  \n".join(f"*{arm_label(k)}*: {v}" for k, v in p.items()))
            return [playable(g.get(k)) for k in ("source", "B_ref", "base", "edit")] + [pretty_scores(res["table"]), text]

        go.click(run, [sa, va, sb, vb, scene, seed, direction, method, block, window, alpha, with_ref],
                 outs + [table, status])


# --- Edit from your video ---------------------------------------------------------------------------

def upload_tab(engine: Engine | None):
    with gr.Tab("Edit from your video"):
        gr.Markdown("Use a real clip as the motion source. It is cut to 16 frames (centre crop, 512 px) and inverted "
                    "onto the sampler's 25 steps; the edit then takes its motion while your prompt names the new "
                    "subject. *Reconstruction* shows how well the inversion reproduces your video.", elem_classes="lede")
        with gr.Row():
            video = gr.Video(label="Your video (mp4, avi or mov)", sources=["upload"])
            with gr.Column():
                src = gr.Textbox("a man running in a park, high quality, detailed", label="Describe your video",
                                 placeholder="a man running in a park")
                tgt = gr.Textbox("a tiger running in a park, high quality, detailed", label="Target prompt",
                                 placeholder="a tiger running in a park")
                method = gr.Dropdown([(METHOD_LABELS[k], k) for k in ("attn_all", "attn", "selected", "tensor")],
                                     value="attn_all", label="Method")
        with gr.Accordion("Advanced settings", open=False):
            gr.Markdown("New clips start from a mix of the inverted noise and the seed's noise, so their layout follows "
                        "the video. From the seed's noise alone the transfer fails (log §22), and the selected patch is "
                        "stable only from the inverted noise.")
            with gr.Row():
                start = gr.Radio([("Mix of inverted and seed noise", "blend"), ("Inverted noise", "inverted"),
                                  ("Seed noise only", "gaussian")], value="blend", label="New clips start from")
                alpha = gr.Slider(0, 1, value=1.0, step=0.05, label="Strength")
                seed = gr.Number(42, precision=0, label="Seed")
        go = gr.Button("Transfer the video's motion", variant="primary", interactive=engine is not None)
        status = gr.Markdown(no_gpu_note(engine))
        with gr.Row():
            outs = video_outputs(("Your video · 16 frames", "Reconstruction", "Prompt only", "Edit"))
        table = gr.Dataframe(headers=UPLOAD_COLS, label="Scores", interactive=False, wrap=True)

        def run(video, src, tgt, method, alpha, seed, start, progress=gr.Progress()):
            if not video:
                raise gr.Error("Add a video first: drop an mp4, avi or mov file in the box above.")
            res = engine.upload_edit(video, src, tgt, method, float(alpha), int(seed), start, progress=progress)
            g = res["gifs"]
            return ([playable(g[k]) for k in ("video", "source", "base", "edit")]
                    + [pretty_scores(res["table"]), f"Done in {res['seconds'] / 60:.1f} min."])

        go.click(run, [video, src, tgt, method, alpha, seed, start], outs + [table, status])


# --- Results --------------------------------------------------------------------------------------------

SUMMARY_NAMES = {"direction": "Keep", "arm": "Clip", "n_pairs": "Pairs", "median_D": "Median D",
                 "median_dD": "Median gain over prompt only", "d16_ci_lo": "95% CI low", "d16_ci_hi": "95% CI high",
                 "frac_dD_pos": "Share of pairs improved"}


def results_tab():
    runs = cached_runs()
    with gr.Tab("Results"):
        if not runs:
            gr.Markdown("No saved edit runs on this machine yet. Copy the Kaggle sweeps into `results/kaggle/` "
                        "(`vdprobe-test`, `vdprobe-dev`) or run `scripts/edit_sweep.py`, then reload.")
            return None
        names = list(runs)
        gr.Markdown("Every clip of every saved edit pair, with its scores. No GPU needed. Each run lists the pairs it "
                    "finished; the summary compares every method with the prompt-only clip of the same pair.",
                    elem_classes="lede")
        with gr.Row():
            run = gr.Dropdown(names, value=names[0], label="Run")
            pair = gr.Dropdown([p.name for p in runs[names[0]]], value=runs[names[0]][0].name, label="Pair")
        prompts = gr.Markdown()
        gallery = gr.Gallery(label="Clips", columns=4, height="auto", object_fit="contain")
        table = gr.Dataframe(label="Scores per clip", interactive=False, wrap=True)
        summary = gr.Dataframe(label="Run summary: median D over pairs and the gain over prompt only "
                                     "(95% paired bootstrap CI)", interactive=False, wrap=True)

        def pairs_of(r):
            ps = [p.name for p in runs[r]]
            return gr.Dropdown(choices=ps, value=ps[0])

        def show(r, name):
            d = next(p for p in runs[r] if p.name == name)
            c = load_cached(d)
            order = ["source", "B_ref", "motion.base", "content.base"]
            keys = [k for k in order if k in c["gifs"]] + sorted(k for k in c["gifs"] if k not in order)
            sc = c["scores"].set_index("arm") if c["scores"] is not None else None

            def cap(k):
                s = arm_label(k)
                if sc is not None and k in sc.index and not np.isnan(sc.loc[k, "D"]):
                    s += f" · D {sc.loc[k, 'D']:.2f} · ▲ moves like A {sc.loc[k, 'sm_A']:.2f} · ■ looks like B {sc.loc[k, 'sc_B']:.2f}"
                return s
            pa = c["pair"]
            md = (f"**A** = {pa['subject_a']} {pa['verb_a']} · **B** = {pa['subject_b']} {pa['verb_b']} · "
                  f"{pa['scene']} · seed {pa['seed']}")
            tab = None
            if sc is not None:
                tab = pretty_scores(sc.reset_index()[["arm", "sc_A", "sc_B", "sm_A", "sm_B", "D", "consistency_ratio"]])
            s = run_summary(r)
            if s is not None:
                s = s[list(SUMMARY_NAMES)].copy()
                s["arm"] = s["arm"].map(arm_label)
                s["direction"] = s["direction"].map({"motion": "▲ Motion", "content": "■ Content"}).fillna(s["direction"])
                s = s.rename(columns=SUMMARY_NAMES).round(3)
            return md, [(playable(c["gifs"][k]), cap(k)) for k in keys], tab, s

        run.change(pairs_of, run, pair)
        pair.change(show, [run, pair], [prompts, gallery, table, summary])
        run.change(show, [run, pair], [prompts, gallery, table, summary])
        return show, run, pair, [prompts, gallery, table, summary]


# --- Where motion lives -------------------------------------------------------------------------------

def _headline(maps, i, name):
    lo, hi = maps["S_dyn_lo"][i], maps["S_dyn_hi"][i]
    mo = np.where(hi < 0)[0]
    co = np.where(lo > 0)[0]
    if len(mo):
        when = f"step {mo.min()}" if mo.min() == mo.max() else f"steps {mo.min()}–{mo.max()}"
        return f"{name}: the frame-varying part leans to motion at {when}"
    if len(co):
        return f"{name}: no step leans to motion; content-leaning at {len(co)} of {len(lo)} steps"
    return f"{name}: no step where the frame-varying part reliably leans either way"


def maps_tab():
    with gr.Tab("Where motion lives"):
        f = ANALYSIS / "analysis.json"
        if not f.exists():
            gr.Markdown("No probing analysis on this machine yet. Run `scripts/analyze_sweep.py`, then reload.")
            return None
        a = json.loads(f.read_text())
        maps = np.load(ANALYSIS / "maps.npz", allow_pickle=True)
        h1, h2, h3 = a["H1"], a["H2"], a["H3"]
        verdict = lambda ok: "survives" if ok else "**falsified**"
        gr.Markdown(
            f"360 clips (15 subjects × 4 verbs × 2 scenes × 3 seeds), 750 pairs that each change one word, 16 layers "
            f"× 25 steps. S compares how much a layer changes when the **verb** is swapped (▲ motion) with when the "
            f"**subject** is swapped (■ content): S < 0 leans to motion, S > 0 to content.\n\n"
            f"- **H1 (which layer)** {verdict(h1['survives'])}: motion selectivity peaks at `{h1['argmax_block']}` "
            f"({h1['bootstrap_argmax_counts'][h1['argmax_block']]} of 1,000 bootstrap resamples).\n"
            f"- **H2 (which steps)** {verdict(h2['survives'])}: changepoint at step {h2['tau']} (95% CI "
            f"{int(h2['tau_ci'][0])}–{int(h2['tau_ci'][1])}): motion is settled in the first 2–3 of 25 steps.\n"
            f"- **H3 (frame mean hides motion)** {verdict(h3['survives'])}: removing the frame mean moves every layer "
            f"towards motion but flips none (f_stat {h3['f_stat']:.2f}, below 0.75).", elem_classes="lede")
        labels = [c["short"] for c in a["coords"]]
        coord = gr.Dropdown(labels, value="up0.mm2", label="Layer")
        plot = gr.Plot(label="Selectivity over the denoising steps")

        def draw(name):
            i = labels.index(name)
            fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=CARD)
            ax.set_facecolor(CARD)
            t = np.arange(maps["S_dyn_median"].shape[1])
            ax.axhline(0, color=INK2, linewidth=0.8)
            styles = (("dyn", MOTION, "-", "^", "frame-varying part (▲)"), ("stat", CONTENT, "--", "s", "frame mean (■)"))
            ends = {k: maps[f"S_{k}_median"][i][-1] for k in ("dyn", "stat")}
            upper = max(ends, key=ends.get)          # nudge the end labels apart: the higher line's label goes up
            for key, color, ls, mk, lab in styles:
                med = maps[f"S_{key}_median"][i]
                ax.fill_between(t, maps[f"S_{key}_lo"][i], maps[f"S_{key}_hi"][i], color=color, alpha=0.14, linewidth=0)
                ax.plot(t, med, color=color, linewidth=2, linestyle=ls, marker=mk, markevery=4, markersize=6)
                ax.annotate(lab, (t[-1], med[-1]), xytext=(6, 7 if key == upper else -7), textcoords="offset points",
                            va="center", fontsize=10, color=INK)
            ax.set_xlabel("denoising step (0 = pure noise)", color=INK)
            ax.set_ylabel("median S, 95% CI", color=INK)
            ax.text(0.005, 0.98, "↑ content-leaning", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
            ax.text(0.005, 0.02, "↓ motion-leaning", transform=ax.transAxes, va="bottom", fontsize=9, color=INK2)
            ax.set_title(_headline(maps, i, name), loc="left", fontsize=12, color=INK)
            ax.tick_params(colors=INK2)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(LINE)
            ax.set_xlim(-0.5, t[-1] + 5.5)
            fig.tight_layout()
            return fig

        coord.change(draw, coord, plot)
        captions = {"sweep_selectivity_maps": "S for every layer and step (frame-varying part left, frame mean right)",
                    "sweep_h1_blocks": "H1: motion selectivity per block",
                    "sweep_h2_timestep": "H2: motion selectivity over the steps, with the changepoint",
                    "sweep_static_energy": "How much of each layer's signal is its frame mean",
                    "sweep_probes": "Linear probes: can the verb be read off the layer?"}
        with gr.Accordion("All Stage 4 figures", open=False):
            for name, cap in captions.items():
                if (FIGS / f"{name}.png").exists():
                    gr.Image(str(FIGS / f"{name}.png"), label=cap, type="filepath")
        return draw, coord, plot


# --- Cross-attention ---------------------------------------------------------------------------------

def exp2_summary() -> str:
    rows = []
    for f in sorted((RESULTS / "exp2").glob("*/exp2_analysis.json")):
        a = json.loads(f.read_text())
        c = a.get("confirm")
        if c:
            rows.append(f"| {a['cell']} | {c['mean_S_dyn']:+.3f} | {c['p_one_sided']:.4f} | "
                        f"{'yes' if c['replicates'] else 'no'} | {a.get('rsa', {}).get('best_motion', '–')} |")
    if not rows:
        return ("Experiment 2 results appear here once `scripts/exp2_analysis.py` has written `results/exp2/`. "
                "The sweep is running on Kaggle.")
    return ("Does Experiment 1's motion site (up_blocks.0, steps 0–2) replicate on other models? One-sided exact "
            "sign flip over subjects, fixed in advance (D-35).\n\n"
            "| Model cell | Mean S (frame-varying) | p | Replicates | Strongest flow-matched layer |\n"
            "|---|---|---|---|---|\n" + "\n".join(rows))


def xattn_tab(engine: Engine | None, vocab: dict):
    with gr.Tab("Cross-attention"):
        gr.Markdown("Where each prompt word looks, frame by frame: the attention every spatial cross-attention layer "
                    "pays to the word, summed over the 16 layers and 25 steps (DAAM, Tang et al., ACL 2023, applied "
                    "per frame). The grid splits it by layer and step. Motion modules do not read the text, so these "
                    "maps show where words land, not how motion is computed.", elem_classes="lede")
        cached = cached_xattn()
        handles = None
        with gr.Tabs():
            if cached:
                with gr.Tab("Four model cells (saved)"):
                    names = list(cached)
                    gr.Markdown(f"Prompt *{cached[names[0]]['prompt']}*, seed 42: base image model "
                                f"(SD 1.5 or epiCRealism) × motion adapter (v1-5-2 or v1-5-3).")
                    with gr.Row():
                        var = gr.Dropdown([(CELL_NAMES.get(n, n), n) for n in names], value=names[0], label="Model cell")
                        phrase = gr.Radio([("Subject", "subject"), ("Verb", "verb"), ("Scene", "scene")],
                                          value="subject", label="Word")
                    with gr.Row():
                        c_clip, c_over = video_outputs(("Clip", "Where the word looks"), height=320)
                    c_stats = gr.Markdown()
                    c_grid = gr.Image(label="Where the word looks, by layer and denoising steps (each cell scaled "
                                            "to its own maximum)", type="filepath")
                    sheet = XATTN_RUN / "variants_contact_sheet.png"
                    if sheet.exists():
                        with gr.Accordion("All four cells side by side", open=False):
                            gr.Image(str(sheet), label="Frames 0, 5, 10, 15; subject attention; verb attention",
                                     type="filepath")

                    def show(v, ph):
                        c = cached[v]
                        st = c["stats"][ph]
                        return (playable(c["clip"]), playable(c["overlay"][ph]), c["grid"][ph],
                                f"The word's mask covers {st['mask_area'] * 100:.0f}% of the frame; frame-to-frame "
                                f"overlap {st['mask_iou_across_frames']:.2f} (it falls when the subject travels or "
                                f"morphs).")
                    var.change(show, [var, phrase], [c_clip, c_over, c_grid, c_stats])
                    phrase.change(show, [var, phrase], [c_clip, c_over, c_grid, c_stats])
                    handles = (show, var, phrase, [c_clip, c_over, c_grid, c_stats])
            with gr.Tab("Try a prompt"):
                gr.Markdown(f"Model: {CELL_NAMES.get(engine.variant, engine.variant) if engine else '–'}, about "
                            f"{SEC_PER_CLIP / 60 + 0.3:.0f} min per clip.")
                with gr.Row():
                    sj = gr.Dropdown(vocab["subjects"], value="golden retriever", label="Subject", allow_custom_value=True)
                    vb = gr.Dropdown(vocab["verbs"], value="running", label="Verb", allow_custom_value=True)
                    sc = gr.Dropdown(vocab["scenes"], value="on a beach", label="Scene", allow_custom_value=True)
                    sd = gr.Number(42, precision=0, label="Seed")
                go = gr.Button("Generate and map the attention", variant="primary", interactive=engine is not None)
                l_info = gr.Markdown(no_gpu_note(engine))
                with gr.Row():
                    l_vids = video_outputs(("Clip", "Subject", "Verb", "Scene"), height=220)
                l_grid = gr.Image(label="Subject, by layer and denoising steps (each cell scaled to its own "
                                        "maximum)", type="filepath")

                def run(sj, vb, sc, sd, progress=gr.Progress()):
                    try:
                        r = engine.cross_attention(sj, vb, sc, int(sd), progress=progress)
                    except ValueError as e:
                        raise gr.Error(str(e))
                    vids = [playable(r["clip"])] + [playable(r["overlay"][ph]) for ph in ("subject", "verb", "scene")]
                    area = ", ".join(f"{k} {v * 100:.0f}%" for k, v in r["mask_area"].items())
                    return vids + [r["grid"]["subject"], f"Done in {r['seconds'] / 60:.1f} min · *{r['prompt']}* · "
                                                         f"share of the frame each word covers: {area}"]
                go.click(run, [sj, vb, sc, sd], [*l_vids, l_grid, l_info])
        with gr.Accordion("Experiment 2: does the motion site replicate?", open=False):
            gr.Markdown(exp2_summary())
        return handles


# --- app -------------------------------------------------------------------------------------------------

def build(engine: Engine | None) -> gr.Blocks:
    with gr.Blocks(title="Content vs. motion in AnimateDiff") as demo:
        gr.HTML(HEADER)
        with gr.Accordion("How the scores work", open=False):
            try:
                pool = reference_pool()[0]
            except FileNotFoundError:
                pool = "none on this machine: edits show the content and motion scores without D"
            gr.Markdown(SCORES_HELP + f"\nReference pool for D: **{pool}**."
                        + (f" Model cell: **{engine.variant}**." if engine else ""))
        edit_tab(engine, vocabulary())
        upload_tab(engine)
        b = results_tab()
        m = maps_tab()
        x = xattn_tab(engine, vocabulary())
        if b:
            demo.load(b[0], [b[1], b[2]], b[3])
        if m:
            demo.load(m[0], m[1], m[2])
        if x:
            demo.load(x[0], [x[1], x[2]], x[3])
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--preload", action="store_true", help="load the models at start, not on the first edit")
    ap.add_argument("--variant", default="sd15_mm2", choices=sorted(VARIANTS), help="model cell (Experiment 2)")
    ap.add_argument("--assets", type=Path, default=None, help="directory laid out like the repo (results/, data/)")
    args = ap.parse_args()
    if args.assets:
        print("linked", link_assets(args.assets), "asset files", flush=True)
    engine = None if args.no_gpu else Engine(variant=args.variant)
    if engine and args.preload:
        engine.ensure()
    demo = build(engine)
    demo.queue(default_concurrency_limit=1)      # one GPU: edits run one at a time
    demo.launch(share=args.share, server_port=args.port, allowed_paths=[str(RESULTS)], theme=THEME, css=CSS)


if __name__ == "__main__":
    main()
