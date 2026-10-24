"""The project interface (Gradio). Five tabs:

  Live edit         two prompts (A, B), a direction and a method -> A, B, the prompt-only baseline and
                    the edit, generated from one noise tensor, with s_c, s_m and D
  Upload a video    a real clip as the motion source (DDIM inversion), moved onto a new subject
  Results browser   the scored Kaggle dev/test sweeps, every arm of every pair (no GPU needed)
  Where motion lives  the Stage 4 selectivity maps, per coordinate, and the H1-H3 verdicts
  Cross-attention     DAAM-style per-frame maps for the prompt's subject / verb / scene (live or
                      cached for the four model variants), and Experiment 2 results when present

A live edit costs about 65 s per generated clip on the laptop GPU (4 clips with the reference),
so the results browser is the fast path during the viva.

    python scripts/app.py                 # http://127.0.0.1:7860
    python scripts/app.py --share         # public Gradio link
    python scripts/app.py --no-gpu        # browser + maps only
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

from vdprobe.demo import (METHODS, RESULTS, SEC_PER_CLIP, VARIANTS, XATTN_RUN, EditRequest, Engine, cached_runs,
                          cached_xattn, link_assets, load_cached, reference_pool, run_summary, selection,
                          vocabulary)

CONTENT, MOTION, INK2, RULE = "#2A78D6", "#EB6834", "#55636E", "#DDE4E9"
BLOCKS = ["down_blocks.0", "down_blocks.1", "down_blocks.2", "down_blocks.3", "mid_block",
          "up_blocks.0", "up_blocks.1", "up_blocks.2", "up_blocks.3", "all"]
WINDOW_LABELS = {"first third (steps 0-8)": "ft", "all 25 steps": "all", "step 0 only": "one0",
                 "steps 0-3": "five1"}
COLS = ["clip", "sc_A", "sc_B", "sm_A", "sm_B", "D", "consistency_ratio", "motion_magnitude"]
ANALYSIS = RESULTS / "analysis"
FIGS = RESULTS / "figures"

ABOUT = """
**IE 643 course project · team Claudn't.** Probing AnimateDiff (SD 1.5 + motion adapter v1-5-2) for
where content and motion live, and editing with what was found.

- **s_c** = per-frame CLIP image cosine (content fidelity), **s_m** = cosine of dense optical-flow
  fields (motion fidelity). Suffix _A / _B = compared with clip A / clip B.
- **D** (disentanglement, roadmap §5.4): in the motion direction the output should look like B and move
  like A, D = ½[ŝc(out,B) + ŝm(out,A)] − ½[ŝc(out,A) + ŝm(out,B)]; reversed for the content direction.
  ŝ = min-max normalised over a fixed reference pool. Higher is better; compare the edit with the
  prompt-only baseline of the same row.
"""


def _table(df):
    keep = [c for c in COLS if c in df.columns]
    return df[keep].round(3)


# --- live edit ----------------------------------------------------------------------------

def live_tab(engine: Engine | None, vocab: dict):
    sel = selection()
    with gr.Tab("Live edit"):
        gr.Markdown("Clip **A** is the source. *Keep A's motion*: the output gets B's subject and A's verb, "
                    "and receives A's frame-varying activations. *Keep A's content*: A's subject with B's verb, "
                    "receiving A's frame-mean activations. Subjects can be typed freely; the dropdowns hold the "
                    "frozen probing vocabulary.")
        with gr.Row():
            with gr.Column():
                sa = gr.Dropdown(vocab["subjects"], value="golden retriever", label="A: subject", allow_custom_value=True)
                va = gr.Dropdown(vocab["verbs"], value="running", label="A: verb", allow_custom_value=True)
            with gr.Column():
                sb = gr.Dropdown(vocab["subjects"], value="horse", label="B: subject", allow_custom_value=True)
                vb = gr.Dropdown(vocab["verbs"], value="walking", label="B: verb", allow_custom_value=True)
            with gr.Column():
                scene = gr.Dropdown(vocab["scenes"], value="on a beach", label="scene", allow_custom_value=True)
                seed = gr.Number(42, precision=0, label="seed (noise tensor)")
        with gr.Row():
            direction = gr.Radio([("keep A's motion, change the subject", "motion"),
                                  ("keep A's content, change the motion", "content")],
                                 value="motion", label="direction")
            method = gr.Dropdown([(v, k) for k, v in METHODS.items()], value="selected", label="method")
        with gr.Accordion("advanced", open=False):
            gr.Markdown(f"Stage 5 selection (dev pairs, fixed rule D-31): motion = `{sel['motion']['arm']}`, "
                        f"content = `{sel['content']['arm']}`. Block and window apply to method *custom*.")
            with gr.Row():
                block = gr.Dropdown(BLOCKS, value="up_blocks.3", label="block (custom)")
                window = gr.Dropdown(list(WINDOW_LABELS), value="first third (steps 0-8)", label="window (custom)")
                alpha = gr.Slider(0, 1, value=1.0, step=0.05, label="alpha (patch / injection strength)")
                with_ref = gr.Checkbox(True, label="also generate B (needed for D, +1 clip)")
        go = gr.Button("Generate and edit", variant="primary", interactive=engine is not None)
        info = gr.Markdown("" if engine else "GPU disabled (`--no-gpu`): use the results browser.")
        with gr.Row():
            outs = [gr.Image(label=l, type="filepath", height=256) for l in
                    ("A (source)", "B (reference)", "prompt only", "edit")]
        table = gr.Dataframe(label="scores", interactive=False)

        def run(sa, va, sb, vb, scene, seed, direction, method, block, window, alpha, with_ref):
            req = EditRequest(sa, va, sb, vb, scene, int(seed), direction, method, block,
                              WINDOW_LABELS[window], float(alpha), bool(with_ref))
            try:
                res = engine.edit(req)
            except ValueError as e:
                raise gr.Error(str(e))
            g = res["gifs"]
            p = res["prompts"]
            text = (f"Done in {res['seconds']:.0f} s. D normalised with the **{res['pool']}** pool.  \n"
                    + "  \n".join(f"`{k}`: {v}" for k, v in p.items()))
            return [g.get("source"), g.get("B_ref"), g.get("base"), g.get("edit"), _table(res["table"]), text]

        def estimate(with_ref):
            n = 3 + bool(with_ref)
            return f"Running {n} clips in lockstep, about {n * SEC_PER_CLIP / 60:.0f} min on the laptop GPU ..."

        go.click(estimate, [with_ref], info).then(
            run, [sa, va, sb, vb, scene, seed, direction, method, block, window, alpha, with_ref],
            outs + [table, info])


# --- upload ------------------------------------------------------------------------------

def upload_tab(engine: Engine | None):
    with gr.Tab("Upload a video"):
        gr.Markdown("A real clip becomes the motion source. It is cut to 16 frames (stride ≤ 2, centre crop, "
                    "512 px), encoded by the VAE and DDIM-inverted onto the sampler's 25 timesteps; the edit then "
                    "takes its temporal attention (or its frame-varying activations) while the prompt names the new "
                    "subject. The new clips start from a mix of the inverted noise and the seed's noise, so their layout "
                    "follows the video: from pure Gaussian noise the transfer fails (§22 of the log). The component "
                    "patch is only stable with the *inverted* start. The *reconstruction* column shows how well the "
                    "inversion reproduces the upload.")
        with gr.Row():
            video = gr.Video(label="video (mp4 / avi / mov)", sources=["upload"])
            with gr.Column():
                src = gr.Textbox("a man running in a park, high quality, detailed", label="describe the video")
                tgt = gr.Textbox("a tiger running in a park, high quality, detailed", label="target prompt")
                method = gr.Dropdown([(METHODS[k], k) for k in ("attn_all", "attn", "selected", "tensor")],
                                     value="attn_all", label="method")
                start = gr.Radio([("blend of inverted + seed noise", "blend"), ("inverted noise", "inverted"),
                                  ("seed noise only", "gaussian")], value="blend", label="new clips start from")
                alpha = gr.Slider(0, 1, value=1.0, step=0.05, label="alpha")
                seed = gr.Number(42, precision=0, label="seed (noise the new clips start from)")
        go = gr.Button("Invert and edit", variant="primary", interactive=engine is not None)
        info = gr.Markdown()
        with gr.Row():
            outs = [gr.Image(label=l, type="filepath", height=256) for l in
                    ("upload (16 frames)", "reconstruction", "prompt only", "edit")]
        table = gr.Dataframe(label="scores (sm_video: motion vs the upload; clip_text_target: CLIP match "
                                   "to the target prompt)", interactive=False)

        def run(video, src, tgt, method, alpha, seed, start):
            if not video:
                raise gr.Error("upload a video first")
            res = engine.upload_edit(video, src, tgt, method, float(alpha), int(seed), start)
            g = res["gifs"]
            return [g["video"], g["source"], g["base"], g["edit"], res["table"].round(3),
                    f"Done in {res['seconds']:.0f} s."]

        go.click(lambda: f"Inverting (25 UNet calls) then 3 clips in lockstep, about "
                         f"{3 * SEC_PER_CLIP / 60 + 0.5:.0f} min ...", None, info).then(
            run, [video, src, tgt, method, alpha, seed, start], outs + [table, info])


# --- cached results --------------------------------------------------------------------------

def browser_tab():
    runs = cached_runs()
    with gr.Tab("Results browser"):
        if not runs:
            gr.Markdown("No sweep outputs found under `results/kaggle/`.")
            return
        names = list(runs)
        gr.Markdown("Every arm of every edit pair from the Kaggle sweeps (dev: Stage 5 localisation, 51 arms; "
                    "test: the 12 held-out pairs with the selected arms, the baselines and the three LoRAs).")
        with gr.Row():
            run = gr.Dropdown(names, value=names[0], label="run")
            pair = gr.Dropdown([p.name for p in runs[names[0]]], value=runs[names[0]][0].name, label="pair")
        prompts = gr.Markdown()
        gallery = gr.Gallery(label="arms", columns=4, height="auto", object_fit="contain")
        table = gr.Dataframe(label="per-arm scores", interactive=False)
        summary = gr.Dataframe(label="run summary: median D over pairs vs the prompt-only baseline "
                                     "(D-16 statistic, 95% paired bootstrap CI)", interactive=False)

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
                s = k
                if sc is not None and k in sc.index and not np.isnan(sc.loc[k, "D"]):
                    s += f"  ·  D {sc.loc[k, 'D']:.2f}  sm_A {sc.loc[k, 'sm_A']:.2f}  sc_B {sc.loc[k, 'sc_B']:.2f}"
                return s
            pa = c["pair"]
            md = (f"**A** = {pa['subject_a']} {pa['verb_a']}, **B** = {pa['subject_b']} {pa['verb_b']}, "
                  f"{pa['scene']}, seed {pa['seed']}.")
            tab = None
            if sc is not None:
                tab = sc.reset_index()[["arm", "sc_A", "sc_B", "sm_A", "sm_B", "D", "consistency_ratio"]].round(3)
            s = run_summary(r)
            if s is not None:
                s = s[["direction", "arm", "n_pairs", "median_D", "median_dD", "d16_ci_lo", "d16_ci_hi",
                       "frac_dD_pos"]].round(3)
            return md, [(c["gifs"][k], cap(k)) for k in keys], tab, s

        run.change(pairs_of, run, pair)
        pair.change(show, [run, pair], [prompts, gallery, table, summary])
        run.change(show, [run, pair], [prompts, gallery, table, summary])
        return show, run, pair, [prompts, gallery, table, summary]


# --- probing maps ------------------------------------------------------------------------------

def maps_tab():
    with gr.Tab("Where motion lives"):
        f = ANALYSIS / "analysis.json"
        if not f.exists():
            gr.Markdown("Run `scripts/analyze_sweep.py` first.")
            return
        a = json.loads(f.read_text())
        maps = np.load(ANALYSIS / "maps.npz", allow_pickle=True)
        h1, h2, h3 = a["H1"], a["H2"], a["H3"]
        verdict = lambda ok: "survives" if ok else "**falsified**"
        gr.Markdown(
            f"360 clips (15 subjects × 4 verbs × 2 scenes × 3 seeds), 750 one-word-swap pairs, 16 coordinates × "
            f"25 steps. S < 0: a verb swap moves the coordinate more than a subject swap (motion-selective).\n\n"
            f"- **H1 (layer)** {verdict(h1['survives'])}: motion selectivity peaks at `{h1['argmax_block']}` "
            f"(argmax in {h1['bootstrap_argmax_counts'][h1['argmax_block']]}/1000 bootstrap resamples).\n"
            f"- **H2 (timestep)** {verdict(h2['survives'])}: changepoint τ = {h2['tau']} (95% CI "
            f"{int(h2['tau_ci'][0])}–{int(h2['tau_ci'][1])}); motion lives in the first 2–3 of 25 steps.\n"
            f"- **H3 (static confound)** {verdict(h3['survives'])}: f_stat = {h3['f_stat']:.2f} (< 0.75), "
            f"no coordinate flips to motion-selective when the frame mean is removed.")
        labels = [c["short"] for c in a["coords"]]
        coord = gr.Dropdown(labels, value="up0.mm2", label="coordinate")
        plot = gr.Plot()

        def draw(name):
            i = labels.index(name)
            fig, ax = plt.subplots(figsize=(7, 3))
            t = np.arange(maps["S_dyn_median"].shape[1])
            ax.axhline(0, color=INK2, linewidth=0.8)
            for key, color, lab in (("dyn", MOTION, "S_dyn (frame-varying x̃)"), ("stat", CONTENT, "S_stat (frame mean x̄)")):
                ax.fill_between(t, maps[f"S_{key}_lo"][i], maps[f"S_{key}_hi"][i], color=color, alpha=0.15, linewidth=0)
                ax.plot(t, maps[f"S_{key}_median"][i], color=color, linewidth=2, label=lab)
            ax.set_xlabel("denoising step")
            ax.set_ylabel("median S (95% CI)")
            ax.set_title(a["coords"][i]["name"], loc="left", fontsize=9)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            return fig

        coord.change(draw, coord, plot)
        with gr.Row():
            for name in ("sweep_selectivity_maps", "sweep_h1_blocks"):
                if (FIGS / f"{name}.png").exists():
                    gr.Image(str(FIGS / f"{name}.png"), show_label=False, type="filepath")
        with gr.Row():
            for name in ("sweep_h2_timestep", "sweep_static_energy", "sweep_probes"):
                if (FIGS / f"{name}.png").exists():
                    gr.Image(str(FIGS / f"{name}.png"), show_label=False, type="filepath")
        return draw, coord, plot


# --- cross-attention and model variants ---------------------------------------------------

def exp2_summary() -> str:
    rows = []
    for f in sorted((RESULTS / "exp2").glob("*/exp2_analysis.json")):
        a = json.loads(f.read_text())
        c = a.get("confirm")
        if c:
            rows.append(f"| {a['cell']} | {c['mean_S_dyn']:+.3f} | {c['p_one_sided']:.4f} | "
                        f"{'yes' if c['replicates'] else 'no'} | {a.get('rsa', {}).get('best_motion', '-')} |")
    if not rows:
        return "Experiment 2 analyses appear here once `scripts/exp2_analysis.py` has run (results/exp2/)."
    return ("**Experiment 2 · does Experiment 1's motion site replicate?** (up_blocks.0 motion module, steps 0-2; "
            "one-sided exact sign flip over subjects, pre-registered D-35)\n\n"
            "| model cell | mean S_dyn | p | replicates | strongest RSA motion coordinate |\n|---|---|---|---|---|\n"
            + "\n".join(rows))


def xattn_tab(engine: Engine | None, vocab: dict):
    with gr.Tab("Cross-attention"):
        gr.Markdown("Where each prompt word looks, frame by frame: the head-averaged attention mass of every spatial "
                    "cross-attention layer on the word's tokens, upsampled and summed over the 16 layers and 25 steps "
                    "(DAAM, Tang et al. ACL 2023, applied per frame). The grid splits it by layer (rows) and step window "
                    "(columns). Motion modules take no text, so these maps cannot show motion processing itself.")
        cached = cached_xattn()
        if cached:
            names = list(cached)
            gr.Markdown(f"**Cached: the four model cells** (base image model × motion adapter), "
                        f"prompt *{cached[names[0]]['prompt']}*, seed 42.")
            sheet = XATTN_RUN / "variants_contact_sheet.png"
            if sheet.exists():
                gr.Image(str(sheet), show_label=False, type="filepath")
            with gr.Row():
                var = gr.Dropdown(names, value=names[0], label="model cell")
                phrase = gr.Radio(["subject", "verb", "scene"], value="subject", label="word")
            with gr.Row():
                c_clip = gr.Image(label="clip", type="filepath", height=256)
                c_over = gr.Image(label="attention overlay", type="filepath", height=256)
                c_grid = gr.Image(label="per layer x step window", type="filepath", height=512)
            c_stats = gr.Markdown()

            def show(v, ph):
                c = cached[v]
                st = c["stats"][ph]
                return (c["clip"], c["overlay"][ph], c["grid"][ph],
                        f"mask area {st['mask_area']:.3f} of the frame; mask IoU across frames "
                        f"{st['mask_iou_across_frames']:.3f} (falls when the subject travels or morphs)")
            var.change(show, [var, phrase], [c_clip, c_over, c_grid, c_stats])
            phrase.change(show, [var, phrase], [c_clip, c_over, c_grid, c_stats])
        gr.Markdown(f"**Live** (model cell: `{engine.variant if engine else '-'}`; about {SEC_PER_CLIP / 60 + 0.3:.0f} "
                    "min per clip)")
        with gr.Row():
            sj = gr.Dropdown(vocab["subjects"], value="golden retriever", label="subject", allow_custom_value=True)
            vb = gr.Dropdown(vocab["verbs"], value="running", label="verb", allow_custom_value=True)
            sc = gr.Dropdown(vocab["scenes"], value="on a beach", label="scene", allow_custom_value=True)
            sd = gr.Number(42, precision=0, label="seed")
        go = gr.Button("Generate with cross-attention maps", variant="primary", interactive=engine is not None)
        with gr.Row():
            l_clip = gr.Image(label="clip", type="filepath", height=256)
            l_overs = [gr.Image(label=f"{ph} overlay", type="filepath", height=256) for ph in ("subject", "verb", "scene")]
        l_grid = gr.Image(label="subject: per layer x step window", type="filepath", height=400)
        l_info = gr.Markdown("" if engine else "GPU disabled (`--no-gpu`): only the cached maps are available.")

        def run(sj, vb, sc, sd):
            try:
                r = engine.cross_attention(sj, vb, sc, int(sd))
            except ValueError as e:
                raise gr.Error(str(e))
            return ([r["clip"]] + [r["overlay"][ph] for ph in ("subject", "verb", "scene")] + [r["grid"]["subject"]]
                    + [f"Done in {r['seconds']:.0f} s · *{r['prompt']}* · mask area "
                       + ", ".join(f"{k} {v:.2f}" for k, v in r["mask_area"].items())])
        go.click(run, [sj, vb, sc, sd], [l_clip, *l_overs, l_grid, l_info])
        gr.Markdown(exp2_summary())
        if cached:
            return show, var, phrase, [c_clip, c_over, c_grid, c_stats]


def build(engine: Engine | None) -> gr.Blocks:
    with gr.Blocks(title="Content vs motion in AnimateDiff") as demo:
        gr.Markdown("# Content vs. motion in a video diffusion U-Net")
        gr.Markdown(ABOUT)
        try:
            gr.Markdown(f"D reference pool: **{reference_pool()[0]}** run."
                        + (f" Model cell: **{engine.variant}**." if engine else ""))
        except FileNotFoundError:
            gr.Markdown("No scored run found: edits show s_c and s_m, without D.")
        live_tab(engine, vocabulary())
        upload_tab(engine)
        b = browser_tab()
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
    demo.launch(share=args.share, server_port=args.port, allowed_paths=[str(RESULTS)])


if __name__ == "__main__":
    main()
