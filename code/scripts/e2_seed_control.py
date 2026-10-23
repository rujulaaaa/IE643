"""E2 · If we fix the seed and change ONE word, does the rest of the clip stay put?

Port of notebooks/kaggle_pilots/E2_seed_control.ipynb. Seven clips share ONE fixed
noise tensor (drawn on the CPU from seed 42, so identical on any GPU): the base
"a golden retriever running on a beach", three noun swaps (cat / horse / bird,
ordered by shape distance; motion held fixed) and three verb swaps
(walking / jumping / standing; subject held fixed).

Each swap is compared with the base clip:
  d_c     1 - mean per-frame CLIP image-image cosine      (appearance)
  d_m     1 - mean cosine of frame-to-frame difference maps (a pixel motion proxy)
  S       (d_c - d_m) / (d_c + d_m), the same sign convention as S(L, t)
  layout  SSIM of frame 0                                  (composition)

Pass condition written into the notebook before it ran: mean S > 0 for noun swaps
and < 0 for verb swaps. A noun swap holds motion fixed, so a valid motion measure
should move LESS under it than under a verb swap; if d_m moves more, the motion
measure is confounded by static content.

    python scripts/e2_seed_control.py [--profile gpu|lowvram]
"""

from statistics import mean

from vdprobe import metrics as M
from vdprobe import runio
from vdprobe.pipeline import generate, make_noise
from vdprobe.summaries import selectivity

SEED, SCENE = 42, "on a beach"
CLIPS = [("BASE", "golden retriever", "running"),
         ("noun:cat", "cat", "running"), ("noun:horse", "horse", "running"), ("noun:bird", "bird", "running"),
         ("verb:walking", "golden retriever", "walking"), ("verb:jumping", "golden retriever", "jumping"),
         ("verb:standing", "golden retriever", "standing")]


def main():
    args = runio.parser(__doc__, "E2").parse_args()
    loaded = runio.load(args)
    cfg = loaded.cfg
    noise = make_noise(cfg, SEED)
    print(f"fixed noise tensor {tuple(noise.shape)}  sum={noise.float().sum():.4f}")

    clips = {}
    for tag, subj, verb in CLIPS:
        prompt = cfg.prompt(subj, verb, SCENE)
        gen = generate(loaded, prompt, noise)
        clips[tag] = {"frames": gen.frames, "prompt": prompt}
        print(f"{tag:<16}{gen.seconds + gen.decode_seconds:6.1f} s   {prompt}")
        runio.save_gif(gen.frames, args.out / "clips" / f"{tag.replace(':', '_')}.gif")

    scorer = M.ClipScorer()
    for c in clips.values():
        c["emb"] = scorer.image(c["frames"])
        c["sig"] = M.motion_signature(c["frames"])
    base = clips["BASE"]

    print(f"\n{'clip':<16}{'d_c appear':>12}{'d_m motion':>12}{'S index':>10}{'layout':>9}")
    rows = []
    for tag, c in clips.items():
        if tag == "BASE":
            continue
        d_c = M.appearance_distance(c["emb"], base["emb"])
        d_m = M.motion_distance(c["sig"], base["sig"])
        lay = M.layout_ssim(c["frames"][0], base["frames"][0])
        rows.append({"clip": tag, "d_c": round(d_c, 4), "d_m": round(d_m, 4), "S": round(selectivity(d_c, d_m), 3),
                     "layout_ssim": round(lay, 3), "prompt": c["prompt"]})
        print(f"{tag:<16}{d_c:>12.4f}{d_m:>12.4f}{rows[-1]['S']:>10.3f}{lay:>9.3f}")

    noun = [r for r in rows if r["clip"].startswith("noun")]
    verb = [r for r in rows if r["clip"].startswith("verb")]
    avg = lambda rs, k: round(mean(r[k] for r in rs), 4)
    summary = {
        "d_c_noun_mean": avg(noun, "d_c"), "d_c_verb_mean": avg(verb, "d_c"),
        "d_c_noun_over_verb": round(avg(noun, "d_c") / avg(verb, "d_c"), 2),
        "d_m_noun_mean": avg(noun, "d_m"), "d_m_verb_mean": avg(verb, "d_m"),
        "layout_ssim_noun_mean": avg(noun, "layout_ssim"), "layout_ssim_verb_mean": avg(verb, "layout_ssim"),
        "S_noun_mean": avg(noun, "S"), "S_verb_mean": avg(verb, "S"),
        "notebook_pass": avg(noun, "S") > 0 > avg(verb, "S"),
        "pixel_motion_proxy_inverted": avg(noun, "d_m") > avg(verb, "d_m"),
        "d_m_standing": next(r["d_m"] for r in rows if r["clip"] == "verb:standing"),
    }
    print(f"\nmean S  noun swaps {summary['S_noun_mean']:+.3f}   verb swaps {summary['S_verb_mean']:+.3f}  -> "
          f"{'E2 PASSES' if summary['notebook_pass'] else 'E2 INCONCLUSIVE - signs did not separate'}")
    print(f"content control: noun swaps move CLIP appearance {summary['d_c_noun_over_verb']}x more than verb swaps")
    print(f"pixel motion proxy: noun {summary['d_m_noun_mean']} vs verb {summary['d_m_verb_mean']} -> "
          f"{'INVERTED (static content confounds the motion measure)' if summary['pixel_motion_proxy_inverted'] else 'not inverted'}")

    M.contact_sheet({t: c["frames"] for t, c in clips.items()}).save(args.out / "E2_contact_sheet.png")
    print("contact sheet rows, top to bottom:", list(clips))
    runio.write_json(args.out / "E2_result.json", runio.record(
        "E2", loaded, seed=SEED, scene=SCENE, noise="fixed CPU-generated fp16 tensor, shared by all 7 clips",
        rows=rows, summary=summary))


if __name__ == "__main__":
    main()
