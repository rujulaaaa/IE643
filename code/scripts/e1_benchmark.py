"""E1 · Can we generate video at all on this hardware, and at what unit cost?

Port of notebooks/kaggle_pilots/E1_animatediff_benchmark.ipynb. One warm-up
generation, then N timed ones, at the frozen protocol (16 frames, 512x512, 25 DDIM
steps, fp16, CFG 7.5). The wall-clock per clip is the unit behind every budget.

--config-sweep repeats the notebook's memory-config sweep (6 extra generations):
what is the minimum VRAM this configuration needs?

    python scripts/e1_benchmark.py [--profile gpu|lowvram] [--runs 3] [--config-sweep]
"""

import gc
import statistics as st

import torch

from vdprobe import runio
from vdprobe.pipeline import build_pipeline, generate, make_noise

SWEEP = [("fp16, no slicing", dict(profile="gpu", vae_slicing=False)),
         ("fp16 + vae slicing", dict(profile="gpu", vae_slicing=True)),
         ("fp16 + slicing + cpu offload", dict(profile="lowvram", vae_slicing=True))]


def config_sweep(cfg, prompt, noise) -> list:
    rows = []
    for name, kw in SWEEP:
        gc.collect()
        torch.cuda.empty_cache()
        try:
            loaded = build_pipeline(cfg, **kw)
            generate(loaded, prompt, noise)                       # warm
            g = generate(loaded, prompt, noise)                   # measured
            rows.append([name, f"{g.seconds + g.decode_seconds:.1f}", f"{g.peak_reserved_gb:.2f}"])
        except torch.cuda.OutOfMemoryError:
            rows.append([name, "OOM", "OOM"])
        finally:
            loaded = None
        print(f"{name:<32} {rows[-1][1]:>7} s   {rows[-1][2]:>5} GB peak")
    return rows

PROMPT_ARGS = ("golden retriever", "running", "on a beach")
SEED = 42


def main():
    p = runio.parser(__doc__, "E1")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--config-sweep", action="store_true", help="also run the notebook's memory-config sweep")
    args = p.parse_args()
    loaded = runio.load(args)
    prompt = loaded.cfg.prompt(*PROMPT_ARGS)
    noise = make_noise(loaded.cfg, SEED)
    n_params = sum(x.numel() for x in loaded.unet.parameters())

    print(f"{'run':<10}{'denoise s':>10}{'decode s':>10}{'alloc GB':>10}{'resv GB':>9}")
    warm = generate(loaded, prompt, noise)
    print(f"{'warm-up':<10}{warm.seconds:>10.1f}{warm.decode_seconds:>10.1f}"
          f"{warm.peak_alloc_gb:>10.2f}{warm.peak_reserved_gb:>9.2f}")
    runs = []
    for i in range(args.runs):
        g = generate(loaded, prompt, noise)
        runs.append(g)
        print(f"{'timed ' + str(i + 1):<10}{g.seconds:>10.1f}{g.decode_seconds:>10.1f}"
              f"{g.peak_alloc_gb:>10.2f}{g.peak_reserved_gb:>9.2f}")

    total = [g.seconds + g.decode_seconds for g in runs]
    mean_s = st.mean(total)
    rec = runio.record(
        "E1", loaded, seed=SEED, prompt=prompt, negative_prompt=loaded.cfg.negative_prompt,
        unet_params_b=round(n_params / 1e9, 3), model_load_s=round(loaded.load_s, 1),
        wall_clock_s_mean=round(mean_s, 1), wall_clock_s_sd=round(st.stdev(total) if len(total) > 1 else 0.0, 2),
        denoise_s_mean=round(st.mean(g.seconds for g in runs), 1),
        decode_s_mean=round(st.mean(g.decode_seconds for g in runs), 1),
        cold_start_s=round(warm.seconds + warm.decode_seconds, 1),
        peak_vram_gb_reserved=round(max(g.peak_reserved_gb for g in runs), 2),
        peak_vram_gb_allocated=round(max(g.peak_alloc_gb for g in runs), 2),
        denoise_peak_vram_gb_allocated=round(max(g.denoise_peak_alloc_gb for g in runs), 2),
        gpu_hours_for_360_clips=round(360 * mean_s / 3600, 2),
    )
    runio.save_gif(runs[-1].frames, args.out / "E1_retriever_running.gif")
    if args.config_sweep:
        print("\nmemory-config sweep")
        cfg = loaded.cfg
        del loaded, runs, warm
        rec["config_sweep"] = config_sweep(cfg, prompt, noise)
    runio.write_json(args.out / "E1_result.json", rec)
    print(f"\nwall-clock per clip {mean_s:.1f} s, peak VRAM {rec['peak_vram_gb_reserved']} GB reserved "
          f"({rec['gpu']}, profile {rec['profile']}); 360-clip probing set = {rec['gpu_hours_for_360_clips']} GPU-h")


if __name__ == "__main__":
    main()
