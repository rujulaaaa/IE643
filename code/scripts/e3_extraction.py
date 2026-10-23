"""E3 · Can we get activations out of all 95 coordinates, and what does storing them cost?

Three generations of the same clip from the same noise tensor:
  1. no hooks                                  baseline wall clock
  2. ShapeAccountant on all 95 coordinates     what a full fp16 dump would cost (E3's original measurement)
  3. SummaryRecorder on all 95 coordinates     the production store, written to disk and measured

    python scripts/e3_extraction.py [--profile gpu|lowvram] [--grid 16]
"""

import time

import torch

from vdprobe import runio
from vdprobe.coords import count_by_kind, enumerate_coordinates
from vdprobe.harness import Harness, ShapeAccountant, SummaryRecorder
from vdprobe.pipeline import generate, make_noise, phrase_token_positions
from vdprobe.probing_set import ProbingSet
from vdprobe.store import load_clip, save_clip

GiB, MiB = 2 ** 30, 2 ** 20
N_CLIPS = 360                       # 120 clips x 3 seeds (D-04)
KAGGLE_OUTPUT_CAP_GB, KAGGLE_DISK_GB = 20, 73


def analytic_summary_bytes(records, frames: int, grid: int) -> tuple[int, int]:
    """E3's formula: (2C + F) fp32 per record, plus F*grid*grid fp32 per attention record."""
    stats = att = 0
    for r in records:
        shape = r["shape"]
        c = shape[1] if r["kind"] == "temporal_module" else shape[-1]
        stats += (2 * c + frames) * 4
        if "attn" in r["kind"]:
            att += grid * grid * frames * 4
    return stats, att


def main():
    p = runio.parser(__doc__, "E3")
    p.add_argument("--grid", type=int, default=16)
    args = p.parse_args()
    loaded = runio.load(args)
    cfg = loaded.cfg
    clip = ProbingSet().clip("golden retriever", "running", "on a beach", seed=42)
    prompt = clip.prompt(cfg)
    noise = make_noise(cfg, clip.seed)
    tokens = phrase_token_positions(loaded.pipe.tokenizer, prompt, clip.phrases)

    coords = enumerate_coordinates(loaded.unet)
    by_kind = count_by_kind(coords)
    print(f"{len(coords)} hookable coordinates: {by_kind}")

    generate(loaded, prompt, noise, decode_frames=False)          # warm-up
    base = generate(loaded, prompt, noise, decode_frames=False)
    print(f"no hooks            {base.seconds:6.1f} s")

    acct = ShapeAccountant()
    with Harness(loaded.unet, coords, [acct]) as ctx:
        meta = generate(loaded, prompt, noise, decode_frames=False)
    assert ctx.step + 1 == cfg.steps, "one UNet call per denoising step expected"
    print(f"metadata hooks      {meta.seconds:6.1f} s  ({100 * (meta.seconds / base.seconds - 1):+.1f}%)")

    rec = SummaryRecorder(grid=args.grid)
    with Harness(loaded.unet, coords, [rec], token_index=tokens):
        summ = generate(loaded, prompt, noise, decode_frames=False)
    print(f"summary recorder    {summ.seconds:6.1f} s  ({100 * (summ.seconds / base.seconds - 1):+.1f}%)")

    t0 = time.perf_counter()
    clip_dir = args.out / "store" / clip.id
    manifest = {"clip_id": clip.id, "prompt": prompt, "seed": clip.seed, "token_index": tokens,
                "timesteps": ctx.timesteps, "grid": args.grid, "branch": "text-conditional",
                "config": cfg.to_dict(), "revisions": loaded.revisions,
                "coordinates": [{"name": c.name, "kind": c.kind, "block": c.block, "level": c.level}
                                for c in coords]}
    sizes = save_clip(clip_dir, rec.tensors(), manifest)
    write_s = time.perf_counter() - t0
    tensors, _ = load_clip(clip_dir)                              # round-trip check
    assert all(torch.equal(tensors[k], v) for k, v in rec.tensors().items())

    full_by_kind = acct.bytes_by_kind()
    full = sum(full_by_kind.values())
    full_cond = sum(acct.bytes_by_kind(cond_only=True).values())
    stats_b, att_b = analytic_summary_bytes(acct.records, cfg.frames, args.grid)
    analytic = stats_b + att_b
    on_disk = sizes["file_bytes"]

    print(f"\n{'kind':<18}{'coords':>7}{'full dump GiB':>15}")
    for k in by_kind:
        print(f"{k:<18}{by_kind[k]:>7}{full_by_kind[k] / GiB:>15.3f}")
    print(f"{'FULL DUMP':<18}{len(coords):>7}{full / GiB:>15.3f}   (text-conditional branch only: {full_cond / GiB:.3f})")
    print(f"\nsummary store, analytic  {analytic / MiB:8.2f} MiB  (stats {stats_b / MiB:.2f} + attention {att_b / MiB:.2f})")
    print(f"summary store, on disk   {on_disk / MiB:8.2f} MiB  ({sizes['n_tensors']} tensors, written in {write_s:.1f} s)")
    print(f"compression vs full dump {full / on_disk:8.0f}x   (vs a conditional-branch-only dump: {full_cond / on_disk:.0f}x)")

    grids = {}
    for g in (32, 16, 8, 4):
        s, a = analytic_summary_bytes(acct.records, cfg.frames, g)
        grids[f"{g}x{g}"] = {"mb_per_clip": round((s + a) / MiB, 2), "gb_360_clips": round(N_CLIPS * (s + a) / GiB, 2)}
        print(f"  grid {g:>2}x{g:<2} {(s + a) / MiB:7.1f} MiB/clip  {N_CLIPS * (s + a) / GiB:6.2f} GiB for {N_CLIPS} clips")

    out = runio.record(
        "E3", loaded, seed=clip.seed, prompt=prompt, unet_calls=ctx.step + 1,
        n_coordinates=len(coords), coordinates_by_kind=by_kind,
        baseline_s_no_hooks=round(base.seconds, 1),
        generation_s_with_metadata_hooks=round(meta.seconds, 1),
        metadata_hook_overhead_pct=round(100 * (meta.seconds / base.seconds - 1), 1),
        generation_s_with_summary_recorder=round(summ.seconds, 1),
        summary_recorder_overhead_pct=round(100 * (summ.seconds / base.seconds - 1), 1),
        full_dump_gb_per_clip=round(full / GiB, 3),
        full_dump_gb_per_clip_cond_branch=round(full_cond / GiB, 3),
        full_dump_gb_by_kind={k: round(v / GiB, 3) for k, v in full_by_kind.items()},
        attention_grid=args.grid,
        summary_store_mb_per_clip_analytic=round(analytic / MiB, 2),
        summary_store_mb_per_clip_on_disk=round(on_disk / MiB, 2),
        summary_split_mb={"channel_stats": round(stats_b / MiB, 2), "attention_movies": round(att_b / MiB, 2)},
        compression_ratio=round(full / analytic, 1),
        compression_ratio_on_disk=round(full / on_disk, 1),
        compression_ratio_vs_cond_branch_dump=round(full_cond / on_disk, 1),
        grid_sensitivity=grids, n_clips_planned=N_CLIPS,
        full_dump_gb_total=round(N_CLIPS * full / GiB, 1),
        summary_store_gb_total=round(N_CLIPS * on_disk / GiB, 2),
        full_dump_one_clip_fits_kaggle_disk=full / GiB < KAGGLE_DISK_GB,
        store_path=str(clip_dir.relative_to(args.out)),
        units="GiB / MiB (powers of 1024), as in the Kaggle E3 record",
    )
    runio.write_json(args.out / "E3_result.json", out)


if __name__ == "__main__":
    main()
