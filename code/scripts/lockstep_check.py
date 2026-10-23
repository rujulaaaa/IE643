"""Verifies the lockstep sampler against diffusers' own pipeline before any sweep uses it.

1. floor        two ordinary `pipe(...)` generations of the same clip (fp16 run-to-run noise)
2. solo         the same clip through `run_lockstep`, alone
3. grouped      the same clip inside a 3-clip group (must not be affected by its neighbours)
4. probed       the same group with StepCache / PairScorer / Sketcher attached (probes are
                read-only, so the output must not move); the clip scored against an
                identical copy of itself must give distances ~0, against a one-word swap not

All comparisons use the final-latent mean absolute difference, the D-11 quantity
(threshold 1e-3).

    python scripts/lockstep_check.py
"""

import torch

from vdprobe import runio
from vdprobe.coords import coarse_coordinates, enumerate_coordinates
from vdprobe.lockstep import PairScorer, Sketcher, StepCache, Track, run_lockstep
from vdprobe.metrics import latent_mad
from vdprobe.pipeline import generate, make_noise

SEED = 42
P = "a golden retriever running on a beach, high quality, detailed"
Q = "a golden retriever walking on a beach, high quality, detailed"
R = "a horse running on a beach, high quality, detailed"


def main():
    args = runio.parser(__doc__, "lockstep_check").parse_args()
    loaded = runio.load(args)
    coords = enumerate_coordinates(loaded.unet)
    coarse = coarse_coordinates(coords)
    noise = make_noise(loaded.cfg, SEED)
    res = {}

    g1 = generate(loaded, P, noise, decode_frames=False)
    g2 = generate(loaded, P, noise, decode_frames=False)
    res["pipe_seconds"] = g1.seconds
    res["floor_mad"] = latent_mad(g1.latents, g2.latents)
    print(f"floor   pipe vs pipe            MAD {res['floor_mad']:.2e}   ({g1.seconds:.1f} s)")

    solo = [Track("P", P)]
    t = run_lockstep(loaded, coords, solo, noise)
    res["solo_mad"] = latent_mad(g1.latents, solo[0].latents)
    res["solo_seconds"] = t["seconds"]
    print(f"solo    lockstep vs pipe        MAD {res['solo_mad']:.2e}   ({t['seconds']:.1f} s)")

    group = [Track("P", P), Track("Q", Q), Track("R", R)]
    t = run_lockstep(loaded, coords, group, noise)
    res["grouped_mad"] = latent_mad(g1.latents, group[0].latents)
    res["grouped_seconds_per_clip"] = t["seconds"] / 3
    print(f"grouped P in [P,Q,R] vs pipe    MAD {res['grouped_mad']:.2e}   ({t['seconds'] / 3:.1f} s per clip)")

    cache = StepCache(coarse)
    sk = Sketcher(coarse, steps=range(25))
    self_pair = PairScorer(coarse, [(0, cache)], per_channel_steps=(0, 12, 24))
    swap_pair = PairScorer(coarse, [(1, cache)], per_channel_steps=(0, 12, 24))
    probed = [Track("P", P, [cache, sk]), Track("P_copy", P, [self_pair]), Track("Q", Q, [swap_pair])]
    t = run_lockstep(loaded, coords, probed, noise, log_every=5)
    res["probed_mad"] = latent_mad(g1.latents, probed[0].latents)
    res["probed_seconds_per_clip"] = t["seconds"] / 3
    print(f"probed  P with probes vs pipe   MAD {res['probed_mad']:.2e}   ({t['seconds'] / 3:.1f} s per clip)")

    def worst(rows, key):
        return max(r[key] for r in rows)

    def median(rows, key):
        return float(torch.tensor([r[key] for r in rows]).median())

    res["self_pair_max_d_dynamic"] = worst(self_pair.rows, "d_dynamic")
    res["self_pair_max_d_static"] = worst(self_pair.rows, "d_static")
    res["swap_pair_median_d_dynamic"] = median(swap_pair.rows, "d_dynamic")
    res["swap_pair_median_d_static"] = median(swap_pair.rows, "d_static")
    res["rows_per_pair"] = len(swap_pair.rows)
    res["per_channel_entries"] = len(swap_pair.per_channel)
    res["sketch_entries"] = len(sk.out)
    print(f"self pair  max d_dynamic {res['self_pair_max_d_dynamic']:.2e}  max d_static {res['self_pair_max_d_static']:.2e}")
    print(f"swap pair  median d_dynamic {res['swap_pair_median_d_dynamic']:.3f}  "
          f"median d_static {res['swap_pair_median_d_static']:.3f}")
    print(f"rows per pair {res['rows_per_pair']} (16 coords x 25 steps = 400), "
          f"per-channel {res['per_channel_entries']} (16 x 3 = 48), sketches {res['sketch_entries']} (400)")

    tol = 1e-3
    checks = {
        "solo matches pipe": res["solo_mad"] < tol,
        "grouping does not leak": res["grouped_mad"] < tol,
        "probes are read-only": res["probed_mad"] < tol,
        "self pair distance ~0": res["self_pair_max_d_dynamic"] < 1e-2,
        "swap pair distance > self pair": res["swap_pair_median_d_dynamic"] > 10 * res["self_pair_max_d_dynamic"],
        "row counts": res["rows_per_pair"] == 400 and res["sketch_entries"] == 400,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    res["checks"] = checks
    runio.write_json(args.out / "lockstep_check.json", runio.record("lockstep_check", loaded, **res))


if __name__ == "__main__":
    main()
