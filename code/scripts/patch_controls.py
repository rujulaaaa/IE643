"""Stage 5 controls on the real model, before any patching result is believed (roadmap §5.3).

Group 1, source A = golden retriever running (beach, seed 42). Target arms:
    null_tensor     A's own activations, 16 coarse coordinates, all 25 steps      must pass D-11
    null_dynamic    the same as a D-29 x̃-only component patch                     must pass D-11
    null_static     the same as an x̄-only component patch                         must pass D-11
    null_attn       A's own temporal attention probabilities, all 42 modules      must pass D-11
                    (the probabilities are stored in fp16, so this is not bit-exact)
    misindexed      null_tensor read one step late (steps 1..24)                  must FAIL D-11
    B_plain         B's prompt, no patch (= the unpatched B)
    A_to_B          B's prompt + A's up_blocks.0 motion modules, first third
Group 2, source B = horse walking (family Q like A; the pair differs in subject AND verb on
purpose, so each direction has something to transfer). Target arms:
    A_plain, B_to_A   the mirror image.

Symmetry: the transfer fraction f(X->Y) = 1 - MAD(out, X) / MAD(Y_plain, X) should be
similar in both directions.

    python -u scripts/patch_controls.py
"""

import torch

from vdprobe import runio
from vdprobe.coords import coarse_coordinates, enumerate_coordinates
from vdprobe.edit import Arm, AttnSpec, EditSpec, run_arms, temporal_attention_modules
from vdprobe.metrics import latent_mad, psnr
from vdprobe.patching import NULL_PATCH_MAX_LATENT_MAD, NULL_PATCH_MIN_PSNR_DB, PatchSpec, window

SEED = 42
A = "a golden retriever running on a beach, high quality, detailed"
B = "a horse walking on a beach, high quality, detailed"


def verdict(ref_lat, ref_frames, lat, frames):
    mad, db = latent_mad(ref_lat, lat), psnr(ref_frames, frames)
    return {"latent_mad": mad, "psnr_db": db,
            "pass": bool(mad < NULL_PATCH_MAX_LATENT_MAD and db > NULL_PATCH_MIN_PSNR_DB)}


def main():
    ap = runio.parser(__doc__, "patch_controls")
    ap.add_argument("--cache-device", default="cuda", help="cpu on 8 GB GPUs")
    args = ap.parse_args()
    loaded = runio.load(args)
    from vdprobe.pipeline import make_noise
    coords = enumerate_coordinates(loaded.unet)
    coarse = tuple(c.name for c in coarse_coordinates(coords))
    up0 = tuple(c.name for c in coords if c.kind == "temporal_module" and c.block == "up_blocks.0")
    tattn = temporal_attention_modules(coords)
    allsteps, ft = window("all"), window("first_third")
    noise = make_noise(loaded.cfg, SEED)

    g1 = [
        Arm("null_tensor", A, EditSpec(PatchSpec(coarse, allsteps))),
        Arm("null_dynamic", A, EditSpec(PatchSpec(coarse, allsteps, component="dynamic"))),
        Arm("null_static", A, EditSpec(PatchSpec(coarse, allsteps, component="static"))),
        Arm("null_attn", A, EditSpec(attn=AttnSpec(tattn, allsteps))),
        Arm("misindexed", A, EditSpec(PatchSpec(coarse, allsteps[1:])), step_offset=-1),
        Arm("B_plain", B, EditSpec()),
        Arm("A_to_B", B, EditSpec(PatchSpec(up0, ft))),
    ]
    print(f"group 1: source A + {len(g1)} arms")
    r1 = run_arms(loaded, coords, A, g1, noise, cache_device=args.cache_device, log_every=5)
    print(f"  {r1.seconds:.0f} s, patched {r1.n_patched}, attn injected {r1.n_attn_injected}")
    g2 = [Arm("A_plain", A, EditSpec()), Arm("B_to_A", A, EditSpec(PatchSpec(up0, ft)))]
    print(f"group 2: source B + {len(g2)} arms")
    r2 = run_arms(loaded, coords, B, g2, noise, cache_device=args.cache_device, log_every=5)

    res = {"seconds": {"group1": r1.seconds, "group2": r2.seconds},
           "seconds_per_clip": (r1.seconds + r2.seconds) / (len(g1) + len(g2) + 2),
           "n_patched": r1.n_patched | r2.n_patched, "n_attn_injected": r1.n_attn_injected}
    srcA_lat, srcA_fr = r1.source_latents, r1.frames["source"]
    for k in ("null_tensor", "null_dynamic", "null_static", "null_attn", "misindexed"):
        res[k] = verdict(srcA_lat, srcA_fr, r1.latents[k], r1.frames[k])
        print(f"  {k:<13} MAD {res[k]['latent_mad']:.2e}  PSNR {res[k]['psnr_db']:.1f} dB  "
              f"{'pass' if res[k]['pass'] else 'FAIL'}")
    # the unpatched arms inside a group must equal the other group's source exactly
    res["B_plain_vs_sourceB_mad"] = latent_mad(r1.latents["B_plain"], r2.source_latents)
    res["A_plain_vs_sourceA_mad"] = latent_mad(r2.latents["A_plain"], srcA_lat)

    def transfer(out, src, plain):
        return 1 - latent_mad(out, src) / latent_mad(plain, src)
    res["transfer_A_to_B"] = transfer(r1.latents["A_to_B"], srcA_lat, r1.latents["B_plain"])
    res["transfer_B_to_A"] = transfer(r2.latents["B_to_A"], r2.source_latents, r2.latents["A_plain"])
    print(f"  symmetry: f(A->B) {res['transfer_A_to_B']:.3f}   f(B->A) {res['transfer_B_to_A']:.3f}")

    checks = {
        "null tensor patch passes D-11": res["null_tensor"]["pass"],
        "null x̃ patch passes D-11": res["null_dynamic"]["pass"],
        "null x̄ patch passes D-11": res["null_static"]["pass"],
        "null attention injection passes D-11": res["null_attn"]["pass"],
        "mis-indexed null patch fails D-11": not res["misindexed"]["pass"],
        "unpatched arm == source of the other group": max(res["B_plain_vs_sourceB_mad"], res["A_plain_vs_sourceA_mad"]) < 1e-3,
        "symmetric transfer (ratio within 2x)": min(res["transfer_A_to_B"], res["transfer_B_to_A"]) > 0
            and max(res["transfer_A_to_B"], res["transfer_B_to_A"]) / min(res["transfer_A_to_B"], res["transfer_B_to_A"]) < 2,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    res["checks"] = checks
    for tag, r in (("g1", r1), ("g2", r2)):
        for k, fr in r.frames.items():
            runio.save_gif(fr, args.out / "gifs" / f"{tag}_{k}.gif")
    runio.write_json(args.out / "patch_controls.json", runio.record("patch_controls", loaded, **res))


if __name__ == "__main__":
    main()
