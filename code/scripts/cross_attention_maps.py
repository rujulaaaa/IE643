"""Cross-attention visualisation (DAAM-style, per frame) for the probing prompts.

    python scripts/cross_attention_maps.py [--subject "golden retriever"] [--verb running] \
        [--scene "on a beach"] [--seed 42] [--config configs/generation.yaml]

For each phrase of the prompt (subject, verb, scene) the attention mass of every spatial
cross-attention layer on the phrase's tokens is upsampled and summed over layers and steps
(vdprobe.harness.CrossAttentionMaps). Writes to <out>/<clip id>/:
  overlay_<phrase>.gif     the aggregated map over the generated frames
  grid_<phrase>.png        frame-mean map per layer (rows) x step window (columns)
  masks.npz                heat maps and tau = 0.4 masks per phrase, (F, 64, 64), for object-only flow
  cross_attention.json     per (layer, step): concentration of each phrase's map; mask IoU across frames
"""


import numpy as np
import torch
from PIL import Image

from vdprobe import runio
from vdprobe.coords import block_order, enumerate_coordinates
from vdprobe.harness import CrossAttentionMaps, Harness
from vdprobe.pipeline import generate, make_noise, phrase_token_positions
from vdprobe.probing_set import Clip

WINDOWS = [(0, 3), (3, 8), (8, 13), (13, 19), (19, 25)]   # step windows for the grid figure


def overlay(frames, heat: torch.Tensor, alpha: float = 0.55, size: int = 256) -> list[Image.Image]:
    import matplotlib

    cmap = matplotlib.colormaps["inferno"]
    out = []
    for f, h in zip(frames, heat):
        img = np.asarray(f.convert("RGB").resize((size, size)), np.float32) / 255
        hm = np.asarray(Image.fromarray((h.numpy() * 255).astype(np.uint8)).resize((size, size), Image.BICUBIC),
                        np.float32) / 255
        mix = (1 - alpha) * img + alpha * cmap(hm)[..., :3]
        out.append(Image.fromarray((mix * 255).clip(0, 255).astype(np.uint8)))
    return out


def grid_figure(rec: CrossAttentionMaps, coords, phrase: str, path, n_steps: int):
    import matplotlib.pyplot as plt

    order = {b: i for i, b in enumerate(block_order(coords))}
    names = sorted((c for c in coords if c.kind == "spatial_attn2"), key=lambda c: (order[c.block], c.name))
    fig, axes = plt.subplots(len(names), len(WINDOWS), figsize=(1.6 * len(WINDOWS), 1.6 * len(names)),
                             squeeze=False)
    for i, c in enumerate(names):
        for j, (a, b) in enumerate(WINDOWS):
            ms = [rec.cells[(c.name, t)][phrase] for t in range(a, min(b, n_steps)) if (c.name, t) in rec.cells]
            ax = axes[i, j]
            ax.imshow(torch.stack(ms).mean(0).numpy() if ms else np.zeros((rec.grid, rec.grid)), cmap="inferno")
            ax.set_xticks([]), ax.set_yticks([])
            if i == 0:
                ax.set_title(f"steps {a}-{b - 1}", fontsize=8)
            if j == 0:
                ax.set_ylabel(c.name.replace(".transformer_blocks.0.attn2", "").replace("attentions.", "a"),
                              fontsize=6)
    fig.suptitle(f"cross-attention on '{phrase}' (frame mean, per layer and step window)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def iou_across_frames(mask: torch.Tensor) -> float:
    """Mean IoU of each frame's mask with frame 0's: does the attended region stay put?"""
    m0 = mask[0]
    ious = [(m0 & m).sum().item() / max((m0 | m).sum().item(), 1) for m in mask[1:]]
    return float(np.mean(ious)) if ious else 1.0


def main():
    p = runio.parser(__doc__, "cross_attention")
    p.add_argument("--subject", default="golden retriever")
    p.add_argument("--verb", default="running")
    p.add_argument("--scene", default="on a beach")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=0.4, help="DAAM mask threshold (0.3-0.5 in the paper)")
    args = p.parse_args()
    loaded = runio.load(args)
    cfg = loaded.cfg
    clip = Clip(args.subject, args.verb, args.scene, args.seed)
    prompt = clip.prompt(cfg)
    tokens = phrase_token_positions(loaded.pipe.tokenizer, prompt, clip.phrases)
    coords = enumerate_coordinates(loaded.unet)
    phrases = list(clip.phrases)

    rec = CrossAttentionMaps(phrases, coords=[c for c in coords if c.kind == "spatial_attn2"])
    with Harness(loaded.unet, coords, [rec], token_index=tokens):
        gen = generate(loaded, prompt, make_noise(cfg, clip.seed))
    print(f"{prompt!r}: {gen.seconds:.0f} s with cross-attention recording")

    out = args.out / clip.id
    out.mkdir(parents=True, exist_ok=True)
    runio.save_gif(gen.frames, out / "clip.gif")
    stats = {"prompt": prompt, "tokens": tokens, "tau": args.tau, "phrases": {}}
    arrays = {}
    for ph in phrases:
        heat, mask = rec.heatmap(ph), rec.mask(ph, args.tau)
        frames = overlay(gen.frames, heat)
        frames[0].save(out / f"overlay_{ph}.gif", save_all=True, append_images=frames[1:], duration=125, loop=0)
        grid_figure(rec, coords, ph, out / f"grid_{ph}.png", cfg.steps)
        arrays[f"heat_{ph}"], arrays[f"mask_{ph}"] = heat.numpy(), mask.numpy()
        stats["phrases"][ph] = {
            "mask_area": float(mask.float().mean()),
            "mask_iou_across_frames": iou_across_frames(mask),
            "concentration": {f"{name}|{t}": rec.concentration(cell[ph]) for (name, t), cell in rec.cells.items()},
        }
        print(f"  {ph:<8} mask area {stats['phrases'][ph]['mask_area']:.3f}  "
              f"IoU across frames {stats['phrases'][ph]['mask_iou_across_frames']:.3f}")
    np.savez_compressed(out / "masks.npz", **arrays)
    runio.write_json(out / "cross_attention.json", runio.record("cross_attention", loaded, **stats))


if __name__ == "__main__":
    main()
