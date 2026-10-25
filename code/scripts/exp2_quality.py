"""Experiment 2 Phase 1 · quality gate over the 2x2 model grid (CPU; reads sweep outputs).

    python scripts/exp2_quality.py results/kaggle/<sweep>/out/<cell> [more cells ...] --out results/exp2/quality

For every clip of every group: VBench subject consistency (DINO), dynamic degree (RAFT), CLIP
text alignment, X-CLIP zero-shot verb recognition (vdprobe.quality). Writes per-clip rows
(quality.csv), a per-cell / per-verb summary (quality.json) and the Phase 1 decision:
  primary cell   highest median subject consistency among cells whose dynamic share is >= 0.5
                 and whose CLIP-T is within 0.01 of the best cell (fixed before the run); when no cell
                 passes both, the CLIP-T tolerance is dropped and the dynamic-share requirement kept
                 (D-39: subject consistency alone favours near-static videos)
  ladder verbs   the two verbs with the highest X-CLIP hit rate on the primary cell
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from vdprobe import metrics as M
from vdprobe import quality as Q


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("cells", nargs="*", type=Path, help="one directory per model cell, holding group dirs")
    p.add_argument("--out", type=Path, default=Path("results/exp2/quality"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--from-csv", action="store_true",
                   help="re-derive the summary and decision from <out>/quality.csv without re-scoring")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.from_csv:
        summarise(pd.read_csv(args.out / "quality.csv"), args.out)
        return

    dino, clip = Q.Dino(args.device), M.ClipScorer(device=args.device)
    rows, xclip = [], None
    for cell in args.cells:
        for g in sorted(x for x in cell.iterdir() if (x / "manifest.json").exists()):
            man = json.loads((g / "manifest.json").read_text())
            meta = man["clip_meta"]
            if xclip is None:
                verbs = sorted({m["verb"] for m in meta.values()})
                xclip = Q.XClipVerbs(verbs, args.device)
            flows = torch.load(g / "flows.pt") if (g / "flows.pt").exists() else {}
            emb = torch.load(g / "clip_emb.pt") if (g / "clip_emb.pt").exists() else {}
            for cid in man["clips"]:
                m, frames = meta[cid], Q.gif_frames(g / "gifs" / f"{cid}.gif")
                text = clip.text([man["prompts"][cid]])[0]
                e = emb[cid].float().to(text.device) if cid in emb else clip.image(frames)
                dyn, top5 = Q.dynamic_degree(flows[cid].numpy()) if cid in flows else (np.nan, np.nan)
                probs = xclip(frames, m["subject"])
                rows.append({"cell": cell.name, "group": g.name, "clip": cid, **m,
                             "subject_consistency": Q.subject_consistency(dino(frames)),
                             "dynamic": dyn, "flow_top5": top5, "clip_t": float((e @ text).mean()),
                             "verb_prob": float(probs[xclip.verbs.index(m["verb"])]),
                             "verb_hit": bool(xclip.verbs[int(probs.argmax())] == m["verb"])})
            print(f"{cell.name}/{g.name}: {len(man['clips'])} clips", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(args.out / "quality.csv", index=False)
    summarise(df, args.out)


def decide(df: pd.DataFrame) -> dict:
    s = df.groupby("cell").agg(sc=("subject_consistency", "median"), dyn=("dynamic", "mean"), ct=("clip_t", "median"))
    moving = s[s.dyn >= 0.5]
    ok = moving[moving.ct >= s.ct.max() - 0.01]
    if len(ok):
        pool, rule = ok, "D-35"
    elif len(moving):
        pool, rule = moving, "D-39 (no cell within the CLIP-T tolerance: dynamic share >= 0.5 kept, CLIP-T dropped)"
    else:
        pool, rule = s, "D-39 (no cell with dynamic share >= 0.5: highest subject consistency overall)"
    primary = pool.sc.idxmax()
    vr = df[df.cell == primary].groupby("verb").verb_hit.mean().sort_values(ascending=False)
    return {"primary_cell": primary, "ladder_verbs": list(vr.index[:2]), "applied": rule,
            "rule": "max median subject consistency among cells with dynamic share >= 0.5 and CLIP-T within "
                    "0.01 of the best; ladder verbs = top-2 X-CLIP hit rate on the primary cell",
            "eligible": list(ok.index), "moving": list(moving.index)}


def summarise(df: pd.DataFrame, out: Path):
    cols = ["subject_consistency", "dynamic", "flow_top5", "clip_t", "verb_prob", "verb_hit"]
    by_cell = df.groupby("cell")[cols].agg(["median", "mean"]).round(4)
    by_verb = df.groupby(["cell", "verb"])[["verb_hit", "verb_prob", "flow_top5"]].mean().round(4)
    decision = decide(df)
    by_cell.columns = [f"{a}_{b}" for a, b in by_cell.columns]
    res = {"by_cell": json.loads(by_cell.to_json(orient="index")),
           "by_cell_verb": json.loads(by_verb.reset_index().to_json(orient="records")),
           "decision": decision, "n_clips": len(df)}
    (out / "quality.json").write_text(json.dumps(res, indent=2))
    print(by_cell.to_string())
    print(by_verb.to_string())
    print("decision:", decision)


if __name__ == "__main__":
    main()
