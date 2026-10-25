"""One-off (D-36 item 3): recompute the `channels` section of exp2_analysis.json with the fp64
channel_selectivity, leaving every other section and maps.npz untouched."""
import json
import sys
from pathlib import Path

C = Path("/home/shivi/IE643/code")
sys.path.insert(0, str(C / "scripts"))
import exp2_analysis as X  # noqa: E402

for cell in ["sd15_mm2", "epicrealism_mm2", "sd15_mm3", "epicrealism_mm3"]:
    out = C / "results/exp2" / cell / "exp2_analysis.json"
    res = json.loads(out.read_text())
    if res.get("channels_fp64"):
        print(cell, "already recomputed", flush=True)
        continue
    groups, _, meta = X.load_cell(C / "results/kaggle" / f"vdprobe-exp2-{cell.replace('_', '-')}" / "out" / cell)
    old = res.get("channels") or {}
    res["channels"] = X.section_channels(groups, meta)
    res["channels_fp64"] = True
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    tmp.replace(out)
    ch = res["channels"] or {}
    best = max(ch.items(), key=lambda kv: kv[1]["S_dyn_all"] - kv[1]["S_dyn_top"], default=(None, None))
    print(cell, "recomputed", len(ch), "entries; old eta_top_mean median",
          sorted(v["eta_top_mean"] for v in old.values())[len(old) // 2] if old else None,
          "new", sorted(v["eta_top_mean"] for v in ch.values())[len(ch) // 2] if ch else None, flush=True)
print("CHANNELS RECOMPUTED")
