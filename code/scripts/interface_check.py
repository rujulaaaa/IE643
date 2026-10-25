"""GPU check of every interface path, headless (no browser): what scripts/app.py runs on a click.

    python -u scripts/interface_check.py [--variant sd15_mm2] [--video <mp4>]

  live edit, motion direction, selected component patch (+ B reference, D)
  live edit, content direction, selected component patch
  live edit, motion direction, probe-guided LoRA
  upload: a real HMDB51 clip inverted, attention injection at all steps onto a new subject
  cross-attention maps for a typed prompt
  the Gradio app builds with this engine (not launched)
Writes timings, scores and output paths to <out>/interface_check.json; GIFs under <out>/interface/.
"""

import importlib.util
import json
import time
import traceback
from pathlib import Path

from vdprobe import runio
from vdprobe.config import REPO_ROOT
from vdprobe.demo import EditRequest, Engine, link_assets


def main():
    p = runio.parser(__doc__, "interface_check")
    p.add_argument("--variant", default="sd15_mm2")
    p.add_argument("--assets", type=Path, default=None, help="directory laid out like the repo (results/, data/)")
    p.add_argument("--video", type=Path,
                   default=REPO_ROOT / "data" / "hmdb51_lora" / "jump" / "jump_Faith_Rewarded_jump_f_nm_np1_ri_med_89.mp4")
    args = p.parse_args()
    if args.assets:
        print("linked", link_assets(args.assets), "asset files", flush=True)
    eng = Engine(out=args.out / "interface", variant=args.variant)
    t0 = time.perf_counter()
    eng.ensure()
    res = {"variant": args.variant, "load_seconds": time.perf_counter() - t0, "loras": eng.loras,
           "lora_skipped": eng.lora_skipped, "checks": {}}
    ticks = []        # what the app's progress bar would receive
    prog = lambda frac, desc="": ticks.append((round(float(frac), 3), desc))
    checks = {
        "edit_motion_selected": lambda: eng.edit(EditRequest("golden retriever", "running", "horse", "walking",
                                                             "on a beach", 42, "motion", "selected"), progress=prog),
        "edit_content_selected": lambda: eng.edit(EditRequest("golden retriever", "running", "horse", "jumping",
                                                              "on a beach", 42, "content", "selected")),
        "edit_motion_lora_probe": lambda: eng.edit(EditRequest("golden retriever", "running", "tiger", "running",
                                                               "in a forest", 1337, "motion", "lora_probe",
                                                               with_reference=False)),
        "upload_attn_all": lambda: eng.upload_edit(str(args.video), "a man jumping, high quality, detailed",
                                                   "a tiger jumping, high quality, detailed"),
        "cross_attention": lambda: eng.cross_attention("red fox", "jumping", "in a snowy forest", 7, progress=prog),
    }
    for name, fn in checks.items():
        t1 = time.perf_counter()
        try:
            r = fn()
            out = {"ok": True, "seconds": time.perf_counter() - t1}
            if "table" in r:
                out["table"] = json.loads(r["table"].round(4).to_json(orient="records"))
                out["pool"] = r.get("pool")
            out["files"] = r.get("gifs") or {k: r[k] for k in ("clip", "overlay", "grid") if k in r}
        except Exception as e:
            out = {"ok": False, "seconds": time.perf_counter() - t1, "error": f"{type(e).__name__}: {e}",
                   "trace": traceback.format_exc()[-3000:]}
        out["progress_ticks"] = len(ticks)
        if ticks:
            out["progress_last"] = ticks[-1]
        ticks.clear()
        res["checks"][name] = out
        print(f"{name}: {'OK' if out['ok'] else 'FAILED ' + out['error']}  {out['seconds']:.0f} s", flush=True)
    try:
        spec = importlib.util.spec_from_file_location("app", REPO_ROOT / "scripts" / "app.py")
        app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app)
        app.build(eng)
        from vdprobe.demo import playable
        res["playable_mp4"] = playable(res["checks"]["edit_motion_selected"].get("files", {}).get("edit"))
        res["checks"]["app_builds"] = {"ok": True}
        print("app_builds: OK", flush=True)
    except Exception as e:
        res["checks"]["app_builds"] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-3000:]}
        print("app_builds: FAILED", e, flush=True)
    runio.write_json(args.out / "interface_check.json", res)
    if not all(c["ok"] for c in res["checks"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
