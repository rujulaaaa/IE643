"""Stage 6 · Fetch the real clips the motion LoRAs are trained on (D-19 as amended 2026-10-20).

HMDB51 (Kuehne et al., ICCV 2011), from the Hugging Face mirror `divm/hmdb51` (individual
mp4s, so no multi-GB archive). Four classes standing in for the four verbs of the probing
vocabulary: run -> running, walk -> walking, jump -> jumping, turn -> spinning.

Selection rule (deterministic, applied to HMDB51's own annotation tags in the file name):
    body fully visible (f), no camera motion (nm), one person (np1), quality not "bad";
    sorted by file name, at most one clip per source film, first 5 per class.
A static camera matters: with camera motion the LoRA would learn panning, not the action.

Writes data/hmdb51_lora/<class>/*.mp4 and data/hmdb51_lora/manifest.json.

    python scripts/fetch_lora_data.py

Experiment 2 (Phases 4/6) fetches a long-tail motion the base model cannot already render:

    python scripts/fetch_lora_data.py --classes "cartwheel=doing a cartwheel" --per-class 10 \
        --per-film 3 --cut-free --out data/hmdb51_cartwheel

(static-camera cartwheels in HMDB51 come from only 4 films, hence up to 3 clips per film), and
--cut-free keeps only clips without a scene cut (inversion.shot_bounds).
"""

import argparse

import hashlib
import json
import re
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "divm/hmdb51"
CLASSES = {"run": "running", "walk": "walking", "jump": "jumping", "turn": "spinning"}
PER_CLASS = 5
TAGS = re.compile(r"^(?P<cls>[a-z_]+?)_(?P<film>.+)_(?P=cls)_(?P<body>[fhlu])_(?P<cam>cm|nm)_np(?P<np>\d+)"
                  r"_(?P<view>[a-z]+)_(?P<q>bad|med|goo)_(?P<n>\d+)\.mp4$")
OUT = Path(__file__).resolve().parent.parent / "data" / "hmdb51_lora"


def select(files: list[str], classes=CLASSES, per_class=PER_CLASS, per_film=1) -> dict[str, list[str]]:
    picked = {}
    for cls in classes:
        chosen, films = [], {}
        for f in sorted(files):
            m = TAGS.match(f.split("/")[-1])
            if not m or m["cls"] != cls:
                continue
            if m["body"] == "f" and m["cam"] == "nm" and m["np"] == "1" and m["q"] != "bad" \
                    and films.get(m["film"], 0) < per_film:
                chosen.append(f)
                films[m["film"]] = films.get(m["film"], 0) + 1
            if len(chosen) == per_class:
                break
        picked[cls] = chosen
    return picked


def has_cut(path: Path) -> bool:
    import cv2

    from vdprobe.inversion import shot_bounds

    cap, frames = cv2.VideoCapture(str(path)), []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    return len(shot_bounds(frames)) > 1


def fetch(classes=CLASSES, per_class=PER_CLASS, per_film=1, out=OUT, cut_free=False) -> Path:
    info = HfApi().dataset_info(REPO)
    # select generously when cut-free filtering will drop some
    picked = select([s.rfilename for s in info.siblings], classes, per_class * (3 if cut_free else 1), per_film)
    manifest = {"source": f"https://huggingface.co/datasets/{REPO}", "revision": info.sha,
                "classes": dict(classes), "rule": __doc__.split("Selection rule")[1].split("Writes")[0].strip(),
                "per_film": per_film, "cut_free": cut_free, "dropped_for_cuts": [], "clips": []}
    for cls, files in picked.items():
        kept = 0
        for f in files:
            if kept == per_class:
                break
            local = Path(hf_hub_download(REPO, f, repo_type="dataset", revision=info.sha))
            if cut_free and has_cut(local):
                manifest["dropped_for_cuts"].append(f)
                continue
            kept += 1
            dst = out / cls / f.split("/")[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dst)
            manifest["clips"].append({"class": cls, "verb": classes[cls], "split": f.split("/")[0],
                                      "path": str(dst.relative_to(out)), "bytes": dst.stat().st_size,
                                      "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
        print(f"{cls}: {kept} clips ({len(manifest['dropped_for_cuts'])} dropped for cuts so far)")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out / 'manifest.json'} ({len(manifest['clips'])} clips)")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classes", nargs="*", default=None, help='"cls=verb phrase" entries (default: the 4 probing classes)')
    ap.add_argument("--per-class", type=int, default=PER_CLASS)
    ap.add_argument("--per-film", type=int, default=1)
    ap.add_argument("--cut-free", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    classes = dict(c.split("=", 1) for c in args.classes) if args.classes else CLASSES
    fetch(classes, args.per_class, args.per_film, args.out, args.cut_free)


if __name__ == "__main__":
    main()
