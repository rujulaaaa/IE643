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
"""

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


def select(files: list[str]) -> dict[str, list[str]]:
    picked = {}
    for cls in CLASSES:
        chosen, films = [], set()
        for f in sorted(files):
            m = TAGS.match(f.split("/")[-1])
            if not m or m["cls"] != cls:
                continue
            if m["body"] == "f" and m["cam"] == "nm" and m["np"] == "1" and m["q"] != "bad" and m["film"] not in films:
                chosen.append(f)
                films.add(m["film"])
            if len(chosen) == PER_CLASS:
                break
        picked[cls] = chosen
    return picked


def main():
    info = HfApi().dataset_info(REPO)
    picked = select([s.rfilename for s in info.siblings])
    manifest = {"source": f"https://huggingface.co/datasets/{REPO}", "revision": info.sha,
                "classes": CLASSES, "rule": __doc__.split("Selection rule")[1].split("Writes")[0].strip(),
                "clips": []}
    for cls, files in picked.items():
        for f in files:
            local = Path(hf_hub_download(REPO, f, repo_type="dataset", revision=info.sha))
            dst = OUT / cls / f.split("/")[-1]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, dst)
            manifest["clips"].append({"class": cls, "verb": CLASSES[cls], "split": f.split("/")[0],
                                      "path": str(dst.relative_to(OUT)), "bytes": dst.stat().st_size,
                                      "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
        print(f"{cls}: {len(files)} clips")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {OUT / 'manifest.json'} ({len(manifest['clips'])} clips)")


if __name__ == "__main__":
    main()
