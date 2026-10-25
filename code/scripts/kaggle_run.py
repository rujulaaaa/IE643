"""Run repository scripts on Kaggle's T4 x2 as a private script kernel, then download the outputs.

The kernel carries this repository's code (src/, configs/, scripts/) as an embedded archive,
pins diffusers to the version every result was produced with, fetches the weights once, and
starts one job per GPU in parallel (CUDA_VISIBLE_DEVICES=i). Each job is a command line
relative to the repository root; outputs go under /kaggle/working/out and are downloaded to
results/kaggle/<slug>/.

    python scripts/kaggle_run.py --slug vdprobe-sweep \
        --job "scripts/probe_sweep.py --scenes 'on a beach'" \
        --job "scripts/probe_sweep.py --scenes 'in a forest'"
    python scripts/kaggle_run.py --slug vdprobe-sweep --download-only

Every job gets `--out /kaggle/working/out/<name>` appended unless it already has --out.
Larger inputs (trained LoRAs) go in a private dataset: --dataset <slug> attaches it and jobs
see it at input/<slug>/.
Needs the Kaggle CLI (~/kaggle/.venv/bin/kaggle) and ~/.kaggle/kaggle.json or ~/.kaggle/access_token.
"""

import argparse
import base64
import io
import json
import os
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = str(Path.home() / "kaggle" / ".venv" / "bin" / "kaggle")
# Kaggle account the kernels run under: --user, else $KAGGLE_USER, else the original account
USER = os.environ.get("KAGGLE_USER", "rujularahate")
DIFFUSERS = "0.40.0"
PEFT = "0.21.2"     # the version the motion LoRAs were trained with

KERNEL = r'''
import base64, io, os, subprocess, sys, tarfile, time
from pathlib import Path

REPO = Path("/kaggle/working/repo")
OUT = Path("/kaggle/working/out")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(PAYLOAD))).extractall(REPO, filter="data")
OUT.mkdir(parents=True, exist_ok=True)
# attached datasets: link each to repo/input/<slug>, wherever this Kaggle image mounts it
for slug in DATASETS:
    hits = [p for p in Path("/kaggle/input").rglob(slug) if p.is_dir()]
    if not hits:
        sys.exit("dataset %s not mounted under /kaggle/input" % slug)
    (REPO / "input").mkdir(exist_ok=True)
    (REPO / "input" / slug).symlink_to(hits[0])
    print("dataset", slug, "->", hits[0], sorted(x.name for x in hits[0].iterdir()), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", "diffusers==%s" % DIFFUSERS,
                "peft==%s" % PEFT, "scikit-learn"], check=True)
# the image's torchao (0.10) makes peft refuse to build LoRA layers; nothing here uses it
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-q", "-y", "torchao"])
# nvidia-smi is not on PATH in every Kaggle image; ask torch in a child process (keeps CUDA
# uninitialised in this parent, which only launches the jobs)
probe = subprocess.run([sys.executable, "-c", "import torch; n = torch.cuda.device_count(); print(n); "
                        "[print(torch.cuda.get_device_name(i)) for i in range(n)]; print(torch.__version__)"],
                       capture_output=True, text=True)
print(probe.stdout, probe.stderr[-2000:], flush=True)
n_gpu = int(probe.stdout.split()[0]) if probe.stdout.strip() else 0
if n_gpu == 0:
    sys.exit("no CUDA device in this session")

env = dict(os.environ, PYTHONPATH=str(REPO / "src"), PYTHONUNBUFFERED="1")
# fetch the weights once, before parallel jobs race on the Hugging Face cache
subprocess.run([sys.executable, "-c", PREFETCH], env=env, cwd=REPO, check=True)

procs, logs = [], []
for i, job in enumerate(JOBS):
    gpu = i % max(n_gpu, 1)
    log = OUT / ("job%d.log" % i)
    cmd = [sys.executable, "-u"] + job
    print("job %d on GPU %d: %s" % (i, gpu, " ".join(cmd)), flush=True)
    if n_gpu < len(JOBS) and i >= n_gpu:   # fewer GPUs than jobs: run the rest after
        for p in procs: p.wait()
    procs.append(subprocess.Popen(cmd, cwd=REPO, env=dict(env, CUDA_VISIBLE_DEVICES=str(gpu)),
                                  stdout=open(log, "w"), stderr=subprocess.STDOUT))
    logs.append(log)

seen = [0] * len(logs)
while any(p.poll() is None for p in procs):
    time.sleep(60)
    for i, log in enumerate(logs):
        lines = log.read_text(errors="replace").splitlines()
        for l in lines[seen[i]:]:
            print("[job%d] %s" % (i, l), flush=True)
        seen[i] = len(lines)
for i, log in enumerate(logs):
    for l in log.read_text(errors="replace").splitlines()[seen[i]:]:
        print("[job%d] %s" % (i, l), flush=True)
codes = [p.returncode for p in procs]
print("exit codes", codes, flush=True)
import shutil; shutil.rmtree(REPO, ignore_errors=True)
if any(codes):
    sys.exit(1)
'''

PREFETCH = (
    "from vdprobe.config import GenerationConfig; from vdprobe import pipeline as P; "
    "c = GenerationConfig.from_yaml(); P._snapshot(c.adapter_id, P.ADAPTER_FILES); P._resolve_base(c); "
    "from huggingface_hub import snapshot_download; snapshot_download('openai/clip-vit-base-patch32')"
)


def kaggle(*args, check=True):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout + r.stderr


def payload(with_data=False) -> str:
    # data/ (the LoRA clips, ~2 MB) only on request: Kaggle rejects a script much over ~1 MB
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for sub in ("src/vdprobe", "configs", "scripts") + (("data",) if with_data else ()):
            p = ROOT / sub
            if p.exists():
                tar.add(p, arcname=sub, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def build(slug, jobs, with_data=False, datasets=()):
    kdir = ROOT / "build" / slug
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    parsed = []
    for i, j in enumerate(jobs):
        argv = shlex.split(j)
        if "--out" not in argv:
            argv += ["--out", f"/kaggle/working/out/job{i}"]
        parsed.append(argv)
    head = (f"PAYLOAD = {payload(with_data)!r}\nDIFFUSERS = {DIFFUSERS!r}\nPEFT = {PEFT!r}\nPREFETCH = {PREFETCH!r}\n"
            f"JOBS = {json.dumps(parsed)}\nDATASETS = {list(datasets)!r}\n")
    (kdir / "script.py").write_text(head + KERNEL)
    meta = {
        "id": f"{USER}/{slug}", "title": slug, "code_file": "script.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "machine_shape": "NvidiaTeslaT4", "enable_internet": True,
        "competition_sources": [], "dataset_sources": [f"{USER}/{d}" for d in datasets],
        "kernel_sources": [], "model_sources": [],
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"built {kdir} ({(kdir / 'script.py').stat().st_size / 1024:.0f} KiB), jobs:")
    for j in parsed:
        print("   ", " ".join(j))
    return kdir


def status(slug) -> str:
    return kaggle("kernels", "status", f"{USER}/{slug}", check=False).strip()


def wait(slug, poll=120):
    while True:
        s = status(slug)
        print(time.strftime("%H:%M:%S"), s.splitlines()[-1] if s else "?", flush=True)
        low = s.lower()
        if any(k in low for k in ("complete", "error", "cancel")):
            return "complete" in low
        time.sleep(poll)


def download(slug):
    out = ROOT / "results" / "kaggle" / slug
    out.mkdir(parents=True, exist_ok=True)
    print(kaggle("kernels", "output", f"{USER}/{slug}", "-p", str(out), "--force"))
    print(f"downloaded to {out}")


def main():
    global USER
    p = argparse.ArgumentParser()
    p.add_argument("--slug", required=True)
    p.add_argument("--job", action="append", default=[], help="command line relative to the repo root")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--no-wait", action="store_true")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--data", action="store_true", help="also ship data/ (LoRA training clips)")
    p.add_argument("--dataset", action="append", default=[],
                   help="private dataset slug of this user to attach; jobs see it at input/<slug>/")
    p.add_argument("--user", help="Kaggle username (default: $KAGGLE_USER, else %s)" % USER)
    args = p.parse_args()
    if args.user:
        USER = args.user
    if args.download_only:
        return download(args.slug)
    kdir = build(args.slug, args.job, args.data, args.dataset)
    if args.build_only:
        return
    print(kaggle("kernels", "push", "-p", str(kdir)))
    if args.no_wait:
        return
    time.sleep(15)
    ok = wait(args.slug)
    print("kernel finished OK" if ok else "kernel FAILED")
    download(args.slug)


if __name__ == "__main__":
    main()
