# Demo video: how to run and what to show

The interface is a Gradio app. Everything except live generation runs from the saved results, without a GPU.

```bash
cd code
pip install -e .                        # once (Python 3.10+)
python scripts/app.py --no-gpu          # saved results only; opens http://127.0.0.1:7860
python scripts/app.py                   # on a CUDA machine: also live edits and live CameraCtrl transfer
```

Clips play as MP4 videos. They are converted from the saved GIFs with `ffmpeg` the first time they are shown, so open
each tab once before recording.

## Suggested recording (about 5 minutes, 1400 x 950 browser window)

| Time | Where | What to show | What to say |
|---|---|---|---|
| 0:00 | header | title and the two colour chips | The question: which layers of a video diffusion U-Net hold what is shown and how it moves. |
| 0:20 | **Where motion lives** | the selectivity map | Intensive phase: content dominates almost every layer; motion is selective only in the first decoder block's motion module, steps 0-2. |
| 0:50 | **Results** | run `vdprobe-test`, a pair; play the clips | Edits on 12 test pairs; attention-based edits beat patches placed by the map. |
| 1:30 | **Viewpoint (novelty)** | the intro text | The review asked whether the motion layers also carry the camera viewpoint, and whether an action can be moved while the target keeps its subject and viewpoint. Two new models: CameraCtrl and MVDream. |
| 1:50 | **1 · Does motion carry viewpoint?** | choose `up_blocks.0`, then `down_blocks.0` | In up block 0, where Experiment 1 found motion, a camera change moves the activations about 5x more than a subject change. In the down blocks the camera dominates the first steps. |
| 2:30 | **2 · Donor with another camera** | play donor, target, patched clips | Copying the motion modules from a donor filmed with a different camera brings that camera: the zoom becomes a pan. True for every block and step window we tried. |
| 3:00 | **3 · Action transfer, viewpoint kept** | pair "deer jumping -> horse running (zoom in)", candidate "Up-block 1"; play all four clips; then switch the candidate to "Up-blocks 1-3, steps 0-12" | Fix: generate the donor with the target's camera. One up block moves the deer's gait onto the horse and the horse stays a horse (0.26 of the way to the donor's motion, identity leak 0.05, same value in a repeat run). Copying more blocks or steps gives a deer-horse hybrid: the trade-off plot. |
| 4:00 | **4 · Multi-view model** | the first two sheets | The same holds in MVDream: with the same view ring, the tiger takes the horse's posture in all four views and stays a tiger. |
| 4:30 | **5 · Pre-registered rules** | the table | Every rule was written before its run; we report the failures too. |
| 4:45 | **6 · Try it live** (GPU only) | pick deer / jumping / horse / zoom-in, Generate | Live run of the same transfer (about 30 s per pair on an H100). Skip this step without a GPU. |

## Where the shown results come from

| Tab | Files | Script |
|---|---|---|
| 1 | `results/action_view/sweep_cells.csv`, `fig_f2_cc_maps.png`, `fig_f3_mv_maps.png` | `scripts/camera_sweep.py --model cc --set configs/av_set.yaml`, `scripts/mvdream_probe.py sweep` |
| 2 | `results/autoresearch/av_patch/` | `scripts/action_view_patch.py` |
| 3 | `results/autoresearch/av_combo/` | `scripts/action_view_patch.py --combo --gifs` |
| 4 | `results/autoresearch/mv_matched_sheets/`, `mv_patch/` | `scripts/mvdream_probe.py patch` |
| 5 | `results/action_view/*/patch_verdict.*`, `sweep.json`, `mv_sweep.json` | `scripts/action_view_analysis.py` |
