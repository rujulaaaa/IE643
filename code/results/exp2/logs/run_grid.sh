#!/bin/bash
# after the channels recompute (RAM), re-run the 2x2 grid
cd /home/shivi/IE643/code; L=results/exp2/logs/orchestrate_local.log
until grep -q "CHANNELS RECOMPUTED\|Traceback (most recent call last):\s*$" <(grep -A0 "CHANNELS RECOMPUTED" $L) || ! pgrep -f "[r]ecompute_channels.py" > /dev/null; do sleep 30; done
R=results/kaggle; cells=""
for c in sd15_mm2 epicrealism_mm2 sd15_mm3 epicrealism_mm3; do cells="$cells $c=$R/vdprobe-exp2-${c//_/-}/out/$c"; done
env -u PYTHONPATH PYTHONPATH=src nice -n 10 .venv/bin/python scripts/exp2_grid.py --cells $cells --out results/exp2/grid && echo "GRID DONE" || echo "GRID FAILED"
