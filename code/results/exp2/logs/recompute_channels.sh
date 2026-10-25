#!/bin/bash
# waits until the orchestrator has passed the RAM-heavy local steps (quality gate, selection, grid) and is
# waiting on Kaggle, then recomputes the channels section for the four cells
E=/home/shivi/IE643/code/results/exp2
until grep -q "queueing\|patching and erasure downloaded" $E/logs/orchestrate.log && \
      ! pgrep -f "[e]xp2_(quality|select|grid).py" > /dev/null; do sleep 60; done
cd /home/shivi/IE643/code && env -u PYTHONPATH PYTHONPATH=src nice -n 10 .venv/bin/python $E/logs/recompute_channels.py
