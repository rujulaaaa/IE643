#!/bin/bash
# local steps only (no KAGGLE_USER): selection + grid via the orchestrator, then the channels recompute (D-36 item 3)
cd /home/shivi/IE643/code
env -u KAGGLE_USER NJOBS=4 bash scripts/exp2_orchestrate.sh || exit 1
env -u PYTHONPATH PYTHONPATH=src nice -n 10 .venv/bin/python results/exp2/logs/recompute_channels.py
echo "LOCAL STEPS DONE"
