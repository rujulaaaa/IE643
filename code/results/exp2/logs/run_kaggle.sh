#!/bin/bash
# Experiment 2 Kaggle stage on the second account (token read from a private file, never logged)
export KAGGLE_USER=rujularahate
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/acct2_token)
export ELSEWHERE="vdprobe-exp2-erase"   # runs on shivanshisme, see run_erase_wait.sh
cd /home/shivi/IE643/code && NJOBS=4 exec bash scripts/exp2_orchestrate.sh
