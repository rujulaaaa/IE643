#!/bin/bash
# erasure runs on shivanshisme (default token): wait, download, mark .download_ok for the orchestrator
unset KAGGLE_API_TOKEN
cd /home/shivi/IE643/code && KAGGLE_USER=shivanshisme exec scripts/kaggle_wait.sh vdprobe-exp2-erase results/kaggle/vdprobe-exp2-erase 2
