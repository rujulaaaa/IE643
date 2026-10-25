#!/bin/bash
# usage: kaggle_queue.sh <slug>...  -- push each built kernel (build/<slug>) when a GPU slot is free, retrying every
# 5 min while Kaggle refuses (two concurrent GPU sessions, weekly quota)
K=~/kaggle/.venv/bin/kaggle; B=$(cd "$(dirname "$0")/.." && pwd)/build
for slug in "$@"; do
  until out=$($K kernels push -p $B/$slug 2>&1); echo "$out" | grep -q "successfully pushed"; do
    echo "$(date +%T) $slug: $(echo "$out" | tail -1)"; sleep 300
  done
  echo "$(date +%T) $slug: pushed"
done
