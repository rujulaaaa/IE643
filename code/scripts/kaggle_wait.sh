#!/bin/bash
# usage: kaggle_wait.sh <slug> <dest> [min_manifests]  -- poll a Kaggle kernel until it ends, then download its
# output, retrying until the CLI exits 0 and at least min_manifests manifest.json files exist; touches
# <dest>/.download_ok on success
K=~/kaggle/.venv/bin/kaggle; slug=$1; dest=$2; need=${3:-0}; U=${KAGGLE_USER:?set KAGGLE_USER}
for i in $(seq 1 1000); do
  s=$($K kernels status $U/$slug 2>&1 | tail -1); echo "$(date +%T) $s"
  case "$s" in *COMPLETE*|*ERROR*|*CANCEL*) break;; esac; sleep 60
done
mkdir -p "$dest"
for t in $(seq 1 8); do
  $K kernels output $U/$slug -p "$dest" > "$dest/.download.log" 2>&1; rc=$?
  n=$(find "$dest" -name manifest.json | wc -l)
  echo "$(date +%T) download attempt $t: exit $rc, $(grep -c 'Output file downloaded' "$dest/.download.log") files, $n manifests"
  [ $rc -eq 0 ] && [ "$n" -ge "$need" ] && { touch "$dest/.download_ok"; echo "DOWNLOAD OK $slug"; exit 0; }
  grep -v "Output file downloaded" "$dest/.download.log" | tail -3
  sleep 60
done
echo "DOWNLOAD FAILED $slug"; exit 1
