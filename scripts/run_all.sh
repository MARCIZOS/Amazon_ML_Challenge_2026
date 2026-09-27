#!/usr/bin/env bash
# Full pipeline on Linux / macOS: indexes -> train -> predict -> 5 validated submissions.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
echo "Using: $PY"
$PY -m pip install -r requirements.txt
$PY -m pytest -q
$PY -m src.preprocessing.build_indexes --config configs/pipeline.yaml
$PY -m src.train
$PY -m src.predict
for v in 2 3 4 5 1; do $PY -m src.make_submission --variant "$v"; done
echo "ALL DONE. output/ = variant 1; other variants in data/interim/submissions/; report in experiments/results/"
