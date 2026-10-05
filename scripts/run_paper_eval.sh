#!/usr/bin/env bash
# Simulation evaluation of the paper (Section 5.1): Iso, Ours and Oracle on the four objects,
# 100 trials each, with the single-task policies and with the multi-task policy; then the
# per-trial scores and the tables (Table 1, the multi-task table and the diagnostics table).
#
# usage: bash scripts/run_paper_eval.sh <checkpoint_dir> <output_dir>
#   <checkpoint_dir>  holds revolute.pth, cylindrical.pth, planar.pth, universal.pth and multitask.pth
#   <output_dir>      receives single_task/<object>/<controller>/, multi_task/<object>/<controller>/
#                     (run directories that already exist there are replaced),
#                     scores_single_task.jsonl, scores_multi_task.jsonl, tables.md and summary.csv
#
# PYTHON selects the interpreter that runs Isaac Lab (default: python), for example
#   PYTHON="/path/to/IsaacLab/isaaclab.sh -p" bash scripts/run_paper_eval.sh checkpoints runs
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: bash scripts/run_paper_eval.sh <checkpoint_dir> <output_dir>" >&2
  exit 1
fi
CHECKPOINTS=$(realpath "$1")
OUT=$(realpath -m "$2")
PYTHON=${PYTHON:-python}
cd "$(dirname "$(realpath "$0")")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

OBJECTS=(revolute cylindrical planar universal)
CONTROLLERS=(iso ours oracle)

evaluate() {  # <policy setting> <object> <controller> <checkpoint>
  local run="$OUT/$1/$2/$3"
  rm -rf "$run"
  mkdir -p "$run"
  if ! $PYTHON scripts/evaluate.py --object "$2" --controller "$3" --checkpoint "$4" --out "$run" --headless \
    > "$run/evaluate.log" 2>&1 || [[ ! -f "$run/results.json" ]]; then
    echo "evaluation of $1/$2/$3 failed, see $run/evaluate.log" >&2
    exit 1
  fi
  grep -E "successes ->" "$run/evaluate.log"
}

for object in "${OBJECTS[@]}"; do
  for controller in "${CONTROLLERS[@]}"; do
    evaluate single_task "$object" "$controller" "$CHECKPOINTS/$object.pth"
  done
done
for object in "${OBJECTS[@]}"; do
  for controller in "${CONTROLLERS[@]}"; do
    evaluate multi_task "$object" "$controller" "$CHECKPOINTS/multitask.pth"
  done
done

$PYTHON scripts/score.py "$OUT/single_task" --out "$OUT/scores_single_task.jsonl"
$PYTHON scripts/score.py "$OUT/multi_task" --out "$OUT/scores_multi_task.jsonl"
$PYTHON scripts/make_tables.py --single_task "$OUT/scores_single_task.jsonl" \
  --multi_task "$OUT/scores_multi_task.jsonl" --out "$OUT/tables.md" --csv "$OUT/summary.csv"
