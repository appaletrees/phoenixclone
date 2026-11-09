#!/usr/bin/env bash
# Run from nohup/setsid; a lock prevents two evaluators sharing the same output.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
output_dir="${1:-runs/piper-dataset-eval/20261028}"
mkdir -p "$output_dir"
exec 9>"$output_dir/evaluation.lock"
if ! flock -n 9; then
    echo "An evaluator is already running for $output_dir" >&2
    exit 1
fi
echo "$$" > "$output_dir/background.pid"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "Background evaluation started: $(date -Is)"
set +e
.venv-piper-inference/bin/python -u tools/evaluate_piper_trajectories.py \
    --task all --output-dir "$output_dir"
result=$?
set -e
echo "$result" > "$output_dir/exit_status.txt"
echo "Background evaluation finished: $(date -Is), exit=$result"
exit "$result"
