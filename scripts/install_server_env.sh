#!/usr/bin/env bash
# Run inside the supplied server image, from the repository root.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-python}"
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != --dry-run ) ]]; then
  echo 'Usage: bash scripts/install_server_env.sh [--dry-run]' >&2
  exit 2
fi
PROFILE=requirements/server-cu130-py312.constraints.txt
IMAGE=resources/compat/server-image.constraints.txt
PLAN=resources/compat/server-install-plan.json
"$PYTHON" tools/check_server_runtime.py --base-only --write-image-constraints "$IMAGE" \
  --output resources/compat/server-base.json
ARGS=(--no-build-isolation -c "$PROFILE" -c "$IMAGE" -e '.[train,logging]' 'vllm==0.19.0')
"$PYTHON" -m pip install --dry-run --report "$PLAN" "${ARGS[@]}"
"$PYTHON" tools/check_server_runtime.py --check-plan "$PLAN" \
  --output resources/compat/server-plan-check.json
if [[ "${1:-}" == --dry-run ]]; then exit 0; fi
"$PYTHON" -m pip install "${ARGS[@]}"
"$PYTHON" -m pip check
"$PYTHON" tools/check_server_runtime.py
