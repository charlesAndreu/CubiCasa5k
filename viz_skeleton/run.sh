#!/usr/bin/env bash
# Start the wall-skeleton post-process viewer (requires conda env charles-cubicasa).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate charles-cubicasa
else
  echo "conda not found; activate charles-cubicasa manually" >&2
  exit 1
fi

exec python viz_skeleton/app.py "$@"
