#!/usr/bin/env bash
# Run the train.jsonl verification UI (from train/ directory).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec uv run python labelling/verify_server.py "$@"
