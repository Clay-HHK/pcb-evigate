#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
exec python -m scripts.run_full_training
