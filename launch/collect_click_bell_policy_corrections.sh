#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

python scripts/collect_click_bell_policy_corrections.py \
  --gym_config configs/click_bell/random/gym_config.json \
  --action_config configs/click_bell/action_config.json \
  --num_envs 1 \
  --headless \
  "$@"
