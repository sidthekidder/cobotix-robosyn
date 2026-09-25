#!/usr/bin/env bash
set -Eeuo pipefail

# Run inside a fresh Runpod workspace containing this repository. This script
# installs the pinned simulator dependency, prepares ACT, downloads the pinned
# checkpoint, and writes a self-contained benchmark directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(dirname "$REPO_ROOT")"
ACT_DIR="$REPO_ROOT/policy/act"
EMBODICHAIN_DIR="$WORKSPACE_ROOT/EmbodiChain"

TASK="${TASK:-click_bell}"
SETTING="${SETTING:-random}"
EPISODES="${EPISODES:-50}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
ACT_N_ACTION_STEPS="${ACT_N_ACTION_STEPS:-}"
ACT_TEMPORAL_ENSEMBLE_COEFF="${ACT_TEMPORAL_ENSEMBLE_COEFF:-}"
ACT_RECOVERY_ENABLED="${ACT_RECOVERY_ENABLED:-false}"
ACT_RECOVERY_MIN_STEP="${ACT_RECOVERY_MIN_STEP:-60}"
ACT_RECOVERY_PLATEAU_STEPS="${ACT_RECOVERY_PLATEAU_STEPS:-10}"
ACT_RECOVERY_MIN_PRESS_DEPTH_M="${ACT_RECOVERY_MIN_PRESS_DEPTH_M:-0.0011}"
ACT_RECOVERY_PRESS_EPSILON_M="${ACT_RECOVERY_PRESS_EPSILON_M:-0.0001}"
ACT_RECOVERY_COOLDOWN_STEPS="${ACT_RECOVERY_COOLDOWN_STEPS:-25}"
ACT_RECOVERY_MAX_REPLANS="${ACT_RECOVERY_MAX_REPLANS:-2}"
ACT_PRESS_ORACLE_ENABLED="${ACT_PRESS_ORACLE_ENABLED:-false}"
ACT_PRESS_ORACLE_MIN_STEP="${ACT_PRESS_ORACLE_MIN_STEP:-35}"
ACT_PRESS_ORACLE_TARGET_PLAN_STEP="${ACT_PRESS_ORACLE_TARGET_PLAN_STEP:-49}"
ACT_PRESS_ORACLE_TRIGGER_MAX_JOINT_ERROR_RAD="${ACT_PRESS_ORACLE_TRIGGER_MAX_JOINT_ERROR_RAD:-0.45}"
ACT_PRESS_ORACLE_INTERPOLATION_STEPS="${ACT_PRESS_ORACLE_INTERPOLATION_STEPS:-12}"
ACT_PRESS_ORACLE_HOLD_STEPS="${ACT_PRESS_ORACLE_HOLD_STEPS:-8}"
ACT_PRESS_ORACLE_TELEMETRY_STRIDE="${ACT_PRESS_ORACLE_TELEMETRY_STRIDE:-2}"
EVAL_EXPERT_CHECK="${EVAL_EXPERT_CHECK:-true}"
EVAL_EPISODE_SEEDS="${EVAL_EPISODE_SEEDS:-}"
RUN_ID="${RUN_ID:-${TASK}_act_$(date -u +%Y%m%dT%H%M%SZ)}"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-RoboSynChallenge/ACT_sim_click_bell}"
CHECKPOINT_REVISION="${CHECKPOINT_REVISION:-677e65fbb15974024ff840893496197ef7db26d4}"
EMBODICHAIN_REVISION="${EMBODICHAIN_REVISION:-9ebee30011f378f94a7cbe78b01d8c2eacba231a}"

if ! [[ "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES must be a positive integer, got: $EPISODES" >&2
    exit 2
fi
if ! [[ "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUN_ID may contain only letters, digits, dots, underscores, and hyphens." >&2
    exit 2
fi
if [[ -n "$ACT_N_ACTION_STEPS" ]] && ! [[ "$ACT_N_ACTION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACT_N_ACTION_STEPS must be a positive integer when set." >&2
    exit 2
fi
if [[ "$EVAL_EXPERT_CHECK" != "true" && "$EVAL_EXPERT_CHECK" != "false" ]]; then
    echo "EVAL_EXPERT_CHECK must be true or false." >&2
    exit 2
fi
if [[ "$ACT_RECOVERY_ENABLED" != "true" && "$ACT_RECOVERY_ENABLED" != "false" ]]; then
    echo "ACT_RECOVERY_ENABLED must be true or false." >&2
    exit 2
fi
if [[ "$ACT_PRESS_ORACLE_ENABLED" != "true" && "$ACT_PRESS_ORACLE_ENABLED" != "false" ]]; then
    echo "ACT_PRESS_ORACLE_ENABLED must be true or false." >&2
    exit 2
fi
if [[ "$ACT_PRESS_ORACLE_ENABLED" == "true" && "$ACT_RECOVERY_ENABLED" == "true" ]]; then
    echo "The diagnostic press oracle and contact recovery must run separately." >&2
    exit 2
fi
if [[ -n "$ACT_TEMPORAL_ENSEMBLE_COEFF" ]] && ! [[ "$ACT_TEMPORAL_ENSEMBLE_COEFF" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ACT_TEMPORAL_ENSEMBLE_COEFF must be a non-negative number." >&2
    exit 2
fi
if [[ -n "$ACT_TEMPORAL_ENSEMBLE_COEFF" && "$ACT_RECOVERY_ENABLED" == "true" ]]; then
    echo "Temporal ensembling and contact recovery must be evaluated separately." >&2
    exit 2
fi
for value_name in ACT_RECOVERY_MIN_STEP ACT_RECOVERY_PLATEAU_STEPS ACT_RECOVERY_MAX_REPLANS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value_name must be a positive integer, got: $value" >&2
        exit 2
    fi
done
if ! [[ "$ACT_RECOVERY_COOLDOWN_STEPS" =~ ^[0-9]+$ ]]; then
    echo "ACT_RECOVERY_COOLDOWN_STEPS must be a non-negative integer." >&2
    exit 2
fi
if [[ -n "$EVAL_EPISODE_SEEDS" ]]; then
    python3 - "$EVAL_EPISODE_SEEDS" "$EPISODES" <<'PY'
import json
import sys

seeds = json.loads(sys.argv[1])
episodes = int(sys.argv[2])
if not isinstance(seeds, list) or not all(isinstance(seed, int) for seed in seeds):
    raise SystemExit("EVAL_EPISODE_SEEDS must be a JSON list of integers.")
if len(seeds) != episodes:
    raise SystemExit(
        f"EVAL_EPISODE_SEEDS contains {len(seeds)} seeds, expected {episodes}."
    )
PY
fi

RUN_DIR="$REPO_ROOT/benchmark_runs/$RUN_ID"
CHECKPOINT_DIR="$REPO_ROOT/checkpoints/ACT_sim_click_bell"
if [[ -e "$RUN_DIR" ]]; then
    echo "Refusing to overwrite existing benchmark directory: $RUN_DIR" >&2
    exit 2
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/bootstrap.log") 2>&1

echo "Benchmark run: $RUN_ID"
echo "Repository: $REPO_ROOT"
echo "Episodes/seed: $EPISODES/$SEED"

missing_packages=()
command -v git >/dev/null 2>&1 || missing_packages+=(git)
command -v curl >/dev/null 2>&1 || missing_packages+=(curl)
command -v ffmpeg >/dev/null 2>&1 || missing_packages+=(ffmpeg)
if (( ${#missing_packages[@]} > 0 )); then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "${missing_packages[@]}"
fi

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
uv --version

if [[ ! -d "$EMBODICHAIN_DIR/.git" ]]; then
    git clone https://github.com/DexForce/EmbodiChain.git "$EMBODICHAIN_DIR"
fi
actual_embodichain_revision="$(git -C "$EMBODICHAIN_DIR" rev-parse HEAD)"
if [[ "$actual_embodichain_revision" != "$EMBODICHAIN_REVISION" ]]; then
    git -C "$EMBODICHAIN_DIR" fetch origin "$EMBODICHAIN_REVISION"
    git -C "$EMBODICHAIN_DIR" checkout --detach "$EMBODICHAIN_REVISION"
fi

cd "$ACT_DIR"
uv sync --frozen
PYTHON_BIN="$ACT_DIR/.venv/bin/python"

export HF_HOME="${HF_HOME:-$WORKSPACE_ROOT/.cache/huggingface}"
"$PYTHON_BIN" - "$CHECKPOINT_REPO" "$CHECKPOINT_REVISION" "$CHECKPOINT_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo_id, revision, destination = sys.argv[1:]
snapshot_download(
    repo_id=repo_id,
    revision=revision,
    local_dir=destination,
)
PY

source_revision="unknown"
if [[ -f "$REPO_ROOT/.cobotix-source-revision" ]]; then
    source_revision="$(cat "$REPO_ROOT/.cobotix-source-revision")"
elif [[ -d "$REPO_ROOT/.git" ]]; then
    source_revision="$(git -C "$REPO_ROOT" rev-parse HEAD)"
fi

SOURCE_REVISION="$source_revision" \
EMBODICHAIN_REVISION="$EMBODICHAIN_REVISION" \
CHECKPOINT_REPO="$CHECKPOINT_REPO" \
CHECKPOINT_REVISION="$CHECKPOINT_REVISION" \
TASK="$TASK" SETTING="$SETTING" EPISODES="$EPISODES" SEED="$SEED" \
ACT_N_ACTION_STEPS="$ACT_N_ACTION_STEPS" \
ACT_TEMPORAL_ENSEMBLE_COEFF="$ACT_TEMPORAL_ENSEMBLE_COEFF" \
ACT_RECOVERY_ENABLED="$ACT_RECOVERY_ENABLED" \
ACT_RECOVERY_MIN_STEP="$ACT_RECOVERY_MIN_STEP" \
ACT_RECOVERY_PLATEAU_STEPS="$ACT_RECOVERY_PLATEAU_STEPS" \
ACT_RECOVERY_MIN_PRESS_DEPTH_M="$ACT_RECOVERY_MIN_PRESS_DEPTH_M" \
ACT_RECOVERY_PRESS_EPSILON_M="$ACT_RECOVERY_PRESS_EPSILON_M" \
ACT_RECOVERY_COOLDOWN_STEPS="$ACT_RECOVERY_COOLDOWN_STEPS" \
ACT_RECOVERY_MAX_REPLANS="$ACT_RECOVERY_MAX_REPLANS" \
ACT_PRESS_ORACLE_ENABLED="$ACT_PRESS_ORACLE_ENABLED" \
ACT_PRESS_ORACLE_MIN_STEP="$ACT_PRESS_ORACLE_MIN_STEP" \
ACT_PRESS_ORACLE_TARGET_PLAN_STEP="$ACT_PRESS_ORACLE_TARGET_PLAN_STEP" \
ACT_PRESS_ORACLE_TRIGGER_MAX_JOINT_ERROR_RAD="$ACT_PRESS_ORACLE_TRIGGER_MAX_JOINT_ERROR_RAD" \
ACT_PRESS_ORACLE_INTERPOLATION_STEPS="$ACT_PRESS_ORACLE_INTERPOLATION_STEPS" \
ACT_PRESS_ORACLE_HOLD_STEPS="$ACT_PRESS_ORACLE_HOLD_STEPS" \
ACT_PRESS_ORACLE_TELEMETRY_STRIDE="$ACT_PRESS_ORACLE_TELEMETRY_STRIDE" \
EVAL_EXPERT_CHECK="$EVAL_EXPERT_CHECK" \
EVAL_EPISODE_SEEDS="$EVAL_EPISODE_SEEDS" \
"$PYTHON_BIN" - "$RUN_DIR/run_manifest.json" <<'PY'
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def output(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "task": os.environ["TASK"],
    "setting": os.environ["SETTING"],
    "episodes": int(os.environ["EPISODES"]),
    "seed": int(os.environ["SEED"]),
    "act_n_action_steps": (
        int(os.environ["ACT_N_ACTION_STEPS"])
        if os.environ.get("ACT_N_ACTION_STEPS")
        else None
    ),
    "act_temporal_ensemble_coeff": (
        float(os.environ["ACT_TEMPORAL_ENSEMBLE_COEFF"])
        if os.environ.get("ACT_TEMPORAL_ENSEMBLE_COEFF")
        else None
    ),
    "act_recovery": {
        "enabled": os.environ["ACT_RECOVERY_ENABLED"] == "true",
        "min_step": int(os.environ["ACT_RECOVERY_MIN_STEP"]),
        "plateau_steps": int(os.environ["ACT_RECOVERY_PLATEAU_STEPS"]),
        "min_press_depth_m": float(os.environ["ACT_RECOVERY_MIN_PRESS_DEPTH_M"]),
        "press_epsilon_m": float(os.environ["ACT_RECOVERY_PRESS_EPSILON_M"]),
        "cooldown_steps": int(os.environ["ACT_RECOVERY_COOLDOWN_STEPS"]),
        "max_replans": int(os.environ["ACT_RECOVERY_MAX_REPLANS"]),
    },
    "act_press_oracle": {
        "enabled": os.environ["ACT_PRESS_ORACLE_ENABLED"] == "true",
        "diagnostic_only": True,
        "min_step": int(os.environ["ACT_PRESS_ORACLE_MIN_STEP"]),
        "target_plan_step": int(os.environ["ACT_PRESS_ORACLE_TARGET_PLAN_STEP"]),
        "trigger_max_joint_error_rad": float(os.environ["ACT_PRESS_ORACLE_TRIGGER_MAX_JOINT_ERROR_RAD"]),
        "interpolation_steps": int(os.environ["ACT_PRESS_ORACLE_INTERPOLATION_STEPS"]),
        "hold_steps": int(os.environ["ACT_PRESS_ORACLE_HOLD_STEPS"]),
        "telemetry_stride": int(os.environ["ACT_PRESS_ORACLE_TELEMETRY_STRIDE"]),
    },
    "eval_expert_check": os.environ["EVAL_EXPERT_CHECK"] == "true",
    "eval_episode_seeds": (
        json.loads(os.environ["EVAL_EPISODE_SEEDS"])
        if os.environ.get("EVAL_EPISODE_SEEDS")
        else None
    ),
    "source_revision": os.environ["SOURCE_REVISION"],
    "embodichain_revision": os.environ["EMBODICHAIN_REVISION"],
    "checkpoint_repo": os.environ["CHECKPOINT_REPO"],
    "checkpoint_revision": os.environ["CHECKPOINT_REVISION"],
    "hostname": platform.node(),
    "nvidia_smi": output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    "vulkan_summary": output(["vulkaninfo", "--summary"]),
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

cd "$REPO_ROOT"
export NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-all}"
export PYTHON_BIN
export EMBODICHAIN_ROOT="$EMBODICHAIN_DIR"

act_overrides=()
if [[ -n "$ACT_N_ACTION_STEPS" ]]; then
    act_overrides+=(--n_action_steps "$ACT_N_ACTION_STEPS")
fi
if [[ -n "$ACT_TEMPORAL_ENSEMBLE_COEFF" ]]; then
    act_overrides+=(
        --act_temporal_ensemble_coeff "$ACT_TEMPORAL_ENSEMBLE_COEFF"
    )
fi
if [[ "$ACT_RECOVERY_ENABLED" == "true" ]]; then
    act_overrides+=(
        --act_recovery_enabled true
        --act_recovery_min_step "$ACT_RECOVERY_MIN_STEP"
        --act_recovery_plateau_steps "$ACT_RECOVERY_PLATEAU_STEPS"
        --act_recovery_min_press_depth_m "$ACT_RECOVERY_MIN_PRESS_DEPTH_M"
        --act_recovery_press_epsilon_m "$ACT_RECOVERY_PRESS_EPSILON_M"
        --act_recovery_cooldown_steps "$ACT_RECOVERY_COOLDOWN_STEPS"
        --act_recovery_max_replans "$ACT_RECOVERY_MAX_REPLANS"
    )
fi
if [[ "$ACT_PRESS_ORACLE_ENABLED" == "true" ]]; then
    act_overrides+=(
        --act_press_oracle_enabled true
        --act_press_oracle_min_step "$ACT_PRESS_ORACLE_MIN_STEP"
        --act_press_oracle_target_plan_step "$ACT_PRESS_ORACLE_TARGET_PLAN_STEP"
        --act_press_oracle_trigger_max_joint_error_rad "$ACT_PRESS_ORACLE_TRIGGER_MAX_JOINT_ERROR_RAD"
        --act_press_oracle_interpolation_steps "$ACT_PRESS_ORACLE_INTERPOLATION_STEPS"
        --act_press_oracle_hold_steps "$ACT_PRESS_ORACLE_HOLD_STEPS"
        --act_press_oracle_telemetry_stride "$ACT_PRESS_ORACLE_TELEMETRY_STRIDE"
    )
fi
if [[ -n "$EVAL_EPISODE_SEEDS" ]]; then
    act_overrides+=(--eval_episode_seeds "$EVAL_EPISODE_SEEDS")
fi

bash policy/act/eval.sh \
    "$TASK" "$SETTING" "$CHECKPOINT_DIR" "$GPU_ID" \
    --max_episodes "$EPISODES" \
    --seed "$SEED" \
    --eval_expert_check "$EVAL_EXPERT_CHECK" \
    --pytorch_device cuda \
    --headless True \
    --eval_result_dir "$RUN_DIR/eval_result" \
    "${act_overrides[@]}" \
    2>&1 | tee "$RUN_DIR/evaluation.log"

"$PYTHON_BIN" scripts/benchmark/summarize_eval.py \
    "$RUN_DIR/eval_result" \
    --output "$RUN_DIR/summary.md" \
    | tee "$RUN_DIR/summary.stdout.txt"

echo "Benchmark complete: $RUN_DIR"
