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
RUN_ID="${RUN_ID:-${TASK}_act_$(date -u +%Y%m%dT%H%M%SZ)}"
CHECKPOINT_REPO="${CHECKPOINT_REPO:-EDEM-AI/ACT_sim_click_bell}"
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

bash policy/act/eval.sh \
    "$TASK" "$SETTING" "$CHECKPOINT_DIR" "$GPU_ID" \
    --max_episodes "$EPISODES" \
    --seed "$SEED" \
    --pytorch_device cuda \
    --headless True \
    --eval_result_dir "$RUN_DIR/eval_result" \
    2>&1 | tee "$RUN_DIR/evaluation.log"

"$PYTHON_BIN" scripts/benchmark/summarize_eval.py \
    "$RUN_DIR/eval_result" \
    --output "$RUN_DIR/summary.md" \
    | tee "$RUN_DIR/summary.stdout.txt"

echo "Benchmark complete: $RUN_DIR"
