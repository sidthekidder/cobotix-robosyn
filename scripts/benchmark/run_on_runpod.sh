#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/benchmark/run_on_runpod.sh \
    --target POD_USER@ssh.runpod.io \
    --key /path/to/private_key \
    [--episodes 50] [--seed 0] [--port 22] [--run-id NAME] [--dry-run]

The script uploads the committed Git revision, runs the pinned ACT benchmark,
and downloads benchmark_runs/NAME into artifacts/benchmarks/NAME.
EOF
}

target=""
key_path=""
episodes=50
seed=0
port=22
run_id="click_bell_act_$(date -u +%Y%m%dT%H%M%SZ)"
dry_run=0

while (( $# > 0 )); do
    case "$1" in
        --target) target="${2:?missing value for --target}"; shift 2 ;;
        --key) key_path="${2:?missing value for --key}"; shift 2 ;;
        --episodes) episodes="${2:?missing value for --episodes}"; shift 2 ;;
        --seed) seed="${2:?missing value for --seed}"; shift 2 ;;
        --port) port="${2:?missing value for --port}"; shift 2 ;;
        --run-id) run_id="${2:?missing value for --run-id}"; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$target" ]]; then
    echo "--target is required." >&2
    exit 2
fi
if ! [[ "$episodes" =~ ^[1-9][0-9]*$ ]]; then
    echo "--episodes must be a positive integer." >&2
    exit 2
fi
if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
    echo "--seed must be a non-negative integer." >&2
    exit 2
fi
if ! [[ "$port" =~ ^[1-9][0-9]*$ ]]; then
    echo "--port must be a positive integer." >&2
    exit 2
fi
if ! [[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "--run-id contains unsupported characters." >&2
    exit 2
fi
if (( dry_run == 0 )) && [[ ! -f "$key_path" ]]; then
    echo "SSH private key not found: $key_path" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain --untracked-files=normal)" ]]; then
    echo "Repository must be clean so the uploaded Git revision matches the recorded revision." >&2
    exit 2
fi
SOURCE_REVISION="$(git -C "$REPO_ROOT" rev-parse HEAD)"
REMOTE_WORKSPACE="/workspace/cobotix-benchmarks/$run_id"
REMOTE_REPO="$REMOTE_WORKSPACE/RoboSynChallenge"
LOCAL_RESULTS_ROOT="$REPO_ROOT/artifacts/benchmarks"
LOCAL_RUN_DIR="$LOCAL_RESULTS_ROOT/$run_id"
SSH_ARGS=(-p "$port" -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
if [[ -n "$key_path" ]]; then
    SSH_ARGS+=(-i "$key_path")
fi

if (( dry_run == 1 )); then
    cat <<EOF
Dry run validated.
Source revision: $SOURCE_REVISION
Remote target:   $target
Remote workspace:$REMOTE_WORKSPACE
Episodes/seed:   $episodes / $seed
Local results:   $LOCAL_RUN_DIR
EOF
    exit 0
fi

if [[ -e "$LOCAL_RUN_DIR" ]]; then
    echo "Refusing to overwrite local results: $LOCAL_RUN_DIR" >&2
    exit 2
fi

echo "Creating isolated remote workspace: $REMOTE_WORKSPACE"
ssh "${SSH_ARGS[@]}" "$target" \
    "test ! -e '$REMOTE_WORKSPACE' && mkdir -p '$REMOTE_REPO'"

echo "Uploading committed revision $SOURCE_REVISION"
git -C "$REPO_ROOT" archive --format=tar "$SOURCE_REVISION" \
    | gzip -1 \
    | ssh "${SSH_ARGS[@]}" "$target" "tar -xzf - -C '$REMOTE_REPO'"
printf '%s\n' "$SOURCE_REVISION" \
    | ssh "${SSH_ARGS[@]}" "$target" "cat > '$REMOTE_REPO/.cobotix-source-revision'"

echo "Running $episodes episodes on Runpod"
set +e
ssh "${SSH_ARGS[@]}" "$target" \
    "cd '$REMOTE_REPO' && RUN_ID='$run_id' EPISODES='$episodes' SEED='$seed' bash scripts/benchmark/run_act_benchmark.sh"
remote_status=$?
set -e

mkdir -p "$LOCAL_RESULTS_ROOT"
echo "Downloading available results to: $LOCAL_RUN_DIR"
set +e
ssh "${SSH_ARGS[@]}" "$target" \
    "test -d '$REMOTE_REPO/benchmark_runs/$run_id' && tar -C '$REMOTE_REPO/benchmark_runs' -czf - '$run_id'" \
    | tar -C "$LOCAL_RESULTS_ROOT" -xzf -
download_status=$?
set -e

if (( download_status != 0 )); then
    echo "Result download failed; data may still be under $REMOTE_REPO on the pod." >&2
    exit "$download_status"
fi
if (( remote_status != 0 )); then
    echo "Remote benchmark failed with status $remote_status; partial logs were downloaded." >&2
    exit "$remote_status"
fi

echo "Benchmark and download complete: $LOCAL_RUN_DIR"
echo "Terminate the pod after reviewing the downloaded artifacts."
