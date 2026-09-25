# Reproducible ACT benchmark workflow

This workflow runs the `click_bell` ACT checkpoint for 50 deterministic
episodes on a temporary Runpod pod and copies every result back before the pod
is terminated.

## What is pinned

- EmbodiChain commit: `9ebee30011f378f94a7cbe78b01d8c2eacba231a`
  (`v0.2.4`)
- ACT checkpoint: `EDEM-AI/ACT_sim_click_bell`
- Checkpoint revision: `677e65fbb15974024ff840893496197ef7db26d4`
- Python dependencies: `policy/act/uv.lock`
- Repository source: the local committed Git revision uploaded by the runner

## Pod configuration

Use the verified EmbodiChain image and enable SSH:

- Image: `dexforce/embodichain:ubuntu22.04-cuda12.8`
- Container disk: at least 80 GB
- Container command: `bash -lc 'sleep infinity'`
- Environment: `NVIDIA_DRIVER_CAPABILITIES=all`
- Port: `22/tcp`

No persistent volume is required. The runner downloads results to the Mac even
when evaluation exits with an error, as long as SSH and the container remain
available.

## Validate locally without connecting

```bash
scripts/benchmark/run_on_runpod.sh \
  --target POD_USER@ssh.runpod.io \
  --episodes 50 \
  --dry-run
```

## Run

Copy the SSH proxy username from the Runpod pod's Connect panel:

```bash
scripts/benchmark/run_on_runpod.sh \
  --target POD_USER@ssh.runpod.io \
  --key /absolute/path/to/private_key \
  --episodes 50 \
  --seed 0
```

The local output is written under `artifacts/benchmarks/<run-id>/` and includes:

- `run_manifest.json`: exact source, simulator, checkpoint, and hardware
- `bootstrap.log`: dependency setup, checkpoint download, and evaluation output
- `evaluation.log`: full evaluator output
- `eval_result/`: raw metrics and episode videos
- `summary.md`: success rate, Wilson interval, bell position, inferred arm use,
  closest end-effector approach, end-effector path length, and maximum button
  depression

The benchmark uses the same seeded candidate sequence every time. RoboSyn's
expert feasibility filter may skip candidates for which its own planner cannot
construct a valid trajectory, but accepted policy episodes remain reproducible.

Terminate the pod only after the local result directory has been inspected.

## Audit released demonstration coverage

The released parquet files do not include the sampled bell pose, so the audit
separates what is measured from what is inferred. It reports joint and
end-effector motion for each arm. It also reports the lowest active-arm
end-effector position as an approach proxy, clearly labeled as such.

```bash
python scripts/benchmark/analyze_lerobot_dataset.py \
  --repo-id RoboSynChallenge/cobotmagic_Sim_click_bell \
  --expected-episodes 1000 \
  --progress \
  --output-json artifacts/dataset_audits/click_bell.json \
  --output-markdown artifacts/dataset_audits/click_bell.md
```

This downloads only parquet data and metadata, excluding the much larger camera
videos.
