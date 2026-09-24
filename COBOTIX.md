# Cobotix competition workspace

## Status
Official repository downloaded on 2026-09-23. Base commit:
7503c7cb000bdf869cdba302e329724b3a223c75.
Shallow clone; use `git fetch --unshallow upstream` if older history is needed.
Local branch: `cobotix/baseline`. `upstream` points to the organizers.
No Cobotix GitHub repository has been created or connected. No cloud spending.
Git identity was already configured and has not been changed.

## Local inspection
- Mac: Apple Silicon. Git available; uv not found on PATH.
- Main evaluator: `scripts/eval_policy.py`.
- ACT: `policy/act/` with separate Python environment and pinned uv lockfile.
- Task scenes and randomization: `configs/<task>/random/`.
- Published baseline results: `evaluation_results/README.md` and JSON.
- Official data download instructions: `docs/tutorials/download_data.md`.
- Simulator dependencies require the Linux/NVIDIA environment; none installed here.
- Fixed the version-file path casing in pyproject.toml to match the tracked
  `robosynchallenge/VERSION`; the original casing works on this Mac but differs
  on case-sensitive Linux.

## First GPU session (maximum initial spend $5)
1. Select a Linux NVIDIA instance with at least 24 GB VRAM and confirm total
   hourly price, storage, and Vulkan/headless rendering support before renting.
2. Follow `docs/getting_started/installation.md`; use EmbodiChain v0.2.4 as
   documented, alongside this repository. Inspect container launch options for
   the provider; do not assume nested Docker is available inside a GPU pod.
3. Set up ACT using `uv sync --frozen` in `policy/act` after simulator setup.
4. Download only the pinned click-bell checkpoint initially, then evaluate:

```bash
# Run on the configured GPU host, from the repository root.
hf download RoboSynChallenge/ACT_sim_click_bell \
  --revision 677e65fbb15974024ff840893496197ef7db26d4 \
  --local-dir checkpoints/ACT_sim_click_bell
bash policy/act/eval.sh click_bell random checkpoints/ACT_sim_click_bell 0 \
  --max_episodes 2 --headless True --pytorch_device cuda
```

Two episodes are an installation check, not a reliable success-rate estimate.
Record time, GPU memory, errors, and costs before deciding the training budget.
Cloud setup and these commands have NOT been run or validated on a GPU.

## Budget and experiments
Target total $90, below the user's $100 limit: setup $5, training/baselines $55,
final evaluation $15, storage/transfers/contingency $15. These are planning
allocations, not enforced billing caps. Save checkpoints before stopping or
terminating instances; retained storage may still incur charges.
Use released ACT checkpoints for initial reproduction. Published results cover
five tasks, not all ten; verify checkpoint availability and competition rules
before deciding the final submission coverage. ACT timings in the published
results are estimates; drawer results used modified physics.

## Before submission
Provide evaluation code, task-to-checkpoint mapping, dependency/run instructions,
training-data provenance, and the official policy adapter files. Validate on
fresh seeds and configurations. Keep test parameters separate from tuning.
Create a Cobotix remote when ready, then add it as `origin`; pushes are configured
to target origin rather than upstream. Nothing has been pushed.
