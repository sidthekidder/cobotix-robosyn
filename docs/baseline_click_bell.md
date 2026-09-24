# ACT `click_bell` baseline

## Reproduced smoke test

The Cobotix team reproduced the official pretrained ACT baseline on a Runpod
Secure Cloud NVIDIA A40. This was a three-episode installation and evaluation
smoke test, not a statistically reliable benchmark.

- Date: 2026-09-24
- Checkpoint: `EDEM-AI/ACT_sim_click_bell`
- Checkpoint revision: `677e65fbb15974024ff840893496197ef7db26d4`
- PyTorch: 2.7.1+cu126
- CUDA: 12.6
- GPU: NVIDIA A40

From `policy/act`:

```bash
bash eval.sh click_bell random checkpoints/ACT_sim_click_bell 0 \
  --max_episodes 3 \
  --pytorch_device cuda \
  --headless True
```

## Results

| Episode | Seed | Outcome | Action steps |
| --- | ---: | --- | ---: |
| 0 | 398764591 | Success | 46 |
| 1 | 924231285 | Failed (timeout) | 361 |
| 2 | 441365315 | Success | 48 |

- Success rate: 2/3 (66.7%)
- Mean action steps: 151.67/361 (42.01%)
- Mean inference latency: 0.207382 seconds over 10 calls
- Mean total inference time per episode: 0.691274 seconds
- Feasibility filter: 3 of 5 candidates accepted

The evaluator's official published ACT result for this task uses many more
episodes, so this three-episode result must only be used to confirm that the
stack and checkpoint execute successfully.

## Preliminary failure observation

In the failed episode, the end effector approached and appeared aligned with
the bell in the front camera, but it did not visibly depress the button. It
then drifted away and continued until the action-step timeout. The two
successful episodes reached and depressed the button within about five seconds.
This is one sample, so treat depth/contact execution under randomized scenes as
a hypothesis to test with a larger evaluation rather than a settled diagnosis.

Raw metrics, logs, and videos are retained locally under
`artifacts/click_bell_act_3ep/` and intentionally excluded from Git because they
include generated binary files.
