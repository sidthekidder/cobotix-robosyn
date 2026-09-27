# Click-bell policy-conditioned correction results

The controlled correction pilot is rejected as a replacement for the current
5K hard-edge ACT checkpoint. On the unchanged 20-seed development set, the
candidate scored 12/20 (60%) versus the baseline's 13/20 (65%).

## Experiment

- Start from `click_bell_hard_edges_x3/checkpoints/005000/pretrained_model`.
- Collect 20 policy-conditioned expert recovery tails: two high-x, two high-y,
  and 16 other policy failures; 400 total frames.
- Merge with the 1,000-episode released dataset and preserve the hard-edge
  sampling plan.
- Allocate 5% expected sample probability to corrections.
- Fine-tune for 1,000 updates with batch size 32, ACT chunk/action horizon 50,
  AMP, and `3e-6` policy and backbone learning rates.
- Evaluate on the same fixed 20 seeds with all oracle and recovery options off.

## Result

| Outcome relative to baseline | Seeds |
| --- | --- |
| Stable successes | 10 |
| New successes | 2: `1284876248`, `1136257699` |
| Regressions | 3: `352272321`, `1879422756`, `1932520490` |
| Stable failures | 5 |

The lower correction share and learning rate avoided the larger regression seen
with the earlier 20%-correction run, but the recovery tails still traded away
three baseline successes to gain two new ones. Keep the 13/20 baseline as the
active checkpoint. Preserve this candidate for analysis because its two new
wins show that policy-state data can change the desired failure modes; the next
method should isolate contact correction without updating the full approach
behavior.

Local experiment artifacts, including the checkpoint, training/evaluation logs,
metrics, videos, correction dataset, and seed-level comparison, are under
`artifacts/experiments/click_bell_policy_corrections_5pct/` (gitignored).
