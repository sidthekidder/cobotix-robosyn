# Click-bell targeted fine-tuning plan

## Dataset audit

The complete released dataset contains 1,000 expert demonstrations. Every
trajectory uses the right arm. The parquet schema does not include the sampled
bell pose, so coverage is measured with the lowest right-end-effector position
as an approach proxy.

| Region | Demonstrations | Share |
| --- | ---: | ---: |
| Proxy x >= 0.70 m | 168 | 16.8% |
| Proxy y >= 0.15 m | 75 | 7.5% |
| Either hard edge | 243 | 24.3% |
| Remaining region | 757 | 75.7% |

The two hard-edge groups do not overlap in this dataset. With 3x episode weight,
the expected hard-edge share becomes 49.1%. This is moderate enough to preserve
central coverage while directly testing the workspace-edge hypothesis.

## First training gate

Initialize from the released `ACT_sim_click_bell` checkpoint and train for 5,000
updates with checkpoints every 1,000 updates. Keep the checkpoint architecture
and 50-action chunk size unchanged. Use a single GPU because the weighted sampler
currently targets one training process.

Evaluate the 1k, 2k, 3k, 4k, and 5k checkpoints on the fixed 20-scene development
set. A candidate advances only if it:

- improves the 10 targeted baseline failures;
- preserves at least 9 of the 10 baseline successes; and
- shows an improvement caused by the trained checkpoint rather than simulator
  variation across repeated reference runs.

Run one separate 100-seed held-out evaluation only after a checkpoint passes the
development gate. This limits GPU spend and avoids selecting directly on the
reported challenge-style baseline seeds.
