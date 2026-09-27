# Click-bell policy-conditioned corrections

This collector creates DAgger-style recovery examples from states the current
ACT policy actually reaches. It runs the policy to completion, discards policy
successes, deterministically replays a failed scene, restores the closest
eligible near-contact robot state, and records only the expert recovery tail.
The fixed 20-scene development seeds are excluded from collection.

The first controlled collection was stopped at 20 validated episodes: two
high-x failures, two high-y failures, and 16 other failures. Twenty examples
are sufficient for the first 5%-sampling, 1,000-update gate, and both scarce
hard-region quotas were already complete. Hard-region quotas are deliberately
small because the scripted expert rejects most hard scenes and ACT solves most
of the hard scenes it accepts. A manifest records every attempted seed, the
failure diagnostics, the selected policy snapshot, and whether recovery data
was saved.

Preview the deterministic seeds and quotas without simulator dependencies:

```bash
python scripts/collect_click_bell_policy_corrections.py \
  --dry-run --episodes 6 --max-attempts 20 \
  --high-x-quota 2 --high-y-quota 2
```

Collect data on the simulator host:

```bash
bash launch/collect_click_bell_policy_corrections.sh \
  --checkpoint /workspace/checkpoints/click_bell_hard_edges_x3_005000 \
  --episodes 40 --seed 20260927
```

Run collection in the checkpoint's native LeRobot 0.3.3 ACT environment. The
collector provides the narrow compatibility shim needed by the current
EmbodiChain recorder; upgrading LeRobot can discard the older checkpoint's
stored normalization buffers. LeRobot 0.3.3 writes this dataset in v2.1
directly, so it can be merged with the released click-bell training set without
a conversion pass.

For the first controlled fine-tune, retain the hard-edge base sampling plan,
allocate 5% of sampled frames to policy-conditioned corrections, and train only
1,000 updates from the current 5K checkpoint. Use `--learning-rate 3e-6` and
`--backbone-learning-rate 3e-6`; this reduces catastrophic forgetting relative
to the rejected 20%-correction, `1e-5` experiment. Promote the result only if it
beats 13/20 on the unchanged development seeds without excessive regressions.
