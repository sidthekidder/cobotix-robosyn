# Click-bell policy-conditioned corrections

This collector creates DAgger-style recovery examples from states the current
ACT policy actually reaches. It runs the policy to completion, discards policy
successes, deterministically replays a failed scene, restores the closest
eligible near-contact robot state, and records only the expert recovery tail.
The fixed 20-scene development seeds are excluded from collection.

The default 100-episode collection balances three scene slices: 25 high-x,
25 high-y, and 50 other scenes. A manifest records every attempted seed, the
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
  --episodes 100 --seed 20260926
```

This collection path needs LeRobot 0.4.4 or newer for incremental dataset
recording. The resulting dataset still needs conversion to v2.1 before it can
be merged with the released click-bell training set by the existing merge
script.
