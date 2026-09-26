# ClickBell corrective demonstrations

The collector creates recovery-only demonstrations from small, controlled
near-contact errors. It plans the normal 74-step expert trajectory, chooses an
anchor from steps 34–44, offsets the right end effector by 3–12 mm, and moves
the simulated robot directly to that state. Recording starts only afterward.
The artificial teleport is therefore setup, not a learned action.

Each saved episode interpolates from the perturbed state to the canonical
pre-press target at step 49 and then retains the expert press suffix. Failed IK
samples and unsuccessful presses are discarded. `correction_manifest.json`
records every attempted offset, geometry, outcome, and episode length.

Preview the deterministic plan on any machine:

```bash
python scripts/collect_click_bell_corrections.py --dry-run --episodes 8 --seed 20260925
```

Collect 100 successful corrections on the simulator pod:

```bash
bash launch/collect_click_bell_corrections.sh --episodes 100 --seed 20260925
```

The default output parent is `lerobot_dataset/click_bell_corrections`. Start by
mixing corrections at 20% of sampled training episodes and the released expert
data at 80%. Keep a held-out correction split and the unchanged 20-scene
challenge evaluation; a gain on only the synthetic correction split is not
enough evidence to keep the change.
