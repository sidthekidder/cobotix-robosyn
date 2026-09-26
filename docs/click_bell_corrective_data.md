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

Run collection from the simulator/EmbodiChain Python environment with LeRobot
0.4.4 or newer. The `policy/act` lockfile pins an older LeRobot revision for
policy training and evaluation; that revision does not implement the recorder
API used by the current EmbodiChain collector. Keep collection and ACT training
environments separate. If a disposable smoke pod already has only the ACT
environment, the tested repair is:

```bash
cd /workspace/RoboSynChallenge/policy/act
uv pip install --python .venv/bin/python --upgrade 'lerobot==0.4.4'
export PATH="$PWD/.venv/bin:$PATH"
cd ../..
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

## Verified smoke run

The 2026-09-25 A40 smoke run used source commit `fb9baa0`, LeRobot 0.4.4,
seed `20260925`, and this command:

```bash
bash launch/collect_click_bell_corrections.sh \
  --episodes 3 --seed 20260925 --max-attempts 20
```

It saved all 3 requested episodes after 7 scene attempts. The four discarded
attempts failed during the challenge's canonical scene planner before any
correction was recorded. Each saved episode contains 36 frames at 25 FPS and
one-axis recovery of 8.8 mm in X, 8.0 mm in Y, or 4.8 mm in Z. The complete
smoke artifact is under `artifacts/corrective_smoke/20260925/`.
