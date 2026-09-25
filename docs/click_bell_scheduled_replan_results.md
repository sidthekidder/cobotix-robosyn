# ClickBell scheduled-replanning ablation

## Question

Can ACT recover from visual or contact error by discarding its queued actions
and predicting a fresh chunk from the current camera image and robot state?

Unlike the press-completion oracle, scheduled replanning uses no expert plan,
button pose, contact signal, or other privileged simulator state. It is valid at
submission time.

## Setup

- Checkpoint: `hard_edges_x3_005000`
- Split: the same 20 fixed development seeds for every condition
- Baseline: unchanged ACT, with a 50-action prediction queue
- Repeated schedule: clear the remaining queue every 5 environment steps from
  step 30 through step 80
- Single schedule: clear the remaining queue once, at step 35
- Hardware: one RunPod A40

The final paired runs disabled the expert feasibility precheck. These 20 seeds
had already passed that filter in the original baseline run, and a repeat of
the precheck rejected one seed nondeterministically before policy evaluation.
The scenes and episode seeds were otherwise unchanged.

## Result

| Policy | Success | Change from baseline | Model calls per episode | Mean action steps |
| --- | ---: | ---: | ---: | ---: |
| Unchanged ACT | 13/20 (65%) | — | 3.90 | 173.50 |
| Replan every 5 steps, 30–80 | 8/20 (40%) | -25 points | 13.40 | 249.70 |
| Replan once at step 35 | 11/20 (55%) | -10 points | 5.00 | 202.75 |

The repeated schedule rescued two baseline failures but broke seven baseline
successes. The single replan rescued one baseline failure and broke three
baseline successes. With only four discordant episodes in the single-replan
comparison, this development result is directional rather than statistically
conclusive, but it gives no reason to replace the 65% baseline.

| # | Seed | Baseline | Repeated | Single | Maximum press depth B/R/S (mm) |
|---:|---:|:---:|:---:|:---:|---:|
| 0 | 1491434855 | ✓ | ✓ | ✓ | 5.000 / 5.000 / 5.000 |
| 1 | 292249176 | ✗ | ✗ | ✗ | 0.958 / 0.935 / 0.935 |
| 2 | 374217481 | ✗ | ✓ | ✓ | 0.941 / 5.000 / 5.000 |
| 3 | 1284876248 | ✗ | ✗ | ✗ | 0.959 / 0.935 / 2.209 |
| 4 | 352272321 | ✓ | ✗ | ✓ | 4.869 / 0.942 / 4.851 |
| 5 | 1136257699 | ✗ | ✗ | ✗ | 3.636 / 3.366 / 2.067 |
| 6 | 580757632 | ✗ | ✓ | ✗ | 4.624 / 5.000 / 0.956 |
| 7 | 1544074682 | ✗ | ✗ | ✗ | 0.960 / 0.959 / 0.959 |
| 8 | 716257571 | ✗ | ✗ | ✗ | 0.958 / 2.571 / 0.941 |
| 9 | 1396067212 | ✓ | ✗ | ✓ | 5.000 / 0.941 / 4.927 |
| 10 | 398764591 | ✓ | ✓ | ✗ | 5.000 / 5.000 / 4.767 |
| 11 | 441365315 | ✓ | ✗ | ✓ | 5.000 / 0.935 / 5.000 |
| 12 | 1537364731 | ✓ | ✗ | ✓ | 5.000 / 0.942 / 5.000 |
| 13 | 1819583497 | ✓ | ✗ | ✗ | 5.000 / 0.942 / 0.935 |
| 14 | 530702035 | ✓ | ✓ | ✓ | 5.000 / 5.000 / 5.000 |
| 15 | 1879422756 | ✓ | ✗ | ✗ | 5.000 / 0.935 / 0.935 |
| 16 | 1682652230 | ✓ | ✓ | ✓ | 5.000 / 5.000 / 4.866 |
| 17 | 1171049868 | ✓ | ✗ | ✓ | 5.000 / 0.935 / 5.000 |
| 18 | 1982038771 | ✓ | ✓ | ✓ | 5.000 / 5.000 / 5.000 |
| 19 | 1932520490 | ✓ | ✓ | ✓ | 5.000 / 5.000 / 5.000 |

## Interpretation

ACT's queue is more than a cache: its 50 actions form one coherent motion.
Clearing it asks the model to begin a new demonstration-like trajectory from a
state partway through the old one. The training data may contain few such
mid-trajectory starting states. Repeating this every five steps stitches many
independently predicted trajectory prefixes together and produces the largest
regression.

One reset is less destructive and does rescue seed `374217481`, so fresh visual
feedback can help in a particular scene. It still harms more seeds than it
helps. The evidence therefore narrows the earlier oracle finding: stale open-loop
actions can cause failure, but blindly refreshing them is not a general fix.
The learned policy also needs experience recovering from the state at which the
refresh occurs, or a reliable trigger that avoids disturbing trajectories that
are already succeeding.

The next experiment should target training data rather than another fixed
schedule: oversample the final contact portion of demonstrations and add
corrective trajectories from small lateral, vertical, and camera-induced
errors. A later adaptive trigger can use policy-observable signals such as
prediction disagreement or image/action change, but it should first be tested
on this development split and must preserve the unchanged baseline path when
confidence is high.

## Reproducibility

The implementation is controlled by the `act_scheduled_replan_*` options in
`policy/act/deploy_policy.yml`. Per-episode metrics, telemetry, and videos are
stored in the local ignored artifact directory:

`artifacts/benchmarks/click_bell_scheduled_replan_dev20`
