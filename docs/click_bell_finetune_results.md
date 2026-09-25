# Click-bell hard-edge fine-tuning results

The released H50 ACT checkpoint was fine-tuned for 5,000 updates on the complete
1,000-episode simulation dataset. Episodes whose approach proxy fell in the
hard x/y edge set received 3x sampling weight. Training used batch size 32, AMP,
and the released checkpoint's `1e-5` learning rate.

## Fixed dev20 checkpoint screen

All candidates used the same 20 seeds as the released H50 baseline. The first
10 were baseline failures and the last 10 were baseline successes.

| Checkpoint | Success | Rescues | Regressions | Preserved baseline wins |
|---|---:|---:|---:|---:|
| Released H50 | 10/20 | - | - | 10/10 |
| 1K | 11/20 | 2 | 1 | 9/10 |
| 3K | 12/20 | 3 | 1 | 9/10 |
| 5K | **13/20** | **3** | **0** | **10/10** |

The 5K checkpoint passed the development gate. Its three favorable and zero
unfavorable paired flips give an exact two-sided McNemar p-value of 0.25, so the
small dev-set improvement is promising rather than conclusive.

## Held-out seed-1 evaluation

The 5K checkpoint scored **55/100 (55%)** on 100 deterministic seed-1 scenes
with the expert filter disabled. These seeds have no overlap with the earlier
seed-0 benchmark. Its Wilson 95% interval is 45.2%-64.4%.

The released checkpoint scored 51/100 on the earlier seed-0 set. Because the
two 100-scene runs use different scenes, the four-point difference is not a
paired estimate. The held-out result supports a modest improvement while also
showing that substantial click-bell failures remain.

Evaluation used H50 execution with temporal ensemble and contact recovery
disabled. All metrics, logs, and 100 held-out videos were copied off RunPod and
verified before the evaluation pod was terminated.
