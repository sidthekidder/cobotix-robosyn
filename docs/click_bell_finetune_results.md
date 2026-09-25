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

## Held-out failure analysis

The 45 failures are strongly concentrated at actual bell-pose extremes. The
checkpoint succeeded in 36/48 central scenes (75.0%) but only 19/52 scenes
(36.5%) beyond either hard threshold. It scored 0/13 for bell x positions at or
above 0.80 m and 0/8 when both the x and positive-y hard thresholds were crossed.

Diagnostic thresholds divide the failures into 20 episodes that never entered
the empirical success-distance envelope, 16 close approaches without measurable
button depression, four partial presses, and five near-threshold presses. Only
9/45 failures depressed the button by at least 1.1 mm. A contact recovery rule
may help those nine cases, but it cannot fix most failures by itself.

The 3x sampler used the dataset's end-effector approach proxy because the source
episodes do not record ground-truth bell pose. The next data revision should log
and explicitly target actual randomized object positions, especially x >= 0.80 m
and positive y. Reproduce the analysis with
`scripts/benchmark/analyze_click_bell_failures.py`.
