# Click-bell expert-feasibility audit

This audit ran the official evaluator's expert plan-generation check on the exact 100 seed-1 scenes used by the unfiltered 5K hard-edge checkpoint evaluation. It tests whether `create_demo_action_list(action_sentence=0)` returns a non-empty plan; it does not execute the expert plan.

## Main finding

- The expert planner accepted **63/100** scenes and rejected **37/100**. This 63% acceptance rate is close to the earlier filtered baseline's 50/78 (64.1%).
- ACT solved **50/63 expert-accepted scenes (79.4%)**.
- Of ACT's 45 failures, **32 (71.1%)** were expert-rejected and **13 (28.9%)** were expert-accepted.
- ACT solved **5/37 expert-rejected scenes**, so rejection means the scripted expert could not generate a plan, not that the task is physically impossible.
- Among the 17 expert-accepted hard-region scenes, ACT solved **15/17 (88.2%)**. The apparent hard-region weakness in the unfiltered 55/100 score is therefore dominated by the expert planner's acceptance boundary.
- All **13** scenes with x >= 0.80 m and all **8** scenes beyond both hard thresholds were expert-rejected; ACT also scored 0 in both groups.

## Cross-tab

| Expert check | ACT success | ACT failure | Total | ACT rate |
|---|---:|---:|---:|---:|
| Accepted | 50 | 13 | 63 | 79.4% |
| Rejected | 5 | 32 | 37 | 13.5% |
| All scenes | 55 | 45 | 100 | 55.0% |

## Region breakdown

| Region | Scenes | Expert accepted | ACT wins (all) | ACT wins / accepted | Rejected-scene ACT wins |
|---|---:|---:|---:|---:|---:|
| Central: x < 0.70, y < 0.15 | 48 | 46 | 36 | 35/46 (76.1%) | 1 |
| x-hard only | 30 | 10 | 13 | 9/10 (90.0%) | 4 |
| y-hard only | 14 | 7 | 6 | 6/7 (85.7%) | 0 |
| Both hard thresholds | 8 | 0 | 0 | 0/0 | 0 |
| Any hard threshold | 52 | 17 | 19 | 15/17 (88.2%) | 4 |
| x >= 0.80 | 13 | 0 | 0 | 0/0 | 0 |

## Interpretation

The unfiltered 100-scene run measured two things at once: policy quality and whether the official scripted expert can produce a demonstration for the sampled pose. Conditioning on the official filter raises the comparable policy score from 55.0% to 79.4%. The remaining policy-specific error set is the 13 accepted scenes ACT failed. Those should drive the next model or control experiment.

The 5 successes on rejected scenes also show that the expert filter is conservative. We should retain an unfiltered challenge-stress score alongside the accepted-scene score rather than treating rejected scenes as impossible.

## Reproduction

```bash
python scripts/audit_expert_feasibility.py \
  --metrics /path/to/evaluation_metrics.json \
  --output /path/to/expert_feasibility_seed1.json
```

Raw results are in `artifacts/benchmarks/click_bell_expert_feasibility_seed1/expert_feasibility_seed1.json`; the joined seed-level table is `episodes.csv`.
