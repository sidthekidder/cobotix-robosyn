# ClickBell press-completion oracle diagnostic

## Question

Can the 13 expert-feasible failures from the held-out 100-episode evaluation be
completed if a controller takes over near the end of ACT's approach and drives
the arm to the scene-specific expert contact pose?

This is a diagnostic experiment. The controller uses simulator ground truth and
the task's expert IK plan, so it is not a submission strategy.

## Setup

- Checkpoint: `hard_edges_x3_005000`
- Episodes: the 13 expert-feasible failures from the held-out evaluation
- Trigger: step 35 or later, when ACT's maximum right-arm joint error relative
  to expert plan step 49 is at most 0.45 radians
- Completion: interpolate the right arm to the expert target over 12 commands,
  then hold for 8 commands
- Unmodified joints: held at ACT's current commanded values
- Hardware: one RunPod A40

## Result

All 13 episodes succeeded (13/13, 100%). Success occurred between simulator
steps 45 and 53, with a mean of 49.15 steps. The button reached the full 5 mm
press depth in every episode.

The oracle triggered between steps 35 and 42 (median 39). At takeover, ACT was
still some distance from the exact expert contact pose:

| Measurement at trigger | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Max right-arm joint error | 0.225 rad | 0.420 rad | 0.447 rad |
| RMS right-arm joint error | 0.129 rad | 0.213 rad | 0.263 rad |
| End-effector lateral error | 26.1 mm | 87.7 mm | 134.2 mm |
| End-effector vertical error | 67.8 mm | 100.8 mm | 126.7 mm |
| End-effector orientation error | 10.82 deg | 16.56 deg | 21.73 deg |
| Button depth before takeover | 0.935 mm | 0.935 mm | 0.942 mm |

ACT performed exactly one model inference per episode before the oracle took
over. Its queued action chunk therefore ran open loop throughout the approach.
This directly supports the hypothesis that stale queued actions are a major
failure mode.

## Interpretation

The 13/13 result shows that the chosen final target and approach direction can
complete every expert-feasible failure. It does not show that a small downward
nudge is sufficient. The oracle replaced roughly the last 11–18 approach steps
while the end effector was still 2.6–13.4 cm laterally and 6.8–12.7 cm
vertically from the exact expert target.

The follow-up deployable ablation cleared ACT's queued actions on a fixed
schedule without expert information or simulator state. It did not improve the
development result: the unchanged baseline scored 13/20, replanning every five
steps from steps 30–80 scored 8/20, and one replan at step 35 scored 11/20. See
`click_bell_scheduled_replan_results.md` for the paired analysis.

The most promising next change is therefore to oversample and weight the last
20–30 demonstration frames and add small corrective motions near the button.
Camera augmentation should be evaluated separately because one of the 13
failures appeared to be a camera/perception exception rather than the dominant
open-loop contact failure.

## Reproducibility

The implementation is controlled by the `act_press_oracle_*` options in
`policy/act/deploy_policy.yml`. The benchmark runner records those values in its
manifest. Per-episode metrics and videos were downloaded to the local ignored
artifact directory:

`artifacts/benchmarks/click_bell_press_oracle_feasible13`
