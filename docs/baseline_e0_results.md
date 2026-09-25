# E0 ACT click-bell baseline

## Protocol

- Policy: released `RoboSynChallenge/ACT_sim_click_bell` checkpoint at revision
  `677e65fbb15974024ff840893496197ef7db26d4`
- Task/setting: `click_bell/random`
- Source revision: `068faa9`
- GPU: RTX 4090; Vulkan renderer
- Candidate RNG seed: `0`
- Evaluated episodes: 50 expert-feasible scenes selected from 78 candidates

This is a policy-debug baseline, not an estimate of the challenge score. The expert
feasibility filter rejected 28 of 78 sampled scenes before policy evaluation, so the
result is conditioned on scenes for which the scripted expert produced a plan. The
organizers' released unfiltered 100-episode ACT result is 37%.

## Result

- Success: 35/50 (70.0%)
- 95% Wilson interval: 56.2%–80.9%
- Mean action steps: 141.62 of 361, counting timeouts as 361
- Mean model latency: 13.46 ms on the RTX 4090
- A prior run of the same accepted seeds scored 36/50, showing one episode of
  run-to-run variation.

## Failure structure

The 15 failures split by maximum bell travel:

- 8 barely displaced the bell: less than 1.5 mm
- 4 made partial contact: 1.5–4.0 mm
- 3 were near misses: 4.0–4.8 mm against the 4.8 mm success threshold

Every success reached at least 4.85 mm. Failed episodes ran a median 1.66 m of
right-arm end-effector path, compared with 0.51 m for successes. The released ACT
checkpoint predicts 50-action chunks and executes all 50 actions before replanning.
Long failed paths and several shallow presses make shorter executed horizons the
first intervention to test.

Position alone did not explain the failures in this sample. Success by broad bin was
58.3% at x >= 0.70 and 50.0% at y >= 0.15, but those bins contain only 12 and 4
episodes respectively. Treat those rates as hypotheses for stratified follow-up, not
stable effects.

## Unfiltered challenge-style baseline

After fixing boolean CLI parsing so `--eval_expert_check false` actually disables
the filter, the same released checkpoint was evaluated on the first 100 sampled
scenes from candidate RNG seed 0. Every sampled scene counted.

- Success: 51/100 (51.0%)
- 95% Wilson interval: 41.3%–60.6%
- Mean action steps: 203.48 of 361
- Mean inference latency: 43.32 ms on the RTX 4090
- Expert filter: disabled; 100 candidates sampled and 100 evaluated

Of the 49 failures, 40 displaced the bell by less than 1.5 mm, 4 reached
1.5–4.0 mm, and 5 stopped at 4.0–4.8 mm. Successful trajectories had a median
right-arm path of 0.53 m, compared with 1.70 m for failures.

The larger unfiltered sample reveals a strong workspace-edge effect. Success was
12/39 (30.8%) for bell x positions at or above 0.70 m and 5/26 (19.2%) for y
positions at or above 0.15 m. Central bins reached 60–79%. These are broad
one-dimensional bins rather than a causal model, but they justify stratified
coverage analysis and targeted data balancing.

The 51% estimate is above the organizers' released 37% ACT result. The released
number may use different seeds, environment state, or evaluation revisions, so
the gap should be treated as a reproducibility question rather than an improvement
claim.

## Shorter executed-horizon experiment

The same 100 unfiltered seeds were evaluated with `ACT_N_ACTION_STEPS=10`, so
the policy replanned after at most 10 actions instead of executing the released
50-action chunk. H10 scored 40/100 (40.0%; 95% Wilson interval 30.9%–49.8%),
compared with 51/100 for H50.

The paired outcomes show why the aggregate score fell: H10 rescued 10 H50
failures but broke 21 H50 successes. Thirty scenes succeeded under both horizons
and 39 failed under both. The two-sided exact McNemar p-value is 0.071, so this
100-scene experiment is evidence against adopting H10, but does not establish a
population-level difference at the conventional 5% threshold.

H10 also made failures longer rather than recovering from them. Median failed
right-arm path increased from 1.70 m to 2.79 m, mean action steps increased from
203.48 to 242.20, and inference calls increased from 455 to 2,496. Of its 60
failures, 51 displaced the bell by less than 1.5 mm, 6 reached 1.5–4.0 mm, and 3
stopped at 4.0–4.8 mm. Success in the broad hard-position bins was unchanged:
12/39 for x >= 0.70 m and 5/26 for y >= 0.15 m.

## Next experiment

Keep H50 as the reference horizon. Inspect matched outcome flips to identify the
visual or trajectory signatures that distinguish the 10 H10 rescues from the 21
regressions. Then test a targeted recovery rule on a fixed development subset,
such as replanning only after stalled end-effector progress or a failed contact
approach. This preserves the checkpoint's useful long-chunk behavior while
testing feedback where the baseline diagnostics show it is needed. Separately,
audit demonstration coverage in the hard x/y regions before retraining.
