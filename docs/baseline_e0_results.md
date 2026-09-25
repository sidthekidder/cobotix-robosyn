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

## Next experiment

Run the same seeds with `ACT_N_ACTION_STEPS` set to 5, 10, 25, and 50. Compare
success, inference calls, press depth, and right-arm path length. Then run an
unfiltered 100-episode baseline to reproduce the organizers' evaluation protocol.
