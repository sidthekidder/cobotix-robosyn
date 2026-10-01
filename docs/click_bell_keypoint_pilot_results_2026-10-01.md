# ClickBell balanced keypoint pilot

This pilot tests whether physically grounded button and tool-tip labels can be
collected across the reachable ClickBell workspace before spending on a larger
ACT training run.

## Collection

- Source commit: `0068cec`
- Simulator: DexSim 0.4.3 through `dpaleyev/robosyn-groot:v1`
- Hardware: one Secure RunPod A40
- Seed: `20261001`
- Grid: 3x3 over x `[0.4, 0.85]` m and y `[-0.3, 0.3]` m
- Excluded cell: `(2,2)` after a separate 500-attempt feasibility run
- Result: eight episodes, one per reachable grid cell
- Episode length: 74 frames
- Total labeled frames: 592

The clean collection completed in 64 attempts. The audit passed with no errors,
and each included cell has exactly one saved episode. LeRobot 0.4.4 packed the
eight logical episodes into one physical MP4 per camera; the audit validates
both the episode metadata and referenced packed files.

## Feasibility finding

The original nine-cell run sampled cell `(2,2)` 53 times. All 53 attempts
failed expert planning across x `[0.70451927, 0.84650952]` m and y
`[0.10816976, 0.28698635]` m. The clean pilot therefore excludes this cell
explicitly instead of silently under-filling its quota.

## Camera-label quality

| Camera | Mask visible | Confident centroid | Press point in frame | Press point visible in mask | Tool tip in frame | Median centroid-to-press error | Median tool-to-press distance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `cam_high` | 586/592 | 426/592 | 592/592 | 426/592 | 366/592 | 0.024 normalized | 0.035 normalized |
| `cam_right_wrist` | 535/592 | 388/592 | 510/592 | 419/592 | 592/592 | 0.037 normalized | 0.229 normalized |

The contact sheets show that `cam_high` provides stable global target coverage
through approach and contact. `cam_right_wrist` frequently begins with the bell
outside its view, then provides a close local view as the tool approaches. This
supports global target supervision from the high camera and contact-stage
supervision from the right wrist camera. Camera losses should use the recorded
visibility/confidence masks rather than penalizing off-screen targets.

## Artifact

The transferred archive contains the clean LeRobot dataset, all 24 overlay
videos, all 24 contact sheets, the clean audit, the collection log, and the
failed-cell evidence. Its SHA-256 is
`301f2cc13d6ea2b4ca4d1ffb8653613fa37aa61c2d336696761f22f6490c4650`.

## Next experiment

Use this pilot to smoke-test the ACT physical press-point auxiliary-loss data
path. It is too small to support a standalone fine-tune. If the loss and camera
masks behave correctly, collect roughly ten demonstrations per reachable cell
and mix them with the original ClickBell demonstrations so the policy retains
its action prior while learning target geometry.
