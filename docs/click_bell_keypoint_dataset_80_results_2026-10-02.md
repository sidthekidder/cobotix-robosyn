# ClickBell 80-episode balanced geometry dataset

This run expands the eight-episode keypoint pilot into a training-sized set for
the ACT physical press-point auxiliary loss. It records successful expert
trajectories only; simulator geometry appears in the training sidecar and is
not a policy input at evaluation time.

## Collection

- Challenge commit: `10389f5b9ec1a98ef915972dbedeee6fa485637c`
- EmbodiChain commit: `9ebee30011f378f94a7cbe78b01d8c2eacba231a`
- Image: `dpaleyev/robosyn-groot:v1`
- Hardware: one Secure RunPod A40 at `$0.49/hour`
- Seed: `20261002`
- Grid: 3x3 over x `[0.4, 0.85]` m and y `[-0.3, 0.3]` m
- Excluded cell: `(2,2)`, based on the earlier feasibility run
- Result: 80 episodes and 5,920 labeled frames
- Balance: exactly 10 successful episodes in each of the eight reachable cells
- Collection loop time: 37 minutes 47 seconds

The collector needed 402 randomized scenes: 80 were saved, 78 failed expert
planning, 39 landed in the excluded cell, and 205 landed in cells whose quota
was already full. The final quota phase is therefore slower by design.

## Audit

`scripts/audit_click_bell_keypoints.py` reported `valid: true` with no errors.
LeRobot 0.4.4 packed the 80 logical episodes into one physical MP4 per camera;
all 240 logical camera episode records resolve to those files.

| Camera | Mask visible | Confident centroid | Press point in frame | Press point visible in mask | Tool tip in frame | Median centroid-to-press error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `cam_high` | 5,901/5,920 | 4,385/5,920 | 5,920/5,920 | 4,426/5,920 | 3,485/5,920 | 0.0233 normalized |
| `cam_right_wrist` | 5,315/5,920 | 3,781/5,920 | 5,067/5,920 | 4,129/5,920 | 5,920/5,920 | 0.0382 normalized |

The high camera provides complete physical press-point coverage. The wrist
camera retains full tool-tip coverage and useful close-range target coverage,
while its confidence mask correctly suppresses off-screen and boundary cases.

## Artifact

The verified local archive is:

`artifacts/keypoint_geometry_80/20261002/cobotix_click_bell_geometry_80_20261002.tar.zst`

- Size: 164,441,486 bytes
- SHA-256: `8b90301f30a49e4410e20ef599c1b15d8fd640131916e12973c3cb43370ade15`
- Contents: dataset, schema-v2 sidecar and manifest, full collection log,
  audit JSON, exit code, and environment metadata

The archive was checksum-verified after transfer, decompressed locally, and
rechecked for 80 episodes, 5,920 sidecar rows, balanced cell counts, three
packed camera videos, and the expected packed parquet data file. The RunPod
audit after transfer found zero pods, zero network volumes, and zero endpoints.

## Next experiment

Mix these 80 demonstrations with the original ClickBell demonstrations and
fine-tune from the current ACT checkpoint with the validated auxiliary loss.
Keep a baseline-matched control run so any gain can be attributed to physical
press-point supervision rather than extra optimizer updates alone.
