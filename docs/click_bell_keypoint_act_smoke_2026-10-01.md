# ClickBell physical press-point ACT integration smoke

Date: 2026-10-01

## Decision

The physical press-point auxiliary loss is ready for a larger data collection
and fine-tuning experiment. The complete ACT path ran for 10 optimizer updates
on an A40 with the clean eight-episode ClickBell pilot and the hard-edge 5K
checkpoint. All reported losses and gradient norms were finite.

This is an integration result, not evidence that challenge success rate has
improved. The pilot has only eight demonstrations and the smoke intentionally
saved no checkpoint.

## Inputs

- Source checkpoint: `artifacts/training/click_bell_hard_edges_x3/checkpoints/005000/pretrained_model`
- Dataset: eight expert-success episodes, 592 frames, eight reachable geometry
  cells, LeRobot v3.0 converted locally to v2.1 for ACT
- Sidecar: schema-v2 `bell_keypoints.jsonl`
- Target: physical `press-point`
- Auxiliary loss weight: `0.2`
- Batch size: `2`
- Updates: `10`
- Source commit: `c2512f11695550c63912b3f66212a3f707ffe379`

The input transfer archive had SHA256
`02c225f03d63c21c783a5a430dc31d3bb57b4ea8a51827b47b15229bb9e3b0b5`.

## Runtime

- Secure A40, 46,068 MiB reported VRAM, `$0.49/hour`
- Image: `dpaleyev/robosyn-groot:v1`
- Image digest: `sha256:23d1c2c7adcea07848663ee0ea3b0c5be6ed686414a03641aacc772e30d6168e`
- Python 3.11.15 in the `robosyn` environment
- PyTorch 2.10.0+cu128
- LeRobot 0.3.3 at commit
  `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`
- Hugging Face Datasets 3.6.0
- Full precision; AMP disabled for this compatibility stack

The uncached image pull and extraction took about 27 minutes. Once the runtime
was ready, the final 10-update smoke took 17 seconds wall time, including model
and dataset setup.

## Result

| Metric | Result |
| --- | ---: |
| Completed updates | 10 / 10 |
| Finite total-loss batches | 10 / 10 |
| Finite gradient-norm batches | 10 / 10 |
| Batches with nonzero coordinate loss | 8 / 10 |
| Total loss, mean (min-max) | 0.0778 (0.026-0.139) |
| Gradient norm, mean (min-max) | 10.1864 (2.945-17.147) |
| Coordinate loss, mean (min-max) | 0.01546 (0-0.05445) |
| Visibility loss, mean (min-max) | 0.64098 (0.45843-0.78102) |
| Weighted auxiliary loss, mean (min-max) | 0.01591 (0.00917-0.02651) |

The two zero coordinate-loss batches contained no visible press point in their
sampled camera labels. They still produced visibility supervision and a
nonzero weighted auxiliary loss. This matches the loss definition: coordinate
error is masked by visibility, while visibility is trained for every camera.

The final command was:

```bash
PYTHONPATH=/workspace/RoboSynChallenge conda run -n robosyn \
  python policy/act/scripts/train.py \
  --dataset-root /workspace/act_geometry_smoke_inputs/datasets/clean_dataset \
  --repo-id clean_dataset \
  --output-dir /workspace/act_geometry_smoke_output \
  --pretrained-policy /workspace/act_geometry_smoke_inputs/pretrained_model \
  --bell-keypoints-jsonl /workspace/act_geometry_smoke_inputs/datasets/clean_dataset/bell_keypoints.jsonl \
  --bell-keypoint-target press-point \
  --bell-keypoint-loss-weight 0.2 \
  --chunk-size 50 --n-action-steps 50 \
  --steps 10 --log-freq 1 --save-freq 1000 \
  --batch-size 2 --num-workers 0 \
  --no-save-checkpoint --overwrite
```

## Compatibility findings

The prebuilt image's LeRobot 0.4.4 cannot run this trainer because it removed
`lerobot.scripts.train`. Installing the repository-pinned LeRobot 0.3.3 commit
with `--no-deps` restored the expected API while preserving the image's working
CUDA stack.

LeRobot 0.3.3 requires `datasets<=3.6.0`. With the image's Datasets 4.8.5,
dataset creation failed because a `Column` reached `torch.stack`; pinning
Datasets 3.6.0 fixed it.

AMP under the mixed old-LeRobot/new-PyTorch stack completed an update but
reported a NaN gradient norm. Full precision produced finite gradient norms and
all expected auxiliary metrics. Future production training should either use
the fully pinned ACT stack (PyTorch below 2.8) or keep AMP off after a finite
gradient preflight.

## Artifacts and cleanup

The local evidence bundle is under
`artifacts/keypoint_pilot/20261001/act_integration_smoke/`. The transferred
archive SHA256 is
`65c62dab07b77b8b77514d00dd38d29a978e588103e221a6d1cfcaa1c982b5db`.
It contains the final log, three failed/diagnostic probe logs, the successful
full-precision probe, and a machine-readable per-step summary.

Pod `itdj2r152x5txe` was terminated after the evidence bundle was downloaded.
The final account audit found zero pods, zero network volumes, and zero
serverless endpoints.

## Next experiment

Collect roughly 10 successful demonstrations per reachable geometry cell
(about 80 episodes), retain the eight-cell feasibility mask, and fine-tune from
the hard-edge 5K checkpoint with the same press-point target. Mix the geometry
set with the broader original demonstrations so the auxiliary signal improves
localization without collapsing trajectory diversity. Evaluate against the
same fixed seed list before deciding whether to keep the auxiliary loss.
