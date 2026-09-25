# Cobotix experiment strategy

This document turns public evidence into testable RoboSyn experiments. Public
results guide the search, but no single competitor result is treated as ground
truth for our policy, simulator version, or final real-robot evaluation.

## Closest prior competition evidence

The closest structural comparison is the 2025 RoboTwin Dual-Arm Collaboration
Challenge: it used the AgileX COBOT-Magic platform, included simulation and
real-world rounds, and evaluated 64 teams. Its technical report highlights two
simulation winners:

- **AnchorDP3** predicted 10–30 sparse, affordance-aligned keyposes instead of
  dense 20–25 Hz actions. It combined task-specific encoders with a shared
  diffusion expert, trained with simulator-generated object masks, and jointly
  supervised joint angles and end-effector poses. The report attributes a 98.7%
  simulation success rate and 14.5× more environmental diversity per compute
  budget to this formulation.
- **SEM** lifted multi-view image features into 3D, encoded the full dual-arm
  joint graph, and used a diffusion action decoder. Its central idea was to make
  spatial and embodiment structure explicit rather than asking a 2D encoder to
  infer both implicitly.

Across teams, the same report identifies several repeated gains: thousands of
episodes instead of hundreds, large/noisy-data pretraining followed by
small/high-quality-data fine-tuning, depth-aware features, lowering control
frequency from 25 Hz to 12 Hz, instruction paraphrases, and filtering jitter,
repeated attempts, irrelevant motion, and inconsistent initial frames.

Sources:

- [RoboTwin challenge report](https://arxiv.org/html/2506.23351v2)
- [RoboTwin 2.0 repository](https://github.com/RoboTwin-Platform/RoboTwin)

The ManiSkill2 2023 winner also used two-stage fine-tuning and won all three
tracks, supporting staged specialization rather than a single undifferentiated
training run. Separately, 3D Diffuser Actor reports large gains on RLBench and
CALVIN from combining action diffusion, 3D scene tokens, relative 3D attention,
and keypose prediction.

Sources:

- [ManiSkill2 winning solution](https://arxiv.org/abs/2307.11343)
- [3D Diffuser Actor](https://arxiv.org/abs/2402.10885)

## Our current evidence

The first 498 official `click_bell` parquet episodes downloaded before the
anonymous Hugging Face rate limit engaged. Direct measurement found right-arm
motion in 498/498 and left-arm motion in 0/498. The dataset does not store the
sampled bell pose. The lowest active end-effector position is therefore only a
spatial proxy; its observed y range was -0.299 m to +0.241 m, with a 95th
percentile of +0.170 m.

This partial audit independently supports investigating arm coverage, but it
does not establish the exact distribution of all 1,000 demonstrations or the
hidden real evaluation. The audit script now refuses to label a partial download
complete when `--expected-episodes 1000` is supplied.

## Experiment ladder

Each experiment changes one major variable and reuses the same 50 episode seeds.
Promote a change only when its Wilson interval and failure distribution improve,
not merely its headline success count.

### E0 — instrumented ACT baseline

Run the released ACT checkpoint for 50 randomized episodes. Record bell x/y,
minimum distance from each end effector to the bell, per-arm path length, maximum
joint displacement, press depth, latency, and video.

This establishes whether failures are dominated by reach, perception, approach
height, insufficient press depth, or control oscillation.

### E1 — ACT executed-horizon sweep

Evaluate `n_action_steps` values 5, 10, 25, and 50 without retraining. The released
checkpoint predicts 50 actions and executes all 50 before observing again. Shorter
executed horizons discard the remainder of that action queue and replan from fresh
images sooner, trading inference cost for recovery from visual and contact error.

Do not sweep `act_step` for this purpose. In this adapter, `act_step` only controls
how many environment steps are grouped inside one evaluator call; it does not change
the policy's 50-action queue or replanning cadence.

### E2 — balanced expert data

Repair or replace the left-arm expert route. Generate a stratified grid of bell
positions and retain explicit metadata for bell pose, chosen arm, expert
planning result, and task success. Balance left-only, right-only, and overlap
regions. Reject failed or marginal demonstrations with a recorded reason instead
of silently resampling them.

Train ACT on: official data, balanced data, and a 50/50 mixture. Use identical
seeds to separate model gains from scene sampling noise.

### E3 — temporal and keypose targets

Try 12.5 Hz temporal subsampling, phase-balanced sampling around approach and
contact, and a keypose representation containing pre-press, contact, full press,
and retract poses. The task is short enough that this can be tested before
building a large general keypose architecture.

### E4 — privileged simulation supervision

Use simulator-only bell masks, bell pose, and end-effector-to-bell vectors as
auxiliary training targets. At evaluation the policy still consumes the allowed
RGB cameras and proprioception. This tests whether privileged labels can teach
spatial features without introducing an unavailable test-time input.

For a stronger version, distill a depth/segmentation teacher into the RGB ACT
encoder or triangulate a bell keypoint from calibrated multi-view RGB. This
adapts the strongest lesson from AnchorDP3 and SEM to the sensors RoboSyn
actually exposes.

### E5 — staged sim-to-real adaptation

Pretrain on broad, aggressively randomized synthetic data, then fine-tune on the
limited high-quality real demonstrations with a lower learning rate and balanced
task sampling. Calibrate camera extrinsics, table height, action latency, button
travel, and contact parameters to measured hardware ranges before narrowing the
final randomization distribution.

### E6 — recovery behavior

Train or script a recovery phase when the end effector reaches the bell region
without achieving the required press depth: reobserve, correct laterally, and
press farther. Binary success makes the final few millimeters disproportionately
valuable, so recovery may outperform a larger backbone at much lower cost.

## Decision order under the current budget

1. E0: one 50-episode baseline.
2. E1: action-chunk sweep, stopping poor settings early after 15–20 paired seeds.
3. Complete the 1,000-episode audit and implement E2 data collection.
4. Train one balanced-data ACT candidate.
5. Test E3 before committing GPU time to a larger VLA.
6. Pursue E4 and a π0.5 comparison only after the dataset/control ablations show
   where ACT is capped.

This order seeks our own advantage in control frequency, data design, privileged
supervision, and recovery rather than assuming that a larger public model is the
answer.
