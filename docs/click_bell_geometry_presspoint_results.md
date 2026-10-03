# ClickBell geometry-supervised ACT results

## Decision

Keep the hard-edge 5K ACT checkpoint as the submission candidate. The
geometry-supervised checkpoint did not pass the fixed-development gate and was
stopped after nine episodes once an improvement over 13/20 became impossible.

## Experiment

- Collected 80 balanced ClickBell demonstrations across eight reachable bell
  position cells, ten episodes per cell. The far-positive cell `(2,2)` was
  excluded after an expert-feasibility sweep found it outside the planner's
  practical reach.
- Recorded 5,920 frame-level press-point labels alongside the demonstrations.
- Appended the 80 episodes to the released 1,000-episode dataset and assigned
  them 10% expected sampling probability. Unlabelled released frames used
  visibility `-1`, so the auxiliary keypoint loss ignored them.
- Fine-tuned from the hard-edge 5K checkpoint for 1,000 updates with batch size
  32, learning rate `1e-5`, full precision, and auxiliary press-point loss
  weight `0.2`.
- Training and a two-update preflight had finite losses and gradients. The
  standard ACT loader accepted the resulting checkpoint without the auxiliary
  head.

## Fixed-dev result

The run completed the first nine fixed dev20 seeds:

| Policy | Success on first 9 | Rescues | Regressions |
|---|---:|---:|---:|
| Hard-edge 5K baseline | 2/9 | — | — |
| Geometry press-point 1K | 2/9 | 1 | 1 |

The candidate rescued seed `1136257699`, succeeding in 48 actions, but
regressed seed `1491434855`. It preserved the other baseline success,
`352272321`, and failed the remaining six baseline-failure seeds.

At that point all seven baseline-failure seeds in dev20 had been evaluated.
Only eleven baseline-success seeds remained, so the candidate could score at
most 13/20 even if it preserved every remaining success. The evaluation was
stopped during episode 10 to save GPU cost rather than complete a run that
could no longer pass the promotion gate.

## Interpretation

Sparse geometric supervision changed the policy and rescued one hard scene,
but it did not improve the paired aggregate on the diagnostic portion that
contained every available rescue opportunity. This rules out promoting this
specific combination of 10% geometry sampling, loss weight `0.2`, and 1,000
updates. It does not establish that object-centric supervision is useless: the
labels affected only an auxiliary training head and the deployed ACT policy
still inferred joint-action chunks directly.

The strongest follow-up is to use the geometry labels in the deployed control
path rather than only as an auxiliary regularizer. A bounded test would predict
the press point from the current image, condition the action decoder on that
prediction, and gate it on the same first-nine seeds before spending on a full
dev20 run. A cheaper diagnostic first is to compare predicted press-point error
and end-effector trajectories in the rescued and regressed videos; this will
show whether the auxiliary head learned localization while the action decoder
failed to use it.

## Paired video diagnostic

A frame-aligned visual audit of the one rescue and one regression shows a local
contact-placement tradeoff rather than a gross perception or arm-selection
failure:

- On rescued seed `1136257699`, the geometry checkpoint makes a direct approach
  and ends the episode at frame 48. The baseline reaches the same bell region
  but continues for all 361 frames without registering a press.
- On regressed seed `1491434855`, both policies select the same arm and reach
  the bell at roughly the same time. The baseline centers its gripper over the
  button and succeeds at frame 48. The geometry checkpoint approaches slightly
  off center, passes across the button, and then wanders for the remainder of
  the timeout.

This makes a few-millimetre contact-target or terminal-depth error more likely
than failure to identify the bell. It also explains why sparse auxiliary
supervision can swap individual outcomes without increasing the aggregate:
small representation changes move the final contact point in either direction,
while the open-loop action chunks lack an explicit mechanism to correct the
last centimetre from a fresh image. The next model change should expose the
predicted press point to the action decoder and add a short visual replan near
contact, then test the rescue and regression seeds together as a two-scene
smoke gate.

The verified local bundle, checkpoint, logs, and nine completed videos are
stored under
`artifacts/experiments/click_bell_geometry_presspoint_1k/remote/` (ignored by
Git). The corrected bundle SHA-256 is
`c574d653ae73e9998bff0dee82e609659d8ed41a50ff69b2ad2902ab3a2f62a8`; the
model SHA-256 is
`ec66a80aeb84f516e51d0be01713713072bd73ce485afe2aeb045b64a18b5146`.
