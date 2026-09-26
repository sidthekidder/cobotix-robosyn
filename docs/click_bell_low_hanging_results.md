# ClickBell low-hanging robustness experiments

## Question

Can inexpensive changes improve the `hard_edges_x3_005000` ACT policy without
collecting new demonstrations or changing the model architecture?

The experiments used the same fixed 20 development seeds as the existing
baseline. The expert feasibility precheck was disabled, and all policies saw
only submission-valid observations.

## Results

| Policy | Evaluation | Success | Decision |
| --- | --- | ---: | --- |
| Unchanged 5K ACT baseline | dev20 | 13/20 (65%) | Keep |
| Uniform average of 3K, 4K, and 5K weights | dev20 | 13/20 (65%) | Reject as standalone policy |
| Step-35 replan without blending | dev20 | 11/20 (55%) | Reject |
| Step-35 replan with a five-step queue blend | dev20 | 13/20 (65%) | Reject as standalone policy |
| Phase-balanced continuation, 1K updates | first-five gate | 0/5 (0%) | Reject |
| Phase-balanced continuation, 2K updates | first-five gate | 2/5 (40%) | Reject |

The unchanged baseline also scores 2/5 on the first-five gate. The 2K
continuation therefore tied the small gate rather than improving it.

## Checkpoint averaging

The uniform parameter average rescued development seeds 3 and 5 but regressed
seeds 4 and 19. It tied the baseline at 13/20. The changed success set shows
that nearby checkpoints contain useful diversity, but uniform weight averaging
does not turn that diversity into a stronger standalone policy.

## Queue-boundary blending

The raw step-35 replan scored 11/20 because it spliced a newly predicted motion
onto the old queue abruptly. Blending five discarded old actions into the new
queue recovered the aggregate score to 13/20. It rescued seeds 2 and 5 relative
to the unchanged policy, while regressing seeds 13 and 15.

This establishes a narrower result: blending fixes much of the discontinuity
introduced by a forced replan, but a fixed replan still does not outperform the
original coherent 50-action trajectory. The blend implementation remains
available behind a disabled-by-default configuration option for future adaptive
triggers.

## Phase-balanced continuation

The sampling plan gave weight 3 to hard-edge episodes and to demonstration
frames 20 through 49, combining overlapping rules with `max`. The plan was
expected to make contact-window frames about 57.9% of samples and hard-edge
episodes about 34.7%. At runtime, 40,692 of 74,000 frames received a custom
weight.

Training continued from the 5K checkpoint for 2,000 updates with batch size 32,
AMP, and the original 50-action horizon. The 1K checkpoint failed all first five
seeds. The 2K checkpoint recovered to 2/5, rescuing seed `292249176` in 41 steps
but losing baseline success `352272321`. The user rejected the continuation at
that gate, so no full dev20 score was run.

The likely failure is distribution shift from replay weighting: emphasizing
mid-trajectory frames teaches contact states but reduces rehearsal of the
initial visual approach. More updates partly recover behavior, yet they do not
provide evidence of a net gain. We should not spend more compute tuning this
sampling recipe.

## Decision and next experiment

The 5K checkpoint remains the submission candidate at 13/20. None of the three
cheap modifications improved the paired aggregate, so they should not replace
it.

The next useful experiment should add information rather than reweight the same
demonstrations: collect corrective trajectories from controlled lateral,
vertical, and camera perturbations near contact. Each correction should include
the original approach distribution as replay, and it should pass a five-seed
regression gate before a full dev20 run. This directly teaches recovery states
that the fixed replan exposed as missing.

## Reproducibility

Local ignored artifacts, including paired metrics and videos, the five-seed
gate, sampling plan, and checkpoint metadata, are stored under:

`artifacts/benchmarks/click_bell_low_hanging_dev20`

The transferred archive has SHA-256:

`87d462a862dac90a8459f8e010a3dbb5886f7d5436538911df8530aa6a3ce270`

The RunPod A40 used for these experiments was terminated after artifact
transfer. No pods remained afterward.
