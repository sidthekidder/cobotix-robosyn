# ClickBell corrective fine-tuning results

## Decision

Keep `click_bell_hard_edges_x3/checkpoints/005000`. The correction-augmented
checkpoints did not pass the fixed-dev gate, so none should be promoted or run
on the 100-episode held-out suite.

## Experiment

- Collected 100 successful corrective demonstrations from seed `20260925`.
- Saved 3,600 frames in 160 attempts; the 60 rejected attempts were canonical
  planner failures.
- Perturbations covered x/y/z almost evenly (32/34/34 episodes), both signs,
  and magnitudes from 3.0 to 12.0 mm (7.5 mm median).
- Appended the demonstrations to the 1,000-episode released ClickBell dataset.
- Retained the prior 3x hard-edge episode weights and assigned corrective
  episodes weight `7.6363889`, giving them an expected 20% of sampled frames.
- Fine-tuned from the hard-edge 5K checkpoint for 3,000 updates at `1e-5`,
  saving checkpoints at 1K, 2K, and 3K.
- Evaluated every checkpoint on the same 20 seeds used to select the current
  hard-edge model, with expert filtering and recovery/oracle features disabled.

## Fixed-dev results

| Policy | Success | Rescues | Regressions | High-x (`x >= 0.70`) | High-y (`y >= 0.15`) |
|---|---:|---:|---:|---:|---:|
| Hard-edge 5K baseline | 13/20 | — | — | 3/7 | 1/2 |
| Corrections 1K | 13/20 | 3 | 3 | 5/7 | 0/2 |
| Corrections 2K | 13/20 | 2 | 2 | 4/7 | 0/2 |
| Corrections 3K | 12/20 | 2 | 3 | 4/7 | 0/2 |

The 2K checkpoint was the best correction candidate. It rescued seeds
`292249176` and `1136257699`, but regressed seeds `352272321` and `1932520490`.
The paired McNemar p-value was 1.0. Its shorter average episode length
(162.65 versus 173.5 actions) does not compensate for the unchanged success
rate and two regressions.

## Interpretation

The corrective data changed behavior rather than uniformly improving it. It
helped the high-x slice, but all correction checkpoints lost the baseline's
remaining high-y success. More updates amplified the tradeoff instead of
resolving it: 3K fell to 12/20.

The next correction experiment should reduce the corrective sample share and
stratify demonstrations by scene region. A useful first test is 5–10% total
correction sampling with separate quotas for high-x and high-y scenes, then a
1K checkpoint gate. This tests whether the recovered high-x behavior can be
added without overwriting high-y coverage.

Raw metrics, comparisons, videos, the 100-episode converted dataset, and all
three policy weights are stored locally under
`artifacts/experiments/click_bell_corrections_corr20/` (ignored by Git).
