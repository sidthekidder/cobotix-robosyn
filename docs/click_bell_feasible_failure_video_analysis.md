# Click-bell expert-feasible failure video analysis

This analysis reviews the 13 scenes that passed the official expert plan-generation check but failed under the 5K hard-edge ACT checkpoint. It uses the front and wrist-camera videos plus the simulator diagnostics from the exact seed-1 evaluation.

## Main finding

The dominant failure is final contact execution, not gross bell localization. In 12 of 13 episodes, the high camera shows the arm approaching the correct bell, and the right-wrist camera usually centers the bell under the tool. The tool then straddles or clips the button, does not produce enough downward displacement, and continues making large corrective motions until timeout.

Four episodes reached 3.3–4.75 mm of the required 4.8 mm depression. The other nine stayed below 1.1 mm even though most visibly reached the bell. Episode 92 is the clearest perception or camera-coverage exception: the bell is near the high-camera boundary and does not become clearly visible in the right-wrist view.

## Seed-level review

| Episode | Seed | Max press | Visual pattern | Working classification |
|---:|---:|---:|---|---|
| 17 | 1870626073 | 3.37 mm | Correct bell reached and centered; repeated contact does not finish the stroke | Near press; insufficient downward margin |
| 33 | 1963679703 | 3.77 mm | Bell is outside most of the high-camera view but becomes centered in the wrist view | Boundary view plus near press |
| 41 | 927586281 | 0.95 mm | Correct bell approached despite cup and spoon; tool remains over the rim | Contact alignment/angle |
| 43 | 1300333575 | 3.31 mm | Correct bell reached next to a cup; repeated centered approaches | Near press; possible adjacent-object constraint |
| 44 | 1346146623 | 0.96 mm | Clear, central bell with little occlusion; tool reaches it but does not depress | Contact alignment/vertical command |
| 53 | 1415635672 | 0.94 mm | Bell and adjacent cup remain distinct; tool targets the bell | Contact alignment, not object confusion |
| 59 | 732412360 | 0.96 mm | Bell remains visible and is approached repeatedly | Contact alignment/vertical command |
| 62 | 1808643459 | 0.96 mm | Correct bell reached with a nearby cup; wrist view centers the target | Contact alignment, possible cup interference |
| 66 | 142443833 | 1.05 mm | Correct bell approached amid spoon/fork clutter | Contact alignment under clutter |
| 75 | 1771840848 | 0.93 mm | Bell reached beside a spoon, followed by retreat and retry | Contact alignment/temporal correction |
| 77 | 146764659 | 4.75 mm | Clean centered approach; misses the 4.8 mm threshold by about 0.05 mm | Threshold-margin failure |
| 79 | 532704722 | 0.96 mm | Bell lies near the high-camera boundary but remains visible in the wrist view | Boundary view plus contact alignment |
| 92 | 1754904329 | 0.94 mm | Bell is near the high-camera boundary; wrist view mainly contains the spoon | Likely camera coverage/localization failure |

## Quantitative evidence

Among expert-feasible scenes, successful and failed episodes have these median diagnostics:

| Outcome | Episodes | Minimum right link-to-button distance | Right-arm path length | Maximum right joint delta |
|---|---:|---:|---:|---:|
| Success | 50 | 0.1687 m | 0.52 m | 2.07 rad |
| Failure | 13 | 0.1550 m | 1.65 m | 2.76 rad |

The failed arm trajectories came at least as close by the recorded link-distance proxy, then traveled about three times farther. The proxy measures `right_link6`, not fingertip contact, so its absolute value cannot prove good contact. Together with the videos, it shows that the policy generally reaches the target area and then fails to convert alignment into a completed press.

## Hypotheses ranked by current evidence

1. **Final tool pose and downward travel.** Strong evidence. The bell is commonly centered in the wrist view while the gripper contacts the rim or stops short of full depression.
2. **Weak closed-loop behavior after first contact.** Strong evidence. Failures accumulate long paths and large joint changes after the initial miss instead of applying a small stable downward correction.
3. **Camera-boundary coverage.** Moderate evidence in episodes 33, 79, and especially 92. The wrist camera compensates in 33 and 79, so boundary placement alone is insufficient to explain most failures.
4. **Distractor adjacency or collision.** Possible in episodes 41, 43, 53, 62, 66, and 75, but the videos do not show systematic selection of the wrong object. Cups and utensils may constrain the approach or obscure the final contact rather than cause semantic confusion.
5. **Lighting, material, and texture randomization.** No consistent failure-only pattern is visible. The failures span bright, dark, smooth, striped, and woven surfaces and many bell colors.
6. **Bell x/y position alone.** Weak explanation for this subset. Eleven of the 13 failures are in the central region, and ACT solved 15/17 expert-feasible hard-region scenes.
7. **Initial arm pose.** Still possible, but the present logs contain only maximum motion, not the initial joint vector. This needs explicit logging in a future evaluation.

## Next diagnostic experiment

Run the same 13 seeds with per-step logging of button depth, fingertip pose relative to the button, commanded action, wrist orientation, and ACT replan boundaries. Add a simulator-only oracle press completion test: after ACT first enters a small geometric neighborhood of the button, hold lateral pose and command a short downward motion. This is a diagnostic control, not a submission policy. If it rescues most seeds, the next model change should emphasize final-approach and press-dwell demonstrations rather than broader position sampling.

For a deployable recovery policy, trigger from observations available to ACT, such as wrist-image target centering plus arm proprioception. The earlier depth-plateau recovery required at least 1.1 mm of depression, so it could not activate on the nine failures that stalled below that threshold.
