# Corrective collector smoke run

- Source commit: `fb9baa0`
- GPU: NVIDIA A40
- LeRobot: 0.4.4
- Seed: `20260925`
- Command: `bash launch/collect_click_bell_corrections.sh --episodes 3 --seed 20260925 --max-attempts 20`
- Result: 3 successful episodes from 7 scene attempts
- Dataset: LeRobot v3, 108 frames total, 36 frames per episode, 25 FPS
- Cameras: high, left wrist, and right wrist at 640x480, AV1

The four rejected attempts were canonical planner failures and produced no
training samples. `dataset/correction_manifest.json` contains every attempted
scene and perturbation. `collection.log` is the complete simulator transcript,
and `runtime.json` records the execution environment.
