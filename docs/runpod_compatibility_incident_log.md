# RunPod and compatibility incident log

This is the running record of issues encountered while developing the Cobotix
RoboSyn click-bell policy on RunPod. Add an entry when a problem is observed,
including the evidence, root cause, verified resolution, and a preflight check
that can prevent the same failure on a fresh pod.

## Preflight checklist

1. Create the pod with `dexforce/embodichain:ubuntu22.04-cuda12.8`,
   `NVIDIA_DRIVER_CAPABILITIES=all`, and SSH enabled.
2. Allow the image pull to finish before diagnosing SSH. Confirm the system log
   says `start container ... begin` and `get-pod` reports runtime ports.
3. Verify `nvidia-smi`, Vulkan device selection, and CUDA from the intended
   Python environment.
4. Install `ffmpeg`, `git`, `curl`, `ca-certificates`, `openssh-server`, and
   `python3.10-dev` before `uv sync --frozen`.
5. Transfer the private challenge source as a Git archive or authenticated
   checkout. Record the exact source revision in `.cobotix-source-revision`.
6. Check out EmbodiChain commit
   `9ebee30011f378f94a7cbe78b01d8c2eacba231a`.
7. Keep ACT inference on the locked LeRobot 0.3.3 environment. Load the
   checkpoint and fail the preflight if normalization keys are reported as
   unexpected or missing.
8. Run one policy-conditioned correction through dataset finalization before a
   full collection. Inspect its manifest and recorded frame count.

## 2026-09-27: Pod reported running before its container existed

- **Symptom:** The control plane reported `RUNNING`, while proxy SSH returned
  `container not found`.
- **Evidence:** Pod system logs were still extracting image layers, including
  7.0 GB and 5.8 GB layers.
- **Root cause:** RunPod allocated and billed the GPU before the large simulator
  image had finished pulling and unpacking on that host.
- **Resolution:** Follow the pod system log until the image digest completes and
  `start container ... begin` appears, then retry SSH.
- **Prevention:** Treat `RUNNING` as allocation state. Gate setup on container
  logs or a successful `nvidia-smi` call.

## 2026-09-27: Proxy SSH required an interactive terminal

- **Symptom:** A noninteractive proxy command returned `Your SSH client doesn't
  support PTY`; an early direct connection was refused.
- **Root cause:** The `ssh.runpod.io` proxy opens an interactive shell for this
  pod. The direct TCP mapping was published before `sshd` was listening in the
  custom container.
- **Resolution:** Use `ssh -tt` for the proxy during bootstrap. Install and
  start `openssh-server`, write `$PUBLIC_KEY` to
  `/root/.ssh/authorized_keys`, and then use the direct host and port for SCP
  and noninteractive commands.
- **Prevention:** Include `openssh-server` in bootstrap and test both an
  interactive proxy shell and the direct SSH endpoint before large transfers.

## 2026-09-27: Private GitHub checkout prompted for credentials

- **Symptom:** `git clone https://github.com/sidthekidder/cobotix-robosyn.git`
  stopped at `Username for 'https://github.com'`.
- **Root cause:** The repository is private and the disposable pod had no GitHub
  credential.
- **Resolution:** Upload `git archive HEAD` from the authenticated Mac and write
  the source revision alongside it.
- **Prevention:** Do not assume the pod can clone the challenge repository.
  Prepare an archive transfer path before provisioning.

## 2026-09-27: Locked environment failed building `evdev`

- **Symptom:** `uv sync --frozen` failed with `fatal error: Python.h: No such
  file or directory` while building `evdev==1.9.3`.
- **Root cause:** The image had Python 3.10 but not its development headers.
- **Resolution:** Install `python3.10-dev`, then rerun the locked sync. Cached
  downloads make the retry incremental.
- **Prevention:** Install the development headers in the initial apt bootstrap.

## 2026-09-27: LeRobot 0.4.4 silently changed ACT checkpoint semantics

- **Symptom:** The checkpoint loaded under LeRobot 0.4.4, but emitted unexpected
  keys for every saved input, target, and output normalization buffer.
- **Root cause:** The current EmbodiChain recorder passes the newer
  `metadata_buffer_size` argument, which encouraged upgrading LeRobot. The
  released ACT checkpoint and our fine-tuned checkpoint use LeRobot 0.3.3's
  normalization layout; the 0.4.4 loader discarded those buffers.
- **Resolution:** Restore the frozen LeRobot 0.3.3 environment. The correction
  collector now applies a narrow compatibility shim that removes only the
  unsupported `metadata_buffer_size` recorder hint. Model loading is clean on
  LeRobot 0.3.3 with PyTorch 2.7.1+cu126.
- **Prevention:** Make checkpoint loading warnings a hard smoke-test failure.
  Never upgrade the inference environment merely to satisfy the dataset writer.

## 2026-09-27: First simulator run repopulated asset caches

- **Symptom:** The first smoke run paused to download CobotMagic and simulator
  resource archives and rebuild the combined robot URDF.
- **Root cause:** The pod has no persistent volume and therefore starts with an
  empty EmbodiChain cache.
- **Resolution:** Let the one-episode smoke run populate caches before timing
  the full collection.
- **Prevention:** Budget first-boot setup time separately from episode runtime.
  If repeated runs justify persistent storage, cache only immutable simulator
  assets and continue copying experiment outputs off-pod before teardown.

## 2026-09-27: LeRobot 0.3.3 recorder method signatures differed

- **Symptom:** The first recovery reached dataset saving, then failed because
  `LeRobotDataset.add_frame()` required a separate `task` argument. The legacy
  class also had no `finalize()` method.
- **Root cause:** Current EmbodiChain emits `task` inside each frame and targets
  LeRobot 0.4's `add_frame(frame)` and `finalize()` API. ACT still needs 0.3.3.
- **Resolution:** Extend the narrow collector shim to remove `task` from the
  frame and pass it as the legacy positional argument. Provide a no-op legacy
  `finalize()` because 0.3.3 writes episode metadata in `save_episode()` and
  EmbodiChain already stops the image writer first.
- **Prevention:** The one-episode smoke must include frame conversion,
  `save_episode()`, and finalization; model loading alone cannot verify the
  recorder boundary.

## 2026-09-27: Hard-region recovery quotas conflicted with oracle coverage

- **Symptom:** A 100-example run requested 25 high-x and 25 high-y failures,
  but its first 31 seeds saved four central corrections; both accepted high-x
  scenes were policy successes and no accepted high-y failure appeared.
- **Root cause:** Policy-conditioned recovery still needs the scripted expert's
  canonical press target. Our held-out audit showed the expert accepts only
  17/52 hard scenes and ACT solves 15/17 of those. Asking for 50 hard-region
  policy failures therefore selects a tiny intersection and can exhaust the
  attempt budget.
- **Resolution:** Stop the invalid quota run and use a 40-example first pilot:
  two high-x failures, two high-y failures, and 36 other failures. Keep the
  unchanged dev20 gate to measure whether even this limited hard coverage
  helps.
- **Prevention:** Derive collection quotas from the intersection of oracle
  coverage and policy failures, rather than from the evaluation distribution
  alone. Report both oracle-rejected and policy-success discard counts.
