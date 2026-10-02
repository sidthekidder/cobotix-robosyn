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

## 2026-09-27: A40 host could not start the simulator container

- **Symptom:** The image finished pulling, but the host repeatedly logged
  `OCI runtime create failed ... can't get final child's PID from pipe: EOF`;
  proxy SSH confirmed that the container was not running.
- **Root cause:** Host container-runtime failure before project setup. No
  experiment code ran.
- **Resolution:** One in-place restart reproduced the failure, so pod
  `ilyoznncnmoyun` was terminated. The account was verified to have zero pods
  afterward.
- **Prevention:** Gate all setup on successful SSH plus `nvidia-smi`. After one
  restart reproduces this OCI error, terminate promptly and use another host;
  do not spend time reinstalling software in a container that never started.

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
- **Resolution:** Stop the invalid quota run and request two high-x failures,
  two high-y failures, and other failures. Freeze the first pilot at 20 total
  examples once both hard quotas are complete; a 5% weighted sampler does not
  require 40 raw corrections for the initial 1,000-update gate. Keep the
  unchanged dev20 gate to measure whether even this limited hard coverage
  helps.
- **Prevention:** Derive collection quotas from the intersection of oracle
  coverage and policy failures, rather than from the evaluation distribution
  alone. Report both oracle-rejected and policy-success discard counts.

## 2026-09-27: Progress manifest hid discarded attempts

- **Symptom:** The on-disk manifest count remained unchanged for several
  minutes even though current simulator logs and GPU utilization showed the
  collector progressing.
- **Root cause:** The collector checkpointed its manifest only after recovery
  attempts, not after expert-plan failures or policy-success discards.
- **Resolution:** Write the manifest after every completed attempt. Determine a
  scene's region immediately after reset and skip it before expert planning or
  policy rollout when that region's quota is already full.
- **Prevention:** Treat progress telemetry as part of long-run reliability and
  smoke-test at least one discard path as well as one saved path.

## 2026-09-27: Graceful interruption could save an incomplete rollout

- **Symptom:** The pilot reached its practical 20-example stopping point while
  the collector was configured to continue to 40 and had already begun the
  next simulator attempt.
- **Root cause:** The collector's `finally` block closes the environment, and
  the legacy recorder can finalize an active rollout during close. Sending an
  ordinary interrupt after the next attempt starts can therefore append a
  partial, policy-only episode.
- **Resolution:** After the 20th episode and manifest were durably written,
  stop the Python child without running finalizers. Validate the resulting
  `meta/info.json`: v2.1, 20 episodes, 400 frames, and 25 FPS.
- **Prevention:** Give future collectors a checkpoint-boundary stop flag that
  exits before the next reset. Until then, validate episode and frame counts
  after any early stop and before merging.

## 2026-09-27: Public Hugging Face snapshot download hit HTTP 429

- **Symptom:** The Hub warned that the released click-bell dataset was being
  downloaded without authentication, then stalled after 898 files. A direct
  API probe returned HTTP 429 while the Python process remained alive.
- **Root cause:** The disposable pod intentionally had no Hugging Face token.
- **Resolution:** Stop the stalled snapshot. Copy the complete 1,000-episode
  parquet and metadata snapshot already cached on the Mac, then fetch the
  3,000 predictable camera-video paths directly from the CDN with bounded
  parallelism and retries. The pod fetched 2,980 before another throttle; the
  final 20 were fetched from the Mac and transferred. Validate 1,000 parquet
  files, 3,000 nonempty MP4 files, and 74,000 declared frames.
- **Prevention:** Cache the released base dataset on a persistent volume for
  repeated experiments, or provide a scoped read-only token when transfer time
  materially affects GPU cost. Monitor completed file counts, not only whether
  the downloader PID exists.

## 2026-09-27: macOS tar added AppleDouble sidecar files

- **Symptom:** After transferring cached data from the Mac, validation counted
  2,000 parquet files instead of 1,000 and 3,020 MP4 files instead of 3,000.
- **Root cause:** BSD tar preserved macOS extended attributes as `._*`
  AppleDouble files. The 20 videos fetched on the Mac added 20 more sidecars.
- **Resolution:** Delete only `._*` files, then repeat the exact parquet, MP4,
  zero-byte, and metadata-count checks before merging.
- **Prevention:** Disable macOS metadata when creating transfer archives or
  filter `._*` files immediately after extraction. Always count canonical
  dataset objects before training.

## 2026-09-27: Research audit — short recovery clips change ACT loss weighting

- **Finding:** The latest corrections have 20 frames per episode versus a
  50-action ACT horizon. Uniformly sampling their frames yields 10.5 valid
  target actions on average. LeRobot 0.3.3 masks padding but averages L1 over
  all positions; KL remains per-sample. A 5% sampling share is therefore not
  a 5% share of unmasked action supervision.
- **Status:** Structural behavior confirmed from dataset metadata and upstream
  source. Its effect on rollout success has not been measured; no loss patch
  has been applied.
- **Prevention:** Log valid-target counts and source-specific L1/KL before
  mixing different episode lengths. Test any loss-normalization change as an
  explicit ablation. Do not convert padding to labels without real simulated
  continuation steps.

## 2026-09-27: Research audit — reconstructed recovery state is incomplete

- **Finding:** The collector resets the scene, restores policy-reached qpos,
  then sets its rollout counter to zero. It does not explicitly restore the
  original velocity/contact/object state or randomization/event state.
- **Status:** Confirmed code behavior, not a proven cause of poor training.
- **Prevention:** Replay the policy prefix with recording disabled and enable
  recording at intervention, or validate a full simulator snapshot restore.
  Compare observation/state at the handover before calling data exact on-policy
  corrections.

## 2026-09-27: Research audit — same seeds have differing appearance histories

- **Finding:** Paired saved videos for seeds 1136257699, 1879422756 and
  352272321 show different later material/light appearances between the 5K
  baseline and correction pilot. Initial scenes look similar. The evaluations
  used an A40 and an A6000 respectively. Local contact sheets are under
  `artifacts/research/2026-09-27/`.
- **Status:** Visual discrepancy observed; its source and outcome impact are
  unresolved. Do not infer that either model benefited from it.
- **Prevention:** Compare identical action replays and identical-policy runs in
  fresh/reordered episodes. Record environment versions and event timelines,
  separate model/environment RNGs, and establish paired reproducibility or
  report repeated-run variability before interpreting small score differences.
## CUDA trajectory conversion in ClickBell expert collection

- **Symptom:** demonstration collection stopped at `Generating edges: 0%` and exited with status 0.
- **Hidden error:** `TypeError: can't convert cuda:0 device type tensor to numpy` in `click_bell/action_bank.py`.
- **Why the traceback disappeared:** EmbodiChain's default `SimulationManager.destroy()` calls `os._exit(0)` during `env.close()`, masking an exception raised before teardown.
- **Diagnosis:** rerun with `EMBODICHAIN_SIM_EXIT_PROCESS=0` to retain the Python traceback. The process may segfault later during native simulator teardown; the earlier traceback is the useful signal.
- **Fix:** convert the generated trajectory with `ret.positions[0].detach().cpu().numpy().T`.

## 2026-09-28: Custom DexSim image did not start an SSH daemon

- **Symptom:** RunPod reported a live runtime and published TCP port 22, but the
  direct endpoint returned `Connection refused`; proxy authentication succeeded
  without yielding a shell.
- **Root cause:** `dexforce/embodichain:ubuntu22.04-cuda12.8` did not contain a
  running SSH daemon under the configured `sleep infinity` command.
- **Resolution:** Update the pod command to install `openssh-server` when
  absent, write `$PUBLIC_KEY` to `/root/.ssh/authorized_keys`, generate host
  keys, and execute `/usr/sbin/sshd -D -e`. After restarting, use the newly
  assigned direct TCP port rather than the stale pre-restart port.
- **Prevention:** Require both `Server listening on ... port 22` in container
  logs and a successful direct SSH probe. Refresh `get-pod` after every restart
  because the public TCP port can change.
- **2026-10-01 confirmation:** RunPod's `startSsh` flag alone did not keep this
  custom image running because its default command exited. Set the SSH-daemon
  entrypoint/command at pod creation; otherwise the already-downloaded image
  can be repaired with one command update and restart.

## 2026-09-28: Community RTX 3090 exposed graphics but broken CUDA compute

- **Symptom:** `nvidia-smi` reported an RTX 3090 and `vulkaninfo` selected the
  NVIDIA proprietary device, while PyTorch reported `CUDA unknown error`.
  Direct `ctypes.CDLL("libcuda.so.1").cuInit(0)` returned error 999.
- **Root cause:** Host-level CUDA driver/device attachment failure. The expected
  `/dev/nvidia*` nodes existed, so this was below the Python environment. An
  in-place container restart reproduced the failure.
- **Resolution:** Terminate the unusable pod after preserving no experiment
  results. Do not reinstall PyTorch or DexSim when the driver-level `cuInit`
  preflight already fails.
- **Prevention:** Run `cuInit(0)` and `torch.cuda.is_available()` immediately
  after SSH becomes available, before downloading policy dependencies or
  datasets. Treat Vulkan rendering, NVML, and CUDA compute as separate gates.

## 2026-09-30: Secure A40 retry passed the keypoint-collector smoke test

- **Evidence:** Secure pod `v9l89rnnwmwmbb` passed `cuInit(0) == 0`, PyTorch
  2.7.1+cu126 reported CUDA available on an NVIDIA A40, and Vulkan selected the
  proprietary A40 device. Source commit `9c1b210` and EmbodiChain commit
  `9ebee30011f378f94a7cbe78b01d8c2eacba231a` ran with LeRobot 0.3.3.
- **Result:** A one-cell, one-episode keypoint collection saved and finalized
  one 74-frame LeRobot episode. Validation found 74 parquet frames, 74 ordered
  keypoint rows, and three nonempty camera videos. The high camera saw the
  button in all 74 frames, the right-wrist camera in 60, and the left-wrist
  camera in none for this scene.
- **Cost note:** The uncached DexSim image took about 11 minutes to pull and
  extract before the container started. The pod used ephemeral storage, the
  3 MB smoke artifact was copied locally, and the pod was terminated. A final
  audit found zero pods, network volumes, or serverless endpoints.
- **Artifact:** `artifacts/keypoint_smoke/20260930/keypoint_smoke_9c1b210.tar.gz`
  (SHA256 `a87ddd80bca1c4fae3e117981172bcdf21bf2fab8cf4c23f90797f0c589a3dbd`).

## 2026-10-01: Interactive proxy SSH let FFmpeg consume queued shell commands

- **Symptom:** Commands following the overlay renderer in a piped proxy-SSH
  session were interpreted by FFmpeg instead of the remote shell, so archive
  creation did not run.
- **Root cause:** FFmpeg inherited the interactive SSH stream as standard input.
- **Resolution:** Re-run the renderer with `</dev/null`, then package the
  dataset and overlays. The transferred archive checksum matched the remote
  checksum.
- **Prevention:** Detach stdin for every FFmpeg/media subprocess in a queued
  proxy-SSH session. RunPod proxy SSH does not support the SFTP subsystem, so
  use authenticated `runpodctl` transfer when available or a checksummed
  base64 stream for small emergency artifacts.

## 2026-10-01: DexSim's private package host was unreachable

- **Symptom:** `uv sync --frozen` downloaded and built the remaining ACT and
  simulator dependencies, then failed to fetch `dexsim-engine==0.4.3` from
  `http://pyp.open3dv.site:2345/`. A retry with `UV_HTTP_RETRIES=10` also
  failed.
- **Evidence:** TCP connection attempts timed out from both the Secure RunPod
  host in `EU-SE-1` and a separate local network. This isolated the failure to
  the upstream package endpoint rather than `uv`, DNS, or one RunPod host.
- **Resolution:** Stop the pod to release the A40 while preserving its
  container disk and completed dependency cache. Resume the same pod and rerun
  `uv sync --frozen` after the wheel endpoint responds.
- **Prevention:** Probe the exact locked `dexsim-engine` wheel URL before
  starting a billable pod. Treat failure of this externally hosted wheel as a
  setup gate, and preserve a legally redistributable, checksum-pinned cache if
  DexForce provides one.
- **Related setup correction:** The public source repository is
  `https://github.com/DexForce/EmbodiChain.git`; `EmbodiChain/EmbodiChain` is
  not the clone URL.

## 2026-10-01: Public simulator image recovered the package-host outage

- **Symptom:** Fresh installation remained blocked because the private
  `dexsim-engine==0.4.3` wheel host was still unreachable.
- **Resolution:** Use `dpaleyev/robosyn-groot:v1`, which already contains
  DexSim 0.4.3, EmbodiChain 0.2.4, PyTorch 2.10.0+cu128, and LeRobot 0.4.4.
  Install the pinned RoboSynChallenge and EmbodiChain source trees editable
  with `--no-deps` so the image's working simulator stack is preserved.
- **Cost note:** This image is about 56 GB and took roughly 14 minutes to pull
  and unpack on an uncached Secure A40 host. Budget that startup time before
  provisioning, and reuse a stopped container only when its disk cost is
  justified.
- **Prevention:** Probe the private wheel host first. If it fails, select the
  prebuilt image immediately and verify its package versions before changing
  anything in the environment.

## 2026-10-01: Stale Kitware apt key prevented SSH bootstrap

- **Symptom:** The prebuilt image restarted before SSH became available because
  `apt-get update` rejected an expired or unavailable Kitware repository key.
- **Resolution:** Remove `/etc/apt/sources.list.d/*kitware*` before installing
  `openssh-server`; the workload does not require that repository at runtime.
- **Prevention:** Put this cleanup in the pod startup command for this exact
  image, and make the SSH install conditional on `command -v sshd`.

## 2026-10-01: LeRobot 0.4.4 packs multiple episodes into shared videos

- **Symptom:** The collector saved valid demonstrations, but the original audit
  and visualizer looked for one `episode_*.mp4` per episode and reported videos
  as missing.
- **Root cause:** LeRobot 0.4.4 stores all episodes in a camera's
  `videos/.../chunk-000/file-000.mp4` and records each episode's timestamp
  window in `meta/episodes/chunk-000/file-000.parquet`.
- **Resolution:** Commit `0068cec` makes the audit distinguish logical episode
  records from physical video files and makes the visualizer decode exactly the
  frame count in the selected episode's timestamp window.
- **Prevention:** Inspect `meta/info.json` and the episode metadata before
  assuming a filesystem layout. Audit both logical coverage and the existence
  of every referenced physical file.

## 2026-10-02: `setsid` reports a wrapper PID, not the detached collector PID

- **Symptom:** The PID captured from `setsid bash -lc '...' &` exited within
  seconds, while the collection log was initially empty, which looked like an
  immediate launch failure.
- **Root cause:** `setsid` forked a session child. The recorded PID belonged to
  the short-lived wrapper; the detached bash, `conda run`, and Python collector
  continued under different PIDs and were adopted by PID 1.
- **Resolution:** Check the exact collector command with `pgrep -af` and inspect
  GPU memory before relaunching. In this run, the real Python process was
  healthy and completed all 80 episodes.
- **Prevention:** Write the collector's own PID from inside the detached shell,
  or monitor a completion sentinel plus an exact command match. Do not infer
  workload failure solely from the outer `setsid` PID disappearing.

## 2026-10-02: prebuilt simulator image startup remained the dominant fixed cost

- **Evidence:** `dpaleyev/robosyn-groot:v1` took about 22 minutes to download
  and unpack on a fresh Secure A40 host before the container started. The
  balanced 80-episode collection then took 37 minutes 47 seconds.
- **Resolution:** Keep the image path because it avoids the unavailable private
  DexSim package host, but budget its startup separately from experiment time.
- **Prevention:** Batch compatible simulator-only collection work on one
  validated pod after transferring each completed artifact. Terminate promptly
  when no further simulator work is ready; retaining a stopped ephemeral
  container is not available as durable storage.

## 2026-10-01: Far-positive ClickBell cell is outside the expert's practical reach

- **Symptom:** A balanced 3x3 collection filled eight cells but never saved
  cell `(2,2)` after 500 total attempts.
- **Evidence:** Cell `(2,2)` was sampled 53 times across x
  `[0.70451927, 0.84650952]` m and y `[0.10816976, 0.28698635]` m. Every sample
  failed expert planning; none reached recording.
- **Resolution:** Collect a truthful eight-cell pilot with `(2,2)` excluded in
  the manifest. The clean run produced eight 74-frame episodes and passed the
  audit with 592 frames and no errors.
- **Prevention:** Run a cheap expert-feasibility sweep before assigning balanced
  quotas. Distinguish unreachable expert geometry from policy failure and keep
  excluded cells explicit in the dataset manifest.

## 2026-10-01: Renderer initialization warnings were non-fatal

- **Symptom:** DexSim printed warnings about distractor placeholders and an
  unrecognized A40 renderer profile during initialization.
- **Evidence:** Vulkan/Hybrid initialized, all eight episodes rendered, and the
  audit found valid camera video and geometry labels.
- **Resolution:** Keep Vulkan enabled and use the successful render/audit gates
  as the source of truth. Do not abort solely on these warnings.

## 2026-10-01: Prebuilt image ACT packages were newer than the trainer API

- **Symptom:** The ACT smoke initially failed because LeRobot 0.4.4 no longer
  provides `lerobot.scripts.train`. After pinning LeRobot, dataset creation
  failed when Datasets 4.8.5 returned a `Column` to `torch.stack`.
- **Resolution:** Install LeRobot from commit
  `b883328e6c95681ca90a18b102e4ae5e1f91e2bf` editable with `--no-deps`, then
  install `datasets==3.6.0`. This preserves the image's working CUDA and DexSim
  packages while restoring the exact ACT training API.
- **Prevention:** Before transferring a training dataset, preflight imports for
  `lerobot.scripts.train` and print the LeRobot, Datasets, PyTorch, and
  torchvision versions. Enforce LeRobot's declared `datasets<=3.6.0` bound.

## 2026-10-01: AMP produced a NaN gradient norm on the mixed ACT stack

- **Symptom:** A one-update ACT probe with AMP completed with finite loss but
  reported `grdn:nan` under PyTorch 2.10.0 and LeRobot 0.3.3.
- **Resolution:** Disable AMP for the integration smoke. The next probe and all
  10 final updates had finite gradient norms from 2.945 to 17.147 and finite
  press-point auxiliary losses.
- **Prevention:** Treat finite total loss as insufficient. Run one update and
  require a finite gradient norm before a long job. Use the fully pinned ACT
  stack (`torch<2.8`) for AMP, or keep AMP disabled on this prebuilt image.

## 2026-10-01: Uncached public image startup dominated the smoke-test cost

- **Evidence:** `dpaleyev/robosyn-groot:v1` spent about 27 minutes pulling and
  extracting before the container started; the final 10-update ACT smoke then
  finished in 17 seconds including setup.
- **Resolution:** Complete transfers, compatibility probes, the bounded run,
  and artifact download in one container lifetime, then terminate it. The final
  audit found zero pods, network volumes, or endpoints.
- **Prevention:** Prefer a cached host or a smaller ACT-only image for training.
  Reserve the 56 GB simulator image for jobs that actually need DexSim/Vulkan.
