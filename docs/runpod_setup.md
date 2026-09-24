# Runpod setup for RoboSynChallenge

## Verified configuration

The first successful compatibility test used:

- Runpod Secure Cloud
- NVIDIA A40 (48 GB VRAM)
- `dexforce/embodichain:ubuntu22.04-cuda12.8`
- `NVIDIA_DRIVER_CAPABILITIES=all`
- Container command: `bash -lc 'sleep infinity'`
- The image's existing Conda environment: `py310`

The pod exposed `/dev/dri/card5` and `/dev/dri/renderD132`. `vulkaninfo`
selected the NVIDIA A40 with the proprietary driver, and PyTorch 2.7.0+cu128
detected CUDA.

## Package-index workaround

Do not install every dependency with Open3DV as an extra index. The Open3DV
index redirects public packages to an Aliyun mirror, which was much slower from
the tested Runpod data center. Install public dependencies from PyPI and install
`dexsim_engine` separately from Open3DV.

From a workspace containing sibling `EmbodiChain` and `RoboSynChallenge`
repositories:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate py310

python -m pip install --index-url https://pypi.org/simple \
  'setuptools>=78.1.1' 'gymnasium>=0.29.1' langchain langchain-openai \
  'toppra==0.6.3' pin pin-pink casadi 'qpsolvers[osqp]==4.8.1' \
  'pytorch_kinematics==0.10.0' 'polars==1.31.0' 'PyYAML>=6.0' \
  'wandb>=0.21.0' 'tensorboard>=2.20.0' ortools prettytable \
  'black==26.3.1' fvcore h5py tensordict 'viser==1.0.21'

python -m pip install --no-deps \
  --index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site \
  'dexsim_engine==0.4.3'

python -m pip install --index-url https://pypi.org/simple \
  'newton[sim]==1.2.1' types-usd usd-core 'coacd>=1.0.7' \
  'numpy==1.26.4' 'pymeshfix>=0.17.2' 'pymeshlab>=2023.12.post3' \
  'pyvista==0.46.4' 'trimesh>=4.9.0,<5' 'warp-lang>=1.10.0' av

python -m pip install --no-deps -e /workspace/EmbodiChain
```

LeRobot and the challenge policy environments still need their policy-specific
installation before collecting datasets or evaluating a checkpoint. They are
not required for the simulator compatibility test below.

## ACT evaluation environment

The ACT policy uses its own `uv` environment. The following setup was verified
on an A40 pod with the official EmbodiChain image:

```bash
apt-get update && apt-get install -y ffmpeg git
curl -LsSf https://astral.sh/uv/install.sh | sh

cd /workspace/RoboSynChallenge/policy/act
uv sync --frozen
```

`ffmpeg` is required by the evaluator when it writes episode videos. The
Open3DV-hosted CPython 3.10 `dexsim_engine==0.4.3` wheel was replaced without a
version change. Its current SHA256 is pinned in `policy/act/uv.lock`; using the
former hash causes `uv sync --frozen` to fail with a checksum mismatch.

The verified smoke-test command is recorded in
[`docs/baseline_click_bell.md`](baseline_click_bell.md).

## Compatibility test

```bash
cd /workspace/EmbodiChain
python scripts/tutorials/sim/create_scene.py --headless
```

The verified run selected the Hybrid Vulkan renderer on the A40, completed
1,000 steps at roughly 83–86 FPS, and wrote a headless recording under
`outputs/videos/`.

## Cost guard

Stop the pod as soon as testing or training finishes. A stopped pod releases
the GPU but retains its persistent disk, which continues to incur storage cost.
Terminate the pod after copying any results that need to be kept.

Changing ports or other pod configuration can recreate a container. On a pod
without a persistent volume, that erases its container disk. Configure required
ports when the pod is created and copy results off the pod before changing its
configuration or terminating it.
