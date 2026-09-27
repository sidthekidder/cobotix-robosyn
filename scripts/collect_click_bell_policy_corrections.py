#!/usr/bin/env python3
"""Collect oracle recoveries from states reached by a failing ACT policy.

The policy first runs without recovery for a complete episode.  Successful
episodes are discarded.  For a failure, the closest eligible near-contact
state is restored after a deterministic scene reset, which clears the recorder.
Only the oracle recovery tail is then recorded as a training demonstration.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import inspect
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Fixed development scenes must never enter training data.
DEV20_SEEDS = {
    1491434855, 292249176, 374217481, 1284876248, 352272321,
    1136257699, 580757632, 1544074682, 716257571, 1396067212,
    398764591, 441365315, 1537364731, 1819583497, 530702035,
    1879422756, 1682652230, 1171049868, 1982038771, 1932520490,
}


@dataclass(frozen=True)
class Snapshot:
    step: int
    qpos: list[float]
    max_joint_error_rad: float
    rms_joint_error_rad: float
    lateral_eef_error_m: float
    vertical_eef_error_m: float
    max_press_depth_m: float

    @property
    def rank(self) -> tuple[float, float, float]:
        # Prefer the state geometrically closest to the expert press target.
        return (
            self.lateral_eef_error_m,
            abs(self.vertical_eef_error_m),
            self.rms_joint_error_rad,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "qpos": self.qpos,
            "max_joint_error_rad": self.max_joint_error_rad,
            "rms_joint_error_rad": self.rms_joint_error_rad,
            "lateral_eef_error_m": self.lateral_eef_error_m,
            "vertical_eef_error_m": self.vertical_eef_error_m,
            "max_press_depth_m": self.max_press_depth_m,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--max-attempts", type=int, default=600)
    parser.add_argument("--rollout-steps", type=int, default=361)
    parser.add_argument("--snapshot-min-step", type=int, default=35)
    parser.add_argument("--target-plan-step", type=int, default=49)
    parser.add_argument("--max-joint-error-rad", type=float, default=0.45)
    parser.add_argument("--recovery-steps", type=int, default=12)
    parser.add_argument("--hold-steps", type=int, default=8)
    parser.add_argument("--high-x-min", type=float, default=0.70)
    parser.add_argument("--high-y-min", type=float, default=0.15)
    parser.add_argument("--high-x-quota", type=int, default=2)
    parser.add_argument("--high-y-quota", type=int, default=2)
    parser.add_argument(
        "--output-root",
        default="lerobot_dataset/click_bell_policy_corrections",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _attempt_seeds(seed: int, count: int) -> list[int]:
    rng = np.random.default_rng(seed)
    result: list[int] = []
    used = set(DEV20_SEEDS)
    while len(result) < count:
        candidate = int(rng.integers(0, 2**31 - 1))
        if candidate not in used:
            used.add(candidate)
            result.append(candidate)
    return result


def _region(button_position: list[float], args: argparse.Namespace) -> str:
    x, y = float(button_position[0]), float(button_position[1])
    if y >= args.high_y_min:
        return "high_y"
    if x >= args.high_x_min:
        return "high_x"
    return "other"


def _quotas(args: argparse.Namespace) -> dict[str, int]:
    other = args.episodes - args.high_x_quota - args.high_y_quota
    if other < 0:
        raise ValueError("high-x and high-y quotas exceed --episodes")
    return {"high_x": args.high_x_quota, "high_y": args.high_y_quota, "other": other}


def _get_recorder(env: Any) -> Any:
    manager = env.unwrapped.dataset_manager
    for mode_cfgs in manager._mode_functor_cfgs.values():
        for functor_cfg in mode_cfgs:
            recorder = getattr(functor_cfg, "func", None)
            if hasattr(recorder, "curr_episode") and hasattr(recorder, "dataset_path"):
                return recorder
    raise RuntimeError("No LeRobot recorder is configured")


def _scalar(value: Any) -> float:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return float(value.detach().reshape(-1)[0].cpu())
    except ImportError:
        pass
    return float(np.asarray(value).reshape(-1)[0])


def _right_indices(base: Any) -> list[int]:
    ids = [int(x) for x in base.robot.get_joint_ids(name="right_arm", remove_mimic=True)]
    lookup = {int(joint_id): i for i, joint_id in enumerate(base.active_joint_ids)}
    return [lookup[joint_id] for joint_id in ids]


def _target_details(base: Any, canonical: Any, target_step: int) -> dict[str, Any]:
    import torch

    target_index = min(target_step, len(canonical) - 1)
    target = torch.as_tensor(canonical[target_index], dtype=torch.float32, device=base.device)
    if target.ndim == 2:
        target = target[0]
    indices = _right_indices(base)
    right = target[indices]
    pose = base.robot.compute_fk(name="right_arm", qpos=right.unsqueeze(0), to_matrix=True)[0]
    return {"action": target, "right_indices": indices, "right_qpos": right, "pose": pose}


def _snapshot(base: Any, target: dict[str, Any], step: int) -> Snapshot:
    import torch

    qpos = base.robot.get_qpos()[:, base.active_joint_ids][0].detach()
    right = qpos[target["right_indices"]]
    joint_error = torch.abs(right - target["right_qpos"])
    pose = base.robot.compute_fk(name="right_arm", qpos=right.unsqueeze(0), to_matrix=True)[0]
    diagnostics = base.get_episode_diagnostics()
    return Snapshot(
        step=step,
        qpos=qpos.cpu().tolist(),
        max_joint_error_rad=float(torch.max(joint_error).cpu()),
        rms_joint_error_rad=float(torch.sqrt(torch.mean(joint_error**2)).cpu()),
        lateral_eef_error_m=float(
            torch.linalg.norm(pose[:2, 3] - target["pose"][:2, 3]).cpu()
        ),
        vertical_eef_error_m=float((pose[2, 3] - target["pose"][2, 3]).cpu()),
        max_press_depth_m=_scalar(diagnostics["max_press_depth_m"]),
    )


def _recovery_actions(base: Any, snapshot: Snapshot, target: dict[str, Any], args: argparse.Namespace):
    import torch

    current = torch.as_tensor(snapshot.qpos, dtype=torch.float32, device=base.device)
    destination = current.clone()
    destination[target["right_indices"]] = target["right_qpos"]
    actions = [
        torch.lerp(current, destination, amount).clone()
        for amount in torch.linspace(
            1.0 / args.recovery_steps,
            1.0,
            args.recovery_steps,
            device=base.device,
        )
    ]
    actions.extend(destination.clone() for _ in range(args.hold_steps))
    return actions


def _is_done(value: Any) -> bool:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return bool(value.any().item())
    except ImportError:
        pass
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


def _is_success(env: Any) -> bool:
    return _is_done(env.get_wrapper_attr("is_task_success")())


def _install_lerobot_recorder_compat(dataset_class: Any) -> list[str]:
    """Let the current recorder call older LeRobot without changing ACT.

    EmbodiChain passes ``metadata_buffer_size`` to LeRobot 0.4+, but the ACT
    checkpoint was trained with LeRobot 0.3.3. Upgrading LeRobot makes the new
    loader discard that checkpoint's normalization buffers. The 0.3.3 dataset
    writer otherwise supports the recorder calls used here, so ignore only the
    unknown buffering hint and keep policy loading byte-compatible.
    """
    installed: list[str] = []
    if (
        "metadata_buffer_size" not in inspect.signature(dataset_class.create).parameters
        and not getattr(dataset_class.create, "_robosyn_metadata_compat", False)
    ):
        original_create = dataset_class.create

        def create_compat(*create_args: Any, **create_kwargs: Any) -> Any:
            create_kwargs.pop("metadata_buffer_size", None)
            return original_create(*create_args, **create_kwargs)

        create_compat._robosyn_metadata_compat = True
        dataset_class.create = create_compat
        installed.append("create.metadata_buffer_size")

    if (
        "task" in inspect.signature(dataset_class.add_frame).parameters
        and not getattr(dataset_class.add_frame, "_robosyn_task_compat", False)
    ):
        original_add_frame = dataset_class.add_frame

        def add_frame_compat(self: Any, frame: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
            copied_frame = dict(frame)
            task = copied_frame.pop("task", "click the bell")
            return original_add_frame(self, copied_frame, str(task), *args, **kwargs)

        add_frame_compat._robosyn_task_compat = True
        dataset_class.add_frame = add_frame_compat
        installed.append("add_frame.task")

    if not hasattr(dataset_class, "finalize"):
        def finalize_compat(self: Any) -> None:
            # EmbodiChain stops the image writer immediately before this call;
            # LeRobot 0.3.3 persists metadata in save_episode itself.
            return None

        dataset_class.finalize = finalize_compat
        installed.append("finalize")

    return installed


def _write_manifest(recorder: Any, manifest: dict[str, Any], counts: dict[str, int]) -> Path:
    manifest["saved_episodes"] = int(recorder.curr_episode)
    manifest["total_attempts"] = len(manifest["attempts"])
    manifest["region_counts"] = counts
    path = Path(recorder.dataset_path) / "policy_correction_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


def _collect(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import torch
    import tqdm

    import robosynchallenge  # noqa: F401
    import scripts.run_env as challenge_run_env  # noqa: F401
    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
    from embodichain.utils.logger import log_info, log_warning
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from policy.act.deploy_policy import eval as act_eval
    from policy.act.deploy_policy import get_model, reset_model
    from scripts.eval_policy import prepare_episode_reset

    legacy_recorder_compat = _install_lerobot_recorder_compat(LeRobotDataset)
    if not args.checkpoint:
        raise ValueError("--checkpoint is required")
    if not args.gym_config or not args.action_config:
        raise ValueError("--gym_config and --action_config are required")
    if args.num_envs not in (None, 1):
        raise ValueError("exactly one environment is supported")
    args.num_envs = 1
    args.max_episodes = args.episodes
    quotas = _quotas(args)

    def modify_config(config: dict) -> None:
        params = config["env"]["dataset"]["lerobot"]["params"]
        params["save_path"] = args.output_root
        params.setdefault("extra", {})["collection_type"] = "policy_conditioned_recovery"
        config["max_episodes"] = args.episodes

    env_cfg, gym_config, action_config = build_env_cfg_from_args(
        args, gym_config_modifier=modify_config
    )
    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    model = get_model(
        {
            "checkpoint_path": args.checkpoint,
            "device": "cuda",
            "act_step": 1,
            "n_action_steps": 50,
            "act_recovery_enabled": False,
            "act_scheduled_replan_enabled": False,
            "act_press_oracle_enabled": False,
        }
    )
    recorder = _get_recorder(env)
    counts = {key: 0 for key in quotas}
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "click_bell",
        "collection_type": "policy_conditioned_recovery",
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "requested_episodes": args.episodes,
        "region_quotas": quotas,
        "excluded_seeds": sorted(DEV20_SEEDS),
        "legacy_lerobot_recorder_compat": legacy_recorder_compat,
        "attempts": [],
    }

    try:
        progress = tqdm.tqdm(total=args.episodes, unit="episode", desc="Saved policy recoveries")
        for attempt_index, scene_seed in enumerate(_attempt_seeds(args.seed, args.max_attempts), 1):
            if int(recorder.curr_episode) >= args.episodes:
                break
            attempt: dict[str, Any] = {"attempt": attempt_index, "seed": scene_seed}
            prepare_episode_reset(env, scene_seed)
            obs, _ = env.reset(seed=scene_seed, options={"save_data": False})
            canonical = env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
            if canonical is None or len(canonical) == 0:
                attempt.update(saved=False, reason="canonical_planning_failed")
                manifest["attempts"].append(attempt)
                continue

            # Some planners query or update simulator-side caches. Replay the
            # exact seed so the policy always starts from the pristine scene.
            prepare_episode_reset(env, scene_seed)
            obs, _ = env.reset(seed=scene_seed, options={"save_data": False})
            base = env.unwrapped
            target = _target_details(base, canonical, args.target_plan_step)
            reset_model(model)
            candidates: list[Snapshot] = []
            policy_success = False
            steps = 0
            for steps in range(1, args.rollout_steps + 1):
                obs, _, truncated, _ = act_eval(env, model, obs)
                if _is_success(env):
                    policy_success = True
                    break
                snap = _snapshot(base, target, steps)
                if (
                    steps >= args.snapshot_min_step
                    and snap.max_joint_error_rad <= args.max_joint_error_rad
                ):
                    candidates.append(snap)
                if _is_done(truncated):
                    break

            diagnostics = base.get_episode_diagnostics()
            button = np.asarray(diagnostics["button_base_position_m"]).reshape(-1, 3)[0].tolist()
            region = _region(button, args)
            attempt.update(
                policy_success=policy_success,
                policy_steps=steps,
                button_base_position_m=button,
                region=region,
                eligible_snapshots=len(candidates),
                final_diagnostics={
                    "max_press_depth_m": _scalar(diagnostics["max_press_depth_m"]),
                    "movement_threshold_m": _scalar(diagnostics["movement_threshold_m"]),
                    "minimum_right_eef_to_button_m": _scalar(
                        diagnostics["minimum_right_eef_to_button_m"]
                    ),
                },
            )
            if policy_success:
                attempt.update(saved=False, reason="policy_succeeded")
                manifest["attempts"].append(attempt)
                continue
            if counts[region] >= quotas[region]:
                attempt.update(saved=False, reason="region_quota_filled")
                manifest["attempts"].append(attempt)
                continue
            if not candidates:
                attempt.update(saved=False, reason="no_near_contact_snapshot")
                manifest["attempts"].append(attempt)
                continue

            chosen = min(candidates, key=lambda item: item.rank)
            attempt["snapshot"] = chosen.to_dict()

            # Recreate the same scene to discard the policy rollout from the
            # recorder, then restore the policy-reached robot state without an
            # environment step.  The first saved label is therefore an oracle
            # recovery command, never a teleport or policy action.
            prepare_episode_reset(env, scene_seed)
            env.reset(seed=scene_seed, options={"save_data": False})
            canonical = env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
            if canonical is None or len(canonical) == 0:
                attempt.update(saved=False, reason="replay_planning_failed")
                manifest["attempts"].append(attempt)
                continue
            base = env.unwrapped
            target = _target_details(base, canonical, args.target_plan_step)
            qpos = torch.as_tensor(chosen.qpos, dtype=torch.float32, device=base.device)
            base.robot.set_qpos(qpos.unsqueeze(0), joint_ids=base.active_joint_ids, target=False)
            base.robot.set_qpos(qpos.unsqueeze(0), joint_ids=base.active_joint_ids, target=True)
            base.current_rollout_step = 0

            for action in _recovery_actions(base, chosen, target, args):
                env.step(action.unsqueeze(0))
            recovery_success = _is_success(env)
            before = int(recorder.curr_episode)
            env.reset(options={"save_data": recovery_success})
            after = int(recorder.curr_episode)
            saved = after > before
            attempt.update(
                saved=saved,
                recovery_success=recovery_success,
                recorded_steps=args.recovery_steps + args.hold_steps,
            )
            manifest["attempts"].append(attempt)
            if saved:
                counts[region] += 1
                progress.update(1)
            else:
                log_warning(f"Oracle recovery failed for seed {scene_seed}")

            # Preserve a usable audit trail if a long collection job is
            # interrupted after any completed recovery attempt.
            _write_manifest(recorder, manifest, counts)

        progress.close()
        path = _write_manifest(recorder, manifest, counts)
        log_info(f"Policy correction manifest: {path}", color="green")
        if int(recorder.curr_episode) < args.episodes:
            raise RuntimeError(
                f"saved {recorder.curr_episode}/{args.episodes} recoveries after "
                f"{len(manifest['attempts'])} attempts"
            )
    finally:
        env.close()


def main() -> None:
    parser = _parser()
    preliminary, _ = parser.parse_known_args()
    if not preliminary.dry_run:
        from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

        add_env_launcher_args_to_parser(parser)
    args = parser.parse_args()
    quotas = _quotas(args)
    if args.dry_run:
        print(json.dumps({"quotas": quotas, "attempt_seeds": _attempt_seeds(args.seed, args.max_attempts)}, indent=2))
        return
    _collect(args)


if __name__ == "__main__":
    main()
