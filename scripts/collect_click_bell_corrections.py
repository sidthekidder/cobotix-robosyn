#!/usr/bin/env python3
"""Collect clean ClickBell recovery demonstrations from controlled perturbations.

The robot is teleported to a perturbed near-contact pose before recording starts.
Only the expert recovery actions and their resulting observations enter LeRobot;
the teleport itself is never used as a training label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

# Direct execution (``python scripts/...``) otherwise exposes only ``scripts/``
# on sys.path when the challenge package has not been installed yet.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from click_bell_correction_plan import (
    CorrectionSpec,
    build_correction_specs,
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--anchor-min", type=int, default=34)
    parser.add_argument("--anchor-max", type=int, default=44)
    parser.add_argument("--target-step", type=int, default=49)
    parser.add_argument("--recovery-steps", type=int, default=12)
    parser.add_argument("--lateral-mm-min", type=float, default=6.0)
    parser.add_argument("--lateral-mm-max", type=float, default=12.0)
    parser.add_argument("--vertical-mm-min", type=float, default=3.0)
    parser.add_argument("--vertical-mm-max", type=float, default=7.0)
    parser.add_argument(
        "--output-root",
        default="lerobot_dataset/click_bell_corrections",
        help="Parent directory for the auto-numbered LeRobot dataset.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=500,
        help="Stop if this many randomized scenes do not yield the requested successes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deterministic perturbation plan without importing the simulator.",
    )
    return parser


def _specs_from_args(args: argparse.Namespace) -> list[CorrectionSpec]:
    return build_correction_specs(
        args.episodes,
        args.seed,
        anchor_min=args.anchor_min,
        anchor_max=args.anchor_max,
        target_step=args.target_step,
        recovery_steps=args.recovery_steps,
        lateral_mm=(args.lateral_mm_min, args.lateral_mm_max),
        vertical_mm=(args.vertical_mm_min, args.vertical_mm_max),
    )


def _get_recorder(env: Any) -> Any:
    manager = env.unwrapped.dataset_manager
    for mode_cfgs in manager._mode_functor_cfgs.values():
        for functor_cfg in mode_cfgs:
            recorder = getattr(functor_cfg, "func", None)
            if hasattr(recorder, "curr_episode") and hasattr(recorder, "dataset_path"):
                return recorder
    raise RuntimeError("No LeRobot recorder is configured")


def _active_indices(env: Any, global_joint_ids: list[int]) -> list[int]:
    active = [int(joint_id) for joint_id in env.unwrapped.active_joint_ids]
    lookup = {joint_id: i for i, joint_id in enumerate(active)}
    missing = [joint_id for joint_id in global_joint_ids if joint_id not in lookup]
    if missing:
        raise RuntimeError(f"right-arm joints are absent from the action space: {missing}")
    return [lookup[joint_id] for joint_id in global_joint_ids]


def _build_recovery(env: Any, canonical: Any, spec: CorrectionSpec) -> tuple[Any, dict]:
    """Teleport to a perturbed anchor and return recovery-only action labels."""

    import torch

    base = env.unwrapped
    if canonical.ndim != 3 or canonical.shape[1] != 1:
        raise RuntimeError(f"expected actions [steps, 1, joints], got {canonical.shape}")
    if spec.target_step >= canonical.shape[0]:
        raise RuntimeError(
            f"target step {spec.target_step} exceeds {canonical.shape[0]} canonical actions"
        )

    right_global = [
        int(joint_id)
        for joint_id in base.robot.get_joint_ids(name="right_arm", remove_mimic=True)
    ]
    right_active = _active_indices(env, right_global)
    anchor_active = canonical[spec.anchor_step, 0].to(base.device).clone()
    anchor_right = anchor_active[right_active].unsqueeze(0)

    anchor_pose = base.robot.compute_fk(anchor_right, "right_arm", to_matrix=True)
    perturbed_pose = anchor_pose.clone()
    perturbed_pose[:, :3, 3] += torch.as_tensor(
        spec.offset_m, dtype=perturbed_pose.dtype, device=perturbed_pose.device
    )
    ik_result = base.robot.compute_ik(
        pose=perturbed_pose, joint_seed=anchor_right, name="right_arm"
    )
    if ik_result is None:
        raise RuntimeError("right-arm IK solver is unavailable")
    ik_ok, perturbed_right = ik_result
    if not bool(ik_ok.all().item()):
        raise ValueError("IK failed for sampled correction")

    perturbed_active = anchor_active.clone()
    perturbed_active[right_active] = perturbed_right[0]

    # Move both physical state and controller target. No env.step occurs here,
    # therefore this artificial setup transition never reaches the recorder.
    base.robot.set_qpos(
        perturbed_active.unsqueeze(0), joint_ids=base.active_joint_ids, target=False
    )
    base.robot.set_qpos(
        perturbed_active.unsqueeze(0), joint_ids=base.active_joint_ids, target=True
    )
    base.current_rollout_step = 0

    target = canonical[spec.target_step, 0].to(base.device)
    alpha = torch.linspace(
        1.0 / spec.recovery_steps,
        1.0,
        spec.recovery_steps,
        dtype=target.dtype,
        device=target.device,
    ).unsqueeze(1)
    recovery = perturbed_active.unsqueeze(0) + alpha * (
        target - perturbed_active
    ).unsqueeze(0)
    suffix = canonical[spec.target_step + 1 :, 0].to(base.device)
    actions = torch.cat([recovery, suffix], dim=0).unsqueeze(1)

    details = spec.to_dict()
    details.update(
        {
            "canonical_steps": int(canonical.shape[0]),
            "recorded_steps": int(actions.shape[0]),
            "anchor_pose_m": anchor_pose[0, :3, 3].detach().cpu().tolist(),
            "perturbed_pose_m": perturbed_pose[0, :3, 3].detach().cpu().tolist(),
        }
    )
    return actions, details


def _collect(args: argparse.Namespace, specs: list[CorrectionSpec]) -> None:
    import gymnasium as gym
    import torch
    import tqdm

    import robosynchallenge  # noqa: F401 - registers challenge environments
    import scripts.run_env as challenge_run_env  # noqa: F401 - registers managers
    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
    from embodichain.utils.logger import log_info, log_warning

    if not args.gym_config or not args.action_config:
        raise ValueError("--gym_config and --action_config are required unless --dry-run")
    if args.num_envs not in (None, 1):
        raise ValueError("corrective collection currently supports exactly one environment")
    args.num_envs = 1
    args.max_episodes = args.episodes

    def modify_config(config: dict) -> None:
        params = config["env"]["dataset"]["lerobot"]["params"]
        params["save_path"] = args.output_root
        params.setdefault("extra", {})["collection_type"] = "near_contact_correction"
        config["max_episodes"] = args.episodes

    env_cfg, gym_config, action_config = build_env_cfg_from_args(
        args, gym_config_modifier=modify_config
    )
    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "task": "click_bell",
        "seed": args.seed,
        "requested_episodes": args.episodes,
        "source_gym_config": str(args.gym_config),
        "source_action_config": str(args.action_config),
        "attempts": [],
    }

    try:
        recorder = _get_recorder(env)
        env.reset(options={"save_data": False})
        saved = int(recorder.curr_episode)
        attempts = 0
        progress = tqdm.tqdm(total=args.episodes, desc="Saved corrections", unit="episode")

        while saved < args.episodes and attempts < args.max_attempts:
            attempts += 1
            spec = specs[saved]
            attempt: dict[str, Any] = {"attempt": attempts, **spec.to_dict()}
            canonical = env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
            if canonical is None or len(canonical) == 0:
                attempt.update(saved=False, reason="canonical_planning_failed")
                manifest["attempts"].append(attempt)
                env.reset(options={"save_data": False})
                continue

            try:
                actions, details = _build_recovery(env, canonical, spec)
                attempt.update(details)
            except ValueError as error:
                attempt.update(saved=False, reason=str(error))
                manifest["attempts"].append(attempt)
                env.reset(options={"save_data": False})
                continue

            for action in actions:
                env.step(action)

            success = bool(env.get_wrapper_attr("is_task_success")().all().item())
            before = int(recorder.curr_episode)
            env.reset(options={"save_data": success})
            after = int(recorder.curr_episode)
            was_saved = after > before
            attempt.update(saved=was_saved, success=success)
            manifest["attempts"].append(attempt)

            if was_saved:
                saved = after
                progress.update(1)
            else:
                log_warning(
                    f"Correction attempt {attempts} failed; retained {saved}/{args.episodes}."
                )

        progress.close()
        manifest["saved_episodes"] = saved
        manifest["total_attempts"] = attempts
        dataset_path = Path(recorder.dataset_path)
        manifest_path = dataset_path / "correction_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        log_info(f"Correction manifest: {manifest_path}", color="green")
        if saved < args.episodes:
            raise RuntimeError(
                f"saved {saved}/{args.episodes} corrections after {attempts} attempts"
            )
    finally:
        env.close()


def main() -> None:
    parser = _create_parser()

    # Simulator launcher arguments are imported only for actual collection, so
    # --dry-run remains usable on a Mac without DexSim/CUDA installed.
    preliminary, _ = parser.parse_known_args()
    if not preliminary.dry_run:
        from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

        add_env_launcher_args_to_parser(parser)

    args = parser.parse_args()
    specs = _specs_from_args(args)
    if args.dry_run:
        print(json.dumps([spec.to_dict() for spec in specs], indent=2))
        return
    _collect(args, specs)


if __name__ == "__main__":
    main()
