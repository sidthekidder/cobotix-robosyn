#!/usr/bin/env python3
"""Collect pose-balanced ClickBell demos with simulator-only keypoint labels.

Mask labels are written to a JSONL sidecar and are consumed only by the ACT
training loss. They never become policy inputs or submission-time metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CAMERAS = ("cam_high", "cam_right_wrist", "cam_left_wrist")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-attempts", type=int, default=1200)
    parser.add_argument("--grid-x", type=int, default=3)
    parser.add_argument("--grid-y", type=int, default=3)
    parser.add_argument("--x-range", nargs=2, type=float, default=(0.4, 0.85))
    parser.add_argument("--y-range", nargs=2, type=float, default=(-0.3, 0.3))
    parser.add_argument(
        "--output-root", default="lerobot_dataset/click_bell_keypoint_balanced"
    )
    return parser


def _as_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value)


def _get_recorder(env: Any) -> Any:
    manager = env.unwrapped.dataset_manager
    for mode_cfgs in manager._mode_functor_cfgs.values():
        for functor_cfg in mode_cfgs:
            recorder = getattr(functor_cfg, "func", None)
            if hasattr(recorder, "curr_episode") and hasattr(recorder, "dataset_path"):
                return recorder
    raise RuntimeError("No LeRobot recorder is configured")


def _button_visual_ids(base: Any) -> np.ndarray:
    values = _as_numpy(base.sim.get_asset("button").get_user_ids()).reshape(-1)
    if values.size == 0:
        raise RuntimeError("button articulation has no visual IDs")
    return values.astype(np.int64)


def _keypoints(obs: dict[str, Any], visual_ids: np.ndarray) -> list[list[float]]:
    result: list[list[float]] = []
    for camera in CAMERAS:
        mask = _as_numpy(obs["sensor"][camera]["mask"])[0]
        if mask.ndim == 3:
            mask = mask[..., 0]
        pixels = np.argwhere(np.isin(mask, visual_ids))
        if len(pixels) == 0:
            result.append([0.5, 0.5, 0.0])
            continue
        height, width = mask.shape
        y, x = pixels.mean(axis=0)
        result.append(
            [float(x / max(width - 1, 1)), float(y / max(height - 1, 1)), 1.0]
        )
    return result


def _cell(position: list[float], args: argparse.Namespace) -> tuple[int, int]:
    x = np.clip(
        (position[0] - args.x_range[0]) / (args.x_range[1] - args.x_range[0]),
        0.0,
        1.0 - np.finfo(np.float64).eps,
    )
    y = np.clip(
        (position[1] - args.y_range[0]) / (args.y_range[1] - args.y_range[0]),
        0.0,
        1.0 - np.finfo(np.float64).eps,
    )
    return int(x * args.grid_x), int(y * args.grid_y)


def _is_success(env: Any) -> bool:
    value = env.get_wrapper_attr("is_task_success")()
    return bool(_as_numpy(value).any())


def _collect(args: argparse.Namespace) -> None:
    import gymnasium as gym
    import tqdm

    import robosynchallenge  # noqa: F401
    import scripts.run_env as challenge_run_env  # noqa: F401
    from embodichain.lab.gym.utils.gym_utils import build_env_cfg_from_args
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from scripts.collect_click_bell_policy_corrections import (
        _install_lerobot_recorder_compat,
    )

    recorder_compat = _install_lerobot_recorder_compat(LeRobotDataset)

    if not args.gym_config or not args.action_config:
        raise ValueError("--gym_config and --action_config are required")
    if args.num_envs not in (None, 1):
        raise ValueError("exactly one environment is supported")
    if args.episodes < args.grid_x * args.grid_y:
        raise ValueError("episodes must be at least the number of grid cells")
    args.num_envs = 1
    args.max_episodes = args.episodes

    def modify_config(config: dict[str, Any]) -> None:
        params = config["env"]["dataset"]["lerobot"]["params"]
        params["save_path"] = args.output_root
        params.setdefault("extra", {})["collection_type"] = "bell_keypoint_balanced"
        config["max_episodes"] = args.episodes
        for sensor in config.get("sensor", []):
            if sensor.get("sensor_type") in ("Camera", "StereoCamera"):
                sensor["enable_mask"] = True

    env_cfg, gym_config, action_config = build_env_cfg_from_args(
        args, gym_config_modifier=modify_config
    )
    env = gym.make(id=gym_config["id"], cfg=env_cfg, **action_config)
    recorder = _get_recorder(env)
    cells = [(x, y) for x in range(args.grid_x) for y in range(args.grid_y)]
    base_quota, extra = divmod(args.episodes, len(cells))
    quotas = {
        cell: base_quota + (index < extra) for index, cell in enumerate(cells)
    }
    counts = {cell: 0 for cell in cells}
    rng = np.random.default_rng(args.seed)
    sidecar_rows: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    pending_scene: tuple[int, Any] | None = None

    try:
        progress = tqdm.tqdm(total=args.episodes, desc="Saved balanced demos", unit="episode")
        for attempt_index in range(1, args.max_attempts + 1):
            if int(recorder.curr_episode) >= args.episodes:
                break
            if pending_scene is None:
                scene_seed = int(rng.integers(0, 2**31 - 1))
                obs, _ = env.reset(seed=scene_seed, options={"save_data": False})
            else:
                # The recorder reset used to save the preceding episode also
                # initializes the next randomized scene. Reuse it instead of
                # resetting twice, which leaves that pose unchanged in DexSim.
                scene_seed, obs = pending_scene
                pending_scene = None
            base = env.unwrapped
            position = (
                _as_numpy(base.get_episode_diagnostics()["button_base_position_m"])
                .reshape(-1, 3)[0]
                .tolist()
            )
            cell = _cell(position, args)
            record = {
                "attempt": attempt_index,
                "seed": scene_seed,
                "button_base_position_m": position,
                "cell": list(cell),
            }
            if counts[cell] >= quotas[cell]:
                record.update(saved=False, reason="cell_quota_filled")
                attempts.append(record)
                continue

            actions = env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
            if actions is None or len(actions) == 0:
                record.update(saved=False, reason="expert_planning_failed")
                attempts.append(record)
                continue
            label = _keypoints(obs, _button_visual_ids(base))
            for action in actions:
                env.step(action)
            success = _is_success(env)
            episode_index = int(recorder.curr_episode)
            next_seed = int(rng.integers(0, 2**31 - 1))
            next_obs, _ = env.reset(seed=next_seed, options={"save_data": success})
            pending_scene = (next_seed, next_obs)
            saved = int(recorder.curr_episode) > episode_index
            record.update(saved=saved, success=success, frames=len(actions))
            attempts.append(record)
            if not saved:
                continue
            counts[cell] += 1
            for frame_index in range(len(actions)):
                sidecar_rows.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "keypoints": label,
                    }
                )
            progress.update(1)
        progress.close()

        dataset_path = Path(recorder.dataset_path)
        sidecar = dataset_path / "bell_keypoints.jsonl"
        with sidecar.open("w", encoding="utf-8") as handle:
            for row in sidecar_rows:
                handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        manifest = {
            "schema_version": 1,
            "task": "click_bell",
            "collection_type": "bell_keypoint_balanced",
            "cameras": list(CAMERAS),
            "coordinate_order": ["x_normalized", "y_normalized", "visible"],
            "legacy_lerobot_recorder_compat": recorder_compat,
            "grid": {"x": args.grid_x, "y": args.grid_y},
            "ranges_m": {"x": args.x_range, "y": args.y_range},
            "quotas": {f"{x},{y}": value for (x, y), value in quotas.items()},
            "counts": {f"{x},{y}": value for (x, y), value in counts.items()},
            "saved_episodes": int(recorder.curr_episode),
            "labeled_frames": len(sidecar_rows),
            "attempts": attempts,
        }
        (dataset_path / "bell_keypoint_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        if int(recorder.curr_episode) < args.episodes:
            raise RuntimeError(
                f"saved {recorder.curr_episode}/{args.episodes} episodes after "
                f"{args.max_attempts} attempts"
            )
        print(f"Keypoint sidecar: {sidecar}")
    finally:
        env.close()


def main() -> None:
    parser = _parser()
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser

    add_env_launcher_args_to_parser(parser)
    _collect(parser.parse_args())


if __name__ == "__main__":
    main()
