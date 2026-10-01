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
GEOMETRY_CAMERAS = ("cam_high", "cam_right_wrist")

# These offsets come from the pinned simulator assets. The button-cover visual
# mesh reaches z=0.02904 m in button_cover coordinates, and CobotMagic's
# right-arm OPW solver defines its TCP 0.143 m along right_link6 local +Z.
BUTTON_PRESS_SURFACE_OFFSET_M = 0.02904
RIGHT_TCP_OFFSET_M = 0.143


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=90)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--max-attempts", type=int, default=1200)
    parser.add_argument("--grid-x", type=int, default=3)
    parser.add_argument("--grid-y", type=int, default=3)
    parser.add_argument(
        "--exclude-cell",
        action="append",
        default=[],
        metavar="X,Y",
        help="Grid cell to exclude from balancing; may be repeated.",
    )
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


def _camera_mask(obs: dict[str, Any], camera: str) -> np.ndarray:
    mask = _as_numpy(obs["sensor"][camera]["mask"])[0]
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise ValueError(
            f"{camera} mask must be 2-D after channel selection, got {mask.shape}"
        )
    return mask


def _mask_measurements(mask: np.ndarray, visual_ids: np.ndarray) -> dict[str, Any]:
    """Measure the visible button mask without inferring physical geometry."""

    selected = np.isin(mask, visual_ids)
    pixels = np.argwhere(selected)
    height, width = mask.shape
    count = int(len(pixels))
    if count == 0:
        return {
            "centroid": [0.5, 0.5],
            "visible": False,
            "area_pixels": 0,
            "area_fraction": 0.0,
            "bbox_xyxy_pixels": None,
            "bbox_xyxy_normalized": None,
            "touches_image_boundary": False,
        }

    y, x = pixels.mean(axis=0)
    y_min, x_min = pixels.min(axis=0)
    y_max, x_max = pixels.max(axis=0)
    x_scale = max(width - 1, 1)
    y_scale = max(height - 1, 1)
    return {
        "centroid": [float(x / x_scale), float(y / y_scale)],
        "visible": True,
        "area_pixels": count,
        "area_fraction": float(count / mask.size),
        "bbox_xyxy_pixels": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "bbox_xyxy_normalized": [
            float(x_min / x_scale),
            float(y_min / y_scale),
            float(x_max / x_scale),
            float(y_max / y_scale),
        ],
        "touches_image_boundary": bool(
            x_min == 0 or y_min == 0 or x_max == width - 1 or y_max == height - 1
        ),
    }


def _project_world_point(
    camera_pose_opengl: np.ndarray,
    intrinsics: np.ndarray,
    point_world: np.ndarray,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Project one arena-frame point using DexSim's OpenGL camera convention."""

    camera_pose_opencv = np.asarray(camera_pose_opengl, dtype=np.float64).copy()
    camera_pose_opencv[:3, 1:3] *= -1.0
    point_h = np.append(np.asarray(point_world, dtype=np.float64), 1.0)
    point_camera = np.linalg.inv(camera_pose_opencv) @ point_h
    depth = float(point_camera[2])
    in_front = depth > 1e-8
    if in_front:
        pixel_h = np.asarray(intrinsics, dtype=np.float64) @ (
            point_camera[:3] / depth
        )
        x_pixel, y_pixel = float(pixel_h[0]), float(pixel_h[1])
    else:
        x_pixel = y_pixel = None

    height, width = image_shape
    in_frame = bool(
        in_front
        and x_pixel is not None
        and y_pixel is not None
        and 0.0 <= x_pixel <= width - 1
        and 0.0 <= y_pixel <= height - 1
    )
    return {
        "xy_normalized": [
            float(x_pixel / max(width - 1, 1)),
            float(y_pixel / max(height - 1, 1)),
        ] if in_front else None,
        "xy_pixels": [x_pixel, y_pixel],
        "depth_m": depth,
        "in_front": bool(in_front),
        "in_frame": in_frame,
    }


def _point_on_mask(
    mask: np.ndarray,
    visual_ids: np.ndarray,
    xy_pixels: list[float] | None,
    radius_pixels: int = 2,
) -> bool:
    if xy_pixels is None or not np.isfinite(xy_pixels).all():
        return False
    x, y = (int(round(value)) for value in xy_pixels)
    height, width = mask.shape
    x0, x1 = max(0, x - radius_pixels), min(width, x + radius_pixels + 1)
    y0, y1 = max(0, y - radius_pixels), min(height, y + radius_pixels + 1)
    return bool(
        x0 < x1
        and y0 < y1
        and np.isin(mask[y0:y1, x0:x1], visual_ids).any()
    )


def _transform_point(pose: np.ndarray, local_xyz: list[float]) -> np.ndarray:
    return (np.asarray(pose) @ np.asarray([*local_xyz, 1.0]))[:3]


def _physical_points(base: Any) -> dict[str, np.ndarray]:
    button = base.sim.get_articulation("button")
    cover_pose = _as_numpy(
        button.get_link_pose("button_cover", to_matrix=True)
    ).reshape(-1, 4, 4)[0]
    right_link6_pose = _as_numpy(
        base.robot.get_link_pose("right_link6", to_matrix=True)
    ).reshape(-1, 4, 4)[0]
    return {
        "press_point": _transform_point(
            cover_pose, [0.0, 0.0, BUTTON_PRESS_SURFACE_OFFSET_M]
        ),
        "right_tool_tip": _transform_point(
            right_link6_pose, [0.0, 0.0, RIGHT_TCP_OFFSET_M]
        ),
    }


def _frame_labels(
    base: Any, obs: dict[str, Any], visual_ids: np.ndarray
) -> tuple[list[list[float]], dict[str, Any]]:
    """Return legacy centroid labels and richer training-only geometry."""

    masks = {camera: _camera_mask(obs, camera) for camera in CAMERAS}
    mask_data = {
        camera: _mask_measurements(mask, visual_ids)
        for camera, mask in masks.items()
    }
    keypoints = [
        [*mask_data[camera]["centroid"], float(mask_data[camera]["visible"])]
        for camera in CAMERAS
    ]

    points = _physical_points(base)
    geometry: dict[str, Any] = {}
    for camera in GEOMETRY_CAMERAS:
        sensor = base.sim.get_sensor(camera)
        if sensor is None:
            raise RuntimeError(f"camera sensor {camera!r} is unavailable")
        camera_pose = _as_numpy(
            sensor.get_arena_pose(to_matrix=True)
        ).reshape(-1, 4, 4)[0]
        intrinsics = _as_numpy(sensor.get_intrinsics()).reshape(-1, 3, 3)[0]
        mask = masks[camera]
        projected = {
            name: _project_world_point(camera_pose, intrinsics, point, mask.shape)
            for name, point in points.items()
        }
        press_on_mask = _point_on_mask(
            mask, visual_ids, projected["press_point"]["xy_pixels"]
        )
        reliable = bool(
            mask_data[camera]["visible"]
            and not mask_data[camera]["touches_image_boundary"]
            and projected["press_point"]["in_frame"]
            and press_on_mask
        )
        press_xy = projected["press_point"]["xy_normalized"]
        tip_xy = projected["right_tool_tip"]["xy_normalized"]
        relative_xy = (
            [
                float(press_xy[0] - tip_xy[0]),
                float(press_xy[1] - tip_xy[1]),
            ]
            if press_xy is not None and tip_xy is not None
            else None
        )
        geometry[camera] = {
            "mask": mask_data[camera],
            "press_point": projected["press_point"],
            "right_tool_tip": projected["right_tool_tip"],
            "tool_to_press_delta_normalized": relative_xy,
            "press_point_visible_in_mask": press_on_mask,
            # Conservative: reject absent, truncated, off-screen, and
            # center-occluded centroid labels.
            "mask_centroid_confidence": float(reliable),
        }
    return keypoints, geometry


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


def _excluded_cells(args: argparse.Namespace) -> set[tuple[int, int]]:
    excluded: set[tuple[int, int]] = set()
    for value in args.exclude_cell:
        try:
            x_text, y_text = value.split(",", maxsplit=1)
            cell = (int(x_text), int(y_text))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid --exclude-cell {value!r}; expected X,Y") from error
        if not (0 <= cell[0] < args.grid_x and 0 <= cell[1] < args.grid_y):
            raise ValueError(
                f"excluded cell {cell} is outside {args.grid_x}x{args.grid_y} grid"
            )
        excluded.add(cell)
    return excluded


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
    excluded_cells = _excluded_cells(args)
    all_cells = [(x, y) for x in range(args.grid_x) for y in range(args.grid_y)]
    cells = [cell for cell in all_cells if cell not in excluded_cells]
    if not cells:
        raise ValueError("at least one grid cell must remain after exclusions")
    if args.episodes < len(cells):
        raise ValueError("episodes must be at least the number of included grid cells")
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
    base_quota, extra = divmod(args.episodes, len(cells))
    quotas = {
        cell: base_quota + (index < extra) for index, cell in enumerate(cells)
    }
    counts = {cell: 0 for cell in all_cells}
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
                _as_numpy(
                    base.sim.get_articulation("button").get_link_pose(
                        "button_base", to_matrix=True
                    )[:, :3, 3]
                )
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
            if cell in excluded_cells:
                record.update(saved=False, reason="excluded_cell")
                attempts.append(record)
                continue
            if counts[cell] >= quotas[cell]:
                record.update(saved=False, reason="cell_quota_filled")
                attempts.append(record)
                continue

            actions = env.get_wrapper_attr("create_demo_action_list")(action_sentence=0)
            if actions is None or len(actions) == 0:
                record.update(saved=False, reason="expert_planning_failed")
                attempts.append(record)
                continue
            visual_ids = _button_visual_ids(base)
            frame_labels: list[tuple[list[list[float]], dict[str, Any]]] = []
            for action in actions:
                obs, *_ = env.step(action)
                frame_labels.append(_frame_labels(base, obs, visual_ids))
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
            for frame_index, (keypoints, geometry) in enumerate(frame_labels):
                sidecar_rows.append(
                    {
                        "episode_index": episode_index,
                        "frame_index": frame_index,
                        "keypoints": keypoints,
                        "geometry": geometry,
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
            "schema_version": 2,
            "task": "click_bell",
            "collection_type": "bell_keypoint_balanced",
            "cameras": list(CAMERAS),
            "coordinate_order": ["x_normalized", "y_normalized", "visible"],
            "geometry_cameras": list(GEOMETRY_CAMERAS),
            "geometry": {
                "camera_pose_convention": "OpenGL converted to OpenCV for projection",
                "press_point": {
                    "link": "button_cover",
                    "local_xyz_m": [0.0, 0.0, BUTTON_PRESS_SURFACE_OFFSET_M],
                },
                "right_tool_tip": {
                    "link": "right_link6",
                    "local_xyz_m": [0.0, 0.0, RIGHT_TCP_OFFSET_M],
                },
                "mask_centroid_confidence": (
                    "1 iff mask is visible, does not touch an image boundary, "
                    "the projected press point is in-frame, and button-mask pixels "
                    "occur within 2 pixels of it; otherwise 0"
                ),
            },
            "legacy_lerobot_recorder_compat": recorder_compat,
            "grid": {"x": args.grid_x, "y": args.grid_y},
            "excluded_cells": [list(cell) for cell in sorted(excluded_cells)],
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
