#!/usr/bin/env python3
"""Validate and summarize a ClickBell keypoint collection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _distance(left: list[float], right: list[float]) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _camera_summary(rows: list[dict[str, Any]], camera: str, index: int) -> dict[str, Any]:
    legacy_visible = sum(row["keypoints"][index][2] > 0.5 for row in rows)
    result: dict[str, Any] = {
        "frames": len(rows),
        "legacy_mask_visible": legacy_visible,
    }
    geometry_rows = [row["geometry"][camera] for row in rows if camera in row.get("geometry", {})]
    if not geometry_rows:
        return result

    centroid_to_press = []
    tool_to_press = []
    for geometry in geometry_rows:
        mask = geometry["mask"]
        press = geometry["press_point"]
        tip = geometry["right_tool_tip"]
        if mask["visible"] and press["in_frame"]:
            centroid_to_press.append(_distance(mask["centroid"], press["xy_normalized"]))
        if press["in_frame"] and tip["in_frame"]:
            tool_to_press.append(_distance(tip["xy_normalized"], press["xy_normalized"]))
    result.update(
        {
            "geometry_frames": len(geometry_rows),
            "mask_visible": sum(g["mask"]["visible"] for g in geometry_rows),
            "mask_touches_boundary": sum(
                g["mask"]["touches_image_boundary"] for g in geometry_rows
            ),
            "mask_centroid_confident": sum(
                g["mask_centroid_confidence"] > 0.5 for g in geometry_rows
            ),
            "press_point_in_frame": sum(
                g["press_point"]["in_frame"] for g in geometry_rows
            ),
            "press_point_visible_in_mask": sum(
                g["press_point_visible_in_mask"] for g in geometry_rows
            ),
            "right_tool_tip_in_frame": sum(
                g["right_tool_tip"]["in_frame"] for g in geometry_rows
            ),
            "median_centroid_to_press_normalized": _median(centroid_to_press),
            "median_tool_to_press_normalized": _median(tool_to_press),
        }
    )
    return result


def audit(dataset: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    manifest = json.loads((dataset / "bell_keypoint_manifest.json").read_text())
    info = json.loads((dataset / "meta" / "info.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset / "bell_keypoints.jsonl").read_text().splitlines()
        if line.strip()
    ]
    errors: list[str] = []

    keys = [(int(row["episode_index"]), int(row["frame_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("duplicate episode/frame labels")
    grouped: dict[int, list[int]] = defaultdict(list)
    for episode, frame in keys:
        grouped[episode].append(frame)
    for episode, frames in grouped.items():
        if sorted(frames) != list(range(len(frames))):
            errors.append(f"episode {episode} has non-contiguous frame labels")

    if len(rows) != int(info["total_frames"]):
        errors.append("sidecar row count does not match meta/info.json total_frames")
    if len(grouped) != int(info["total_episodes"]):
        errors.append("labeled episode count does not match meta/info.json")
    if int(manifest["labeled_frames"]) != len(rows):
        errors.append("manifest labeled_frames does not match sidecar")
    if int(manifest["saved_episodes"]) != len(grouped):
        errors.append("manifest saved_episodes does not match sidecar")

    cameras = list(manifest["cameras"])
    video_counts = {}
    for camera in cameras:
        videos = list(dataset.glob(f"videos/chunk-*/observation.images.{camera}/*.mp4"))
        video_counts[camera] = len(videos)
        if len(videos) != len(grouped):
            errors.append(f"{camera} has {len(videos)} videos for {len(grouped)} episodes")
        if any(video.stat().st_size == 0 for video in videos):
            errors.append(f"{camera} contains an empty video")

    saved_attempts = [attempt for attempt in manifest["attempts"] if attempt["saved"]]
    episode_cells = {
        episode: f"{attempt['cell'][0]},{attempt['cell'][1]}"
        for episode, attempt in enumerate(saved_attempts)
    }
    actual_cells = Counter(episode_cells.values())
    expected_cells = {key: int(value) for key, value in manifest["quotas"].items()}
    if dict(actual_cells) != expected_cells:
        errors.append(
            f"saved cell counts {dict(actual_cells)} do not match quotas {expected_cells}"
        )

    camera_summary = {
        camera: _camera_summary(rows, camera, index)
        for index, camera in enumerate(cameras)
    }
    per_cell = {}
    for cell in sorted(expected_cells):
        cell_rows = [row for row in rows if episode_cells[int(row["episode_index"])] == cell]
        per_cell[cell] = {
            "episodes": actual_cells[cell],
            "frames": len(cell_rows),
            "cameras": {
                camera: _camera_summary(cell_rows, camera, index)
                for index, camera in enumerate(cameras)
            },
        }

    return {
        "dataset": str(dataset),
        "schema_version": manifest.get("schema_version"),
        "episodes": len(grouped),
        "frames": len(rows),
        "attempts": len(manifest["attempts"]),
        "cell_counts": dict(sorted(actual_cells.items())),
        "video_counts": video_counts,
        "cameras": camera_summary,
        "per_cell": per_cell,
        "errors": errors,
        "valid": not errors,
    }


def main() -> None:
    args = _parser().parse_args()
    result = audit(args.dataset)
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
