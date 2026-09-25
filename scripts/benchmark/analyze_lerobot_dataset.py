#!/usr/bin/env python3
"""Measure arm-use and spatial coverage in a RoboSyn LeRobot dataset.

The released click_bell parquet files do not store the sampled bell pose. This
script therefore reports arm motion directly and labels the lowest active-arm
end-effector position as an approach proxy, never as the true bell position.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np


def resolve_dataset(args) -> Path:
    if args.dataset:
        return args.dataset.expanduser().resolve()
    # Many small parquet files can otherwise trigger one Xet token request per
    # worker. Direct HTTP plus low concurrency is slower but reliably resumable.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required with --repo-id; pass --dataset instead."
        ) from exc
    return Path(
        snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            revision=args.revision,
            allow_patterns=["data/**/*.parquet", "meta/*"],
            max_workers=args.max_workers,
        )
    )


def load_columns(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required. Install it in the analysis environment first."
        ) from exc
    return pq.read_table(
        path,
        columns=["observation.qpos", "left_ee_pose", "right_ee_pose"],
    ).to_pydict()


def path_length(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())


def classify_arm(left_path, right_path, left_delta, right_delta, args):
    left_active = left_path >= args.path_threshold or left_delta >= args.joint_threshold
    right_active = right_path >= args.path_threshold or right_delta >= args.joint_threshold
    if left_active and right_active:
        return "both"
    if left_active:
        return "left"
    if right_active:
        return "right"
    return "neither"


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(values)),
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def analyze(dataset: Path, args):
    parquet_files = sorted(dataset.glob("data/**/*.parquet"))
    if not parquet_files:
        parquet_files = sorted(dataset.glob("**/*.parquet"))
    if not parquet_files:
        raise SystemExit(f"No parquet episode files found under {dataset}")
    if args.expected_episodes is not None and len(parquet_files) != args.expected_episodes:
        raise SystemExit(
            f"Expected {args.expected_episodes} parquet episodes, found "
            f"{len(parquet_files)} under {dataset}. The download is incomplete; "
            "rerun after the Hugging Face rate limit clears or authenticate with HF_TOKEN."
        )

    episodes = []
    for index, path in enumerate(parquet_files):
        columns = load_columns(path)
        qpos = np.asarray(columns["observation.qpos"], dtype=float)
        left_pose = np.asarray(columns["left_ee_pose"], dtype=float)
        right_pose = np.asarray(columns["right_ee_pose"], dtype=float)
        left_positions = left_pose[:, :3, 3]
        right_positions = right_pose[:, :3, 3]

        left_path = path_length(left_positions)
        right_path = path_length(right_positions)
        left_delta = float(np.max(np.abs(qpos[:, args.left_joints] - qpos[0, args.left_joints])))
        right_delta = float(np.max(np.abs(qpos[:, args.right_joints] - qpos[0, args.right_joints])))
        active_arm = classify_arm(left_path, right_path, left_delta, right_delta, args)

        if active_arm == "left" or (active_arm == "both" and left_path > right_path):
            approach_arm = "left"
            approach_position = left_positions[np.argmin(left_positions[:, 2])]
        elif active_arm in {"right", "both"}:
            approach_arm = "right"
            approach_position = right_positions[np.argmin(right_positions[:, 2])]
        else:
            approach_arm = None
            approach_position = np.array([np.nan, np.nan, np.nan])

        episodes.append(
            {
                "episode_file": str(path.relative_to(dataset)),
                "frames": int(len(qpos)),
                "active_arm": active_arm,
                "left_eef_path_length_m": left_path,
                "right_eef_path_length_m": right_path,
                "max_left_arm_joint_delta_rad": left_delta,
                "max_right_arm_joint_delta_rad": right_delta,
                "approach_proxy_arm": approach_arm,
                "approach_proxy_position_m": approach_position.tolist(),
            }
        )
        if args.progress and (index + 1) % 100 == 0:
            print(f"Analyzed {index + 1}/{len(parquet_files)} episodes", flush=True)

    counts = Counter(episode["active_arm"] for episode in episodes)
    proxy_positions = np.asarray(
        [
            episode["approach_proxy_position_m"]
            for episode in episodes
            if episode["approach_proxy_arm"] is not None
        ],
        dtype=float,
    )
    summary = {
        "episode_count": len(episodes),
        "arm_use_counts": {
            arm: counts.get(arm, 0) for arm in ("left", "right", "both", "neither")
        },
        "arm_use_rates": {
            arm: counts.get(arm, 0) / len(episodes)
            for arm in ("left", "right", "both", "neither")
        },
        "left_eef_path_length_m": summarize_values(
            [episode["left_eef_path_length_m"] for episode in episodes]
        ),
        "right_eef_path_length_m": summarize_values(
            [episode["right_eef_path_length_m"] for episode in episodes]
        ),
        "approach_proxy_position_m": {
            axis: summarize_values(proxy_positions[:, axis_index])
            for axis_index, axis in enumerate(("x", "y", "z"))
        },
        "bell_position_available": False,
        "bell_position_note": (
            "The released parquet schema contains robot state/action and end-effector "
            "poses, but no bell pose. approach_proxy_position_m is the lowest active-arm "
            "end-effector position and must not be treated as ground-truth bell location."
        ),
    }
    return {"dataset": str(dataset), "summary": summary, "episodes": episodes}


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    count = summary["episode_count"]
    lines = [
        "# RoboSyn dataset coverage",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Episodes: {count}",
        "- Bell pose stored in parquet: no",
        "",
        "## Arm use",
        "",
        "| Classification | Episodes | Rate |",
        "| --- | ---: | ---: |",
    ]
    for arm in ("left", "right", "both", "neither"):
        lines.append(
            f"| {arm} | {summary['arm_use_counts'][arm]} | "
            f"{100 * summary['arm_use_rates'][arm]:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Lowest end-effector approach proxy",
            "",
            "| Axis | Min | 5th percentile | Median | 95th percentile | Max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for axis in ("x", "y", "z"):
        values = summary["approach_proxy_position_m"][axis]
        lines.append(
            f"| {axis} | {values['min']:.3f} | {values['p05']:.3f} | "
            f"{values['median']:.3f} | {values['p95']:.3f} | {values['max']:.3f} |"
        )
    lines.extend(["", summary["bell_position_note"], ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", type=Path, help="Local LeRobot dataset root")
    source.add_argument(
        "--repo-id",
        help="Hugging Face dataset repo; downloads parquet and metadata only",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Concurrent Hugging Face downloads (low by default to avoid rate limits)",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--left-joints", type=int, nargs="+", default=list(range(0, 6)))
    parser.add_argument("--right-joints", type=int, nargs="+", default=list(range(7, 13)))
    parser.add_argument("--path-threshold", type=float, default=0.02)
    parser.add_argument("--joint-threshold", type=float, default=0.05)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--expected-episodes",
        type=int,
        help="Fail rather than silently analyze an incomplete download",
    )
    args = parser.parse_args()

    dataset = resolve_dataset(args)
    report = analyze(dataset, args)
    markdown = render_markdown(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
