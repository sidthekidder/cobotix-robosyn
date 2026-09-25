#!/usr/bin/env python3
"""Summarize a RoboSyn evaluation metrics file for quick comparison."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054):
    if episodes == 0:
        return None, None
    p = successes / episodes
    denominator = 1 + z * z / episodes
    center = (p + z * z / (2 * episodes)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * episodes)) / episodes) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def find_metrics(path: Path) -> Path:
    if path.is_file():
        return path
    matches = sorted(path.rglob("evaluation_metrics.json"))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one evaluation_metrics.json under {path}, found {len(matches)}."
        )
    return matches[0]


def fmt_percent(value: float) -> str:
    return f"{100 * value:.1f}%"


def fmt_millimeters(value) -> str:
    return f"{1000 * float(value):.1f} mm" if value is not None else "n/a"


def fmt_meters(value) -> str:
    return f"{float(value):.3f} m" if value is not None else "n/a"


def infer_active_arm(diagnostics: dict) -> str:
    """Describe arm use from raw motion metrics without hiding their values."""
    left_path = float(diagnostics.get("left_eef_path_length_m", 0.0))
    right_path = float(diagnostics.get("right_eef_path_length_m", 0.0))
    left_delta = float(diagnostics.get("max_left_arm_joint_delta_rad", 0.0))
    right_delta = float(diagnostics.get("max_right_arm_joint_delta_rad", 0.0))
    left_active = left_path >= 0.02 or left_delta >= 0.05
    right_active = right_path >= 0.02 or right_delta >= 0.05
    if left_active and right_active:
        return "both"
    if left_active:
        return "left"
    if right_active:
        return "right"
    return "neither"


def build_summary(metrics: dict, metrics_path: Path) -> str:
    summary = metrics["summary"]
    episodes = metrics.get("episodes", [])
    count = int(summary["episode_count"])
    successes = int(summary["success_count"])
    low, high = wilson_interval(successes, count)
    lines = [
        "# RoboSyn benchmark summary",
        "",
        f"- Metrics: `{metrics_path}`",
        f"- Policy: `{metrics['config']['policy']}`",
        f"- Task: `{metrics['config']['task']}` (`{metrics['config']['setting']}`)",
        f"- Episodes: {count}",
        f"- Successes: {successes}/{count} ({fmt_percent(summary['success_rate'])})",
        f"- 95% Wilson interval: {fmt_percent(low)}–{fmt_percent(high)}",
        f"- Mean action steps: {summary['average_action_steps']:.2f}",
        f"- Mean inference latency: {1000 * summary['average_inference_time_seconds']:.2f} ms",
        "",
        "## Episodes",
        "",
        "| Episode | Seed | Result | Bell x/y | Arm | Closest L/R | Path L/R | Max press |",
        "| ---: | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    arm_counts = {"left": 0, "right": 0, "both": 0, "neither": 0}
    for episode in episodes:
        diagnostics = episode.get("diagnostics") or {}
        max_depth = diagnostics.get("max_press_depth_m")
        button_position = diagnostics.get("button_base_position_m")
        if isinstance(button_position, list) and len(button_position) >= 2:
            position_text = f"{float(button_position[0]):.3f}, {float(button_position[1]):.3f}"
        else:
            position_text = "n/a"
        arm = infer_active_arm(diagnostics)
        arm_counts[arm] += 1
        closest_text = (
            f"{fmt_millimeters(diagnostics.get('minimum_left_eef_to_button_m'))} / "
            f"{fmt_millimeters(diagnostics.get('minimum_right_eef_to_button_m'))}"
        )
        path_text = (
            f"{fmt_meters(diagnostics.get('left_eef_path_length_m'))} / "
            f"{fmt_meters(diagnostics.get('right_eef_path_length_m'))}"
        )
        result = "success" if episode["success"] else "fail"
        lines.append(
            f"| {episode['episode_index']} | {episode['seed']} | {result} | "
            f"{position_text} | {arm} | {closest_text} | {path_text} | "
            f"{fmt_millimeters(max_depth)} |"
        )
    if episodes:
        lines.extend(
            [
                "",
                "## Arm use",
                "",
                *[
                    f"- {arm}: {arm_counts[arm]}/{len(episodes)} "
                    f"({fmt_percent(arm_counts[arm] / len(episodes))})"
                    for arm in ("left", "right", "both", "neither")
                ],
                "",
                "An arm is classified as active when its end-effector path is at least "
                "2 cm or one of its arm joints moves at least 0.05 rad. The raw values "
                "remain in `evaluation_metrics.json`.",
            ]
        )
    feasibility = metrics.get("feasibility_filter") or {}
    if feasibility:
        lines.extend(
            [
                "",
                "## Feasibility filter",
                "",
                f"- Candidate seeds checked: {feasibility.get('candidate_seed_attempt_count', 0)}",
                f"- Accepted: {feasibility.get('accepted_episode_count', count)}",
                f"- Skipped: {feasibility.get('skipped_seed_count', 0)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Metrics JSON or a directory containing it")
    parser.add_argument("--output", type=Path, help="Write Markdown summary here")
    args = parser.parse_args()

    metrics_path = find_metrics(args.path.expanduser().resolve())
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rendered = build_summary(metrics, metrics_path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
