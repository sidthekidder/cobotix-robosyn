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
        "| Episode | Seed | Result | Steps | Max press | Threshold |",
        "| ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for episode in episodes:
        diagnostics = episode.get("diagnostics") or {}
        max_depth = diagnostics.get("max_press_depth_m")
        threshold = diagnostics.get("movement_threshold_m")
        max_text = f"{1000 * float(max_depth):.3f} mm" if max_depth is not None else "n/a"
        threshold_text = (
            f"{1000 * float(threshold):.3f} mm" if threshold is not None else "n/a"
        )
        result = "success" if episode["success"] else "fail"
        lines.append(
            f"| {episode['episode_index']} | {episode['seed']} | {result} | "
            f"{episode['action_steps']} | {max_text} | {threshold_text} |"
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
