#!/usr/bin/env python3
"""Build an ACT episode-sampling plan from a dataset coverage audit."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


EPISODE_PATTERN = re.compile(r"episode_(\d+)\.parquet$")


def episode_index(episode):
    match = EPISODE_PATTERN.search(episode["episode_file"])
    if match is None:
        raise ValueError(f"Cannot parse episode index: {episode['episode_file']}")
    return int(match.group(1))


def build_plan(report, *, hard_x, hard_y, hard_weight):
    episodes = report["episodes"]
    selected = []
    x_selected = []
    y_selected = []
    for episode in episodes:
        index = episode_index(episode)
        x, y, _ = episode["approach_proxy_position_m"]
        if x >= hard_x:
            x_selected.append(index)
        if y >= hard_y:
            y_selected.append(index)
        if x >= hard_x or y >= hard_y:
            selected.append(index)

    selected_set = set(selected)
    if len(selected_set) != len(selected):
        raise ValueError("The audit contains duplicate episode indices.")
    central_count = len(episodes) - len(selected)
    weighted_mass = hard_weight * len(selected)
    expected_hard_rate = weighted_mass / (central_count + weighted_mass)
    return {
        "schema_version": 1,
        "description": "Oversample click_bell hard-edge expert demonstrations.",
        "proxy_warning": (
            "Thresholds use the lowest active-arm end-effector position, not the "
            "unrecorded ground-truth bell pose."
        ),
        "thresholds": {"proxy_x_gte_m": hard_x, "proxy_y_gte_m": hard_y},
        "default_weight": 1.0,
        "hard_weight": hard_weight,
        "episode_count": len(episodes),
        "hard_episode_count": len(selected),
        "x_hard_episode_count": len(x_selected),
        "y_hard_episode_count": len(y_selected),
        "expected_hard_sample_rate": expected_hard_rate,
        "episode_weights": {str(index): hard_weight for index in sorted(selected)},
    }


def render_markdown(plan):
    thresholds = plan["thresholds"]
    return "\n".join(
        [
            "# ACT hard-edge sampling plan",
            "",
            f"- Episodes audited: {plan['episode_count']}",
            f"- Hard-edge episodes: {plan['hard_episode_count']}",
            f"- Proxy x >= {thresholds['proxy_x_gte_m']:.2f} m: "
            f"{plan['x_hard_episode_count']}",
            f"- Proxy y >= {thresholds['proxy_y_gte_m']:.2f} m: "
            f"{plan['y_hard_episode_count']}",
            f"- Hard-edge weight: {plan['hard_weight']:.1f}x",
            f"- Expected hard-edge sampling share: "
            f"{100 * plan['expected_hard_sample_rate']:.1f}%",
            "",
            plan["proxy_warning"],
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path)
    parser.add_argument("--hard-x", type=float, default=0.70)
    parser.add_argument("--hard-y", type=float, default=0.15)
    parser.add_argument("--hard-weight", type=float, default=3.0)
    args = parser.parse_args()

    if args.hard_weight <= 0:
        parser.error("--hard-weight must be positive")
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    plan = build_plan(
        report,
        hard_x=args.hard_x,
        hard_y=args.hard_y,
        hard_weight=args.hard_weight,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(plan)
    if args.output_markdown:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
