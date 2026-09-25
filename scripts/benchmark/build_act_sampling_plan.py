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


def build_plan(
    report,
    *,
    hard_x,
    hard_y,
    hard_weight,
    phase_start_frame=None,
    phase_end_frame=None,
    phase_weight=None,
    combine="multiply",
):
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
    frame_ranges = []
    if phase_start_frame is not None:
        frame_ranges.append(
            {
                "start_frame": phase_start_frame,
                "end_frame": phase_end_frame,
                "weight": phase_weight,
            }
        )

    total_mass = 0.0
    hard_mass = 0.0
    phase_mass = 0.0
    for episode in episodes:
        index = episode_index(episode)
        episode_weight = hard_weight if index in selected_set else 1.0
        for frame_index in range(int(episode["frames"])):
            in_phase = (
                phase_start_frame is not None
                and phase_start_frame <= frame_index <= phase_end_frame
            )
            weight = episode_weight
            if in_phase:
                weight = (
                    weight * phase_weight
                    if combine == "multiply"
                    else max(weight, phase_weight)
                )
            total_mass += weight
            if index in selected_set:
                hard_mass += weight
            if in_phase:
                phase_mass += weight

    plan = {
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
        "expected_hard_sample_rate": hard_mass / total_mass,
        "episode_weights": {str(index): hard_weight for index in sorted(selected)},
        "combine": combine,
    }
    if frame_ranges:
        plan.update(
            {
                "frame_ranges": frame_ranges,
                "phase_frame_range": [phase_start_frame, phase_end_frame],
                "phase_weight": phase_weight,
                "expected_phase_sample_rate": phase_mass / total_mass,
            }
        )
    return plan


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
            *(
                [
                    f"- Phase anchor frames: {plan['phase_frame_range'][0]}–"
                    f"{plan['phase_frame_range'][1]}",
                    f"- Phase weight: {plan['phase_weight']:.1f}x",
                    f"- Expected phase sampling share: "
                    f"{100 * plan['expected_phase_sample_rate']:.1f}%",
                    f"- Episode/frame weight combination: {plan['combine']}",
                ]
                if "phase_frame_range" in plan
                else []
            ),
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
    parser.add_argument("--phase-start-frame", type=int)
    parser.add_argument("--phase-end-frame", type=int)
    parser.add_argument("--phase-weight", type=float)
    parser.add_argument("--combine", choices=("multiply", "max"), default="multiply")
    args = parser.parse_args()

    if args.hard_weight <= 0:
        parser.error("--hard-weight must be positive")
    phase_values = (
        args.phase_start_frame,
        args.phase_end_frame,
        args.phase_weight,
    )
    if any(value is not None for value in phase_values):
        if any(value is None for value in phase_values):
            parser.error(
                "--phase-start-frame, --phase-end-frame, and --phase-weight "
                "must be supplied together"
            )
        if (
            args.phase_start_frame < 0
            or args.phase_end_frame < args.phase_start_frame
            or args.phase_weight <= 0
        ):
            parser.error("Invalid phase frame range or weight")
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    plan = build_plan(
        report,
        hard_x=args.hard_x,
        hard_y=args.hard_y,
        hard_weight=args.hard_weight,
        phase_start_frame=args.phase_start_frame,
        phase_end_frame=args.phase_end_frame,
        phase_weight=args.phase_weight,
        combine=args.combine,
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
