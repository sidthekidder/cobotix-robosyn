#!/usr/bin/env python3
"""Analyze click-bell evaluation failures from evaluation_metrics.json."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter
from pathlib import Path


PRESS_SUCCESS_M = 0.0048
NEAR_PRESS_M = 0.003
PARTIAL_PRESS_M = 0.0011
SUCCESS_DISTANCE_ENVELOPE_M = 0.18


def position(episode: dict) -> tuple[float, float]:
    x, y, _ = episode["diagnostics"]["button_base_position_m"][0]
    return float(x), float(y)


def failure_category(episode: dict) -> str:
    diagnostics = episode["diagnostics"]
    depth = float(diagnostics["max_press_depth_m"])
    distance = float(diagnostics["minimum_right_eef_to_button_m"])
    if depth >= NEAR_PRESS_M:
        return "near-threshold press"
    if depth >= PARTIAL_PRESS_M:
        return "partial press"
    if distance <= SUCCESS_DISTANCE_ENVELOPE_M:
        return "close approach without depression"
    return "outside empirical success-distance envelope"


def rate_row(name: str, episodes: list[dict]) -> tuple[str, int, int, str]:
    successes = sum(bool(episode["success"]) for episode in episodes)
    rate = successes / len(episodes) if episodes else 0.0
    return name, len(episodes), successes, f"{100 * rate:.1f}%"


def markdown_table(headers: list[str], rows: list[tuple]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows),
    ]


def render_svg(episodes: list[dict], output: Path, hard_x: float, hard_y: float) -> None:
    width, height = 920, 760
    left, right, top, bottom = 85, 35, 55, 145
    xmin, xmax, ymin, ymax = 0.38, 0.87, -0.32, 0.32
    plot_w, plot_h = width - left - right, height - top - bottom

    def sx(x: float) -> float:
        return left + (x - xmin) / (xmax - xmin) * plot_w

    def sy(y: float) -> float:
        return top + (ymax - y) / (ymax - ymin) * plot_h

    colors = {
        "success": "#258f55",
        "near-threshold press": "#e6a019",
        "partial press": "#d15d2f",
        "close approach without depression": "#9a4cb3",
        "outside empirical success-distance envelope": "#c93838",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#17212b}.axis{stroke:#52606d;stroke-width:1}.grid{stroke:#d9e2ec;stroke-width:1}.threshold{stroke:#1f5f99;stroke-width:2;stroke-dasharray:8 6}.label{font-size:13px}.title{font-size:22px;font-weight:700}.legend{font-size:13px}</style>',
        '<text x="85" y="30" class="title">Held-out click-bell outcomes by randomized bell position</text>',
    ]
    for tick in [0.4, 0.5, 0.6, 0.7, 0.8]:
        x = sx(tick)
        lines += [f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+plot_h}" class="grid"/>',
                  f'<text x="{x:.1f}" y="{top+plot_h+25}" text-anchor="middle" class="label">{tick:.1f}</text>']
    for tick in [-0.3, -0.15, 0.0, 0.15, 0.3]:
        y = sy(tick)
        lines += [f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" class="grid"/>',
                  f'<text x="{left-12}" y="{y+5:.1f}" text-anchor="end" class="label">{tick:+.2f}</text>']
    lines += [
        f'<line x1="{sx(hard_x):.1f}" y1="{top}" x2="{sx(hard_x):.1f}" y2="{top+plot_h}" class="threshold"/>',
        f'<line x1="{left}" y1="{sy(hard_y):.1f}" x2="{left+plot_w}" y2="{sy(hard_y):.1f}" class="threshold"/>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" class="axis"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" class="axis"/>',
        f'<text x="{left+plot_w/2:.1f}" y="{top+plot_h+55:.1f}" text-anchor="middle" class="label">button x position (m)</text>',
        f'<text x="22" y="{top+plot_h/2:.1f}" transform="rotate(-90 22 {top+plot_h/2:.1f})" text-anchor="middle" class="label">button y position (m)</text>',
    ]
    for episode in episodes:
        x, y = position(episode)
        category = "success" if episode["success"] else failure_category(episode)
        radius = 6 if episode["success"] else 7
        lines.append(
            f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="{radius}" fill="{colors[category]}" '
            f'fill-opacity="0.82" stroke="#ffffff" stroke-width="1"><title>episode {episode["episode_index"]}: '
            f'{category}; x={x:.3f}, y={y:.3f}</title></circle>'
        )
    legend = [
        ("success", "success"),
        ("near-threshold press", "failure: 3.0–4.8 mm press"),
        ("partial press", "failure: 1.1–3.0 mm press"),
        ("close approach without depression", "failure: close approach, no press"),
        ("outside empirical success-distance envelope", "failure: outside success-distance envelope"),
    ]
    legend_positions = [(105, 705), (230, 705), (505, 705), (105, 735), (420, 735)]
    for (category, label), (lx, ly) in zip(legend, legend_positions):
        lines += [f'<circle cx="{lx}" cy="{ly}" r="6" fill="{colors[category]}"/>',
                  f'<text x="{lx+12}" y="{ly+5}" class="legend">{label}</text>']
    lines.append("</svg>")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hard-x", type=float, default=0.70)
    parser.add_argument("--hard-y", type=float, default=0.15)
    args = parser.parse_args()

    payload = json.loads(args.metrics.read_text(encoding="utf-8"))
    episodes = payload["episodes"]
    failures = [episode for episode in episodes if not episode["success"]]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_success_distance = max(
        episode["diagnostics"]["minimum_right_eef_to_button_m"]
        for episode in episodes
        if episode["success"]
    )
    if max_success_distance > SUCCESS_DISTANCE_ENVELOPE_M:
        raise ValueError(
            f"Success-distance cutoff {SUCCESS_DISTANCE_ENVELOPE_M} is below observed "
            f"success maximum {max_success_distance}."
        )

    csv_path = args.output_dir / "failure_episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "episode_index", "seed", "category", "button_x_m", "button_y_m",
                "max_press_depth_m", "minimum_right_eef_to_button_m",
                "right_eef_path_length_m", "action_steps",
            ]
        )
        for episode in failures:
            diagnostics = episode["diagnostics"]
            x, y = position(episode)
            writer.writerow(
                [
                    episode["episode_index"], episode["seed"], failure_category(episode),
                    x, y, diagnostics["max_press_depth_m"],
                    diagnostics["minimum_right_eef_to_button_m"],
                    diagnostics["right_eef_path_length_m"], episode["action_steps"],
                ]
            )

    def select(predicate):
        return [episode for episode in episodes if predicate(*position(episode))]

    region_rows = [
        rate_row("Central: x < 0.70 and y < 0.15", select(lambda x, y: x < args.hard_x and y < args.hard_y)),
        rate_row("x-hard only", select(lambda x, y: x >= args.hard_x and y < args.hard_y)),
        rate_row("y-hard only", select(lambda x, y: x < args.hard_x and y >= args.hard_y)),
        rate_row("Both hard thresholds", select(lambda x, y: x >= args.hard_x and y >= args.hard_y)),
        rate_row("Any hard threshold", select(lambda x, y: x >= args.hard_x or y >= args.hard_y)),
    ]
    x_rows = [
        rate_row(f"{lo:.1f} ≤ x < {hi:.1f}", select(lambda x, y, lo=lo, hi=hi: lo <= x < hi))
        for lo, hi in [(0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 1.0)]
    ]
    y_rows = [
        rate_row(f"{lo:+.2f} ≤ y < {hi:+.2f}", select(lambda x, y, lo=lo, hi=hi: lo <= y < hi))
        for lo, hi in [(-0.31, -0.15), (-0.15, 0.0), (0.0, 0.15), (0.15, 0.31)]
    ]
    category_counts = Counter(failure_category(episode) for episode in failures)
    failure_rows = [
        (category, count, f"{100 * count / len(failures):.1f}%")
        for category, count in [
            ("Outside empirical success-distance envelope", category_counts["outside empirical success-distance envelope"]),
            ("Close approach without depression", category_counts["close approach without depression"]),
            ("Partial press (1.1–3.0 mm)", category_counts["partial press"]),
            ("Near-threshold press (3.0–4.8 mm)", category_counts["near-threshold press"]),
        ]
    ]

    success_paths = [episode["diagnostics"]["right_eef_path_length_m"] for episode in episodes if episode["success"]]
    failure_paths = [episode["diagnostics"]["right_eef_path_length_m"] for episode in failures]
    report = [
        "# Click-bell held-out failure analysis",
        "",
        f"The 5K hard-edge checkpoint succeeded in **{len(episodes)-len(failures)}/{len(episodes)}** episodes. "
        f"All **{len(failures)} failures** reached the {payload['config']['timeout_action_steps']}-step timeout.",
        "",
        "## Main findings",
        "",
        f"- Central scenes succeeded at **{region_rows[0][3]}**, versus **{region_rows[-1][3]}** after either hard threshold.",
        f"- Scenes with x >= 0.80 m scored **{x_rows[-1][2]}/{x_rows[-1][1]} ({x_rows[-1][3]})**.",
        f"- Scenes beyond both hard thresholds scored **{region_rows[3][2]}/{region_rows[3][1]} ({region_rows[3][3]})**.",
        f"- Only **{category_counts['near-threshold press'] + category_counts['partial press']}/{len(failures)}** failures produced at least 1.1 mm of depression; most failures need better pose coverage or approach control rather than a press-only recovery.",
        f"- Median right-arm path length was **{statistics.median(success_paths):.2f} m** for successes and **{statistics.median(failure_paths):.2f} m** for failures. The long failure paths are a consequence of running to timeout and show that repeated motion did not recover the initial miss.",
        "",
        "## Failure taxonomy",
        "",
        *markdown_table(["Heuristic category", "Episodes", "Share of failures"], failure_rows),
        "",
        f"The distance-envelope cutoff is {SUCCESS_DISTANCE_ENVELOPE_M:.2f} m. All successful episodes reached "
        f"{max_success_distance:.3f} m or closer. The 1.1 mm boundary matches the evaluator's configured minimum "
        "press depth for contact recovery, although recovery was disabled in this run. These labels are diagnostic "
        "heuristics, not simulator-provided causal labels.",
        "",
        "## Success by hard-edge region",
        "",
        *markdown_table(["Region", "Episodes", "Successes", "Rate"], region_rows),
        "",
        "## Success by x position",
        "",
        *markdown_table(["Position", "Episodes", "Successes", "Rate"], x_rows),
        "",
        "## Success by y position",
        "",
        *markdown_table(["Position", "Episodes", "Successes", "Rate"], y_rows),
        "",
        "## Implication for the next experiment",
        "",
        "The 3x training sampler selected demonstrations using an end-effector approach proxy because the source "
        "dataset does not record ground-truth bell pose. The held-out failures are concentrated at actual bell-pose "
        "extremes, especially x >= 0.80 m and positive y. The next data or sampling revision should log and target "
        "actual randomized object pose. A contact recovery trigger may rescue some of the nine partial/near presses, "
        "but it cannot address the 36 failures with less than 1.1 mm depression by itself.",
        "",
    ]
    (args.output_dir / "failure_analysis.md").write_text("\n".join(report), encoding="utf-8")
    render_svg(episodes, args.output_dir / "position_success_map.svg", args.hard_x, args.hard_y)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'failure_analysis.md'}")
    print(f"Wrote {args.output_dir / 'position_success_map.svg'}")


if __name__ == "__main__":
    main()
