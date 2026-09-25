#!/usr/bin/env python3
"""Compare two ACT evaluations episode-by-episode on identical seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half_width, center + half_width


def exact_mcnemar_p(a_wins: int, b_wins: int) -> float:
    """Two-sided exact binomial p-value for discordant matched pairs."""
    n = a_wins + b_wins
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(a_wins, b_wins) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def press_bins(episodes: list[dict]) -> dict[str, int]:
    bins = {"<1.5 mm": 0, "1.5-4.0 mm": 0, "4.0-4.8 mm": 0}
    for episode in episodes:
        if episode["success"]:
            continue
        depth = episode["diagnostics"]["max_press_depth_m"]
        if depth < 0.0015:
            bins["<1.5 mm"] += 1
        elif depth < 0.004:
            bins["1.5-4.0 mm"] += 1
        else:
            bins["4.0-4.8 mm"] += 1
    return bins


def rate(episodes: list[dict], predicate) -> tuple[int, int]:
    selected = [e for e in episodes if predicate(e)]
    return sum(bool(e["success"]) for e in selected), len(selected)


def describe(label: str, data: dict) -> dict:
    episodes = data["episodes"]
    successes = sum(bool(e["success"]) for e in episodes)
    interval = wilson(successes, len(episodes))
    successful_paths = [e["diagnostics"]["right_eef_path_length_m"] for e in episodes if e["success"]]
    failed_paths = [e["diagnostics"]["right_eef_path_length_m"] for e in episodes if not e["success"]]
    x_edge = rate(episodes, lambda e: e["diagnostics"]["button_base_position_m"][0][0] >= 0.70)
    y_edge = rate(episodes, lambda e: e["diagnostics"]["button_base_position_m"][0][1] >= 0.15)
    return {
        "label": label,
        "successes": successes,
        "episodes": len(episodes),
        "success_rate": successes / len(episodes),
        "wilson_95": interval,
        "failure_press_bins": press_bins(episodes),
        "median_right_path_success_m": statistics.median(successful_paths),
        "median_right_path_failure_m": statistics.median(failed_paths),
        "x_ge_0.70": {"successes": x_edge[0], "episodes": x_edge[1]},
        "y_ge_0.15": {"successes": y_edge[0], "episodes": y_edge[1]},
        "summary": data["summary"],
        "config": data["config"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--candidate-label", default="candidate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    baseline_by_seed = {e["seed"]: e for e in baseline["episodes"]}
    candidate_by_seed = {e["seed"]: e for e in candidate["episodes"]}
    if baseline_by_seed.keys() != candidate_by_seed.keys():
        raise SystemExit("Evaluations do not contain identical seed sets")

    baseline_wins = 0
    candidate_wins = 0
    unchanged_success = 0
    unchanged_failure = 0
    flips = []
    for seed, before in baseline_by_seed.items():
        after = candidate_by_seed[seed]
        pair = (bool(before["success"]), bool(after["success"]))
        if pair == (True, False):
            baseline_wins += 1
            flips.append({"seed": seed, "from": "success", "to": "failure"})
        elif pair == (False, True):
            candidate_wins += 1
            flips.append({"seed": seed, "from": "failure", "to": "success"})
        elif pair == (True, True):
            unchanged_success += 1
        else:
            unchanged_failure += 1

    result = {
        args.baseline_label: describe(args.baseline_label, baseline),
        args.candidate_label: describe(args.candidate_label, candidate),
        "paired": {
            "baseline_only_successes": baseline_wins,
            "candidate_only_successes": candidate_wins,
            "both_success": unchanged_success,
            "both_failure": unchanged_failure,
            "exact_mcnemar_p": exact_mcnemar_p(baseline_wins, candidate_wins),
            "flips": flips,
        },
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
