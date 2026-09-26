#!/usr/bin/env python3
"""Append corrective ClickBell episodes to the released LeRobot v2.1 dataset.

The released dataset uses legacy feature names (``observation.qpos`` and
``cam_*.color``), while the current EmbodiChain recorder uses canonical names.
This script normalizes the corrective dataset during the append and optionally
builds a WeightedRandomSampler plan with a requested correction sample share.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import pyarrow as pa
import pyarrow.parquet as pq


FEATURE_ALIASES = {
    "observation.state": "observation.qpos",
    "observation.images.cam_high": "cam_high.color",
    "observation.images.cam_right_wrist": "cam_right_wrist.color",
    "observation.images.cam_left_wrist": "cam_left_wrist.color",
}


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _format(template: str, episode_index: int, video_key: str | None = None) -> str:
    values = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // 1000,
        "chunk_index": episode_index // 1000,
        "video_key": video_key,
    }
    return template.format(**values)


def _rename_stats(stats: dict) -> dict:
    return {FEATURE_ALIASES.get(key, key): value for key, value in stats.items()}


def _replace_scalar_column(table: pa.Table, name: str, values: list[int]) -> pa.Table:
    if name not in table.column_names:
        return table
    index = table.column_names.index(name)
    return table.set_column(index, name, pa.array(values, type=table.schema.field(name).type))


def _sampling_plan(
    base_episodes: list[dict],
    correction_episodes: list[dict],
    base_plan_path: Path | None,
    target_share: float,
    first_correction_index: int,
) -> dict:
    if not 0 < target_share < 1:
        raise ValueError("--correction-sample-share must be between 0 and 1")
    base_plan = json.loads(base_plan_path.read_text()) if base_plan_path else {}
    if base_plan.get("frame_ranges"):
        raise ValueError("base plans with frame_ranges are not supported")
    default_weight = float(base_plan.get("default_weight", 1.0))
    episode_weights = {
        int(index): float(weight)
        for index, weight in base_plan.get("episode_weights", {}).items()
    }
    base_mass = sum(
        int(row["length"]) * episode_weights.get(int(row["episode_index"]), default_weight)
        for row in base_episodes
    )
    correction_frames = sum(int(row["length"]) for row in correction_episodes)
    if correction_frames <= 0:
        raise ValueError("corrective dataset contains no frames")
    correction_weight = target_share * base_mass / ((1.0 - target_share) * correction_frames)
    for offset in range(len(correction_episodes)):
        episode_weights[first_correction_index + offset] = correction_weight
    return {
        "schema_version": 1,
        "description": "Retain hard-edge sampling and allocate a fixed share to corrective episodes.",
        "default_weight": default_weight,
        "episode_weights": {str(k): v for k, v in sorted(episode_weights.items())},
        "combine": "max",
        "frame_ranges": [],
        "base_weighted_frame_mass": base_mass,
        "correction_frames": correction_frames,
        "correction_weight": correction_weight,
        "expected_correction_sample_share": target_share,
        "first_correction_episode": first_correction_index,
        "correction_episode_count": len(correction_episodes),
    }


def merge(args: argparse.Namespace) -> None:
    base = args.base.resolve()
    corrections = args.corrections.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")

    base_info = json.loads((base / "meta/info.json").read_text())
    correction_info = json.loads((corrections / "meta/info.json").read_text())
    if base_info.get("codebase_version") != "v2.1":
        raise ValueError("base dataset must use LeRobot v2.1")
    if correction_info.get("codebase_version") != "v2.1":
        raise ValueError("corrective dataset must be converted to LeRobot v2.1 first")

    base_episodes = _read_jsonl(base / "meta/episodes.jsonl")
    correction_episodes = _read_jsonl(corrections / "meta/episodes.jsonl")
    base_stats = _read_jsonl(base / "meta/episodes_stats.jsonl")
    correction_stats = _read_jsonl(corrections / "meta/episodes_stats.jsonl")
    if len(correction_episodes) != len(correction_stats):
        raise ValueError("corrective episode and stats counts differ")

    shutil.copytree(base, output)
    first_index = len(base_episodes)
    global_index = int(base_info["total_frames"])
    appended_episodes: list[dict] = []
    appended_stats: list[dict] = []

    correction_video_keys = [
        key
        for key, feature in correction_info["features"].items()
        if feature.get("dtype") == "video"
    ]
    for offset, (episode, stats) in enumerate(zip(correction_episodes, correction_stats)):
        source_index = int(episode["episode_index"])
        destination_index = first_index + offset
        length = int(episode["length"])

        source_data = corrections / _format(correction_info["data_path"], source_index)
        table = pq.read_table(source_data)
        renamed = [FEATURE_ALIASES.get(name, name) for name in table.column_names]
        table = table.rename_columns(renamed)
        table = _replace_scalar_column(table, "episode_index", [destination_index] * table.num_rows)
        table = _replace_scalar_column(
            table, "index", list(range(global_index, global_index + table.num_rows))
        )
        destination_data = output / _format(base_info["data_path"], destination_index)
        destination_data.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.replace_schema_metadata(None), destination_data)
        global_index += table.num_rows

        for source_key in correction_video_keys:
            destination_key = FEATURE_ALIASES.get(source_key, source_key)
            source_video = corrections / _format(
                correction_info["video_path"], source_index, source_key
            )
            destination_video = output / _format(
                base_info["video_path"], destination_index, destination_key
            )
            destination_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, destination_video)

        episode_row = dict(episode)
        episode_row["episode_index"] = destination_index
        appended_episodes.append(episode_row)
        stats_row = dict(stats)
        stats_row["episode_index"] = destination_index
        stats_row["stats"] = _rename_stats(stats_row["stats"])
        appended_stats.append(stats_row)

        if table.num_rows != length:
            raise ValueError(
                f"episode {source_index} declares {length} frames but contains {table.num_rows}"
            )

    all_episodes = base_episodes + appended_episodes
    _write_jsonl(output / "meta/episodes.jsonl", all_episodes)
    _write_jsonl(output / "meta/episodes_stats.jsonl", base_stats + appended_stats)

    base_info["total_episodes"] = len(all_episodes)
    base_info["total_frames"] = global_index
    base_info["total_chunks"] = math.ceil(len(all_episodes) / int(base_info["chunks_size"]))
    base_info["total_videos"] = int(base_info.get("total_videos", len(base_episodes) * 3)) + (
        len(appended_episodes) * len(correction_video_keys)
    )
    base_info["splits"] = {"train": f"0:{len(all_episodes)}"}
    (output / "meta/info.json").write_text(json.dumps(base_info, indent=2) + "\n")

    plan = _sampling_plan(
        base_episodes,
        appended_episodes,
        args.base_sampling_plan,
        args.correction_sample_share,
        first_index,
    )
    args.output_sampling_plan.parent.mkdir(parents=True, exist_ok=True)
    args.output_sampling_plan.write_text(json.dumps(plan, indent=2) + "\n")
    print(
        f"Merged {len(base_episodes)} base + {len(appended_episodes)} corrections; "
        f"correction weight={plan['correction_weight']:.6f}, "
        f"expected share={plan['expected_correction_sample_share']:.1%}"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--corrections", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-sampling-plan", type=Path)
    parser.add_argument("--output-sampling-plan", type=Path, required=True)
    parser.add_argument("--correction-sample-share", type=float, default=0.20)
    return parser


if __name__ == "__main__":
    merge(_parser().parse_args())
