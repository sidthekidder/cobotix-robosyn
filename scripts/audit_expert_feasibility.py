#!/usr/bin/env python
"""Audit expert planning feasibility for the exact seeds in an eval result."""

import argparse
import json
import time
from pathlib import Path

from eval_policy import (
    collect_episode_diagnostics,
    find_action_config,
    find_gym_config,
    make_env_from_configs,
    prepare_episode_reset,
    suppress_expert_output,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    source = json.loads(Path(args.metrics).read_text())
    episodes = source["episodes"]
    seeds = [int(item["seed"]) for item in episodes]

    config = {
        "task_name": "click_bell",
        "setting": "random",
        "num_envs": 1,
        "device": args.device,
        "gpu_id": args.gpu_id,
        "headless": True,
        "renderer": "hybrid",
        "filter_dataset_saving": True,
    }
    gym_config = find_gym_config(config)
    action_config = find_action_config(config)
    env, _ = make_env_from_configs(config, gym_config, action_config)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.time()

    def write_checkpoint():
        output = {
            "schema_version": 1,
            "source_metrics": str(args.metrics),
            "task": "click_bell",
            "setting": "random",
            "episode_count": len(results),
            "requested_episode_count": len(seeds),
            "feasible_count": sum(item["feasible"] for item in results),
            "infeasible_count": sum(not item["feasible"] for item in results),
            "elapsed_seconds": time.time() - started,
            "complete": len(results) == len(seeds),
            "episodes": results,
        }
        output_path.write_text(json.dumps(output, indent=2) + "\n")

    try:
        for index, (seed, policy_episode) in enumerate(zip(seeds, episodes)):
            result = {
                "episode_index": index,
                "seed": seed,
                "policy_success": bool(policy_episode["success"]),
                "feasible": False,
                "reason": "expert_action_generation_failed",
                "expert_plan_steps": 0,
                "source_diagnostics": policy_episode.get("diagnostics", {}),
            }
            item_started = time.time()
            try:
                actual_seed = prepare_episode_reset(env, seed)
                with suppress_expert_output():
                    env.reset(seed=actual_seed, options={"save_data": False})
                result["diagnostics"] = collect_episode_diagnostics(env)
                with suppress_expert_output():
                    action_list = env.get_wrapper_attr("create_demo_action_list")(
                        action_sentence=0
                    )
                if action_list is None:
                    result["reason"] = "expert_action_list_none"
                elif len(action_list) == 0:
                    result["reason"] = "expert_action_list_empty"
                else:
                    result["expert_plan_steps"] = len(action_list)
                    result["feasible"] = True
                    result["reason"] = "expert_planning_succeeded"
            except Exception as exc:
                message = " ".join(str(exc).split())
                result["reason"] = (
                    f"expert_exception:{type(exc).__name__}:{message}"
                )
            result["elapsed_seconds"] = time.time() - item_started
            results.append(result)
            write_checkpoint()
            feasible = sum(item["feasible"] for item in results)
            print(
                f"[{index + 1:03d}/{len(seeds)}] seed={seed} "
                f"feasible={result['feasible']} cumulative={feasible}/{index + 1} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
    finally:
        env.close()

    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
