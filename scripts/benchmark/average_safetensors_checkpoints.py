#!/usr/bin/env python3
"""Average compatible model.safetensors checkpoints into one checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        help="Optional non-negative weights, one per input (default: uniform).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.name != "model.safetensors":
            raise ValueError(f"Expected model.safetensors, got: {path}")
        return path.parent
    if not (path / "model.safetensors").is_file():
        raise ValueError(f"Checkpoint has no model.safetensors: {path}")
    return path


def normalized_weights(count: int, values: list[float] | None) -> list[float]:
    values = values or [1.0] * count
    if len(values) != count:
        raise ValueError(f"Expected {count} weights, got {len(values)}")
    if any(value < 0 for value in values) or sum(values) <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum.")
    total = sum(values)
    return [value / total for value in values]


def average_state_dicts(
    state_dicts: list[dict[str, torch.Tensor]], weights: list[float]
) -> dict[str, torch.Tensor]:
    reference_keys = set(state_dicts[0])
    for index, state in enumerate(state_dicts[1:], start=1):
        if set(state) != reference_keys:
            missing = sorted(reference_keys - set(state))
            extra = sorted(set(state) - reference_keys)
            raise ValueError(
                f"Checkpoint {index} has incompatible keys; missing={missing[:5]}, "
                f"extra={extra[:5]}"
            )

    averaged: dict[str, torch.Tensor] = {}
    for key in sorted(reference_keys):
        tensors = [state[key].cpu() for state in state_dicts]
        reference = tensors[0]
        for index, tensor in enumerate(tensors[1:], start=1):
            if tensor.shape != reference.shape or tensor.dtype != reference.dtype:
                raise ValueError(
                    f"Tensor {key!r} differs in checkpoint {index}: "
                    f"{tensor.shape}/{tensor.dtype} vs "
                    f"{reference.shape}/{reference.dtype}"
                )

        if reference.is_floating_point():
            accumulator = torch.zeros_like(reference, dtype=torch.float64)
            for weight, tensor in zip(weights, tensors):
                accumulator.add_(tensor.to(torch.float64), alpha=weight)
            averaged[key] = accumulator.to(reference.dtype)
        else:
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                raise ValueError(f"Non-floating tensor {key!r} is not identical.")
            averaged[key] = reference
    return averaged


def main() -> None:
    args = parse_args()
    inputs = [checkpoint_dir(path) for path in args.inputs]
    weights = normalized_weights(len(inputs), args.weights)
    output = args.output.expanduser().resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    states = [load_file(str(path / "model.safetensors"), device="cpu") for path in inputs]
    averaged = average_state_dicts(states, weights)
    save_file(averaged, str(output / "model.safetensors"))

    source = inputs[-1]
    for filename in ("config.json", "train_config.json"):
        if (source / filename).is_file():
            shutil.copy2(source / filename, output / filename)

    manifest = {
        "method": "weighted_parameter_mean",
        "inputs": [str(path) for path in inputs],
        "normalized_weights": weights,
        "companion_config_source": str(source),
        "tensor_count": len(averaged),
    }
    (output / "averaging_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote averaged checkpoint to {output}")


if __name__ == "__main__":
    main()
