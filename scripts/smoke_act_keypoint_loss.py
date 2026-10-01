#!/usr/bin/env python3
"""Backpropagate the ACT bell keypoint loss through a real label sidecar."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).resolve().parents[1] / "policy" / "act" / "auxiliary_keypoints.py"
SPEC = importlib.util.spec_from_file_location("auxiliary_keypoints", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load {MODULE_PATH}")
auxiliary_keypoints = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auxiliary_keypoints)
CAMERAS = auxiliary_keypoints.CAMERAS
SpatialKeypointHead = auxiliary_keypoints.SpatialKeypointHead
keypoint_loss = auxiliary_keypoints.keypoint_loss
load_keypoint_sidecar = auxiliary_keypoints.load_keypoint_sidecar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sidecar", type=Path)
    parser.add_argument(
        "--target", choices=("mask-centroid", "press-point"), default="press-point"
    )
    parser.add_argument("--loss-weight", type=float, default=0.2)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--width", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    labels = load_keypoint_sidecar(args.sidecar, target_mode=args.target)
    ordered_keys = sorted(labels)
    target = torch.stack([labels[key] for key in ordered_keys])

    head = SpatialKeypointHead(channels=args.channels)
    feature_maps = [
        torch.randn(
            len(target),
            args.channels,
            args.height,
            args.width,
            requires_grad=True,
        )
        for _camera in CAMERAS
    ]
    predictions = [head(feature_map) for feature_map in feature_maps]
    raw_loss, metrics = keypoint_loss(predictions, target)
    weighted_loss = args.loss_weight * raw_loss
    weighted_loss.backward()

    head_gradient = math.sqrt(
        sum(
            float((parameter.grad**2).sum())
            for parameter in head.parameters()
            if parameter.grad is not None
        )
    )
    feature_gradients = [float(feature_map.grad.norm()) for feature_map in feature_maps]
    frames_per_episode = Counter(episode for episode, _frame in ordered_keys)
    summary = {
        "sidecar": str(args.sidecar.resolve()),
        "target": args.target,
        "labels": len(labels),
        "episodes": len(frames_per_episode),
        "frames_per_episode": dict(sorted(frames_per_episode.items())),
        "target_shape": list(target.shape),
        "visible_frames": {
            camera: int(target[:, index, 2].sum())
            for index, camera in enumerate(CAMERAS)
        },
        "raw_aux_loss": float(raw_loss.detach()),
        "weighted_aux_loss": float(weighted_loss.detach()),
        "metrics": metrics,
        "finite": bool(torch.isfinite(raw_loss)),
        "shared_head_gradient_l2": head_gradient,
        "feature_map_gradient_l2": dict(zip(CAMERAS, feature_gradients)),
        "all_gradients_nonzero": head_gradient > 0
        and all(gradient > 0 for gradient in feature_gradients),
    }
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")

    if not summary["finite"] or not summary["all_gradients_nonzero"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
