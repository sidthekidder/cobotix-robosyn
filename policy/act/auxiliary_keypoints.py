"""Training-only bell localization supervision for ACT.

The auxiliary head stays outside the policy module tree. Its parameters take
part in optimization but are absent from exported ACT checkpoints, so the
standard challenge inference loader remains compatible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


TARGET_KEY = "auxiliary.bell_keypoints"


def load_keypoint_sidecar(path: str | Path) -> dict[tuple[int, int], torch.Tensor]:
    """Load normalized ``[x, y, visible]`` labels keyed by episode and frame."""

    result: dict[tuple[int, int], torch.Tensor] = {}
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["episode_index"]), int(row["frame_index"]))
            value = torch.as_tensor(row["keypoints"], dtype=torch.float32)
            if value.ndim != 2 or value.shape[-1] != 3:
                raise ValueError(
                    f"{path}:{line_number}: keypoints must have shape [cameras, 3]"
                )
            if key in result:
                raise ValueError(f"{path}:{line_number}: duplicate frame {key}")
            result[key] = value
    if not result:
        raise ValueError(f"No keypoint labels found in {path}")
    return result


class KeypointSidecarDataset(torch.utils.data.Dataset):
    """Add frame-aligned keypoint labels to a LeRobot dataset."""

    def __init__(self, base_dataset: Any, labels: dict[tuple[int, int], torch.Tensor]):
        self._base_dataset = base_dataset
        self._labels = labels
        self.meta = base_dataset.meta

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base_dataset, name)

    def __len__(self) -> int:
        return len(self._base_dataset)

    @staticmethod
    def _scalar(value: Any) -> int:
        return int(value.item() if hasattr(value, "item") else value)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self._base_dataset[index]
        key = (
            self._scalar(item["episode_index"]),
            self._scalar(item["frame_index"]),
        )
        try:
            item[TARGET_KEY] = self._labels[key]
        except KeyError as error:
            raise KeyError(f"Missing bell keypoint label for frame {key}") from error
        return item


class SpatialKeypointHead(nn.Module):
    """Predict normalized image coordinates with a differentiable soft argmax."""

    def __init__(self, channels: int, temperature: float = 0.1):
        super().__init__()
        self.heatmap = nn.Conv2d(channels, 1, kernel_size=1)
        self.visibility = nn.Linear(channels, 1)
        self.temperature = float(temperature)

    def forward(self, feature_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = feature_map.shape
        logits = self.heatmap(feature_map).reshape(batch, -1) / self.temperature
        probability = logits.softmax(dim=-1).reshape(batch, height, width)
        xs = torch.linspace(0.0, 1.0, width, device=feature_map.device, dtype=feature_map.dtype)
        ys = torch.linspace(0.0, 1.0, height, device=feature_map.device, dtype=feature_map.dtype)
        x = (probability.sum(dim=1) * xs).sum(dim=-1)
        y = (probability.sum(dim=2) * ys).sum(dim=-1)
        visible_logit = self.visibility(feature_map.mean(dim=(-2, -1))).squeeze(-1)
        return torch.stack((x, y), dim=-1), visible_logit


def keypoint_loss(
    predictions: list[tuple[torch.Tensor, torch.Tensor]], target: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    if len(predictions) != target.shape[1]:
        raise ValueError(
            f"Captured {len(predictions)} camera maps, but labels contain {target.shape[1]} cameras"
        )
    coordinate_losses = []
    visibility_losses = []
    for camera_index, (coordinates, visibility_logit) in enumerate(predictions):
        camera_target = target[:, camera_index].to(coordinates.device)
        visible = camera_target[:, 2].clamp(0, 1)
        per_sample = F.smooth_l1_loss(
            coordinates, camera_target[:, :2], reduction="none"
        ).mean(-1)
        coordinate_losses.append(
            (per_sample * visible).sum() / visible.sum().clamp_min(1.0)
        )
        visibility_losses.append(
            F.binary_cross_entropy_with_logits(visibility_logit, visible)
        )
    coordinate_loss = torch.stack(coordinate_losses).mean()
    visibility_loss = torch.stack(visibility_losses).mean()
    loss = coordinate_loss + 0.1 * visibility_loss
    return loss, {
        "bell_keypoint_loss": float(coordinate_loss.detach()),
        "bell_visibility_loss": float(visibility_loss.detach()),
    }


def install_auxiliary_keypoint_training(policy: Any, loss_weight: float) -> Any:
    """Patch an ACT policy instance with a training-only localization loss."""

    backbone = policy.model.backbone
    channels = int(policy.model.encoder_img_feat_input_proj.in_channels)
    device = next(policy.parameters()).device
    head = SpatialKeypointHead(channels).to(device)
    captures: list[torch.Tensor] = []

    def capture_features(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captures.append(output["feature_map"])

    backbone.register_forward_hook(capture_features)
    original_forward = policy.forward
    original_get_optim_params = policy.get_optim_params

    def forward_with_keypoints(batch: dict[str, Any]):
        target = batch.get(TARGET_KEY)
        captures.clear()
        loss, loss_dict = original_forward(batch)
        if target is None:
            raise KeyError(f"Training batch is missing {TARGET_KEY}")
        head.train(policy.training)
        predictions = [head(feature_map) for feature_map in captures]
        auxiliary_loss, auxiliary_metrics = keypoint_loss(predictions, target)
        loss = loss + float(loss_weight) * auxiliary_loss
        loss_dict.update(auxiliary_metrics)
        loss_dict["bell_aux_weighted_loss"] = float(
            (float(loss_weight) * auxiliary_loss).detach()
        )
        return loss, loss_dict

    def get_optim_params_with_head():
        groups = list(original_get_optim_params())
        groups.append({"params": list(head.parameters())})
        return groups

    # Bypass nn.Module.__setattr__: this keeps the training-only head out of
    # model.safetensors and therefore out of the challenge submission.
    policy.__dict__["_bell_keypoint_head"] = head
    policy.forward = forward_with_keypoints
    policy.get_optim_params = get_optim_params_with_head
    return policy
