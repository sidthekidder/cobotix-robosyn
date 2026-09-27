import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch

MODULE_PATH = Path(__file__).parents[1] / "policy" / "act" / "auxiliary_keypoints.py"
SPEC = importlib.util.spec_from_file_location("auxiliary_keypoints", MODULE_PATH)
auxiliary_keypoints = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(auxiliary_keypoints)
KeypointSidecarDataset = auxiliary_keypoints.KeypointSidecarDataset
SpatialKeypointHead = auxiliary_keypoints.SpatialKeypointHead
keypoint_loss = auxiliary_keypoints.keypoint_loss
load_keypoint_sidecar = auxiliary_keypoints.load_keypoint_sidecar
install_auxiliary_keypoint_training = (
    auxiliary_keypoints.install_auxiliary_keypoint_training
)


class _Dataset(torch.utils.data.Dataset):
    meta = object()

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return {"episode_index": torch.tensor(2), "frame_index": torch.tensor(7)}


class AuxiliaryKeypointTests(unittest.TestCase):
    def test_sidecar_load_and_dataset_join(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "episode_index": 2,
                        "frame_index": 7,
                        "keypoints": [[0.25, 0.75, 1.0], [0.5, 0.5, 0.0]],
                    }
                )
                + "\n"
            )
            labels = load_keypoint_sidecar(path)
            item = KeypointSidecarDataset(_Dataset(), labels)[0]
            self.assertEqual(item["auxiliary.bell_keypoints"].shape, (2, 3))

    def test_spatial_keypoint_loss_backpropagates(self):
        head = SpatialKeypointHead(channels=4)
        maps = [torch.randn(3, 4, 5, 7, requires_grad=True) for _ in range(2)]
        predictions = [head(feature_map) for feature_map in maps]
        target = torch.tensor(
            [
                [[0.2, 0.8, 1.0], [0.5, 0.5, 1.0]],
                [[0.3, 0.7, 1.0], [0.5, 0.5, 0.0]],
                [[0.4, 0.6, 1.0], [0.6, 0.4, 1.0]],
            ]
        )
        loss, metrics = keypoint_loss(predictions, target)
        loss.backward()
        self.assertTrue(loss.isfinite())
        self.assertGreaterEqual(metrics["bell_keypoint_loss"], 0)
        self.assertTrue(all(feature_map.grad is not None for feature_map in maps))

    def test_sidecar_rejects_duplicate_frame(self):
        row = {
            "episode_index": 0,
            "frame_index": 0,
            "keypoints": [[0.5, 0.5, 1.0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.jsonl"
            path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate frame"):
                load_keypoint_sidecar(path)

    def test_training_head_is_not_exported(self):
        class Backbone(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 4, 1)

            def forward(self, image):
                return {"feature_map": self.conv(image)}

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = Backbone()
                self.encoder_img_feat_input_proj = torch.nn.Conv2d(4, 8, 1)

        class Policy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Model()

            def forward(self, batch):
                outputs = [self.model.backbone(image)["feature_map"] for image in batch["images"]]
                return sum(output.mean() * 0 for output in outputs), {}

            def get_optim_params(self):
                return [{"params": list(self.parameters())}]

        policy = install_auxiliary_keypoint_training(Policy(), loss_weight=0.2)
        self.assertFalse(any("keypoint" in key for key in policy.state_dict()))
        batch = {
            "images": [torch.randn(2, 3, 5, 7), torch.randn(2, 3, 5, 7)],
            "auxiliary.bell_keypoints": torch.tensor(
                [
                    [[0.2, 0.8, 1.0], [0.5, 0.5, 1.0]],
                    [[0.3, 0.7, 1.0], [0.5, 0.5, 0.0]],
                ]
            ),
        }
        loss, metrics = policy(batch)
        loss.backward()
        self.assertIn("bell_aux_weighted_loss", metrics)
        self.assertTrue(all(parameter.grad is not None for parameter in policy.__dict__["_bell_keypoint_head"].parameters()))


if __name__ == "__main__":
    unittest.main()
