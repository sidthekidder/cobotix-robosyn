from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "collect_click_bell_keypoint_demos.py"
SPEC = importlib.util.spec_from_file_location("click_bell_keypoints", MODULE_PATH)
click_bell_keypoints = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(click_bell_keypoints)


class MaskMeasurementsTest(unittest.TestCase):
    def test_empty_mask(self):
        result = click_bell_keypoints._mask_measurements(
            np.zeros((4, 6), dtype=np.int32), np.array([7])
        )
        self.assertFalse(result["visible"])
        self.assertEqual(result["area_pixels"], 0)
        self.assertIsNone(result["bbox_xyxy_pixels"])

    def test_centered_and_truncated_masks(self):
        mask = np.zeros((5, 7), dtype=np.int32)
        mask[1:4, 2:5] = 7
        centered = click_bell_keypoints._mask_measurements(mask, np.array([7]))
        self.assertEqual(centered["centroid"], [0.5, 0.5])
        self.assertEqual(centered["area_pixels"], 9)
        self.assertFalse(centered["touches_image_boundary"])

        mask[:, 0] = 7
        truncated = click_bell_keypoints._mask_measurements(mask, np.array([7]))
        self.assertTrue(truncated["touches_image_boundary"])


class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.pose = np.eye(4)
        self.intrinsics = np.array(
            [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]]
        )

    def test_opengl_forward_projects_to_center(self):
        result = click_bell_keypoints._project_world_point(
            self.pose, self.intrinsics, np.array([0.0, 0.0, -2.0]), (81, 101)
        )
        self.assertEqual(result["xy_pixels"], [50.0, 40.0])
        self.assertEqual(result["xy_normalized"], [0.5, 0.5])
        self.assertAlmostEqual(result["depth_m"], 2.0)
        self.assertTrue(result["in_front"])
        self.assertTrue(result["in_frame"])

    def test_behind_and_outside_points_are_flagged(self):
        behind = click_bell_keypoints._project_world_point(
            self.pose, self.intrinsics, np.array([0.0, 0.0, 1.0]), (81, 101)
        )
        self.assertFalse(behind["in_front"])
        self.assertFalse(behind["in_frame"])

        outside = click_bell_keypoints._project_world_point(
            self.pose, self.intrinsics, np.array([2.0, 0.0, -1.0]), (81, 101)
        )
        self.assertTrue(outside["in_front"])
        self.assertFalse(outside["in_frame"])

    def test_mask_lookup_has_small_rasterization_radius(self):
        mask = np.zeros((9, 9), dtype=np.int32)
        mask[4, 6] = 3
        self.assertTrue(click_bell_keypoints._point_on_mask(mask, np.array([3]), [4, 4]))
        self.assertFalse(click_bell_keypoints._point_on_mask(mask, np.array([3]), [0, 0]))


class FrameSchemaTest(unittest.TestCase):
    class _PoseSource:
        def __init__(self, pose):
            self.pose = pose

        def get_link_pose(self, _name, to_matrix=False):
            assert to_matrix
            return self.pose[None]

    class _Sensor:
        def get_arena_pose(self, to_matrix=False):
            assert to_matrix
            return np.eye(4)[None]

        def get_intrinsics(self):
            return np.array(
                [[[100.0, 0.0, 4.0], [0.0, 100.0, 4.0], [0.0, 0.0, 1.0]]]
            )

    def test_legacy_keypoints_and_geometry_coexist(self):
        cover_pose = np.eye(4)
        cover_pose[2, 3] = -1.0 - click_bell_keypoints.BUTTON_PRESS_SURFACE_OFFSET_M
        tool_pose = np.eye(4)
        tool_pose[2, 3] = -1.0 - click_bell_keypoints.RIGHT_TCP_OFFSET_M
        sensor = self._Sensor()

        class Sim:
            def get_articulation(self, _uid):
                return FrameSchemaTest._PoseSource(cover_pose)

            def get_sensor(self, _uid):
                return sensor

        class Base:
            sim = Sim()
            robot = FrameSchemaTest._PoseSource(tool_pose)

        mask = np.zeros((1, 9, 9), dtype=np.int32)
        mask[0, 3:6, 3:6] = 7
        obs = {
            "sensor": {
                camera: {"mask": mask.copy()}
                for camera in click_bell_keypoints.CAMERAS
            }
        }
        keypoints, geometry = click_bell_keypoints._frame_labels(
            Base(), obs, np.array([7])
        )
        self.assertEqual(keypoints, [[0.5, 0.5, 1.0]] * 3)
        self.assertEqual(set(geometry), set(click_bell_keypoints.GEOMETRY_CAMERAS))
        for camera_geometry in geometry.values():
            self.assertEqual(camera_geometry["press_point"]["xy_normalized"], [0.5, 0.5])
            self.assertEqual(camera_geometry["right_tool_tip"]["xy_normalized"], [0.5, 0.5])
            self.assertEqual(camera_geometry["mask_centroid_confidence"], 1.0)
        json.dumps(
            {"keypoints": keypoints, "geometry": geometry}, allow_nan=False
        )


if __name__ == "__main__":
    unittest.main()
