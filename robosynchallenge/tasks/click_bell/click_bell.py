# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

import torch
from typing import Dict, Optional, Sequence

from embodichain.lab.gym.envs import EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger
from embodichain.lab.sim.cfg import MarkerCfg

from embodichain_tasks.tableware.base_agent_env import BaseAgentEnv
from .action_bank import (
    ClickBellActionBank,
)

__all__ = ["ClickBellEnv", "ClickBellTestEnv", "ClickBellAgentEnv"]



@register_env("ClickBell", max_episode_steps=600)
class ClickBellEnv(EmbodiedEnv):

    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)

        action_config = kwargs.get("action_config", None)
        if action_config is not None:
            self.action_config = action_config

        self._success_flag = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self._max_press_depth = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._left_arm_joint_ids = self.robot.get_joint_ids(
            name="left_arm", remove_mimic=True
        )
        self._right_arm_joint_ids = self.robot.get_joint_ids(
            name="right_arm", remove_mimic=True
        )
        self._initialize_episode_diagnostics()

    def _initialize_episode_diagnostics(self):
        """Capture an episode's initial state for read-only benchmark metrics."""
        qpos = self.robot.get_qpos().detach()
        left_eef = self.robot.get_link_pose("left_link6", to_matrix=True)[:, :3, 3]
        right_eef = self.robot.get_link_pose("right_link6", to_matrix=True)[:, :3, 3]
        button = self.sim.get_articulation("button")
        button_base = button.get_link_pose("button_base", to_matrix=True)[:, :3, 3]

        self._initial_left_arm_qpos = qpos[:, self._left_arm_joint_ids].clone()
        self._initial_right_arm_qpos = qpos[:, self._right_arm_joint_ids].clone()
        self._max_left_arm_joint_delta = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._max_right_arm_joint_delta = torch.zeros_like(
            self._max_left_arm_joint_delta
        )
        self._left_eef_path_length = torch.zeros_like(self._max_left_arm_joint_delta)
        self._right_eef_path_length = torch.zeros_like(self._max_left_arm_joint_delta)
        self._previous_left_eef_position = left_eef.clone()
        self._previous_right_eef_position = right_eef.clone()
        self._minimum_left_eef_to_button = torch.full_like(
            self._max_left_arm_joint_delta, torch.inf
        )
        self._minimum_right_eef_to_button = torch.full_like(
            self._max_left_arm_joint_delta, torch.inf
        )
        self._button_base_position = button_base.clone()
        self._episode_diagnostic_steps = 0

    def _update_episode_diagnostics(self, button):
        """Update arm-use and geometric diagnostics after a simulator step."""
        qpos = self.robot.get_qpos().detach()
        left_eef = self.robot.get_link_pose("left_link6", to_matrix=True)[:, :3, 3]
        right_eef = self.robot.get_link_pose("right_link6", to_matrix=True)[:, :3, 3]
        button_cover = button.get_link_pose("button_cover", to_matrix=True)[:, :3, 3]

        left_delta = torch.amax(
            torch.abs(qpos[:, self._left_arm_joint_ids] - self._initial_left_arm_qpos),
            dim=-1,
        )
        right_delta = torch.amax(
            torch.abs(qpos[:, self._right_arm_joint_ids] - self._initial_right_arm_qpos),
            dim=-1,
        )
        self._max_left_arm_joint_delta = torch.maximum(
            self._max_left_arm_joint_delta, left_delta
        )
        self._max_right_arm_joint_delta = torch.maximum(
            self._max_right_arm_joint_delta, right_delta
        )
        self._left_eef_path_length += torch.linalg.norm(
            left_eef - self._previous_left_eef_position, dim=-1
        )
        self._right_eef_path_length += torch.linalg.norm(
            right_eef - self._previous_right_eef_position, dim=-1
        )
        self._previous_left_eef_position = left_eef.clone()
        self._previous_right_eef_position = right_eef.clone()
        self._minimum_left_eef_to_button = torch.minimum(
            self._minimum_left_eef_to_button,
            torch.linalg.norm(left_eef - button_cover, dim=-1),
        )
        self._minimum_right_eef_to_button = torch.minimum(
            self._minimum_right_eef_to_button,
            torch.linalg.norm(right_eef - button_cover, dim=-1),
        )
        self._episode_diagnostic_steps += 1

    def _snapshot_episode_diagnostics(self) -> Dict[str, torch.Tensor | float]:
        """Copy metrics before an autoreset replaces the terminal state."""
        button = self.sim.get_articulation("button")
        final_press_depth = -button.get_qpos()[:, 0]
        return {
            "max_press_depth_m": self._max_press_depth.clone(),
            "final_press_depth_m": final_press_depth.clone(),
            "movement_threshold_m": 0.0048,
            "button_base_position_m": self._button_base_position.clone(),
            "minimum_left_eef_to_button_m": self._minimum_left_eef_to_button.clone(),
            "minimum_right_eef_to_button_m": self._minimum_right_eef_to_button.clone(),
            "left_eef_path_length_m": self._left_eef_path_length.clone(),
            "right_eef_path_length_m": self._right_eef_path_length.clone(),
            "max_left_arm_joint_delta_rad": self._max_left_arm_joint_delta.clone(),
            "max_right_arm_joint_delta_rad": self._max_right_arm_joint_delta.clone(),
        }
    def create_demo_action_list(self, *args, **kwargs):
        """
        Create a demonstration action list for the current task.

        Returns:
            list: A list of demo actions generated by the task.
        """
        logger.log_info("Create demo action list for ClickBellTask.")

        if getattr(self, "action_config") is not None:
            self._init_action_bank(ClickBellActionBank, self.action_config)
            action_list = self.create_expert_demo_action_list(*args, **kwargs)
        else:
            logger.log_error("No action_config found in env, please check again.")

        if action_list is None:
            return action_list

        logger.log_info(
            f"Demo action list created with {len(action_list)} steps.", color="green"
        )
        return action_list

    def create_expert_demo_action_list(self, **kwargs):
        """
        Create an expert demonstration action list using the action bank.

        This function generates a trajectory based on expert knowledge, mapping joint and end-effector
        states to the required action format for the environment and robot type.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            list: A list of actions, each containing joint positions ("qpos").
        """

        if hasattr(self, "action_bank") is False or self.action_bank is None:
            logger.log_error(
                "Action bank is not initialized. Cannot create expert demo action list."
            )

        ret = self.action_bank.create_action_list(
            self, self.graph_compose, self.packages
        )

        if ret is None:
            logger.log_warning("Failed to generate expert demo action list.")
            return None

        # TODO: to be removed, need a unified interface in robot class
        left_arm_joints = self.robot.get_joint_ids(name="left_arm", remove_mimic=True)
        right_arm_joints = self.robot.get_joint_ids(name="right_arm", remove_mimic=True)
        left_eef_joints = self.robot.get_joint_ids(name="left_eef", remove_mimic=True)
        right_eef_joints = self.robot.get_joint_ids(name="right_eef", remove_mimic=True)


        total_traj_num = ret[list(ret.keys())[0]].shape[-1]
        num_active_joints = len(self.active_joint_ids)
        actions = torch.zeros(
            (total_traj_num, self.num_envs, num_active_joints), dtype=torch.float32
        )

        # 建立一个从全局 joint_id 到 active_joint_id 在 action 数组中正确存放位置的映射
        global_to_active_idx = {
            joint_id: active_idx for active_idx, joint_id in enumerate(self.active_joint_ids)
        }

        for key, joints in [
            ("left_arm", left_arm_joints),
            ("left_eef", left_eef_joints),
            ("right_arm", right_arm_joints),
            ("right_eef", right_eef_joints),
        ]:
            if key in ret:
                # TODO: only 1 env supported now
                local_action_data = torch.as_tensor(ret[key].T, dtype=torch.float32)

                # 【修改重点2】：使用映射精准定位它在 action tensor 中的正确位置存放
                for i, joint_id in enumerate(joints):
                    if joint_id in global_to_active_idx:
                        active_idx = global_to_active_idx[joint_id]
                        actions[:, 0, active_idx] = local_action_data[:, i]
        return actions
    def compute_task_state(self, **kwargs):
        button = self.sim.get_articulation("button")
        button_qpos = button.get_qpos()

        # button.urdf uses a single prismatic joint with range [-0.005, 0.0].
        # Treat any detectable displacement as success (with tiny epsilon to avoid numerical noise).
        press_depth = -button_qpos[:, 0]
        movement_threshold = 0.0048
        self._max_press_depth = torch.maximum(self._max_press_depth, press_depth)
        self._update_episode_diagnostics(button)
        current_success = press_depth >= movement_threshold

        # 粘滞锁存：回合内任意一步按到位即记为成功
        self._success_flag |= current_success

        metrics = {
            "press_depth": press_depth,
            "movement_threshold": movement_threshold,
        }
        fail = torch.zeros_like(self._success_flag, dtype=torch.bool)
        success = torch.zeros_like(fail, dtype=torch.bool)
        return success, fail, metrics

    def is_task_success(self, **kwargs) -> torch.Tensor:
        return self._success_flag

    def get_episode_diagnostics(self) -> Dict[str, torch.Tensor | float]:
        """Return read-only contact diagnostics for benchmark reporting."""
        if getattr(self, "_episode_diagnostic_steps", 0) > 0:
            return self._snapshot_episode_diagnostics()
        previous = getattr(self, "_last_episode_diagnostics", None)
        return previous if previous is not None else self._snapshot_episode_diagnostics()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        if getattr(self, "_episode_diagnostic_steps", 0) > 0:
            self._last_episode_diagnostics = self._snapshot_episode_diagnostics()
        obs, info = super().reset(seed=seed, options=options)

        if options is None:
            options = {}
        reset_ids = options.get(
            "reset_ids",
            torch.arange(self.num_envs, dtype=torch.int32, device=self.device),
        )
        self._success_flag[reset_ids] = False
        self._max_press_depth[reset_ids] = 0.0
        self._initialize_episode_diagnostics()

        return obs, info

@register_env("ClickBellTest", max_episode_steps=600)
class ClickBellTestEnv(ClickBellEnv):
    def compute_task_state(self, **kwargs):
        button = self.sim.get_articulation("button")
        button_qpos = button.get_qpos()

        # button.urdf uses a single prismatic joint with range [-0.005, 0.0].
        # Treat any detectable displacement as success (with tiny epsilon to avoid numerical noise).
        press_depth = -button_qpos[:, 0]
        movement_threshold = 0.004
        success = press_depth >= movement_threshold
        # print(f"press_depth: {press_depth}, movement_threshold: {movement_threshold}")
        self._success_flag |= success
        fail = torch.zeros_like(success, dtype=torch.bool)

        return success, fail, {}
    def is_task_success(self, **kwargs) -> torch.Tensor:
        return torch.ones_like(self._success_flag, dtype=torch.bool)

@register_env("ClickBellAgent", max_episode_steps=600)
class ClickBellAgentEnv(BaseAgentEnv, ClickBellEnv):
    def __init__(self, cfg: EmbodiedEnvCfg = None, **kwargs):
        super().__init__(cfg, **kwargs)
        super()._init_agents(**kwargs)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        obs, info = super().reset(seed=seed, options=options)
        super().get_states()
        return obs, info
