# ----------------------------------------------------------------------------
# LeRobot ACT Policy Adapter for RoboSynChallenge
#
# Follows the same unified evaluation interface as policy/pi0:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from policy.inference_timing import finish_inference, start_inference
from policy.act.recovery import ContactPlateauRecovery


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


def _metric_scalar(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().reshape(-1)[0].cpu())
    array = np.asarray(value)
    return float(array.reshape(-1)[0])


def get_model(usr_args):
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = [f"--device={device}"]
    temporal_ensemble_coeff = usr_args.get("act_temporal_ensemble_coeff")
    recovery_enabled = _as_bool(usr_args.get("act_recovery_enabled", False))
    if temporal_ensemble_coeff is not None and recovery_enabled:
        raise ValueError(
            "Contact recovery and temporal ensembling must be evaluated separately."
        )
    n_action_steps = usr_args.get("n_action_steps")
    if temporal_ensemble_coeff is not None:
        temporal_ensemble_coeff = float(temporal_ensemble_coeff)
        if temporal_ensemble_coeff < 0:
            raise ValueError(
                "act_temporal_ensemble_coeff must be non-negative, got "
                f"{temporal_ensemble_coeff}."
            )
        if n_action_steps not in (None, 1, "1"):
            raise ValueError(
                "Temporal ensembling requires n_action_steps=1 or unset."
            )
        cli_overrides.extend(
            [
                "--n_action_steps=1",
                f"--temporal_ensemble_coeff={temporal_ensemble_coeff}",
            ]
        )
    elif n_action_steps is not None:
        n_action_steps = int(n_action_steps)
        if n_action_steps <= 0:
            raise ValueError(
                f"n_action_steps must be positive, got {n_action_steps}."
            )
        cli_overrides.append(f"--n_action_steps={n_action_steps}")
    try:
        policy = ACTPolicy.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
        policy = ACTPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(
            image_key,
            image_key.removeprefix("observation.images."),
        )

    policy.act_device = next(policy.parameters()).device
    policy.act_step = int(usr_args.get("act_step", 8))
    policy.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.act_image_keys = image_keys
    policy.image_key_map = image_key_map
    policy.act_recovery = None
    if recovery_enabled:
        policy.act_recovery = ContactPlateauRecovery(
            min_step=int(usr_args.get("act_recovery_min_step", 60)),
            plateau_steps=int(usr_args.get("act_recovery_plateau_steps", 10)),
            min_press_depth_m=float(
                usr_args.get("act_recovery_min_press_depth_m", 0.0011)
            ),
            press_epsilon_m=float(
                usr_args.get("act_recovery_press_epsilon_m", 0.0001)
            ),
            cooldown_steps=int(usr_args.get("act_recovery_cooldown_steps", 25)),
            max_replans=int(usr_args.get("act_recovery_max_replans", 2)),
        )

    if policy.act_step <= 0:
        raise ValueError(f"act_step must be positive, got {policy.act_step}.")

    return policy


def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.act_step):
        runs_model_inference = (
            model.config.temporal_ensemble_coeff is not None
            or len(model._action_queue) == 0
        )
        started_at = start_inference(model.act_device) if runs_model_inference else None
        state = final_obs
        for key in str(model.state_obs_path).split("/"):
            if key:
                state = state[key]
        state_tensor = state.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=torch.float32, device=model.act_device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch = {"observation.state": state_tensor}
        for image_key in model.act_image_keys:
            camera_name = model.image_key_map.get(
                image_key,
                image_key.removeprefix("observation.images."),
            )
            image = final_obs["sensor"][camera_name]["color"]
            image_tensor = image.detach().to(device=model.act_device, dtype=torch.float32) if isinstance(image, torch.Tensor) else torch.as_tensor(image, dtype=torch.float32, device=model.act_device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
            if torch.max(image_tensor) > 1.5:
                image_tensor = image_tensor / 255.0
            batch[image_key] = image_tensor

        action = model.select_action(batch)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action shape [B, D], but got {tuple(action.shape)}.")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = f"Policy action has dim {policy_action_dim}, but env expects {env_action_dim}."
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        action_tensor = action.detach().to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        if runs_model_inference:
            finish_inference(started_at, inference_times_s, model.act_device)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            break
        if model.act_recovery is not None:
            diagnostics = env.unwrapped.get_episode_diagnostics()
            queue_size = len(model._action_queue)
            should_replan = model.act_recovery.observe(
                max_press_depth_m=_metric_scalar(
                    diagnostics["max_press_depth_m"]
                ),
                success_depth_m=_metric_scalar(
                    diagnostics["movement_threshold_m"]
                ),
                queued_actions=queue_size,
            )
            if should_replan:
                model._action_queue.clear()
        if isinstance(truncated, torch.Tensor):
            is_truncated = truncated.any().item()
        elif isinstance(truncated, np.ndarray):
            is_truncated = truncated.any()
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()
    if model.act_recovery is not None:
        model.act_recovery.reset()


def get_episode_metrics(model):
    if model.act_recovery is None:
        return {"contact_plateau_recovery": {"enabled": False}}
    return {"contact_plateau_recovery": model.act_recovery.metrics()}
