import os

import numpy as np

from .DataKey import DataKey
from .RmbData import RmbData


def _to_python_attr_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _to_python_attr_value(value.item())
        return [_to_python_attr_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_python_attr_value(v) for v in value]
    return value


def _normalize_observation_value(key, value):
    value = np.asarray(value)
    if DataKey.is_rgb_image_key(key):
        return value.astype(np.uint8)
    if DataKey.is_depth_image_key(key):
        return value.astype(np.float32)
    if value.dtype.kind == "f":
        return value.astype(np.float32)
    return value


def _normalize_action_value(value):
    value = np.asarray(value)
    if value.dtype.kind == "f":
        return value.astype(np.float32)
    return value


def _zero_like_step_value(value):
    value = np.asarray(value)
    return np.zeros_like(value)


def classify_rmb_keys(rmb_keys):
    observation_keys = []
    action_keys = []
    extra_step_keys = []

    for key in sorted(rmb_keys):
        if key in (DataKey.TIME, DataKey.REWARD):
            continue
        if key.startswith("command_"):
            action_keys.append(key)
        elif key.startswith("measured_") or DataKey.is_rgb_image_key(
            key
        ) or DataKey.is_depth_image_key(key) or DataKey.is_pointcloud_key(key):
            observation_keys.append(key)
        else:
            extra_step_keys.append(key)

    return observation_keys, action_keys, extra_step_keys


def infer_rmb_rlds_schema(rmb_path):
    with RmbData(rmb_path) as rmb_data:
        observation_keys, action_keys, extra_step_keys = classify_rmb_keys(
            rmb_data.keys()
        )
        schema = {
            "observation": {},
            "action": {},
            "extra_step": {},
            "episode_metadata": {},
        }

        for key in observation_keys:
            data = rmb_data[key][0]
            schema["observation"][key] = {
                "shape": tuple(np.asarray(data).shape),
                "dtype": np.asarray(data).dtype,
            }
        for key in action_keys:
            data = rmb_data[key][0]
            schema["action"][key] = {
                "shape": tuple(np.asarray(data).shape),
                "dtype": np.asarray(data).dtype,
            }
        for key in extra_step_keys:
            data = rmb_data[key][0]
            schema["extra_step"][key] = {
                "shape": tuple(np.asarray(data).shape),
                "dtype": np.asarray(data).dtype,
            }

        for key in rmb_data.attrs.keys():
            schema["episode_metadata"][key] = _to_python_attr_value(rmb_data.attrs[key])

    return schema


def make_rlds_episode_from_rmb(
    rmb_path,
    *,
    pad_terminal_step=True,
    task_desc_default="",
):
    with RmbData(rmb_path) as rmb_data:
        observation_keys, action_keys, extra_step_keys = classify_rmb_keys(
            rmb_data.keys()
        )

        data_seq = {key: rmb_data[key][:] for key in rmb_data.keys()}
        episode_len = len(data_seq[DataKey.TIME])

        attrs = {
            key: _to_python_attr_value(rmb_data.attrs[key]) for key in rmb_data.attrs.keys()
        }

    task_desc = attrs.get("task_desc", "")
    if task_desc == "":
        task_desc = task_desc_default

    episode_metadata = {
        "episode_id": os.path.splitext(os.path.basename(rmb_path.rstrip("/")))[0],
        "source_path": rmb_path,
        "env": attrs.get("env", ""),
        "demo_name": attrs.get("demo_name", ""),
        "task_desc": task_desc,
        "camera_names": attrs.get("camera_names", []),
        "rgb_tactile_names": attrs.get("rgb_tactile_names", []),
        "pointcloud_camera_names": attrs.get("pointcloud_camera_names", []),
        "world_idx": attrs.get("world_idx", -1),
        "version": attrs.get("version", ""),
        "attrs": attrs,
    }

    steps = []
    for step_idx in range(episode_len):
        step = {
            "observation": {
                key: _normalize_observation_value(key, data_seq[key][step_idx])
                for key in observation_keys
            },
            "action": {
                key: _normalize_action_value(data_seq[key][step_idx]) for key in action_keys
            },
            "reward": np.float32(data_seq.get(DataKey.REWARD, np.zeros(episode_len))[step_idx]),
            "discount": np.float32(1.0),
            "is_first": step_idx == 0,
            "is_last": False,
            "is_terminal": False,
            "language_instruction": task_desc,
            "time": np.float32(data_seq[DataKey.TIME][step_idx]),
        }
        for key in extra_step_keys:
            step[key] = np.asarray(data_seq[key][step_idx])
        steps.append(step)

    if pad_terminal_step and len(steps) > 0:
        terminal_step = {
            "observation": {
                key: _normalize_observation_value(key, data_seq[key][-1])
                for key in observation_keys
            },
            "action": {
                key: _zero_like_step_value(data_seq[key][-1]) for key in action_keys
            },
            "reward": np.float32(0.0),
            "discount": np.float32(0.0),
            "is_first": False,
            "is_last": True,
            "is_terminal": False,
            "language_instruction": task_desc,
            "time": np.float32(data_seq[DataKey.TIME][-1]),
        }
        for key in extra_step_keys:
            terminal_step[key] = np.asarray(data_seq[key][-1])
        steps.append(terminal_step)
    elif len(steps) > 0:
        steps[-1]["is_last"] = True

    return {
        "steps": steps,
        **episode_metadata,
    }
