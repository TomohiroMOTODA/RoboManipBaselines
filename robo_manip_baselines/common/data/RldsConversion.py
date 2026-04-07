import copy
from collections.abc import Mapping

import numpy as np
import yaml

from robo_manip_baselines import __version__

from .DataKey import DataKey


_MISSING = object()


def load_rlds_conversion_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return normalize_rlds_conversion_config(config)


def normalize_rlds_conversion_config(config):
    normalized = copy.deepcopy(config)
    normalized.setdefault("step_mappings", {})
    normalized.setdefault("episode_metadata_mappings", {})
    normalized.setdefault("constant_episode_metadata", {})
    normalized.setdefault("conversion", {})

    conversion = normalized["conversion"]
    conversion.setdefault("fps", 30.0)
    conversion.setdefault("drop_last_step_if_is_last", True)
    conversion.setdefault("skip_invalid_episodes", True)

    normalized["step_mappings"] = {
        key: normalize_mapping_spec(spec, f"step_mappings.{key}")
        for key, spec in normalized["step_mappings"].items()
    }
    normalized["episode_metadata_mappings"] = {
        key: normalize_mapping_spec(spec, f"episode_metadata_mappings.{key}")
        for key, spec in normalized["episode_metadata_mappings"].items()
    }

    return normalized


def normalize_mapping_spec(spec, spec_name):
    if isinstance(spec, str):
        return {"path": spec, "optional": False}

    if not isinstance(spec, Mapping):
        raise TypeError(
            f"[normalize_mapping_spec] '{spec_name}' must be a string or mapping: {type(spec)}"
        )

    normalized = dict(spec)
    if "path" not in normalized and "value" not in normalized:
        raise ValueError(
            f"[normalize_mapping_spec] '{spec_name}' must have either 'path' or 'value'."
        )
    normalized.setdefault("optional", False)
    return normalized


def resolve_nested_value(data, path):
    if path is None or path == "":
        return data

    current = data
    for key in path.split("."):
        if isinstance(current, Mapping):
            if key not in current:
                return _MISSING
            current = current[key]
        else:
            try:
                current = current[key]
            except Exception:
                return _MISSING

    return current


def to_python_scalar_or_list(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_python_scalar_or_list(value.item())
        return [to_python_scalar_or_list(v) for v in value.tolist()]
    if isinstance(value, list):
        return [to_python_scalar_or_list(v) for v in value]
    if isinstance(value, tuple):
        return [to_python_scalar_or_list(v) for v in value]
    return value


def normalize_step_value(key, value):
    value_np = np.asarray(value)

    if DataKey.is_rgb_image_key(key):
        if value_np.dtype.kind == "f":
            max_value = float(np.max(value_np)) if value_np.size > 0 else 0.0
            if max_value <= 1.0:
                value_np = np.clip(np.round(255.0 * value_np), 0, 255).astype(np.uint8)
            else:
                value_np = np.clip(np.round(value_np), 0, 255).astype(np.uint8)
        else:
            value_np = value_np.astype(np.uint8)
    elif DataKey.is_depth_image_key(key) or DataKey.is_pointcloud_key(key):
        value_np = value_np.astype(np.float32)
    elif value_np.dtype.kind == "f":
        value_np = value_np.astype(np.float64)

    if value_np.ndim == 0:
        return value_np.item()
    return value_np


def is_truthy(value):
    if value is _MISSING:
        return False

    value_np = np.asarray(value)
    if value_np.ndim == 0:
        return bool(value_np.item())
    return bool(np.any(value_np))


class RldsEpisodeConverter:
    def __init__(self, config):
        self.config = normalize_rlds_conversion_config(config)
        self.step_mappings = self.config["step_mappings"]
        self.episode_metadata_mappings = self.config["episode_metadata_mappings"]
        self.constant_episode_metadata = self.config["constant_episode_metadata"]
        self.conversion = self.config["conversion"]

    def should_skip_episode(self, episode):
        if not self.conversion["skip_invalid_episodes"]:
            return False

        invalid = resolve_nested_value(episode, "invalid")
        return is_truthy(invalid)

    def convert_episode(self, episode, episode_index):
        steps = list(episode["steps"])
        if len(steps) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] RLDS episode has no steps: index={episode_index}"
            )

        if self.conversion["drop_last_step_if_is_last"] and is_truthy(
            resolve_nested_value(steps[-1], "is_last")
        ):
            steps = steps[:-1]

        if len(steps) == 0:
            raise ValueError(
                f"[{self.__class__.__name__}] RLDS episode became empty after dropping the terminal last step: index={episode_index}"
            )

        all_data_seq = {}
        for key, spec in self.step_mappings.items():
            data_seq = self._collect_step_sequence(steps, key, spec, episode_index)
            if data_seq is not None:
                all_data_seq[key] = data_seq

        if DataKey.TIME not in all_data_seq:
            fps = float(self.conversion["fps"])
            all_data_seq[DataKey.TIME] = [step_idx / fps for step_idx in range(len(steps))]

        meta_data = self._collect_episode_metadata(episode, episode_index)
        self._finalize_meta_data(meta_data, all_data_seq, episode_index)

        return all_data_seq, meta_data

    def _collect_step_sequence(self, steps, key, spec, episode_index):
        data_seq = []

        for step_idx, step in enumerate(steps):
            value = self._resolve_mapping_value(step, spec)
            if value is _MISSING:
                if spec["optional"]:
                    return None
                raise ValueError(
                    f"[{self.__class__.__name__}] Missing required RLDS path '{spec.get('path')}' for RMB key '{key}' at episode={episode_index}, step={step_idx}."
                )
            data_seq.append(normalize_step_value(key, value))

        return data_seq

    def _collect_episode_metadata(self, episode, episode_index):
        meta_data = copy.deepcopy(self.constant_episode_metadata)

        for key, spec in self.episode_metadata_mappings.items():
            value = self._resolve_mapping_value(episode, spec)
            if value is _MISSING:
                if spec["optional"]:
                    continue
                raise ValueError(
                    f"[{self.__class__.__name__}] Missing required RLDS episode metadata path '{spec.get('path')}' for RMB attr '{key}' at episode={episode_index}."
                )
            meta_data[key] = to_python_scalar_or_list(value)

        return meta_data

    def _resolve_mapping_value(self, source, spec):
        if "value" in spec:
            return spec["value"]

        value = resolve_nested_value(source, spec.get("path"))
        if value is _MISSING and "default" in spec:
            return spec["default"]
        return value

    def _finalize_meta_data(self, meta_data, all_data_seq, episode_index):
        meta_data.setdefault("version", __version__)
        meta_data.setdefault("source_format", "RLDS")
        meta_data.setdefault("source_episode_index", episode_index)

        if meta_data.get("episode_id") is None:
            meta_data["episode_id"] = f"rlds_episode_{episode_index:06d}"

        camera_names = meta_data.get("camera_names")
        if camera_names is None:
            camera_names = []
            for key in all_data_seq:
                if DataKey.is_rgb_image_key(key) or DataKey.is_depth_image_key(key):
                    camera_name = DataKey.get_camera_name(key)
                    if camera_name not in camera_names:
                        camera_names.append(camera_name)
            meta_data["camera_names"] = camera_names
        else:
            meta_data["camera_names"] = to_python_scalar_or_list(camera_names)

        pointcloud_camera_names = meta_data.get("pointcloud_camera_names")
        if pointcloud_camera_names is None:
            pointcloud_camera_names = []
            for key in all_data_seq:
                if DataKey.is_pointcloud_key(key):
                    camera_name = key[: -len("_pointcloud")]
                    if camera_name not in pointcloud_camera_names:
                        pointcloud_camera_names.append(camera_name)
            meta_data["pointcloud_camera_names"] = pointcloud_camera_names
        else:
            meta_data["pointcloud_camera_names"] = to_python_scalar_or_list(
                pointcloud_camera_names
            )

        rgb_tactile_names = meta_data.get("rgb_tactile_names")
        if rgb_tactile_names is None:
            meta_data["rgb_tactile_names"] = []
        else:
            meta_data["rgb_tactile_names"] = to_python_scalar_or_list(rgb_tactile_names)

        task_desc = meta_data.get("task_desc")
        if task_desc is None:
            meta_data["task_desc"] = ""
