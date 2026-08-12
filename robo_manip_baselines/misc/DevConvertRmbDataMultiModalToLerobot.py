"""Convert RmbData to LeRobot while preserving Cobotta multi-sensor data.

In addition to the regular LeRobot state/action/RGB features, this converter
stores every HDF5 time-series in the RMB file (including the high-rate sensor
chunks) as a LeRobot feature.  Depth videos and MP3 recordings do not have a
native LeRobot feature type, so their original files are copied losslessly to
``assets/`` and indexed by ``meta/rmb_assets.json``.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from robo_manip_baselines.common import DataKey, RmbData
from robo_manip_baselines.misc.ConvertRmbDataToLerobot import (
    DEFAULT_DATASET_CONFIG,
    DatasetConfig,
    ConvertRmbDataToLerobot,
)


class ConvertRmbDataToLerobotV2(ConvertRmbDataToLerobot):
    """RMB converter that retains all recorded Cobotta multi-sensor modalities."""

    def _feature_name(self, key: str) -> str:
        # Keep the policy-facing features compatible with the original converter.
        names = {
            DataKey.MEASURED_JOINT_POS: "observation.state",
            DataKey.COMMAND_JOINT_POS: "action",
            DataKey.MEASURED_JOINT_VEL: "observation.velocity",
            DataKey.TIME: "observation.source_time",
            DataKey.REWARD: "reward",
        }
        return names.get(key, f"observation.rmb.{key}")

    @staticmethod
    def _feature_from_dataset(dataset) -> dict:
        """Build a LeRobot numeric feature without changing RMB precision."""
        shape = dataset.shape[1:] or (1,)
        feature = {"dtype": np.dtype(dataset.dtype).name, "shape": shape}
        if len(shape) == 1:
            feature["names"] = [f"value_{idx}" for idx in range(shape[0])]
        return feature

    def create_empty_dataset(
        self,
        repo_id: str,
        root: str,
        mode: Literal["video", "image"] = "video",
        *,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ):
        with RmbData(self.rmb_path_list[0]) as rmb_data:
            self.camera_name_list = list(rmb_data.attrs["camera_names"])
            self.hdf5_feature_map = {
                key: self._feature_name(key) for key in rmb_data.h5file.keys()
            }
            self.rmb_feature_schema = {
                feature_name: {
                    "rmb_key": key,
                    "shape": list(rmb_data.h5file[key].shape[1:]),
                    "dtype": np.dtype(rmb_data.h5file[key].dtype).name,
                }
                for key, feature_name in self.hdf5_feature_map.items()
            }
            features = {
                feature_name: self._feature_from_dataset(rmb_data.h5file[key])
                for key, feature_name in self.hdf5_feature_map.items()
            }

            # The original converter exposes these names; retain the joint names.
            num_joints = rmb_data[DataKey.COMMAND_JOINT_POS].shape[1]
            self.joint_name_list = [f"joint_{idx}" for idx in range(num_joints)]
            for key in ("observation.state", "action", "observation.velocity"):
                if key in features:
                    features[key]["names"] = self.joint_name_list

            for camera_name in self.camera_name_list:
                rgb_key = DataKey.get_rgb_image_key(camera_name)
                if rgb_key not in rmb_data:
                    continue
                features[f"observation.images.{camera_name}_rgb"] = {
                    "dtype": mode,
                    "shape": rmb_data[rgb_key][0].shape,
                    "names": ["height", "width", "channels"],
                }

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=root,
            fps=30,
            features=features,
            use_videos=dataset_config.use_videos,
            tolerance_s=dataset_config.tolerance_s,
            image_writer_processes=dataset_config.image_writer_processes,
            image_writer_threads=dataset_config.image_writer_threads,
            video_backend=dataset_config.video_backend,
        )

    def _task_description(self, rmb_data) -> str:
        if self.task_desc is not None:
            return self.task_desc
        if "task_desc" in rmb_data.attrs:
            return str(rmb_data.attrs["task_desc"])
        # A task string is required by LeRobot.  Preserve unknown RMB tasks rather
        # than rejecting a conversion solely because their environment is new.
        return str(rmb_data.attrs.get("env", ""))

    def _copy_episode_assets(self, rmb_path: str, episode_index: int, attrs: dict):
        source = Path(rmb_path)
        assets = {"source_rmb": str(source), "audio": [], "rgb": [], "depth": []}
        episode_dir = self.dataset.root / "assets" / f"episode_{episode_index:06d}"

        for source_file in sorted(source.glob("*.rmb.mp3")):
            destination = episode_dir / "audio" / source_file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            assets["audio"].append(
                {
                    "path": str(destination.relative_to(self.dataset.root)),
                    "sample_rate_hz": 4950,
                }
            )

        for camera_name in self.camera_name_list:
            rgb_key = DataKey.get_rgb_image_key(camera_name)
            source_file = source / f"{rgb_key}.rmb.mp4"
            if not source_file.is_file():
                continue
            destination = episode_dir / "rgb" / source_file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            assets["rgb"].append(
                {"camera": camera_name, "path": str(destination.relative_to(self.dataset.root))}
            )

        for camera_name in self.camera_name_list:
            depth_key = DataKey.get_depth_image_key(camera_name)
            source_file = source / f"{depth_key}.rmb.mp4"
            if not source_file.is_file():
                continue
            destination = episode_dir / "depth" / source_file.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)
            assets["depth"].append(
                {
                    "camera": camera_name,
                    "path": str(destination.relative_to(self.dataset.root)),
                    "fovy": attrs.get(f"{depth_key}_fovy"),
                }
            )
        return assets

    def populate_dataset(self, episodes: list[int] | None = None):
        selected = set(range(len(self.rmb_path_list)) if episodes is None else episodes)
        self.asset_manifest = {"format": "rmb-v2-assets", "episodes": {}}

        for rmb_idx, rmb_path in tqdm.tqdm(enumerate(self.rmb_path_list)):
            if rmb_idx not in selected:
                continue
            with RmbData(rmb_path) as rmb_data:
                task_desc = self._task_description(rmb_data)
                images = {
                    camera_name: rmb_data[DataKey.get_rgb_image_key(camera_name)][:]
                    for camera_name in self.camera_name_list
                    if DataKey.get_rgb_image_key(camera_name) in rmb_data
                }
                num_frames = len(rmb_data[DataKey.TIME])
                episode_index = self.dataset.meta.total_episodes

                for frame_idx in range(num_frames):
                    frame = {"task": task_desc}
                    for hdf5_key, feature_name in self.hdf5_feature_map.items():
                        value = np.asarray(rmb_data.h5file[hdf5_key][frame_idx])
                        # LeRobot represents scalar numeric features as shape (1,).
                        frame[feature_name] = value.reshape(1) if value.ndim == 0 else value
                    for camera_name, image in images.items():
                        frame[f"observation.images.{camera_name}_rgb"] = image[frame_idx]
                    self.dataset.add_frame(frame)

                self.dataset.save_episode()
                attrs = {key: rmb_data.attrs[key] for key in rmb_data.attrs.keys()}
                self.asset_manifest["episodes"][str(episode_index)] = self._copy_episode_assets(
                    rmb_path, episode_index, attrs
                )

        with open(self.dataset.root / "meta" / "rmb_assets.json", "w") as file:
            json.dump(self.asset_manifest, file, indent=2, default=lambda value: value.item())
        with open(self.dataset.root / "meta" / "rmb_schema.json", "w") as file:
            json.dump(self.rmb_feature_schema, file, indent=2)

    def port_data(
        self,
        episodes: list[int] | None = None,
        push_to_hub: bool = False,
        mode: Literal["video", "image"] = "video",
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ):
        root_dir = HF_LEROBOT_HOME if self.output_dir is None else Path(self.output_dir)
        dataset_path = root_dir / self.repo_id
        if dataset_path.exists():
            shutil.rmtree(dataset_path)

        print(f"[{self.__class__.__name__}] Start dataset conversion: {dataset_path.resolve()}")
        self.create_empty_dataset(self.repo_id, dataset_path, mode, dataset_config=dataset_config)
        self.populate_dataset(episodes)
        self.dataset.finalize()
        if push_to_hub:
            LeRobotDataset(repo_id=self.repo_id, root=dataset_path).push_to_hub()
        print(f"[{self.__class__.__name__}] Complete dataset conversion.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--task_desc", type=str, default=None)
    parser.add_argument("--enable_mobile", action="store_true")
    args = parser.parse_args()
    ConvertRmbDataToLerobotV2(**vars(args)).port_data()
