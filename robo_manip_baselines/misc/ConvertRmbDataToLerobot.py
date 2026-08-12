import argparse
import concurrent.futures
import dataclasses
import inspect
import json
import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import HF_LEROBOT_HOME

from robo_manip_baselines.common import DataKey, RmbData, find_rmb_files


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None
    vcodec: str = "libsvtav1"
    prefetch_episodes: int = 1


DEFAULT_DATASET_CONFIG = DatasetConfig()
CONVERSION_MANIFEST = "rmb_conversion_manifest.json"


@dataclasses.dataclass(frozen=True)
class LoadedRmbEpisode:
    rmb_idx: int
    source_id: str
    source_fingerprint: dict
    task_desc: str
    images_per_camera: dict[str, np.ndarray]
    state: torch.Tensor
    action: torch.Tensor
    velocity: torch.Tensor | None
    effort: torch.Tensor | None
    num_frames: int


class ConvertRmbDataToLerobot:
    def __init__(
        self,
        path,
        output_dir,
        repo_id,
        task_desc,
        enable_mobile,
        camera_names=None,
    ):
        self.rmb_path_list = find_rmb_files(path)
        self.task_desc = task_desc
        self.enable_mobile = enable_mobile
        self.camera_names = camera_names

        if repo_id is None:
            if output_dir is None:
                raise ValueError("Either repo_id or output_dir must be specified")
            output_path = Path(output_dir)
            repo_name = output_path.name
            if repo_name == "":
                raise ValueError(f"Invalid output_dir: {output_dir}")
            self.repo_id = repo_name
            self.output_dir = str(output_path.parent)
        else:
            self.repo_id = repo_id
            self.output_dir = output_dir

    def get_supported_kwargs(self, callable_obj, kwargs: dict) -> dict:
        signature = inspect.signature(callable_obj)
        if any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        ):
            return kwargs

        unsupported_keys = [
            key for key in kwargs.keys() if key not in signature.parameters
        ]
        if unsupported_keys:
            print(
                f"[{self.__class__.__name__}] Ignoring unsupported kwargs for "
                f"{callable_obj.__qualname__}: {unsupported_keys}"
            )

        return {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }

    def get_camera_name_list(self, rmb_data) -> list[str]:
        if self.camera_names is not None:
            return list(self.camera_names)

        if "camera_names" in rmb_data.attrs:
            camera_names = rmb_data.attrs["camera_names"]
            return [
                camera_name.decode("utf-8")
                if isinstance(camera_name, bytes)
                else str(camera_name)
                for camera_name in camera_names
            ]

        camera_names = [
            DataKey.get_camera_name(key)
            for key in rmb_data.keys()
            if DataKey.is_rgb_image_key(key)
        ]
        if len(camera_names) == 0:
            raise ValueError(
                "camera_names attr is missing and no RGB image keys were found. "
                "Specify camera names with --camera_names."
            )

        print(
            f"[{self.__class__.__name__}] camera_names attr is missing. "
            f"Inferred camera names from RGB image keys: {camera_names}"
        )
        return camera_names

    def create_empty_dataset(
        self,
        repo_id: str,
        root: str,
        mode: Literal["video", "image"] = "video",
        *,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ):
        with RmbData(self.rmb_path_list[0]) as rmb_data:
            num_joints = rmb_data[DataKey.COMMAND_JOINT_POS].shape[1]
            self.joint_name_list = [
                f"joint_{joint_idx}" for joint_idx in range(num_joints)
            ]
            self.camera_name_list = self.get_camera_name_list(rmb_data)

            features = {
                "observation.state": {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                },
                "action": {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                },
            }

            self.has_velocity = DataKey.MEASURED_JOINT_VEL in rmb_data
            if self.has_velocity:
                features["observation.velocity"] = {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                }

            self.has_effort = DataKey.MEASURED_JOINT_TORQUE in rmb_data
            if self.has_effort:
                features["observation.effort"] = {
                    "dtype": "float64",
                    "shape": (len(self.joint_name_list),),
                    "names": [
                        self.joint_name_list,
                    ],
                }

            for camera_name in self.camera_name_list:
                rgb_image_key = DataKey.get_rgb_image_key(camera_name)
                rgb_image_shape = rmb_data[rgb_image_key][0].shape
                features[f"observation.images.{camera_name}_rgb"] = {
                    "dtype": mode,
                    "shape": rgb_image_shape,
                    "names": [
                        "height",
                        "width",
                        "channels",
                    ],
                }

        self.dataset = LeRobotDataset.create(
            **self.get_supported_kwargs(
                LeRobotDataset.create,
                {
                    "repo_id": repo_id,
                    "root": root,
                    "fps": 30,
                    "features": features,
                    "use_videos": dataset_config.use_videos,
                    "tolerance_s": dataset_config.tolerance_s,
                    "image_writer_processes": dataset_config.image_writer_processes,
                    "image_writer_threads": dataset_config.image_writer_threads,
                    "video_backend": dataset_config.video_backend,
                    "vcodec": dataset_config.vcodec,
                },
            )
        )

    def load_existing_dataset(
        self,
        dataset_path: Path,
        *,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ):
        with RmbData(self.rmb_path_list[0]) as rmb_data:
            num_joints = rmb_data[DataKey.COMMAND_JOINT_POS].shape[1]
            self.joint_name_list = [
                f"joint_{joint_idx}" for joint_idx in range(num_joints)
            ]
            self.camera_name_list = self.get_camera_name_list(rmb_data)
            self.has_velocity = DataKey.MEASURED_JOINT_VEL in rmb_data
            self.has_effort = DataKey.MEASURED_JOINT_TORQUE in rmb_data

        self.dataset = LeRobotDataset(
            **self.get_supported_kwargs(
                LeRobotDataset,
                {
                    "repo_id": self.repo_id,
                    "root": dataset_path,
                    "tolerance_s": dataset_config.tolerance_s,
                    "video_backend": dataset_config.video_backend,
                    "vcodec": dataset_config.vcodec,
                },
            )
        )
        if dataset_config.image_writer_processes or dataset_config.image_writer_threads:
            self.dataset.start_image_writer(
                dataset_config.image_writer_processes,
                dataset_config.image_writer_threads,
            )

    def get_manifest_path(self) -> Path:
        return self.dataset.root / "meta" / CONVERSION_MANIFEST

    def load_conversion_manifest(self) -> dict:
        manifest_path = self.get_manifest_path()
        if not manifest_path.exists():
            return {"version": 1, "sources": {}}
        with open(manifest_path) as f:
            return json.load(f)

    def save_conversion_manifest(self, manifest: dict):
        manifest_path = self.get_manifest_path()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=4)

    def get_source_id(self, rmb_path: str) -> str:
        return str(Path(rmb_path).resolve())

    def get_source_fingerprint(self, rmb_path: str) -> dict:
        path = Path(rmb_path)
        files = [path]
        if path.is_dir():
            files = [path / "main.rmb.hdf5", *sorted(path.glob("*.rmb.mp4"))]

        fingerprint_files = []
        for file_path in files:
            stat = file_path.stat()
            fingerprint_files.append(
                {
                    "path": file_path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return {"files": fingerprint_files}

    def load_raw_images_per_camera(self, rmb_data) -> dict[str, np.ndarray]:
        images_per_camera = {}
        for camera_name in self.camera_name_list:
            rgb_image_key = DataKey.get_rgb_image_key(camera_name)
            images_per_camera[f"{camera_name}_rgb"] = rmb_data[rgb_image_key][:]
        return images_per_camera

    def load_raw_episode_data(
        self, rmb_data
    ) -> tuple[
        dict[str, np.ndarray],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        state_joint = rmb_data[DataKey.MEASURED_JOINT_POS][:]
        action_joint = rmb_data[DataKey.COMMAND_JOINT_POS][:]
        if self.enable_mobile:
            state_omni = rmb_data[DataKey.MEASURED_MOBILE_OMNI_VEL][:]
            state_all = np.concatenate([state_joint, state_omni], axis=1)

            action_omni = rmb_data[DataKey.COMMAND_MOBILE_OMNI_VEL][:]
            action_all = np.concatenate([action_joint, action_omni], axis=1)

            state = torch.from_numpy(state_all)
            action = torch.from_numpy(action_all)
        else:
            state = torch.from_numpy(state_joint)
            action = torch.from_numpy(action_joint)

        if self.has_velocity:
            velocity_joint = rmb_data[DataKey.MEASURED_JOINT_VEL][:]
            if self.enable_mobile:
                velocity_omni = rmb_data[DataKey.MEASURED_MOBILE_OMNI_VEL][:]
                velocity_all = np.concatenate([velocity_joint, velocity_omni], axis=1)
                velocity = torch.from_numpy(velocity_all)
            else:
                velocity = torch.from_numpy(velocity_joint)
        else:
            velocity = None

        if self.has_effort:
            raise NotImplementedError(
                f"[{self.__class__.__name__}] Conversion of effort data is not supported."
            )
        else:
            effort = None

        images_per_camera = self.load_raw_images_per_camera(rmb_data)

        return images_per_camera, state, action, velocity, effort

    def get_task_desc(self, rmb_data) -> str:
        if self.task_desc is not None:
            return self.task_desc
        if "task_desc" in rmb_data.attrs:
            return rmb_data.attrs["task_desc"]

        env_name = rmb_data.attrs["env"]
        if env_name == "MujocoUR5eCableEnv":
            return "pass the cable between two poles"
        if env_name == "MujocoUR5eRingEnv":
            return "pick a ring and put it around the pole"
        if env_name == "MujocoUR5eParticleEnv":
            return "scoop up particles"
        if env_name == "MujocoUR5eClothEnv":
            return "roll up the cloth"
        if env_name == "MujocoUR5eDoorEnv":
            return "open the door"
        if env_name == "MujocoHsrTidyupEnv":
            return "Bring the object to the box"

        raise ValueError(
            f"[{self.__class__.__name__}] Failed to retrieve the task description."
        )

    def load_rmb_episode(
        self,
        rmb_idx: int,
        rmb_path: str,
        source_id: str,
        source_fingerprint: dict,
    ) -> LoadedRmbEpisode:
        with RmbData(rmb_path) as rmb_data:
            task_desc = self.get_task_desc(rmb_data)
            images_per_camera, state, action, velocity, effort = (
                self.load_raw_episode_data(rmb_data)
            )
            num_frames = len(rmb_data[DataKey.TIME])

        return LoadedRmbEpisode(
            rmb_idx=rmb_idx,
            source_id=source_id,
            source_fingerprint=source_fingerprint,
            task_desc=task_desc,
            images_per_camera=images_per_camera,
            state=state,
            action=action,
            velocity=velocity,
            effort=effort,
            num_frames=num_frames,
        )

    def write_loaded_episode(self, episode: LoadedRmbEpisode, manifest: dict):
        if self.dataset.meta.total_episodes == 0:
            with open(self.dataset.root / "meta" / "modality.json", "w") as f:
                modality = {
                    "state": {
                        "single_arm": {
                            "start": 0,
                            "end": len(self.joint_name_list),
                        }
                    },
                    "action": {
                        "single_arm": {
                            "start": 0,
                            "end": len(self.joint_name_list),
                        }
                    },
                    "video": {
                        f"{camera_name}_rgb": {
                            "original_key": f"observation.images.{camera_name}_rgb"
                        }
                        for camera_name in self.camera_name_list
                    },
                    "annotation": {
                        "human.task_description": {"original_key": "task_index"}
                    },
                }
                json.dump(modality, f, indent=4)

        for i in range(episode.num_frames):
            frame = {
                "observation.state": episode.state[i],
                "action": episode.action[i],
            }

            for camera_name, image in episode.images_per_camera.items():
                frame[f"observation.images.{camera_name}"] = image[i]

            if self.has_velocity:
                frame["observation.velocity"] = episode.velocity[i]
            if self.has_effort:
                frame["observation.effort"] = episode.effort[i]

            frame["task"] = episode.task_desc

            self.dataset.add_frame(frame)

        output_episode_index = self.dataset.meta.total_episodes
        self.dataset.save_episode()
        manifest["sources"][episode.source_id] = {
            "fingerprint": episode.source_fingerprint,
            "episode_index": output_episode_index,
            "num_frames": episode.num_frames,
        }
        self.save_conversion_manifest(manifest)

    def populate_dataset(
        self,
        episodes: list[int] | None = None,
        skip_existing: bool = False,
        manifest: dict | None = None,
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
    ):
        if episodes is None:
            episode_indices = list(range(len(self.rmb_path_list)))
        else:
            episode_indices = list(episodes)

        if manifest is None:
            manifest = {"version": 1, "sources": {}}

        converted_count = 0
        skipped_count = 0
        initial_total_episodes = self.dataset.meta.total_episodes
        episodes_to_convert = []

        for rmb_idx in episode_indices:
            rmb_path = self.rmb_path_list[rmb_idx]
            source_id = self.get_source_id(rmb_path)
            source_fingerprint = self.get_source_fingerprint(rmb_path)
            manifest_entry = manifest["sources"].get(source_id)
            already_in_manifest = (
                manifest_entry is not None
                and manifest_entry.get("fingerprint") == source_fingerprint
            )
            already_in_legacy_dataset = (
                manifest_entry is None and rmb_idx < initial_total_episodes
            )
            if skip_existing and (already_in_manifest or already_in_legacy_dataset):
                if already_in_legacy_dataset:
                    manifest["sources"][source_id] = {
                        "fingerprint": source_fingerprint,
                        "episode_index": rmb_idx,
                        "num_frames": None,
                    }
                    self.save_conversion_manifest(manifest)
                skipped_count += 1
                continue

            episodes_to_convert.append(
                (rmb_idx, rmb_path, source_id, source_fingerprint)
            )

        prefetch_episodes = max(1, dataset_config.prefetch_episodes)
        if prefetch_episodes == 1:
            for episode_args in tqdm.tqdm(episodes_to_convert):
                episode = self.load_rmb_episode(*episode_args)
                self.write_loaded_episode(episode, manifest)
                converted_count += 1
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=prefetch_episodes
            ) as executor:
                futures = {}
                next_to_submit = 0
                while (
                    next_to_submit < len(episodes_to_convert)
                    and len(futures) < prefetch_episodes
                ):
                    futures[next_to_submit] = executor.submit(
                        self.load_rmb_episode, *episodes_to_convert[next_to_submit]
                    )
                    next_to_submit += 1

                for episode_pos in tqdm.tqdm(range(len(episodes_to_convert))):
                    episode = futures.pop(episode_pos).result()
                    if next_to_submit < len(episodes_to_convert):
                        futures[next_to_submit] = executor.submit(
                            self.load_rmb_episode,
                            *episodes_to_convert[next_to_submit],
                        )
                        next_to_submit += 1

                    self.write_loaded_episode(episode, manifest)
                    converted_count += 1

        return converted_count, skipped_count

    def get_stats_einops_patterns(self, num_workers=0):
        """These einops patterns will be used to aggregate batches and compute statistics.

        Note: We assume the images are in channel first format
        """

        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=num_workers,
            batch_size=2,
            shuffle=False,
        )
        batch = next(iter(dataloader))

        stats_patterns = {}

        for key in self.dataset.features:
            # sanity check that tensors are not float64
            assert batch[key].dtype != torch.float64

            # if isinstance(feats_type, (VideoFrame, Image)):
            if key in self.dataset.meta.camera_keys:
                # sanity check that images are channel first
                _, c, h, w = batch[key].shape
                assert (
                    c < h and c < w
                ), f"expect channel first images, but instead {batch[key].shape}"

                # sanity check that images are float32 in range [0,1]
                assert (
                    batch[key].dtype == torch.float32
                ), f"expect torch.float32, but instead {batch[key].dtype=}"
                assert (
                    batch[key].max() <= 1
                ), f"expect pixels lower than 1, but instead {batch[key].max()=}"
                assert (
                    batch[key].min() >= 0
                ), f"expect pixels greater than 1, but instead {batch[key].min()=}"

                stats_patterns[key] = "b c h w -> c 1 1"
            elif batch[key].ndim == 2:
                stats_patterns[key] = "b c -> c "
            elif batch[key].ndim == 1:
                stats_patterns[key] = "b -> 1"
            else:
                raise ValueError(f"{key}, {batch[key].shape}")

        return stats_patterns

    def create_seeded_dataloader(self, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            num_workers=8,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    def flatten_dict(self, d: dict, parent_key: str = "", sep: str = "/") -> dict:
        """Flatten a nested dictionary structure by collapsing nested keys into one key with a separator.

        For example:
        ```
        >>> dct = {"a": {"b": 1, "c": {"d": 2}}, "e": 3}`
        >>> print(flatten_dict(dct))
        {"a/b": 1, "a/c/d": 2, "e": 3}
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self.flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def unflatten_dict(self, d: dict, sep: str = "/") -> dict:
        outdict = {}
        for key, value in d.items():
            parts = key.split(sep)
            d = outdict
            for part in parts[:-1]:
                if part not in d:
                    d[part] = {}
                d = d[part]
            d[parts[-1]] = value
        return outdict

    def serialize_dict(
        self, stats: dict[str, torch.Tensor | np.ndarray | dict]
    ) -> dict:
        serialized_dict = {
            key: value.tolist() for key, value in self.flatten_dict(stats).items()
        }
        return self.unflatten_dict(serialized_dict)

    def port_data(
        self,
        episodes: list[int] | None = None,
        push_to_hub: bool = False,
        mode: Literal["video", "image"] = "video",
        dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
        resume: bool = False,
        skip_existing: bool = False,
        exact_stats: bool = False,
    ):
        root_dir = HF_LEROBOT_HOME if self.output_dir is None else Path(self.output_dir)
        dataset_path = root_dir / self.repo_id
        if dataset_path.exists() and not resume:
            shutil.rmtree(dataset_path)

        print(
            f"[{self.__class__.__name__}] Start dataset conversion: {dataset_path.resolve()}"
        )

        if resume and dataset_path.exists():
            self.load_existing_dataset(
                dataset_path,
                dataset_config=dataset_config,
            )
        else:
            self.create_empty_dataset(
                repo_id=self.repo_id,
                root=dataset_path,
                mode=mode,
                dataset_config=dataset_config,
            )

        manifest = self.load_conversion_manifest()
        converted_count, skipped_count = self.populate_dataset(
            episodes=episodes,
            skip_existing=skip_existing,
            manifest=manifest,
            dataset_config=dataset_config,
        )
        self.dataset.finalize()

        if converted_count == 0 and skipped_count > 0:
            print(
                f"[{self.__class__.__name__}] Skipped {skipped_count} already converted episodes."
            )
            return

        meta_stats = self.dataset.meta.stats
        if not exact_stats:
            if push_to_hub:
                self.dataset.push_to_hub()
            print(
                f"[{self.__class__.__name__}] Complete dataset conversion. "
                f"converted={converted_count}, skipped={skipped_count}"
            )
            return

        self.dataset = LeRobotDataset(repo_id=self.repo_id, root=dataset_path)

        stats_patterns = self.get_stats_einops_patterns(8)

        data_num = len(self.dataset)
        q01, q99 = {}, {}
        data_dir = {
            key: []
            for key, pattern in stats_patterns.items()
            if key not in self.dataset.meta.camera_keys
        }

        for i in range(data_num):
            sample = self.dataset[i]
            for key in data_dir:
                data_dir[key].append(sample[key].float())

        for key in data_dir:
            data_dir[key] = torch.stack(data_dir[key], dim=0)

            q01[key] = torch.quantile(data_dir[key], 0.01, dim=0)
            q99[key] = torch.quantile(data_dir[key], 0.99, dim=0)

        for key in stats_patterns:
            if key in self.dataset.meta.camera_keys:
                continue
            meta_stats[key]["q01"] = np.atleast_1d(q01[key].numpy())
            meta_stats[key]["q99"] = np.atleast_1d(q99[key].numpy())

        serialized_stats = self.serialize_dict(meta_stats)

        with open(self.dataset.root / "meta" / "stats.json", "w") as f:
            json.dump(serialized_stats, f, indent=4)

        if push_to_hub:
            self.dataset.push_to_hub()

        print(
            f"[{self.__class__.__name__}] Complete dataset conversion. "
            f"converted={converted_count}, skipped={skipped_count}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--repo_id", type=str, default=None)
    parser.add_argument("--task_desc", type=str, default=None)
    parser.add_argument("--enable_mobile", action="store_true")
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Camera names to use when the RMB data does not have the "
            "'camera_names' attribute. If omitted, names are inferred from "
            "*_rgb_image keys when possible."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing LeRobot dataset instead of deleting it.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help=(
            "Skip RMB sources already recorded in the conversion manifest. "
            "This implies --resume."
        ),
    )
    parser.add_argument(
        "--exact_stats",
        action="store_true",
        help=(
            "Re-read the final dataset to compute exact q01/q99 for numeric keys. "
            "Slower; by default LeRobot's episode-aggregated stats are kept."
        ),
    )
    parser.add_argument(
        "--image_writer_processes",
        type=int,
        default=DEFAULT_DATASET_CONFIG.image_writer_processes,
    )
    parser.add_argument(
        "--image_writer_threads",
        type=int,
        default=DEFAULT_DATASET_CONFIG.image_writer_threads,
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default=DEFAULT_DATASET_CONFIG.vcodec,
        choices=["h264", "hevc", "libsvtav1"],
        help="Video codec. h264 is usually faster than the default libsvtav1.",
    )
    parser.add_argument(
        "--prefetch_episodes",
        type=int,
        default=DEFAULT_DATASET_CONFIG.prefetch_episodes,
        help=(
            "Number of RMB episodes to load/decode in parallel. "
            "Each prefetched episode is held fully in RAM."
        ),
    )
    args = parser.parse_args()

    dataset_config = dataclasses.replace(
        DEFAULT_DATASET_CONFIG,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        vcodec=args.vcodec,
        prefetch_episodes=args.prefetch_episodes,
    )
    rmb_to_lerobot = ConvertRmbDataToLerobot(
        path=args.path,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        task_desc=args.task_desc,
        enable_mobile=args.enable_mobile,
        camera_names=args.camera_names,
    )
    rmb_to_lerobot.port_data(
        dataset_config=dataset_config,
        resume=args.resume or args.skip_existing,
        skip_existing=args.skip_existing,
        exact_stats=args.exact_stats,
    )
