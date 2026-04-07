import argparse
import os
import shutil

import numpy as np

from robo_manip_baselines.common import find_rmb_files
from robo_manip_baselines.common.data.DataKey import DataKey
from robo_manip_baselines.common.data.RmbToRldsConversion import (
    infer_rmb_rlds_schema,
    make_rlds_episode_from_rmb,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "path",
        type=str,
        help="path to RMB data (*.rmb or *.hdf5) or directory containing them",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output TFDS data directory",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="RLDS dataset name",
    )
    parser.add_argument(
        "--task_desc_default",
        type=str,
        default="",
        help="default task description used when RMB attr 'task_desc' is empty",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="train",
        help="split name used in the generated RLDS dataset",
    )
    parser.add_argument(
        "--no_pad_terminal_step",
        action="store_true",
        help="do not append an extra terminal step. By default, an extra terminal step is appended to preserve all original RMB action labels.",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="remove output_dir before conversion if it already exists",
    )

    return parser.parse_args()


class ConvertRmbDataToRlds:
    def __init__(
        self,
        path,
        output_dir,
        dataset_name,
        task_desc_default="",
        split_name="train",
        no_pad_terminal_step=False,
        force_overwrite=False,
    ):
        self.rmb_path_list = find_rmb_files(path)
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.task_desc_default = task_desc_default
        self.split_name = split_name
        self.pad_terminal_step = not no_pad_terminal_step
        self.force_overwrite = force_overwrite

    def run(self):
        tf = self._import_tensorflow()
        tfds = self._import_tfds()

        self._prepare_output_dir()

        if len(self.rmb_path_list) < 10:
            print(
                f"[{self.__class__.__name__}] Warning: only {len(self.rmb_path_list)} episode(s) were found. This is enough for conversion testing, but not for meaningful policy training."
            )

        schema = infer_rmb_rlds_schema(self.rmb_path_list[0])
        builder_class = self._make_builder_class(tf, tfds, schema)
        builder = builder_class(data_dir=self.output_dir)
        builder.download_and_prepare()

        dataset_dir = os.path.join(
            self.output_dir, self.dataset_name, str(builder.VERSION)
        )
        print(
            f"[{self.__class__.__name__}] Complete RMB to RLDS conversion: {dataset_dir}"
        )

    def _prepare_output_dir(self):
        if os.path.exists(self.output_dir):
            if not self.force_overwrite:
                raise ValueError(
                    f"[{self.__class__.__name__}] Output directory already exists: {self.output_dir} (use --force_overwrite to remove it)"
                )
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir, exist_ok=False)

    def _import_tensorflow(self):
        try:
            import tensorflow as tf
        except Exception as exc:
            raise RuntimeError(
                "[ConvertRmbDataToRlds] Failed to import tensorflow. Install tensorflow in the conversion environment."
            ) from exc
        return tf

    def _import_tfds(self):
        try:
            import tensorflow_datasets as tfds
        except Exception as exc:
            raise RuntimeError(
                "[ConvertRmbDataToRlds] Failed to import tensorflow_datasets. Install tensorflow_datasets in the conversion environment."
            ) from exc
        return tfds

    def _make_builder_class(self, tf, tfds, schema):
        converter = self

        class RmbRldsBuilder(tfds.core.GeneratorBasedBuilder):
            VERSION = tfds.core.Version("1.0.0")
            RELEASE_NOTES = {"1.0.0": "Initial RMB to RLDS conversion."}
            name = converter.dataset_name

            def _info(self):
                return tfds.core.DatasetInfo(
                    builder=self,
                    features=tfds.features.FeaturesDict(
                        {
                            "steps": tfds.features.Dataset(
                                self._step_feature_dict(tf, tfds, schema)
                            ),
                            "episode_id": tfds.features.Text(),
                            "source_path": tfds.features.Text(),
                            "env": tfds.features.Text(),
                            "demo_name": tfds.features.Text(),
                            "task_desc": tfds.features.Text(),
                            "camera_names": tfds.features.Sequence(tfds.features.Text()),
                            "rgb_tactile_names": tfds.features.Sequence(
                                tfds.features.Text()
                            ),
                            "pointcloud_camera_names": tfds.features.Sequence(
                                tfds.features.Text()
                            ),
                            "world_idx": np.int32,
                            "version": tfds.features.Text(),
                        }
                    ),
                )

            def _split_generators(self, dl_manager):
                del dl_manager
                return {
                    converter.split_name: self._generate_examples(converter.rmb_path_list)
                }

            def _generate_examples(self, rmb_path_list):
                for rmb_path in rmb_path_list:
                    episode = make_rlds_episode_from_rmb(
                        rmb_path,
                        pad_terminal_step=converter.pad_terminal_step,
                        task_desc_default=converter.task_desc_default,
                    )
                    episode_id = episode["episode_id"]
                    yield episode_id, {
                        "steps": episode["steps"],
                        "episode_id": episode["episode_id"],
                        "source_path": episode["source_path"],
                        "env": episode["env"],
                        "demo_name": episode["demo_name"],
                        "task_desc": episode["task_desc"],
                        "camera_names": episode["camera_names"],
                        "rgb_tactile_names": episode["rgb_tactile_names"],
                        "pointcloud_camera_names": episode["pointcloud_camera_names"],
                        "world_idx": np.int32(episode["world_idx"]),
                        "version": episode["version"],
                    }

            def _step_feature_dict(self, tf, tfds, schema):
                return {
                    "observation": tfds.features.FeaturesDict(
                        {
                            key: self._feature_from_schema_entry(tf, tfds, key, entry)
                            for key, entry in schema["observation"].items()
                        }
                    ),
                    "action": tfds.features.FeaturesDict(
                        {
                            key: self._feature_from_schema_entry(tf, tfds, key, entry)
                            for key, entry in schema["action"].items()
                        }
                    ),
                    "reward": np.float32,
                    "discount": np.float32,
                    "is_first": np.bool_,
                    "is_last": np.bool_,
                    "is_terminal": np.bool_,
                    "language_instruction": tfds.features.Text(),
                    "time": np.float32,
                    **{
                        key: self._feature_from_schema_entry(tf, tfds, key, entry)
                        for key, entry in schema["extra_step"].items()
                    },
                }

            def _feature_from_schema_entry(self, tf, tfds, key, entry):
                shape = entry["shape"]
                dtype = entry["dtype"]
                if DataKey.is_rgb_image_key(key):
                    return tfds.features.Image(shape=shape, dtype=dtype, encoding_format="png")
                if isinstance(dtype, np.dtype):
                    dtype = dtype.type
                return tfds.features.Tensor(shape=shape, dtype=dtype)

        return RmbRldsBuilder


if __name__ == "__main__":
    convert_rmb = ConvertRmbDataToRlds(**vars(parse_argument()))
    convert_rmb.run()
