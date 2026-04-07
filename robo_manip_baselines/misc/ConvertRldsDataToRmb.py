import argparse
import os
import shutil

from robo_manip_baselines.common.manager.DataManager import DataManager
from robo_manip_baselines.common.data.RldsConversion import (
    RldsEpisodeConverter,
    load_rlds_conversion_config,
)


def parse_argument():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "rlds_dir",
        type=str,
        help="TFDS/RLDS dataset directory readable by tensorflow_datasets.builder_from_directory",
    )
    parser.add_argument(
        "config",
        type=str,
        help="YAML config that maps RLDS paths to RMB keys",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="output directory that will contain episode_******.rmb or .hdf5 files",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        help="dataset split passed to TFDS as_dataset",
    )
    parser.add_argument(
        "--output_ext",
        type=str,
        default=".rmb",
        choices=[".rmb", ".hdf5"],
        help="output episode format",
    )
    parser.add_argument(
        "--force_overwrite",
        action="store_true",
        help="remove output_dir before conversion if it already exists",
    )

    return parser.parse_args()


class ConvertRldsDataToRmb:
    def __init__(
        self,
        rlds_dir,
        config,
        output_dir,
        split="all",
        output_ext=".rmb",
        force_overwrite=False,
    ):
        self.rlds_dir = rlds_dir
        self.output_dir = output_dir
        self.split = split
        self.output_ext = output_ext
        self.force_overwrite = force_overwrite
        self.converter = RldsEpisodeConverter(load_rlds_conversion_config(config))
        self.data_manager = DataManager(env=None)

    def run(self):
        self._prepare_output_dir()

        tfds = self._import_tfds()
        builder = tfds.builder_from_directory(self.rlds_dir)
        episode_dataset = builder.as_dataset(split=self.split)

        converted_count = 0
        skipped_count = 0
        for episode_index, episode in enumerate(episode_dataset):
            episode_data = self._convert_tfds_episode_to_python(episode)
            if self.converter.should_skip_episode(episode_data):
                skipped_count += 1
                print(
                    f"[{self.__class__.__name__}] Skip invalid episode: index={episode_index}"
                )
                continue

            all_data_seq, meta_data = self.converter.convert_episode(
                episode_data, episode_index
            )
            output_path = os.path.join(
                self.output_dir, f"episode_{converted_count:06d}{self.output_ext}"
            )
            self.data_manager.save_data(
                output_path,
                all_data_seq=all_data_seq,
                meta_data=meta_data,
                increment_episode_idx=False,
            )
            converted_count += 1
            print(
                f"[{self.__class__.__name__}] Convert episode index={episode_index} -> {output_path}"
            )

        print(
            f"[{self.__class__.__name__}] Complete RLDS to RMB conversion. converted={converted_count}, skipped={skipped_count}"
        )

    def _prepare_output_dir(self):
        if os.path.exists(self.output_dir):
            if not self.force_overwrite:
                raise ValueError(
                    f"[{self.__class__.__name__}] Output directory already exists: {self.output_dir} (use --force_overwrite to remove it)"
                )
            shutil.rmtree(self.output_dir)

        os.makedirs(self.output_dir, exist_ok=False)

    def _import_tfds(self):
        try:
            import tensorflow_datasets as tfds
        except Exception as exc:
            raise RuntimeError(
                "[ConvertRldsDataToRmb] Failed to import tensorflow_datasets. Install tensorflow and tensorflow_datasets in the conversion environment."
            ) from exc

        return tfds

    def _convert_tfds_episode_to_python(self, episode):
        episode_data = {}
        for key, value in episode.items():
            if key == "steps":
                episode_data[key] = [
                    self._to_numpy_structure(step) for step in value.as_numpy_iterator()
                ]
            else:
                episode_data[key] = self._to_numpy_structure(value)
        return episode_data

    def _to_numpy_structure(self, value):
        if isinstance(value, dict):
            return {k: self._to_numpy_structure(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return tuple(self._to_numpy_structure(v) for v in value)
        if isinstance(value, list):
            return [self._to_numpy_structure(v) for v in value]
        if hasattr(value, "numpy"):
            return value.numpy()
        return value


if __name__ == "__main__":
    convert_rlds = ConvertRldsDataToRmb(**vars(parse_argument()))
    convert_rlds.run()
