import glob
import json
import os
import random


def is_lerobot_dataset_dir(path):
    """Check if the path is a local LeRobot dataset directory."""
    return os.path.isfile(os.path.join(path, "meta", "info.json"))


def parse_lerobot_episode_path(path):
    """Parse a virtual episode path '<dataset_dir>@<episode_idx>' of a LeRobot dataset.

    Returns (dataset_dir, episode_idx) if the path matches, otherwise None.
    """
    if not isinstance(path, str) or "@" not in path:
        return None
    dataset_dir, _, episode_idx_str = path.rpartition("@")
    if not episode_idx_str.isdigit() or not is_lerobot_dataset_dir(dataset_dir):
        return None
    return dataset_dir, int(episode_idx_str)


def find_rmb_files(base_path, num_files=None):
    if parse_lerobot_episode_path(base_path) is not None:
        rmb_path_list = [base_path]
    elif os.path.isdir(base_path) and is_lerobot_dataset_dir(base_path):
        with open(os.path.join(base_path, "meta", "info.json")) as f:
            total_episodes = json.load(f)["total_episodes"]
        rmb_path_list = [
            f"{base_path.rstrip('/')}@{episode_idx}"
            for episode_idx in range(total_episodes)
        ]
    elif base_path.rstrip("/").endswith((".rmb", ".hdf5")):
        rmb_path_list = [base_path]
    elif os.path.isdir(base_path):
        rmb_path_list = sorted(
            [
                f
                for f in glob.glob(f"{base_path}/**/*.*", recursive=True)
                if f.endswith(".rmb")
                or (f.endswith(".hdf5") and not f.endswith(".rmb.hdf5"))
            ]
        )
    else:
        raise ValueError(f"[find_rmb_files] RMB file not found: {base_path}")

    if num_files is not None:
        if num_files > len(rmb_path_list):
            raise ValueError(
                f"[find_rmb_files] Requested num_files={num_files} exceeds total available files={len(rmb_path_list)}."
            )
        rmb_path_list = sorted(random.sample(rmb_path_list, num_files))

    return rmb_path_list
