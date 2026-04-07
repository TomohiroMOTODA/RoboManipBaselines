import importlib.util
import pathlib
import sys
import types

import numpy as np


def load_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def prepare_fake_package_root():
    repo = pathlib.Path(__file__).resolve().parents[2]
    package_root = repo / "robo_manip_baselines"
    common_root = package_root / "common"
    data_root = common_root / "data"

    root_pkg = types.ModuleType("robo_manip_baselines")
    root_pkg.__path__ = [str(package_root)]
    sys.modules["robo_manip_baselines"] = root_pkg

    common_pkg = types.ModuleType("robo_manip_baselines.common")
    common_pkg.__path__ = [str(common_root)]
    sys.modules["robo_manip_baselines.common"] = common_pkg

    data_pkg = types.ModuleType("robo_manip_baselines.common.data")
    data_pkg.__path__ = [str(data_root)]
    sys.modules["robo_manip_baselines.common.data"] = data_pkg

    return repo, data_root


def test_make_rlds_episode_from_rmb_with_terminal_padding():
    repo, data_root = prepare_fake_package_root()
    data_key_mod = load_module(
        "robo_manip_baselines.common.data.DataKey", data_root / "DataKey.py"
    )
    rmb_to_rlds_mod = load_module(
        "robo_manip_baselines.common.data.RmbToRldsConversion",
        data_root / "RmbToRldsConversion.py",
    )

    DataKey = data_key_mod.DataKey

    class FakeRmbData:
        def __init__(self, path):
            self.path = path
            self.attrs = {
                "env": "DummyEnv",
                "demo_name": "DummyDemo",
                "task_desc": "",
                "camera_names": np.array(["front"], dtype=object),
                "rgb_tactile_names": np.array([], dtype=object),
                "pointcloud_camera_names": np.array([], dtype=object),
                "world_idx": 2,
                "version": "3.0.0",
            }
            self._data = {
                DataKey.TIME: np.array([0.0, 0.1], dtype=np.float64),
                DataKey.REWARD: np.array([1.0, 2.0], dtype=np.float64),
                DataKey.MEASURED_JOINT_POS: np.array(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=np.float64
                ),
                DataKey.COMMAND_JOINT_POS: np.array(
                    [[5.0, 6.0], [7.0, 8.0]], dtype=np.float64
                ),
                DataKey.get_rgb_image_key("front"): np.array(
                    [
                        [[[0, 1, 2]]],
                        [[[3, 4, 5]]],
                    ],
                    dtype=np.uint8,
                ),
                "measured_tactile_sensor": np.array(
                    [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64
                ),
            }

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def keys(self):
            return self._data.keys()

        def __getitem__(self, key):
            return self._data[key]

    original_rmb_data = rmb_to_rlds_mod.RmbData
    rmb_to_rlds_mod.RmbData = FakeRmbData
    try:
        episode = rmb_to_rlds_mod.make_rlds_episode_from_rmb(
            "dummy_episode.rmb",
            pad_terminal_step=True,
            task_desc_default="fallback instruction",
        )
    finally:
        rmb_to_rlds_mod.RmbData = original_rmb_data

    assert len(episode["steps"]) == 3
    assert episode["steps"][0]["is_first"] is True
    assert episode["steps"][-1]["is_last"] is True
    assert np.allclose(
        episode["steps"][0]["action"][DataKey.COMMAND_JOINT_POS],
        np.array([5.0, 6.0], dtype=np.float32),
    )
    assert np.allclose(
        episode["steps"][-1]["action"][DataKey.COMMAND_JOINT_POS],
        np.array([0.0, 0.0], dtype=np.float64),
    )
    assert episode["steps"][0]["language_instruction"] == "fallback instruction"
    assert episode["camera_names"] == ["front"]
