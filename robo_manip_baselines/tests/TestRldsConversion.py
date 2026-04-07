from robo_manip_baselines.common import DataKey
from robo_manip_baselines.common.data.RldsConversion import RldsEpisodeConverter


def test_convert_episode_with_auto_time():
    converter = RldsEpisodeConverter(
        {
            "step_mappings": {
                DataKey.MEASURED_JOINT_POS: "observation.state",
                DataKey.COMMAND_JOINT_POS: "action",
                DataKey.get_rgb_image_key("front"): "observation.image_front",
            },
            "episode_metadata_mappings": {
                "task_desc": {"path": "task", "optional": True},
                "episode_id": {"path": "episode_id", "optional": True},
            },
            "conversion": {
                "fps": 10.0,
                "drop_last_step_if_is_last": True,
                "skip_invalid_episodes": True,
            },
        }
    )

    episode = {
        "episode_id": b"ep-0001",
        "task": b"pick object",
        "steps": [
            {
                "is_first": True,
                "is_last": False,
                "observation": {
                    "state": [0.1, 0.2],
                    "image_front": [[[0, 1, 2]]],
                },
                "action": [1.0, 2.0],
            },
            {
                "is_first": False,
                "is_last": True,
                "observation": {
                    "state": [0.3, 0.4],
                    "image_front": [[[3, 4, 5]]],
                },
                "action": [9.0, 9.0],
            },
        ],
    }

    all_data_seq, meta_data = converter.convert_episode(episode, episode_index=7)

    assert len(all_data_seq[DataKey.TIME]) == 1
    assert all_data_seq[DataKey.TIME][0] == 0.0
    assert all_data_seq[DataKey.MEASURED_JOINT_POS][0].shape == (2,)
    assert all_data_seq[DataKey.COMMAND_JOINT_POS][0].shape == (2,)
    assert all_data_seq[DataKey.get_rgb_image_key("front")][0].dtype.name == "uint8"
    assert meta_data["task_desc"] == "pick object"
    assert meta_data["episode_id"] == "ep-0001"
    assert meta_data["camera_names"] == ["front"]
    assert meta_data["pointcloud_camera_names"] == []


def test_skip_invalid_episode():
    converter = RldsEpisodeConverter(
        {
            "step_mappings": {
                DataKey.MEASURED_JOINT_POS: "observation.state",
            },
            "conversion": {
                "skip_invalid_episodes": True,
            },
        }
    )

    assert converter.should_skip_episode({"invalid": True}) is True
    assert converter.should_skip_episode({"invalid": False}) is False
