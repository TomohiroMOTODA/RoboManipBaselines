from os import path
import numpy as np
import mujoco
from gymnasium.spaces import Box, Dict

from ..MujocoEnvBase import MujocoEnvBase


class MujocoVx300sEnvBase(MujocoEnvBase):
    default_camera_config = {
        "azimuth": 0.0,
        "elevation": -20.0,
        "distance": 1.8,
        "lookat": [0.0, 0.0, 0.3],
    }
    observation_space = Dict(
        {
            "right/joint_pos": Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            ),
            "right/joint_vel": Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64
            ),
        }
    )

    def setup_robot(self, init_qpos):
        mujoco.mj_kinematics(self.model, self.data)
        self.arm_urdf_path = path.join(
            path.dirname(__file__), "../../assets/common/robots/aloha/vx300s.urdf"
        )
        self.arm_root_pose = self.get_body_pose("right/base_link")
        self.ik_eef_joint_id = 6
        self.ik_arm_joint_ids = slice(0, 6)
        self.init_qpos[0 : len(init_qpos)] = init_qpos
        # self.init_qpos[len(init_qpos) : 2 * len(init_qpos)] = init_qpos
        self.init_qvel[:] = 0.0

        self.gripper_action_idx = 6
        self.arm_action_idxes = slice(0, 6)

    def step(self, action):
        return super().step(action)

    def _get_obs(self):
        obs = {
            "right/joint_pos": np.zeros(7),
            "right/joint_vel": np.zeros(7),
        }

        single_arm_joint_name_list = [
            "waist",
            "shoulder",
            "elbow",
            "forearm_roll",
            "wrist_angle",
            "wrist_rotate",
        ]
        single_gripper_joint_name_list = [
            "right_finger",
            "left_finger",
        ]

        for arm_idx, arm_name in enumerate([("right")]):
            joint_pos_key = f"{arm_name}/joint_pos"
            joint_vel_key = f"{arm_name}/joint_vel"

            for joint_idx, joint_name in enumerate(single_arm_joint_name_list):
                joint_name = f"{arm_name}/{joint_name}"
                obs[joint_pos_key][joint_idx] = self.data.joint(joint_name).qpos[0]
                obs[joint_vel_key][joint_idx] = self.data.joint(joint_name).qvel[0]

            gripper_joint_qpos = np.zeros(2)
            gripper_joint_qvel = np.zeros(2)
            for joint_idx, joint_name in enumerate(single_gripper_joint_name_list):
                joint_name = f"{arm_name}/{joint_name}"
                gripper_joint_qpos[joint_idx] = self.data.joint(joint_name).qpos[0]
                gripper_joint_qvel[joint_idx] = self.data.joint(joint_name).qvel[0]
            obs[joint_pos_key][-1] = gripper_joint_qpos.mean()
            obs[joint_vel_key][-1] = gripper_joint_qvel.mean()

        return obs

    def get_joint_pos_from_obs(self, obs, exclude_gripper=False):
        """Get joint position from observation."""
        if exclude_gripper:
            return obs["right/joint_pos"][self.arm_action_idxes]
        else:
            return obs["right/joint_pos"]

    def get_joint_vel_from_obs(self, obs, exclude_gripper=False):
        """Get joint velocity from ob   servation."""
        if exclude_gripper:
            return obs["right/joint_vel"][self.arm_action_idxes]
        else:
            return obs["right/joint_vel"]

    def get_eef_wrench_from_obs(self, obs):
        """Get end-effector wrench (fx, fy, fz, nx, ny, nz) from observation."""
        return np.zeros(6)
