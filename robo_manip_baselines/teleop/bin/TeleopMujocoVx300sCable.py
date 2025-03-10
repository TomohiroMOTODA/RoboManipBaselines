import numpy as np
import gymnasium as gym
import pinocchio as pin
from robo_manip_baselines.teleop import TeleopBase
from robo_manip_baselines.common import MotionStatus


class TeleopMujocoVx300sCable(TeleopBase):
    def __init__(self):
        super().__init__()

        # Command configuration
        self.command_rpy_scale = 2e-2

    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoVx300sCableEnv-v0", render_mode="human"
        )
        self.demo_name = self.args.demo_name or "MujocoVx300sCable"

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            super().set_arm_command()
            # target_pos = self.env.unwrapped.get_body_pose("B2")[0:3]
            # target_pos[1] += 0.05  # [m]
            # if self.data_manager.status == MotionStatus.PRE_REACH:
            #     target_pos[2] = 0.3  # [m]
            #     target_rpy = np.array([0.0, np.deg2rad(30), -np.pi / 2])
            # elif self.data_manager.status == MotionStatus.REACH:
            #     target_pos[2] = 0.2  # [m]
            #     target_rpy = np.array([0.0, np.deg2rad(60), -np.pi / 2])
            # self.motion_manager.target_se3 = pin.SE3(
            #     pin.rpy.rpyToMatrix(*target_rpy), target_pos
            # )
        else:
            # target_pos = self.env.unwrapped.get_body_pose("B1")[0:3]
            # target_pos = self.env.unwrapped.get_body_pose("B1")[0:3]
            # target_pos[1] += 0.05  # [m]
            # if self.data_manager.status == MotionStatus.PRE_REACH:
            #     target_pos[2] = 0.3  # [m]
            #     target_rpy = np.array([0.0, np.deg2rad(30), -np.pi / 2])
            # elif self.data_manager.status == MotionStatus.REACH:
            #     target_pos[2] = 0.2  # [m]
            #     target_rpy = np.array([0.0, np.deg2rad(60), -np.pi / 2])
            # self.motion_manager.target_se3 = pin.SE3(
            #     pin.rpy.rpyToMatrix(*target_rpy), target_pos
            # )
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopMujocoVx300sCable()
    teleop.run()
