import numpy as np
import pinocchio as pin
import gymnasium as gym
from robo_manip_baselines.common import MotionStatus
from .RolloutBase import RolloutBase


class RolloutMujocoVx300sCable(RolloutBase):
    def setup_env(self):
        self.env = gym.make(
            "robo_manip_baselines/MujocoVx300sCableEnv-v0", render_mode="human"
        )

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            pass
        else:
            super().set_arm_command()
