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
            "robo_manip_baselines/RealVx300sCableEnv-v0"
        )
        self.demo_name = self.args.demo_name or "RealVx300sCable"

    def set_arm_command(self):
        if self.data_manager.status in (MotionStatus.PRE_REACH, MotionStatus.REACH):
            pass
        else:
            super().set_arm_command()


if __name__ == "__main__":
    teleop = TeleopMujocoVx300sCable()
    teleop.run()
