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
        elif self.data_manager.status == MotionStatus.TELEOP:
            action = self.pred_action[self.env.unwrapped.arm_action_idxes]    
            self.motion_manager.target_se3 = pin.SE3(
                pin.rpy.rpyToMatrix(action[3],action[4],action[5]), np.array(action[:3])
            )
            self.motion_manager.inverse_kinematics()
        else:
            super().set_arm_command()

