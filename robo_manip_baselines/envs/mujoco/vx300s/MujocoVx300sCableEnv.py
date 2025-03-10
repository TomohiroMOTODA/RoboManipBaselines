from os import path
import numpy as np

from .MujocoVx300sEnvBase import MujocoVx300sEnvBase


class MujocoVx300sCableEnv(MujocoVx300sEnvBase):
    def __init__(
        self,
        **kwargs,
    ):
        MujocoVx300sEnvBase.__init__(
            self,
            path.join(
                path.dirname(__file__),
                "../../assets/mujoco/envs/vx300s/env_vx300s_cable.xml",
            ),
            np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0, 0.0084, 0.0084]),
            **kwargs,
        )

        # self.original_pole_pos = self.model.geom("pole1").pos.copy()
        self.pole_pos_offsets = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.015, 0.0, 0.0],
                [0.03, 0.0, 0.0],
                [0.045, 0.0, 0.0],
                [0.06, 0.0, 0.0],
                [0.075, 0.0, 0.0],
            ]
        )  # [m]

    def modify_world(self, world_idx=None, cumulative_idx=None):
        pass
        # if world_idx is None:
        #     world_idx = cumulative_idx % len(self.pole_pos_offsets)
        # self.model.geom("pole1").pos = (
        #     self.original_pole_pos + self.pole_pos_offsets[world_idx]
        # )
        # self.model.geom("pole2").pos = self.model.geom("pole1").pos
        # self.model.geom("pole2").pos[0] *= -1
        # return world_idx
