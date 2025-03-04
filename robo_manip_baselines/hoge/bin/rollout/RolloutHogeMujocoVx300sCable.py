from robo_manip_baselines.hoge import RolloutHoge
from robo_manip_baselines.common.rollout import RolloutMujocoVx300sCable


class RolloutHogeMujocoVx300sCable(RolloutHoge, RolloutMujocoVx300sCable):
    pass


if __name__ == "__main__":
    rollout = RolloutHogeMujocoVx300sCable()
    rollout.run()