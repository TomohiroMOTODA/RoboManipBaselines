import os
import sys
import argparse
import pickle
import numpy as np
import matplotlib.pylab as plt
import cv2
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/ACT"))
from ..config.constants import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import *

from robo_manip_baselines.common.rollout import RolloutBase

class RolloutHoge(RolloutBase):
    def setup_args(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()

        # parse the task name via command line
        parser = argparse.ArgumentParser()
        parser.add_argument('--task', type=str, default='task1')
        # task = self.args.task

        # config
        self.cfg = TASK_CONFIG
        self.policy_config = POLICY_CONFIG
        self.train_cfg = TRAIN_CONFIG
        self.device = os.environ['DEVICE']

        self.action_dim = TASK_CONFIG['action_dim']
        self.skip_draw = 1

        super().setup_args(parser)


    def setup_policy(self):
        
        ckpt_path = os.path.join(self.train_cfg['checkpoint_dir'], self.train_cfg['eval_ckpt_name'])
        self.policy = make_policy(self.policy_config['policy_class'], self.policy_config)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(self.device)))
        
        # Load weight
        print(f"[RolloutAct] Load {ckpt_path}")
        print(loading_status)
        self.policy.to(self.device)
        self.policy.eval()

        stats_path = os.path.join(self.train_cfg['checkpoint_dir'], f'dataset_stats.pkl')
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # Set variables
        self.joint_scales = [1.0] * (7 - 1) + [0.01]
        self.pred_action_list = np.empty((0, 7))
        self.all_actions_history = []

    def setup_plot(self):
        pass
        # fig_ax = plt.subplots(
        #     2,
        #     max(2, self.policy_config["enc_layers"]),
        #     figsize=(13.5, 6.0),
        #     dpi=60,
        #     squeeze=False,
        # )
        # super().setup_plot(fig_ax=fig_ax)

    def infer_policy(self):
        # if self.auto_time_idx % self.args.skip != 0:
        #     return False

        # Preprocess
        self.front_image = self.info["rgb_images"]["wrist_cam_left"]
        front_image_input = self.front_image.transpose(2, 0, 1)
        front_image_input = front_image_input.astype(np.float32) / 255.0
        front_image_input = (
            torch.Tensor(np.expand_dims(front_image_input, 0)).cuda().unsqueeze(0)
        )
        joint_input = self.motion_manager.get_joint_pos(self.obs)
        joint_input = (joint_input - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        joint_input = torch.Tensor(np.expand_dims(joint_input, 0)).cuda()

        # Infer
        all_actions = self.policy(joint_input, front_image_input)[0]
        self.all_actions_history.append(all_actions.cpu().detach().numpy())
        chunk_size = self.policy_config['num_queries']
        if len(self.all_actions_history) > chunk_size:
            self.all_actions_history.pop(0)

        # Postprocess (temporal ensembling)
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(self.all_actions_history)))
        exp_weights = exp_weights / exp_weights.sum()
        action = np.zeros(self.action_dim)
        for action_idx, _all_actions in enumerate(reversed(self.all_actions_history)):
            action += exp_weights[::-1][action_idx] * _all_actions[action_idx]
        self.pred_action = action * self.stats["action_std"] + self.stats["action_mean"]

        self.pred_action_list = np.concatenate(
            [self.pred_action_list, np.expand_dims(self.pred_action, 0)]
        )


        return True

    def draw_plot(self):
        pass
        # if self.auto_time_idx % self.skip_draw != 0:
        #     return

        # for _ax in np.ravel(self.ax):
        #     _ax.cla()
        #     _ax.axis("off")

        # # Draw observed image
        # self.ax[0, 0].imshow(self.front_image)
        # self.ax[0, 0].set_title("Observed image", fontsize=20)


        # # Plot joint
        # xlim = 500 // self.args.skip
        # self.ax[0, 1].set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        # self.ax[0, 1].set_xlim(0, xlim)
        # for joint_idx in range(self.pred_action_list.shape[1]):
        #     self.ax[0, 1].plot(
        #         np.arange(self.pred_action_list.shape[0]),
        #         self.pred_action_list[:, joint_idx] * self.joint_scales[joint_idx],
        #     )
        # self.ax[0, 1].set_xlabel("Step", fontsize=20)
        # self.ax[0, 1].set_title("Joint", fontsize=20)
        # self.ax[0, 1].tick_params(axis="x", labelsize=16)
        # self.ax[0, 1].tick_params(axis="y", labelsize=16)
        # self.ax[0, 1].axis("on")

        # # Draw attention images
        # for layer_idx, layer in enumerate(self.policy.model.transformer.encoder.layers):
        #     if layer.self_attn.correlation_mat is None:
        #         continue
        #     self.ax[1, layer_idx].imshow(
        #         layer.self_attn.correlation_mat[2:, 1].reshape((15, 20))
        #     )
        #     self.ax[1, layer_idx].set_title(f"Attention ({layer_idx})", fontsize=20)

        # self.fig.tight_layout()
        # self.canvas.draw()
        # cv2.imshow(
        #     "Policy image",
        #     cv2.cvtColor(np.asarray(self.canvas.buffer_rgba()), cv2.COLOR_RGB2BGR),
        # )
