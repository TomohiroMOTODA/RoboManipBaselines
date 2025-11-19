from html import parser
import os
import sys

import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/bict-robotics/src"))
import os
from bict.lib.policy import BictConsensusPolicy

from robo_manip_baselines.common import TrainBase
from .MyPolicyDataset import MyPolicyDataset


class TrainMyPolicy(TrainBase):
    DatasetClass = MyPolicyDataset

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(image_aug_std=0.1)

        parser.set_defaults(batch_size=8)
        parser.set_defaults(num_epochs=1000)
        parser.set_defaults(lr=1e-5)

        # My policy specific args
        parser.add_argument('--task', '-t', default='config', type=str, help="Task name to run, e.g., 'act', 'dp', 'config', 'bict_split_refactored', 'bict_split_diffusion', 'bict_split_llm'")
        parser.add_argument('--base', '-b', default='base', type=str, help="Base configuration to use, e.g., 'default', 'base_config', 'bict_refactored'")
        parser.add_argument('--sampler', '-s', default='step', type=str, help="Sampler type: 'epoch' or 'step'")
        parser.add_argument('--accelerate', action='store_true', help="Use Huggingface Accelerate for multi-GPU or distributed training")
        parser.add_argument('--arm', '-a', default=None, type=str, help="Train specific arm only ('left_arm', 'right_arm'), or 'both' for coordinated training")

    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["chunk_size"] = self.args.chunk_size

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            'lr': self.args.lr,
            'num_queries': self.args.chunk_size,
            'kl_weight':  self.args.kl_weight,
            'hidden_dim': self.args.hidden_dim,
            'dim_feedforward': self.args.dim_feedforward,
            'lr_backbone': 1e-5,
            'backbone': "resnet50",
            'enc_layers': 4,
            'dec_layers': 7,
            'nheads': 8,
            'camera_names': self.args.camera_names,
            'action_dim': 14,
            'multi_task': False,
        }

        # Construct policy
        self.policy = BictConsensusPolicy(self.model_meta_info["policy"]["args"]).cuda()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {self.args.chunk_size}")

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                batch_result = self.policy(*[d.cuda() for d in data])
                loss = batch_result["loss"]
                loss.backward()
                self.optimizer.step()
                batch_result_list.append(self.detach_batch_result(batch_result))
            self.log_epoch_summary(batch_result_list, "train", epoch)

            # Run validation step
            with torch.inference_mode():
                self.policy.eval()
                batch_result_list = []
                for data in self.val_dataloader:
                    batch_result = self.policy(*[d.cuda() for d in data])
                    batch_result_list.append(self.detach_batch_result(batch_result))
                epoch_summary = self.log_epoch_summary(batch_result_list, "val", epoch)

                # Update best checkpoint
                self.update_best_ckpt(epoch_summary)

            # Save current checkpoint
            if epoch % max(self.args.num_epochs // 10, 1) == 0:
                self.save_current_ckpt(f"epoch{epoch:0>3}")

        # Save last checkpoint
        self.save_current_ckpt("last")

        # Save best checkpoint
        self.save_best_ckpt()
