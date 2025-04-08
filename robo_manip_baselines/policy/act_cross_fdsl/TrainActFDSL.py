import os
import sys

import torch
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
os.environ['DEVICE'] = device


sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/act-cross-fdsl"))
from detr.models.detr_vae import DETRVAE
from training.policy import ACTPolicy
from training.utils import load_data

from robo_manip_baselines.common import TrainBase

from .ActDataset import ActDataset


class TrainActCrossFDSL(TrainBase):
    DatasetClass = ActDataset # TODO hardcoded for now

    def set_additional_args(self, parser):
        parser.set_defaults(enable_rmb_cache=True)

        parser.set_defaults(image_aug_std=0.1)

        parser.set_defaults(batch_size=8)
        parser.set_defaults(num_epochs=1000)
        parser.set_defaults(lr=1e-5)

        parser.add_argument("--kl_weight", type=int, default=10, help="KL weight")
        parser.add_argument(
            "--chunk_size", type=int, default=100, help="action chunking size"
        )
        parser.add_argument(
            "--hidden_dim", type=int, default=512, help="hidden dimension"
        )
        parser.add_argument(
            "--dim_feedforward", type=int, default=3200, help="feedforward dimension"
        )
    
    def setup_model_meta_info(self):
        super().setup_model_meta_info()

        self.model_meta_info["data"]["chunk_size"] = self.args.chunk_size

    def setup_policy(self):
        # Set policy args
        self.model_meta_info["policy"]["args"] = {
            "lr": self.args.lr,
            "num_queries": self.args.chunk_size,
            "kl_weight": self.args.kl_weight,
            "hidden_dim": self.args.hidden_dim,
            "dim_feedforward": self.args.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "camera_names": ["camera"], # TODO hardcoded for now. e.g., self.args.camera_names,
        }

        # Construct policy
        self.state_dim = 7 # TODO hardcoded for now
        self.action_dim = 7 # TODO hardcoded for now
        DETRVAE.set_state_dim(self.state_dim) 
        DETRVAE.set_action_dim(self.action_dim)
        self.policy = ACTPolicy(self.model_meta_info["policy"]["args"])
        self.policy.cuda()

        # Construct optimizer
        self.optimizer = self.policy.configure_optimizers()

        # Print policy information
        self.print_policy_info()
        print(f"  - chunk size: {self.args.chunk_size}")

    def setup_dataset(self):

        dataset_dir = os.path.join(self.args.dataset_dir)
        num_episodes = len(os.listdir(dataset_dir))
        camera_names = ["camera"] # TODO hardcoded for now
        batch_size_train = self.args.batch_size
        batch_size_val = self.args.batch_size
        data_type = "no_pc"

        self.train_dataloader, self.val_dataloader, self.norm_stats, is_sim = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, data_type=data_type)

        # Setup tensorboard
        self.writer = SummaryWriter(self.args.checkpoint_dir)


    def set_data_stats(self, all_filenames):
        import h5py
        import numpy as np
        dataset_dir = os.path.join(self.args.dataset_dir)
        num_episodes = len(os.listdir(dataset_dir))
        all_qpos_data = []
        all_action_data = []
        for episode_idx in range(num_episodes):
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
                # qvel = root['/observations/qvel'][()]
                action = root['/action'][()]
            all_qpos_data.append(torch.from_numpy(qpos))
            all_action_data.append(torch.from_numpy(action))
        all_qpos_data = torch.stack(all_qpos_data)
        all_action_data = torch.stack(all_action_data)
        all_action_data = all_action_data
        

        data_stats_state = {
            "norm_config": {
                "type": self.args.norm_type,
                **self.get_extra_norm_config(),
            },
            "min": all_qpos_data.min(axis=0),
            "max": all_qpos_data.min(axis=0),
            "range": np.clip(all_qpos_data.min(axis=0) - all_qpos_data.min(axis=0), 1e-3, 1e10),
            "mean": self.norm_stats["qpos_mean"],
            "std": self.norm_stats["qpos_std"],
            "example": all_qpos_data[0],
        }
        self.model_meta_info["state"].update(data_stats_state)

        data_stats_action = {
            "norm_config": {
                "type": self.args.norm_type,
                **self.get_extra_norm_config(),
            },
            "min": all_action_data.min(axis=0), # TODO hardcoded for now
            "max": all_action_data.max(axis=0), # TODO hardcoded for now
            "range": np.clip(all_action_data.min(axis=0) - all_action_data.min(axis=0), 1e-3, 1e10),
            "mean": self.norm_stats["action_mean"],
            "std": self.norm_stats["action_std"],
            "example": all_action_data[0],
        }

        ## TBD
        # self.model_meta_info["action"].update(data_stats_action)
        # self.model_meta_info["image"].update(
        #     {
        #         "rgb_example": rgb_image_example,
        #         "depth_example": depth_image_example,
        #     }
        # )
        # self.model_meta_info["data"].update(
        #     {
        #         "mean_episode_len": np.mean(episode_len_list),
        #         "min_episode_len": np.min(episode_len_list),
        #         "max_episode_len": np.max(episode_len_list),
        #     }
        # )

    # hardcoded for now
    def forward_pass(self, data, policy):
        image_data, qpos_data, action_data, is_pad = data
        image_data, qpos_data, action_data, is_pad = image_data.to(device), qpos_data.to(device), action_data.to(device), is_pad.to(device)
        return policy(qpos_data, image_data, action_data, is_pad)

    def train_loop(self):
        for epoch in tqdm(range(self.args.num_epochs)):
            # Run train step
            self.policy.train()
            batch_result_list = []
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                batch_result = self.forward_pass(data, self.policy)
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
                    batch_result = self.forward_pass(data, self.policy)
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


    def print_policy_info(self):
            print(
                f"[{self.__class__.__name__}] Construct {self.policy_name} policy.\n"
                f"  - state dim: {self.state_dim}, action dim: {self.action_dim}, camera num: {len(self.args.camera_names)}\n"
                f"  - state keys: {self.args.state_keys}\n"
                f"  - action keys: {self.args.action_keys}\n"
                f"  - camera names: {self.args.camera_names}\n"
                f"  - skip: {self.args.skip}, batch size: {self.args.batch_size}, num epochs: {self.args.num_epochs}"
            )
