import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
# DATA_DIR = 'data/'
# DATA_DIR = '/home/projects/ACT/data/output/hdf5'
DATA_DIR = '/home/projects/ACT/data/output/hdf5/eef_task'

# checkpoint directory
CHECKPOINT_DIR = './data/put_banana/scratch_put_banana_no_pc_2000'




# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
#if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/tty.usbmodem57380045221',
    'follower': '/dev/tty.usbmodem57380046991'
}


ROBOT_ENV='real' # 'default', 'sim', 'real'
CAMERA_SERIAL_NO='128422270547'

# task config (you can add new tasks)
# TASK_CONFIG = {
#     'dataset_dir': DATA_DIR,
#     'episode_len': 300,
#     'state_dim': 5,
#     'action_dim': 5,
#     'cam_width': 640,
#     'cam_height': 480,
#     'camera_names': ['front'],
#     'camera_port': 0
# }
TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 2000,
    'state_dim': 7,
    'action_dim': 7,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['camera'],
    'camera_port': 0
}


# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['camera'],
    'policy_class': 'ACT',
    'temporal_agg': False
}

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 100,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}

# finetune config
FINETUNE_CONFIG = {
    'seed': 42,
    'num_epochs': 100,
    'batch_size_val': 8,
    'batch_size_train': 8,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}
