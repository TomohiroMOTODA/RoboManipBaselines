
# LatentTOM Policy

## Install
TBD

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

> [!NOTE]
> If you are using `pyenv` and encounter the error `No module named '_bz2'`, apply the following solution.  
> https://stackoverflow.com/a/71457141

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py LatentTOMPolicy --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/LatentTOMPolicy/<checkpoint_name>
```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py LatentTOMPolicy MujocoUR5eCable --checkpoint ./checkpoint/LatentTOMPolicy/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@article{he2025latent,
  title={Latent Theory of Mind: A Decentralized Diffusion Architecture for Cooperative Manipulation},
  author={He, Chengyang and Camps, Gadiel Sznaier and Liu, Xu and Schwager, Mac and Sartoretti, Guillaume},
  journal={arXiv preprint arXiv:2505.09144},
  year={2025}
}
```
