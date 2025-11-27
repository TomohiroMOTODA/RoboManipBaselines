# MyPolicy

This README is for MyPolicy, a custom policy implementation for robotic manipulation tasks.

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py MyPolicy --dataset_dir ./dataset/MujocoAlohaHandover_Dataset50 --checkpoint_dir ./checkpoint/MyPolicy/test --camera_names overhead_cam worms_eye_cam wrist_cam_left wrist_cam_right
```

> [!NOTE]
> The following error will occur if the chunk_size is larger than the time series length of the training data.
> In such a case, either set the `--skip` option to a small value, or set the `--chunk_size` option to a small value.
> ```console
> RuntimeError: The size of tensor a (70) must match the size of tensor b (102) at non-singleton dimension 0
> ```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Act MujocoUR5eCable --checkpoint ./checkpoint/Act/<checkpoint_name>/policy_last.ckpt
```


## Prepared Dataset

My custom policy is implemented in <https://github.com/TomohiroMOTODA/RoboManipBaselines/blob/master/doc/dataset_list.md>

This is a custom policy implementation for robotic manipulation tasks. Below are the instructions to download a sample dataset and set up the environment for training and evaluation.
```bash
# Download the sample dataset (MujocoUR5eCable_Dataset30)
cd robo_manip_baselines/dataset
wget -O MujocoUR5eCable_Dataset30.zip "https://www.dropbox.com/scl/fo/sykc20cnax2scom1u8sc6/AM-zLM8dAZ5h6EQ8eDXcZic?rlkey=7icbmjc6wdqnp0tngfjqlhwoh&dl=1"

mkdir -p MujocoUR5eCable
unzip MujocoUR5eCable_Dataset30.zip -d MujocoUR5eCable
rm -r MujocoUR5eCable_Dataset30.zip
```

- ALOHA
    - <https://www.dropbox.com/scl/fo/erev1799doh0dbw51of81/AO-V-NNO007bemcPEJNGcYc?rlkey=yrsbjxxqrplqm6d48le24dwrl&dl=1>

```bash
# Download the sample dataset (MujocoUR5eCable_Dataset30)
cd robo_manip_baselines/dataset
wget -O MujocoAlohaHandover_Dataset50.zip "https://www.dropbox.com/scl/fo/erev1799doh0dbw51of81/AO-V-NNO007bemcPEJNGcYc?rlkey=yrsbjxxqrplqm6d48le24dwrl&dl=1"

mkdir -p MujocoAlohaHandover_Dataset50
unzip MujocoAlohaHandover_Dataset50.zip -d MujocoAlohaHandover_Dataset50
rm -r MujocoAlohaHandover_Dataset50.zip
```