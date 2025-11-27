# MyPolicy

Forked from official RoboManipBaselines repository

My custom policy is implemented in <https://github.com/TomohiroMOTODA/RoboManipBaselines/blob/master/doc/dataset_list.md>

This is a custom policy implementation for robotic manipulation tasks. Below are the instructions to download a sample dataset and set up the environment for training and evaluation.
```bash
# Download the sample dataset (MujocoUR5eCable_Dataset30)
cd robo_manip_baselines/dataset
wget -O MujocoUR5eCable_Dataset30.zip "https://www.dropbox.com/scl/fo/sykc20cnax2scom1u8sc6/AM-zLM8dAZ5h6EQ8eDXcZic?rlkey=7icbmjc6wdqnp0tngfjqlhwoh&dl=1"

! mkdir -p MujocoUR5eCable
! unzip MujocoUR5eCable_Dataset30.zip -d MujocoUR5eCable
! rm -r MujocoUR5eCable_Dataset30.zip
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
