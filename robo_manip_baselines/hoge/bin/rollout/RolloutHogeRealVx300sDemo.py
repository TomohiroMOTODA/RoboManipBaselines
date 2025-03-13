from robo_manip_baselines.hoge.config.constants import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG, ROBOT_PORTS # must import first
from robo_manip_baselines.hoge.config.constants import ROBOT_ENV, CAMERA_SERIAL_NO

from robo_manip_baselines.common import MotionManager, MotionStatus, DataManager

import os
import sys
import cv2
import torch
import pickle
import argparse
from time import time
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../third_party/ACT"))
from training.utils import *

import glob
from tqdm import tqdm

# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='task1')
args = parser.parse_args()
task = args.task

# config
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']

def evaluate():


    print("\n===== Deploy Configuration =====")
    print(f"ROBOT_ENV           : {ROBOT_ENV}")
    print(f"CHECK_POINTS        : {os.path.join(train_cfg['checkpoint_dir'], train_cfg['eval_ckpt_name'])}")
    print("==================================\n")


    if ROBOT_ENV=='default':
        
        # libraries
        from robot import Robot
        from common.camera.default import capture_image

        # init camera
        cam = cv2.VideoCapture(cfg['camera_port'])
        # Check if the camera opened successfully
        if not cam.isOpened():
            raise IOError("Cannot open camera")
        # init follower
        follower = Robot(device_name=ROBOT_PORTS['follower'])
    
    elif ROBOT_ENV=='sim':
        from common.environment.vx300s_sim.arm import RobotSimEnv
        follower = RobotSimEnv(viewer_mode="window")
        cam=None
        capture_image = follower.get_wrist_image

    elif ROBOT_ENV=='real':
        # init camera
        from common.camera.realsense_d405 import RealSenseCamera
        cam = RealSenseCamera(serial_no=CAMERA_SERIAL_NO, camera_name="d405", align=False)

        # init robot 
        from common.environment.vx300s_real.arm import RobotRealEnv
        from common.camera.default import capture_image
        follower = RobotRealEnv.make_real_env()

        import pinocchio as pin
        import gymnasium as gym
        env = gym.make(
            "robo_manip_baselines/MujocoVx300sCableEnv-v0", render_mode="human"
        )

        motion_manager = MotionManager(env)

    else:
        raise NotImplemented

    # load the policy
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], train_cfg['eval_ckpt_name'])
    policy = make_policy(policy_config['policy_class'], policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    print(loading_status)
    policy.to(device)
    policy.eval()

    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(train_cfg['checkpoint_dir'], f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # print (stats['qpos_mean'].shape, stats['qpos_std'].shape)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    query_frequency = policy_config['num_queries']
    if policy_config['temporal_agg']:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    # bring the follower to the leader
    # for i in range(90):
    #     follower.read_position()
    #     _ = capture_image(cam)
    
    obs = {
        'qpos': follower.read_position(), #pwm2pos(follower.read_position()),
        'qvel': follower.read_velocity(), #vel2pwm(follower.read_velocity()),
        'images': {cn: capture_image(cam, enable_pointcloud=True)[0] for cn in cfg['camera_names']},
        'points': {cn: capture_image(cam, enable_pointcloud=True)[1] for cn in cfg['camera_names']}
    }

    n_rollouts = 1
    for i in range(n_rollouts):
        ### evaluation loop
        if policy_config['temporal_agg']:
            all_time_actions = torch.zeros([cfg['episode_len'], cfg['episode_len']+num_queries, cfg['state_dim']]).to(device)
        # qpos_history = torch.zeros((1, cfg['episode_len'], cfg['state_dim'])).to(device)
        qpos_history = torch.zeros((1, cfg['episode_len'], 7)).to(device)
        
        img_cv = []
        with torch.inference_mode():
             # init buffers
            obs_replay = []
            action_replay = []
            for t in tqdm(range(cfg['episode_len'])):
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(obs['images'], cfg['camera_names'], device)
                
                if "points" in obs.keys():
                    pc = torch.from_numpy(obs['points']['camera']).float().to("cuda")

                if t % query_frequency == 0:  
                    all_actions = policy(qpos, curr_image,point_cloud=pc) # TODO 第３項目のところにポイントクラウドの入力を入れる。現在は、None
                if policy_config['temporal_agg']:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights.astype(np.float32)).to(device).unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]
                    # print ("raw_action",raw_action)
                    # print ("query_frequency",query_frequency)


                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                # print ("action",action)

                ### take action
                
                follower.set_goal_pos(action)

                # import pinocchio as pin
                # motion_manager.target_se3 = pin.SE3(
                #     pin.rpy.rpyToMatrix(action[3],action[4],action[5]), np.array(action[:3])
                # )
                # motion_manager.inverse_kinematics()
                # action = motion_manager.get_action()
                # follower.step(action)
                
                if ROBOT_ENV=="sim":
                    follower.render() # rendering
                    img  = cv2.resize(obs['images']['camera'], (640, 480))
                    img_overview = cv2.resize(follower.get_overview_image(), (640, 480))
                    img  = np.concatenate([img, img_overview], axis=1)
                    img  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    outimg_file = '{}/{:05d}_video.png'.format("/tmp", t)
                    cv2.imwrite(outimg_file, img)
                    img_cv.append(outimg_file) 
                    
                ### update obs
                obs = {
                    'qpos': (follower.read_position()),
                    'qvel': (follower.read_velocity()),
                    'images': {cn: capture_image(cam) for cn in cfg['camera_names']}
                }
                ### store data
                obs_replay.append(obs)
                action_replay.append(action)
            

            if ROBOT_ENV=="sim":
                # Output video
                fourcc = cv2.VideoWriter_fourcc('m','p','4', 'v')
                video  = cv2.VideoWriter('./data/ImgVideo.mp4', fourcc, 20.0, (1280, 480))

                for img_file in img_cv:
                    img = cv2.imread(img_file)
                    video.write(img)
                video.release()
                
                for p in glob.glob('/tmp/*_video.png', recursive=True):
                    if os.path.isfile(p):
                        os.remove(p)

        # create a dictionary to store the data
        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        # there may be more than one camera
        for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'] = []

        # store the observations and actions
        for o, a in zip(obs_replay, action_replay):
            data_dict['/observations/qpos'].append(o['qpos'])
            data_dict['/observations/qvel'].append(o['qvel'])
            data_dict['/action'].append(a)
            # store the images
            for cam_name in cfg['camera_names']:
                data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

        t0 = time()
        max_timesteps = len(data_dict['/observations/qpos'])
        # create data dir if it doesn't exist
        data_dir = cfg['dataset_dir']  
        if not os.path.exists(data_dir): os.makedirs(data_dir)
        # count number of files in the directory
        idx = len([name for name in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, name))])
        dataset_path = os.path.join(data_dir, f'episode_{idx}')
        # save the data
        with h5py.File("data/demo/trained.hdf5", 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in cfg['camera_names']:
                _ = image.create_dataset(cam_name, (max_timesteps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8',
                                        chunks=(1, cfg['cam_height'], cfg['cam_width'], 3), )
            qpos = obs.create_dataset('qpos', (max_timesteps, cfg['state_dim']))
            qvel = obs.create_dataset('qvel', (max_timesteps, cfg['state_dim']))
            # image = obs.create_dataset("image", (max_timesteps, 240, 320, 3), dtype='uint8', chunks=(1, 240, 320, 3))
            action = root.create_dataset('action', (max_timesteps, cfg['action_dim']))
            
            for name, array in data_dict.items():
                root[name][...] = array
    
    # disable torque
    follower._disable_torque()

if __name__ == "__main__":
    evaluate()
