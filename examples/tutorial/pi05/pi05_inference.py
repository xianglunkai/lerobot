import torch
import time

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.cobot_magic.config_cobot_magic import CobotMagicConfig
from lerobot.robots.cobot_magic.cobot_magic import CobotMagic
from lerobot.robots.agilex_cobot.config_agilex_cobot import AgilexCobotConfig
from lerobot.robots.agilex_cobot.agilex_cobot import AgilexCobot
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
MAX_EPISODES = 1
MAX_STEPS_PER_EPISODE = 10000
CONTROL_FREQUENCY = 50  # Hz

device = torch.device("cuda")  # or "cuda" or "cpu" or "mps"
model_id = "/home/xlk/work/lerobot/checkpoints/fold_towel/30k-50hz/pretrained_model"


model = PI05Policy.from_pretrained(pretrained_name_or_path=model_id)

preprocess, postprocess = make_pre_post_processors(
    model.config,
    pretrained_path=model_id,
    # This overrides allows to run on MPS, otherwise defaults to CUDA (if available)
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

robot_action_processor = make_default_robot_action_processor()

# Robot and environment configuration
# Camera keys must match the name and resolutions of the ones used for training!
# You can check the camera keys expected by a model in the info.json card on the model card on the Hub
camera_config = {
    "high": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=30),
    "left": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=30),
    "right": OpenCVCameraConfig(index_or_path=2, width=640, height=480, fps=30),
}

robot_cfg = AgilexCobotConfig(use_external_commands=False)
robot = AgilexCobot(robot_cfg)
robot.connect()


task = "Carefully fold the towel and then place the folded towel on the black notebook"  # something like "pick the red block"
robot_type = "agilex_cobot"  # something like "so100_follower" for multi-embodiment datasets


# This is used to match the raw observation keys to the keys expected by the policy
action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

reset_position_left= [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 0.05]
reset_position_right = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 0.05]

# create reset action accroding to the robot's action features
reset_action = {}
print(f"robot.action_features : {robot.action_features}")

for i, key in enumerate(robot.action_features):
    # print(f"i: {i}, key:{key}")
    if "left" in key:
        reset_action[key] = reset_position_left[i]
    elif "right" in key:
        reset_action[key] = reset_position_right[i-7]

for _ in range(MAX_EPISODES):
    robot.reset_to_default_positions()
    time.sleep(3)
    
    step_time = 1 / CONTROL_FREQUENCY
    last_step_time = time.time()
    for _ in range(MAX_STEPS_PER_EPISODE):
        t0 = time.perf_counter()
        
        obs = robot.get_observation()
     
        obs_frame = build_inference_frame(
            observation=obs, ds_features=dataset_features, device=device, task=task, robot_type=robot_type
        )

        obs = preprocess(obs_frame)

        action = model.select_action(obs)
        action = postprocess(action)
        action = make_robot_action(action, dataset_features)
        robot_action_to_send = robot_action_processor((action, obs))
        robot.send_action(action)
        
        t1 = time.perf_counter()
        print(f"Step time: {t1 - t0:.3f} seconds")
        

        # Sleep to maintain the desired frame rate
        now = time.time()
        dt = now - last_step_time
        if dt < step_time:
            time.sleep(step_time - dt)
            last_step_time = time.time()
        else:
            last_step_time = now

    print("Episode finished! Starting new episode...")
