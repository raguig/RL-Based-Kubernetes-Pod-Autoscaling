from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import envs
from envs.data_loader import load_azure_day, rescale_to_range

DATA_DIR = "data/workload_traces"
day_files = sorted(glob.glob(f"{DATA_DIR}/invocations_per_function_md.anon.d*.csv"))
assert len(day_files) >= 14, f"Expected 14 daily files, found {len(day_files)}"

train_files = day_files[:10]   # same 10 training days used for both DQN real-data runs
train_traces = [rescale_to_range(load_azure_day(f)) for f in train_files]

env = gym.make("AutoscalingContinuous-v0", load_traces=train_traces)

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.0 * np.ones(n_actions))

model = DDPG(
    "MlpPolicy", env,
    learning_rate=1e-3, buffer_size=50_000, tau=0.005,
    action_noise=action_noise,
    train_freq=(1, "episode"),
    verbose=1, tensorboard_log="./logs/ddpg_realdata/",
)
model.learn(total_timesteps=150_000)   # matches your real-data DQN runs for a fair comparison
model.save("models/ddpg_autoscaling_realdata_v0")