# agents/train_dqn_realdata.py
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import gymnasium as gym
from stable_baselines3 import DQN
import envs
from envs.data_loader import load_azure_day, rescale_to_range

DATA_DIR = "data/workload_traces"
day_files = sorted(glob.glob(f"{DATA_DIR}/invocations_per_function_md.anon.d*.csv"))
assert len(day_files) >= 14, f"Expected 14 daily files, found {len(day_files)} — check your data folder"

train_files = day_files[:10]     # days 1-10 for training
held_out_files = day_files[10:]  # days 11-14 reserved for evaluation only

train_traces = [rescale_to_range(load_azure_day(f, n_bins=200)) for f in train_files]

env = gym.make("Autoscaling-v0", load_traces=train_traces)
model = DQN(
    "MlpPolicy", env,
    learning_rate=1e-3, buffer_size=50_000,
    target_update_interval=500, exploration_fraction=0.3,
    verbose=1, tensorboard_log="./logs/dqn_realdata/",
)
model.learn(total_timesteps=150_000)   # bumped up from 100k since it's now learning across 10 distinct patterns, not one repeated sine wave
model.save("models/dqn_autoscaling_realdata_v0")