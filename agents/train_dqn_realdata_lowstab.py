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
assert len(day_files) >= 14, f"Expected 14 daily files, found {len(day_files)}"

train_files = day_files[:10]   # same 10 training days as before — keeps this a clean, isolated comparison
train_traces = [rescale_to_range(load_azure_day(f)) for f in train_files]

env = gym.make("Autoscaling-v0", load_traces=train_traces, gamma_stab=0.1)
# ^ only this changed from train_dqn_realdata.py: gamma_stab 0.3 -> 0.1

model = DQN(
    "MlpPolicy", env,
    learning_rate=1e-3, buffer_size=50_000,
    target_update_interval=500, exploration_fraction=0.3,
    verbose=1, tensorboard_log="./logs/dqn_realdata_lowstab/",
)
model.learn(total_timesteps=150_000)
model.save("models/dqn_autoscaling_realdata_lowstab_v0")