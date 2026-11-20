from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
import envs

env = gym.make("AutoscalingContinuous-v0")

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=1.0 * np.ones(n_actions))

model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50_000,
    tau=0.005,
    action_noise=action_noise,
    train_freq=(1, "episode"),   # the fix — train once per episode instead of every step
    verbose=1,
    tensorboard_log="./logs/ddpg/",
)

model.learn(total_timesteps=100_000)
model.save("models/ddpg_autoscaling_v0")