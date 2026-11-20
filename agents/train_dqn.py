from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
from stable_baselines3 import DQN
import envs  # now resolves to your local package

env = gym.make("Autoscaling-v0")

model = DQN(
    "MlpPolicy",       # small feedforward network — fine here, since your state is 8 numbers, not raw pixels (no CNN needed)
    env,
    learning_rate=1e-3,
    buffer_size=50_000,       # replay buffer capacity
    target_update_interval=500,  # equivalent to "C" from the DQN paper
    exploration_fraction=0.3,    # how much of training uses annealing epsilon
    verbose=1,
    tensorboard_log="./logs/dqn/",
)

model.learn(total_timesteps=100_000)
model.save("models/dqn_autoscaling_v2")