from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import envs
from envs.data_loader import load_azure_day, rescale_to_range


SEED = 42
DATA_DIR = "data/workload_traces"
MODEL_PATH = "models/ddpg_autoscaling_tuned_delta_realdata_v0"
VECNORMALIZE_PATH = "models/ddpg_autoscaling_tuned_delta_realdata_v0_vecnormalize.pkl"


def make_env(train_traces):
    return Monitor(gym.make("AutoscalingContinuousDelta-v0", load_traces=train_traces))


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)

    day_files = sorted(glob.glob(f"{DATA_DIR}/invocations_per_function_md.anon.d*.csv"))
    assert len(day_files) >= 14, f"Expected 14 daily files, found {len(day_files)}"

    train_files = day_files[:10]
    train_traces = [rescale_to_range(load_azure_day(f)) for f in train_files]

    env = DummyVecEnv([lambda: make_env(train_traces)])
    env.seed(SEED)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.35 * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        action_noise=action_noise,
        train_freq=(1, "step"),
        gradient_steps=1,
        policy_kwargs={"net_arch": [128, 128]},
        seed=SEED,
        verbose=1,
        tensorboard_log="./logs/ddpg_tuned_realdata/",
    )

    model.learn(total_timesteps=150_000)
    model.save(MODEL_PATH)
    env.save(VECNORMALIZE_PATH)
    env.close()
