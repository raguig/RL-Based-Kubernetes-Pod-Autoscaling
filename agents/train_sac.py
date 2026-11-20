from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import envs


SEED = 42
MODEL_PATH = "models/sac_autoscaling_delta_v0"
VECNORMALIZE_PATH = "models/sac_autoscaling_delta_v0_vecnormalize.pkl"


def make_env():
    return Monitor(gym.make("AutoscalingContinuousDelta-v0"))


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)

    env = DummyVecEnv([make_env])
    env.seed(SEED)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=5_000,
        batch_size=128,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        policy_kwargs={"net_arch": [128, 128]},
        seed=SEED,
        verbose=1,
        tensorboard_log="./logs/sac/",
    )

    model.learn(total_timesteps=100_000)
    model.save(MODEL_PATH)
    env.save(VECNORMALIZE_PATH)
    env.close()
