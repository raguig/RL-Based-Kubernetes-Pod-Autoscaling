from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import envs
from envs.data_loader import load_azure_day, rescale_to_range


SEED = 42
DATA_DIR = "data/workload_traces"
MODEL_PATH = "models/ppo_autoscaling_discrete_realdata_v0"
VECNORMALIZE_PATH = "models/ppo_autoscaling_discrete_realdata_v0_vecnormalize.pkl"


def make_env(train_traces):
    return Monitor(gym.make("Autoscaling-v0", load_traces=train_traces))


if __name__ == "__main__":
    Path("models").mkdir(exist_ok=True)

    day_files = sorted(glob.glob(f"{DATA_DIR}/invocations_per_function_md.anon.d*.csv"))
    assert len(day_files) >= 14, f"Expected 14 daily files, found {len(day_files)}"

    train_files = day_files[:10]
    train_traces = [rescale_to_range(load_azure_day(f)) for f in train_files]

    env = DummyVecEnv([lambda: make_env(train_traces)])
    env.seed(SEED)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=SEED,
        verbose=1,
        tensorboard_log="./logs/ppo_discrete_realdata/",
    )

    model.learn(total_timesteps=150_000)
    model.save(MODEL_PATH)
    env.save(VECNORMALIZE_PATH)
    env.close()
