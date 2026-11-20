from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import envs
from envs.data_loader import ALIBABA_DEFAULT_CACHE, get_alibaba_train_test_traces


SEED = 42
DEFAULT_TIMESTEPS = 150_000

MODEL_PATHS = {
    "dqn": "models/dqn_autoscaling_alibaba_v0",
    "ddpg": "models/ddpg_autoscaling_alibaba_v0",
    "tuned-ddpg": "models/ddpg_autoscaling_tuned_delta_alibaba_v0",
    "sac": "models/sac_autoscaling_delta_alibaba_v0",
    "td3": "models/td3_autoscaling_delta_alibaba_v0",
    "ppo-discrete": "models/ppo_autoscaling_discrete_alibaba_v0",
    "ppo-delta": "models/ppo_autoscaling_delta_alibaba_v0",
}

VECNORMALIZE_PATHS = {
    "tuned-ddpg": "models/ddpg_autoscaling_tuned_delta_alibaba_v0_vecnormalize.pkl",
    "sac": "models/sac_autoscaling_delta_alibaba_v0_vecnormalize.pkl",
    "td3": "models/td3_autoscaling_delta_alibaba_v0_vecnormalize.pkl",
    "ppo-discrete": "models/ppo_autoscaling_discrete_alibaba_v0_vecnormalize.pkl",
    "ppo-delta": "models/ppo_autoscaling_delta_alibaba_v0_vecnormalize.pkl",
}

TRAIN_ORDER = [
    "dqn",
    "ddpg",
    "tuned-ddpg",
    "sac",
    "td3",
    "ppo-discrete",
    "ppo-delta",
]


def make_raw_env(env_id, train_traces):
    return Monitor(gym.make(env_id, load_traces=train_traces))


def make_normalized_env(env_id, train_traces):
    env = DummyVecEnv([lambda: make_raw_env(env_id, train_traces)])
    env.seed(SEED)
    return VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
    )


def train_dqn(train_traces, timesteps):
    env = make_raw_env("Autoscaling-v0", train_traces)
    env.action_space.seed(SEED)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        target_update_interval=500,
        exploration_fraction=0.3,
        seed=SEED,
        verbose=1,
        tensorboard_log="./logs/dqn_alibaba/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATHS["dqn"])
    env.close()


def train_ddpg(train_traces, timesteps):
    env = make_raw_env("AutoscalingContinuous-v0", train_traces)
    env.action_space.seed(SEED)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=1.0 * np.ones(n_actions),
    )

    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        tau=0.005,
        action_noise=action_noise,
        train_freq=(1, "episode"),
        seed=SEED,
        verbose=1,
        tensorboard_log="./logs/ddpg_alibaba/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATHS["ddpg"])
    env.close()


def train_tuned_ddpg(train_traces, timesteps):
    env = make_normalized_env("AutoscalingContinuousDelta-v0", train_traces)

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
        tensorboard_log="./logs/ddpg_tuned_alibaba/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATHS["tuned-ddpg"])
    env.save(VECNORMALIZE_PATHS["tuned-ddpg"])
    env.close()


def train_sac(train_traces, timesteps):
    env = make_normalized_env("AutoscalingContinuousDelta-v0", train_traces)

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
        tensorboard_log="./logs/sac_alibaba/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATHS["sac"])
    env.save(VECNORMALIZE_PATHS["sac"])
    env.close()


def train_td3(train_traces, timesteps):
    env = make_normalized_env("AutoscalingContinuousDelta-v0", train_traces)

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.35 * np.ones(n_actions),
    )

    model = TD3(
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
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs={"net_arch": [128, 128]},
        seed=SEED,
        verbose=1,
        tensorboard_log="./logs/td3_alibaba/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATHS["td3"])
    env.save(VECNORMALIZE_PATHS["td3"])
    env.close()


def train_ppo_discrete(train_traces, timesteps):
    env = make_normalized_env("Autoscaling-v0", train_traces)

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
        tensorboard_log="./logs/ppo_discrete_alibaba/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATHS["ppo-discrete"])
    env.save(VECNORMALIZE_PATHS["ppo-discrete"])
    env.close()


def train_ppo_delta(train_traces, timesteps):
    env = make_normalized_env("AutoscalingContinuousDelta-v0", train_traces)

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
        tensorboard_log="./logs/ppo_delta_alibaba/",
    )
    model.learn(total_timesteps=timesteps)
    model.save(MODEL_PATHS["ppo-delta"])
    env.save(VECNORMALIZE_PATHS["ppo-delta"])
    env.close()


TRAINERS = {
    "dqn": train_dqn,
    "ddpg": train_ddpg,
    "tuned-ddpg": train_tuned_ddpg,
    "sac": train_sac,
    "td3": train_td3,
    "ppo-discrete": train_ppo_discrete,
    "ppo-delta": train_ppo_delta,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train autoscaling agents on Alibaba machine usage traces.")
    parser.add_argument(
        "--algo",
        choices=TRAIN_ORDER + ["all"],
        default="sac",
        help="Algorithm to train. Use 'all' to train every Alibaba model sequentially.",
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--csv-path", default="data/alibaba/machine_usage.csv")
    parser.add_argument("--cache-path", default=ALIBABA_DEFAULT_CACHE)
    parser.add_argument("--max-machines", type=int, default=256)
    parser.add_argument("--min-points", type=int, default=200)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only build/load the compact Alibaba trace cache; do not train a model.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    Path("models").mkdir(exist_ok=True)

    traces = get_alibaba_train_test_traces(
        csv_path=args.csv_path,
        cache_path=args.cache_path,
        max_machines=args.max_machines,
        min_points=args.min_points,
        chunksize=args.chunksize,
        force=args.force_cache,
    )
    train_traces = traces["train_traces"]
    test_traces = traces["test_traces"]
    print(
        "Alibaba cache ready: "
        f"{len(train_traces)} train machines, {len(test_traces)} held-out machines, "
        f"{train_traces.shape[1]} steps per trace."
    )

    if args.prepare_only:
        raise SystemExit(0)

    algos = TRAIN_ORDER if args.algo == "all" else [args.algo]
    for algo in algos:
        print(f"\n=== Training {algo} on Alibaba traces for {args.timesteps} timesteps ===")
        TRAINERS[algo](train_traces, args.timesteps)
