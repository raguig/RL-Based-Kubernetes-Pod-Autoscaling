from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
import envs
from baselines.hpa_policy import HPAPolicy
from envs.data_loader import (
    ALIBABA_MICROSERVICE_DEFAULT_CACHE,
    get_alibaba_microservice_train_test_traces,
)
from evaluation.evaluate import add_vecnormalized_policy_if_available, evaluate_policy


N_EPISODES = 20


def model_exists(model_path):
    return Path(model_path).with_suffix(".zip").exists()


def name_to_algo(name):
    return {
        "DQN": "dqn",
        "DDPG": "ddpg",
        "Tuned DDPG": "tuned-ddpg",
        "SAC": "sac",
        "TD3": "td3",
        "PPO Discrete": "ppo-discrete",
        "PPO Delta": "ppo-delta",
    }.get(name, name.lower().replace(" ", "-"))


def add_raw_policy_if_available(policies, name, env, model_class, model_path):
    if not model_exists(model_path):
        print(
            "Skipping "
            f"{name}: train it first with agents/train_alibaba_microservices.py "
            f"--algo {name_to_algo(name)}"
        )
        return

    model = model_class.load(model_path)

    def policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return action

    policies[name] = (env, policy)


def make_random_policy(obs, env):
    return env.action_space.sample()


if __name__ == "__main__":
    try:
        traces = get_alibaba_microservice_train_test_traces(
            cache_path=ALIBABA_MICROSERVICE_DEFAULT_CACHE,
            prepare_if_missing=False,
        )
    except FileNotFoundError as exc:
        print(exc)
        print("Build it with: python agents\\train_alibaba_microservices.py --prepare-only")
        raise SystemExit(1)

    held_out_traces = traces["test_traces"]
    held_out_service_ids = traces["test_service_ids"]

    discrete_env = gym.make("Autoscaling-v0", load_traces=held_out_traces)
    continuous_env = gym.make("AutoscalingContinuous-v0", load_traces=held_out_traces)
    delta_env = gym.make("AutoscalingContinuousDelta-v0", load_traces=held_out_traces)

    hpa = HPAPolicy()

    def hpa_policy(obs, env):
        current_replicas = env.unwrapped.replicas
        current_cpu = env.unwrapped.cpu
        desired = hpa.act(current_replicas, current_cpu)
        delta = int(np.clip(desired - current_replicas, -2, 2))
        return delta + 2

    policies = {
        "Random": (discrete_env, make_random_policy),
        "HPA": (discrete_env, hpa_policy),
    }

    add_raw_policy_if_available(
        policies,
        "DQN",
        discrete_env,
        DQN,
        "models/dqn_autoscaling_alibaba_micro_v0",
    )
    add_raw_policy_if_available(
        policies,
        "DDPG",
        continuous_env,
        DDPG,
        "models/ddpg_autoscaling_alibaba_micro_v0",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "Tuned DDPG",
        delta_env,
        DDPG,
        "models/ddpg_autoscaling_tuned_delta_alibaba_micro_v0",
        "models/ddpg_autoscaling_tuned_delta_alibaba_micro_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
        train_hint="agents/train_alibaba_microservices.py --algo tuned-ddpg",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "SAC",
        delta_env,
        SAC,
        "models/sac_autoscaling_delta_alibaba_micro_v0",
        "models/sac_autoscaling_delta_alibaba_micro_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
        train_hint="agents/train_alibaba_microservices.py --algo sac",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "TD3",
        delta_env,
        TD3,
        "models/td3_autoscaling_delta_alibaba_micro_v0",
        "models/td3_autoscaling_delta_alibaba_micro_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
        train_hint="agents/train_alibaba_microservices.py --algo td3",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "PPO Discrete",
        discrete_env,
        PPO,
        "models/ppo_autoscaling_discrete_alibaba_micro_v0",
        "models/ppo_autoscaling_discrete_alibaba_micro_v0_vecnormalize.pkl",
        "Autoscaling-v0",
        env_kwargs={"load_traces": held_out_traces},
        train_hint="agents/train_alibaba_microservices.py --algo ppo-discrete",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "PPO Delta",
        delta_env,
        PPO,
        "models/ppo_autoscaling_delta_alibaba_micro_v0",
        "models/ppo_autoscaling_delta_alibaba_micro_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
        train_hint="agents/train_alibaba_microservices.py --algo ppo-delta",
    )

    print(
        "Evaluating on Alibaba microservice held-out services: "
        f"{len(held_out_service_ids)} services, {held_out_traces.shape[1]} steps per trace\n"
    )
    print(f"{'Policy':<16} {'Reward':>10} {'SLA viol.':>12} {'Cost':>10} {'Churn':>8}")

    for name, (policy_env, fn) in policies.items():
        stats = evaluate_policy(policy_env, fn, n_episodes=N_EPISODES)
        print(
            f"{name:<16} {stats['reward']:>10.1f} {stats['sla_violation']:>12.1f} "
            f"{stats['cost']:>10.1f} {stats['churn']:>8.1f}"
        )
