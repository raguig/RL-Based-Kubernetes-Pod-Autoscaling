from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import envs
from baselines.hpa_policy import HPAPolicy


def _normalize_action(action, env):
    if isinstance(action, tuple):
        action = action[0]

    if isinstance(env.action_space, gym.spaces.Box):
        action = np.asarray(action, dtype=np.float32)
        if action.shape == ():
            action = action.reshape(1)
        action = action.reshape(env.action_space.shape)
        return np.clip(action, env.action_space.low, env.action_space.high).astype(env.action_space.dtype)

    if isinstance(env.action_space, gym.spaces.Discrete):
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action = action.reshape(-1)[0]
            else:
                action = action[0]

        action = int(action)
        return int(np.clip(action, 0, env.action_space.n - 1))

    return action


def run_episode(env, policy_fn, seed):
    obs, info = env.reset(seed=seed)
    total_reward = 0
    total_sla_violation = 0
    total_cost = 0
    total_churn = 0
    prev_replicas = env.unwrapped.replicas

    for _ in range(env.unwrapped.episode_length):
        action = policy_fn(obs, env)
        action = _normalize_action(action, env)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        total_sla_violation += max(0, info["latency"] - 200)
        total_cost += info["replicas"]
        total_churn += abs(info["replicas"] - prev_replicas)
        prev_replicas = info["replicas"]

        if terminated or truncated:
            break

    return {
        "reward": total_reward,
        "sla_violation": total_sla_violation,
        "cost": total_cost,
        "churn": total_churn,
    }


def evaluate_policy(env, policy_fn, n_episodes=20, base_seed=1000):
    results = [run_episode(env, policy_fn, seed=base_seed + i) for i in range(n_episodes)]
    return {
        key: np.mean([r[key] for r in results])
        for key in results[0]
    }


def make_vecnormalized_policy(model_class, model_path, vecnormalize_path, env_id, env_kwargs=None):
    env_kwargs = env_kwargs or {}
    model = model_class.load(model_path)
    norm_env = DummyVecEnv([lambda: gym.make(env_id, **env_kwargs)])
    norm_env = VecNormalize.load(vecnormalize_path, norm_env)
    norm_env.training = False
    norm_env.norm_reward = False

    def policy(obs, env):
        obs_batch = np.asarray(obs, dtype=np.float32).reshape(1, -1)
        normalized_obs = norm_env.normalize_obs(obs_batch)
        action, _ = model.predict(normalized_obs, deterministic=True)
        return action[0]

    return policy


def make_vecnormalized_ddpg_policy(model_path, vecnormalize_path, env_id, env_kwargs=None):
    return make_vecnormalized_policy(DDPG, model_path, vecnormalize_path, env_id, env_kwargs)


def make_vecnormalized_sac_policy(model_path, vecnormalize_path, env_id, env_kwargs=None):
    return make_vecnormalized_policy(SAC, model_path, vecnormalize_path, env_id, env_kwargs)


def make_vecnormalized_td3_policy(model_path, vecnormalize_path, env_id, env_kwargs=None):
    return make_vecnormalized_policy(TD3, model_path, vecnormalize_path, env_id, env_kwargs)


def make_vecnormalized_ppo_policy(model_path, vecnormalize_path, env_id, env_kwargs=None):
    return make_vecnormalized_policy(PPO, model_path, vecnormalize_path, env_id, env_kwargs)


def add_vecnormalized_policy_if_available(
    policies,
    name,
    env,
    model_class,
    model_path,
    vecnormalize_path,
    env_id,
    env_kwargs=None,
    train_hint=None,
):
    if Path(model_path).with_suffix(".zip").exists() and Path(vecnormalize_path).exists():
        policies[name] = (
            env,
            make_vecnormalized_policy(model_class, model_path, vecnormalize_path, env_id, env_kwargs),
        )
    else:
        if train_hint is None:
            train_hint = "the matching script"
        print(f"Skipping {name}: train it first with {train_hint}.")


if __name__ == "__main__":
    discrete_env = gym.make("Autoscaling-v0")
    continuous_env = gym.make("AutoscalingContinuous-v0")
    tuned_continuous_env = gym.make("AutoscalingContinuousDelta-v0")

    # --- policy 1: random ---
    def random_policy(obs, env):
        return env.action_space.sample()

    # --- policy 2: trained DQN ---
    dqn_model = DQN.load("models/dqn_autoscaling_v0")
    def dqn_policy(obs, env):
        action, _ = dqn_model.predict(obs, deterministic=True)
        return action

    # --- policy 3: HPA baseline ---
    hpa = HPAPolicy()
    def hpa_policy(obs, env):
        current_replicas = env.unwrapped.replicas
        current_cpu = env.unwrapped.cpu
        desired = hpa.act(current_replicas, current_cpu)
        delta = desired - current_replicas
        delta = int(np.clip(delta, -2, 2))       # map back to your DQN's discrete delta range
        return delta + 2                          # convert delta back to action index (0-4)

    # --- policy 4: trained DDPG ---
    ddpg_model = DDPG.load("models/ddpg_autoscaling_v0")
    def ddpg_policy(obs, env):
        action, _ = ddpg_model.predict(obs, deterministic=True)
        return action

    # --- policy 5: tuned DDPG with continuous delta actions ---
    tuned_ddpg_policy = make_vecnormalized_ddpg_policy(
        "models/ddpg_autoscaling_tuned_delta_v0",
        "models/ddpg_autoscaling_tuned_delta_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
    )

    policies = {
        "Random": (discrete_env, random_policy),
        "DQN": (discrete_env, dqn_policy),
        "HPA": (discrete_env, hpa_policy),
        "DDPG": (continuous_env, ddpg_policy),
        "Tuned DDPG": (tuned_continuous_env, tuned_ddpg_policy),
    }

    add_vecnormalized_policy_if_available(
        policies,
        "SAC",
        tuned_continuous_env,
        SAC,
        "models/sac_autoscaling_delta_v0",
        "models/sac_autoscaling_delta_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "TD3",
        tuned_continuous_env,
        TD3,
        "models/td3_autoscaling_delta_v0",
        "models/td3_autoscaling_delta_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "PPO Discrete",
        discrete_env,
        PPO,
        "models/ppo_autoscaling_discrete_v0",
        "models/ppo_autoscaling_discrete_v0_vecnormalize.pkl",
        "Autoscaling-v0",
    )
    add_vecnormalized_policy_if_available(
        policies,
        "PPO Delta",
        tuned_continuous_env,
        PPO,
        "models/ppo_autoscaling_delta_v0",
        "models/ppo_autoscaling_delta_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
    )

    print(f"{'Policy':<16} {'Reward':>10} {'SLA viol.':>12} {'Cost':>10} {'Churn':>8}")
    for name, (policy_env, fn) in policies.items():
        stats = evaluate_policy(policy_env, fn, n_episodes=20)
        print(f"{name:<16} {stats['reward']:>10.1f} {stats['sla_violation']:>12.1f} "
              f"{stats['cost']:>10.1f} {stats['churn']:>8.1f}")
