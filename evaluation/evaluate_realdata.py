from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import glob
import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG, DQN, PPO, SAC, TD3
import envs
from envs.data_loader import load_azure_day, rescale_to_range
from baselines.hpa_policy import HPAPolicy
from evaluation.evaluate import (
    add_vecnormalized_policy_if_available,
    evaluate_policy,
    make_vecnormalized_ddpg_policy,
)


if __name__ == "__main__":
    # --- load the held-out days only (11-14) ---
    day_files = sorted(glob.glob("data/workload_traces/invocations_per_function_md.anon.d*.csv"))
    assert len(day_files) >= 14, f"Expected 14 daily files, found {len(day_files)}"
    held_out_files = day_files[10:]
    held_out_traces = [rescale_to_range(load_azure_day(f)) for f in held_out_files]

    discrete_env = gym.make("Autoscaling-v0", load_traces=held_out_traces)
    continuous_env = gym.make("AutoscalingContinuous-v0", load_traces=held_out_traces)
    tuned_continuous_env = gym.make("AutoscalingContinuousDelta-v0", load_traces=held_out_traces)

    # --- policy 1: random ---
    def random_policy(obs, env):
        return env.action_space.sample()

    # --- policy 2: trained DQN (real-data model, gamma_stab=0.3) ---
    dqn_model = DQN.load("models/dqn_autoscaling_realdata_v0")
    def dqn_policy(obs, env):
        action, _ = dqn_model.predict(obs, deterministic=True)
        return action

    # --- policy 3: HPA baseline ---
    hpa = HPAPolicy()
    def hpa_policy(obs, env):
        current_replicas = env.unwrapped.replicas
        current_cpu = env.unwrapped.cpu
        desired = hpa.act(current_replicas, current_cpu)
        delta = int(np.clip(desired - current_replicas, -2, 2))
        return delta + 2

    # --- policy 4: trained DQN (real-data model, gamma_stab=0.1) ---
    lowstab_model = DQN.load("models/dqn_autoscaling_realdata_lowstab_v0")
    def lowstab_policy(obs, env):
        action, _ = lowstab_model.predict(obs, deterministic=True)
        return action

    # --- policy 5: trained DDPG (real-data model) ---
    ddpg_model = DDPG.load("models/ddpg_autoscaling_realdata_v0")
    def ddpg_policy(obs, env):
        action, _ = ddpg_model.predict(obs, deterministic=True)
        return action

    # --- policy 6: tuned DDPG with continuous delta actions ---
    tuned_ddpg_policy = make_vecnormalized_ddpg_policy(
        "models/ddpg_autoscaling_tuned_delta_realdata_v0",
        "models/ddpg_autoscaling_tuned_delta_realdata_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
    )

    # all policies defined above, now collected together
    policies = {
        "Random": (discrete_env, random_policy),
        "DQN (real data, g=0.3)": (discrete_env, dqn_policy),
        "HPA": (discrete_env, hpa_policy),
        "DQN (real data, g=0.1)": (discrete_env, lowstab_policy),
        "DDPG (real data)": (continuous_env, ddpg_policy),
        "Tuned DDPG (real data)": (tuned_continuous_env, tuned_ddpg_policy),
    }

    add_vecnormalized_policy_if_available(
        policies,
        "SAC (real data)",
        tuned_continuous_env,
        SAC,
        "models/sac_autoscaling_delta_realdata_v0",
        "models/sac_autoscaling_delta_realdata_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
    )
    add_vecnormalized_policy_if_available(
        policies,
        "TD3 (real data)",
        tuned_continuous_env,
        TD3,
        "models/td3_autoscaling_delta_realdata_v0",
        "models/td3_autoscaling_delta_realdata_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
    )
    add_vecnormalized_policy_if_available(
        policies,
        "PPO Disc. (real data)",
        discrete_env,
        PPO,
        "models/ppo_autoscaling_discrete_realdata_v0",
        "models/ppo_autoscaling_discrete_realdata_v0_vecnormalize.pkl",
        "Autoscaling-v0",
        env_kwargs={"load_traces": held_out_traces},
    )
    add_vecnormalized_policy_if_available(
        policies,
        "PPO Delta (real data)",
        tuned_continuous_env,
        PPO,
        "models/ppo_autoscaling_delta_realdata_v0",
        "models/ppo_autoscaling_delta_realdata_v0_vecnormalize.pkl",
        "AutoscalingContinuousDelta-v0",
        env_kwargs={"load_traces": held_out_traces},
    )

    print(f"Evaluating on held-out days: {held_out_files}\n")
    print(f"{'Policy':<28} {'Reward':>10} {'SLA viol.':>12} {'Cost':>10} {'Churn':>8}")
    results = {}
    for name, (policy_env, fn) in policies.items():
        stats = evaluate_policy(policy_env, fn, n_episodes=20)
        results[name] = stats
        print(f"{name:<28} {stats['reward']:>10.1f} {stats['sla_violation']:>12.1f} "
              f"{stats['cost']:>10.1f} {stats['churn']:>8.1f}")
