from pathlib import Path
import sys



ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from stable_baselines3 import DQN
import envs
from baselines.hpa_policy import HPAPolicy
from evaluation.evaluate import evaluate_policy


def make_dqn_policy(model_path):
    model = DQN.load(model_path)
    def policy(obs, env):
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def make_hpa_policy():
    hpa = HPAPolicy()
    def policy(obs, env):
        current_replicas = env.unwrapped.replicas
        current_cpu = env.unwrapped.cpu
        desired = hpa.act(current_replicas, current_cpu)
        delta = int(np.clip(desired - current_replicas, -2, 2))
        return delta + 2
    return policy


def make_random_policy():
    def policy(obs, env):
        return env.action_space.sample()
    return policy


if __name__ == "__main__":
    env = gym.make("Autoscaling-v0")

    # --- define every policy you want plotted ---
    runs = {
        "Random":            make_random_policy(),
        "HPA":               make_hpa_policy(),
        "DQN (g=0.3)":       make_dqn_policy("models/dqn_autoscaling_v0"),
        "DQN (g=0.7)":       make_dqn_policy("models/dqn_autoscaling_v1"),
        "DQN (g=1.0)":       make_dqn_policy("models/dqn_autoscaling_v2"),
        # add more here as you sweep further gammas, e.g.:
        # "DQN (g=1.5)":     make_dqn_policy("models/dqn_autoscaling_v2"),
    }

    results = {name: evaluate_policy(env, fn, n_episodes=20) for name, fn in runs.items()}

    for name, stats in results.items():
        print(f"{name:<16} reward={stats['reward']:>9.1f}  sla={stats['sla_violation']:>9.1f}  "
              f"cost={stats['cost']:>8.1f}  churn={stats['churn']:>6.1f}")

    # ---------- Figure 1: churn vs SLA violation scatter ----------
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    for name, stats in results.items():
        marker = "*" if name == "HPA" else ("x" if name == "Random" else "o")
        size = 200 if name == "HPA" else 100
        ax1.scatter(stats["churn"], stats["sla_violation"], s=size, marker=marker, label=name)
        ax1.annotate(name, (stats["churn"], stats["sla_violation"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=9)

    ax1.set_xlabel("Churn (total |Δreplicas| per episode)")
    ax1.set_ylabel("SLA violation (summed ms over threshold, per episode)")
    ax1.set_title("Stability vs SLA compliance tradeoff")
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig("evaluation/churn_vs_sla.png", dpi=150)

    # ---------- Figure 2: grouped bar chart, all metrics ----------
    metrics = ["reward", "sla_violation", "cost", "churn"]
    names = list(results.keys())
    fig2, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, metric in zip(axes, metrics):
        values = [results[n][metric] for n in names]
        ax.bar(names, values)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig2.tight_layout()
    fig2.savefig("evaluation/policy_comparison_bars.png", dpi=150)

    print("\nSaved plots to evaluation/churn_vs_sla.png and evaluation/policy_comparison_bars.png")