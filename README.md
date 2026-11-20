# RL-Based Kubernetes Pod Autoscaling

Reinforcement-learning agents (DQN, DDPG, TD3, SAC, PPO) that learn to scale the number of pod
replicas of a service, compared against a Kubernetes HPA-style baseline. Training and evaluation
use a synthetic sinusoidal workload and three public real-world traces.

## Environment (`envs/autoscaling_env.py`)

- **State:** last 5 load values, latency, CPU utilisation, current replicas (8 values)
- **Actions:** discrete `{-2,-1,0,+1,+2}` replicas, continuous absolute target, or continuous delta in `[-2, 2]`
- **Replicas:** 2 to 10, each pod handles 150 req/step
- **Latency model:** `50 / (1 - utilisation)`, capped at 1000 ms. The SLA target is 200 ms
- **Reward:** `-(alpha * SLA_violation + beta * replicas + gamma * |change in replicas|)`, clipped to [-50, 50]

Registered environments: `Autoscaling-v0`, `AutoscalingContinuous-v0`, `AutoscalingContinuousDelta-v0`.

## Datasets

| Dataset | Loader | Raw location (not in repo) |
|---|---|---|
| Azure Functions 2019 (14 days) | `load_azure_day` | `data/workload_traces/` |
| Alibaba cluster-trace 2018 (machines) | `get_alibaba_train_test_traces` | `data/alibaba/machine_usage.csv` |
| Alibaba microservices 2021 | `get_alibaba_microservice_train_test_traces` | `data/alibaba_microservices/` |
| GWA-T-12 Bitbrains | `get_gwat_train_test_traces` | `data/gwat/` |

The raw traces are about 18 GB, so they are not committed. The preprocessed `.npz` caches (200-step
traces with a train/test split) are included, which is enough to run the Alibaba, Alibaba-microservice
and GWA-T evaluations. The Azure evaluation needs the raw Azure CSVs.

## Layout

```
envs/        Gymnasium environment and dataset loaders
baselines/   HPA-style threshold policy
agents/      training scripts, one per algorithm and dataset
evaluation/  evaluation scripts and plots
models/      trained Stable-Baselines3 models (+ VecNormalize stats)
```

## Usage

```bash
pip install -r requirements.txt

# train (examples)
python agents/train_dqn.py
python agents/train_gwat.py --help

# evaluate
python evaluation/evaluate.py            # synthetic workload
python evaluation/evaluate_realdata.py   # Azure held-out days 11-14
python evaluation/evaluate_alibaba.py
python evaluation/evaluate_alibaba_microservices.py
python evaluation/evaluate_gwat.py
```

Evaluation reports the mean episode reward, cumulative SLA violation (ms above 200), cost
(replica-steps) and churn (total replica changes) over 20 episodes.
