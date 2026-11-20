# Reinforcement Learning for Kubernetes Pod Autoscaling

Full technical report for the RL-autoscale project.

## 1. Executive Summary

This project studies whether reinforcement learning can improve Kubernetes-style pod autoscaling compared with simple baselines such as Random scaling and HPA, the Horizontal Pod Autoscaler. The system models an autoscaling controller that observes recent workload, latency, CPU usage, and replica count, then decides whether to change the number of running pods.

The project compares multiple algorithms:

- Random baseline
- HPA baseline
- DQN
- DQN with lower stability penalty
- Original DDPG
- Tuned DDPG
- SAC
- TD3
- PPO Discrete
- PPO Delta

The key result is that not all RL methods work equally well. The best results come from algorithms that use bounded delta scaling actions, normalization, and stable actor-critic training. SAC, TD3, and Tuned DDPG are usually the strongest methods because they learn smooth scaling behavior without collapsing to minimum or maximum replicas.

The original DDPG performs badly because it predicts an absolute replica count directly. In several datasets it collapses to a boundary policy, usually minimum replicas, producing very low cost and zero churn but very large SLA violation. After tuning DDPG to use delta actions and normalized training, it becomes one of the strongest policies.

The DQN gamma_stab experiment shows that reward design matters. Lowering gamma_stab from 0.3 to 0.1 reduced the penalty for changing replicas. This allowed DQN to react more aggressively to real workload spikes, improving reward and reducing SLA violation, but it also increased cost and churn.

Overall, the project shows a meaningful research conclusion:

> RL autoscaling can outperform HPA, but only when the action space and reward design match the behavior of real autoscaling systems.

## 2. Project Goal

The goal of the project is to design and evaluate RL-based autoscaling policies for Kubernetes pod autoscaling. In Kubernetes, an autoscaler must decide how many pod replicas should run at each moment. Too few replicas cause high latency and SLA violations. Too many replicas increase resource cost. Frequent scaling changes cause instability and operational overhead.

The project treats autoscaling as a reinforcement learning problem:

- The environment represents an application receiving changing workload.
- The agent observes the current system state.
- The agent chooses a scaling action.
- The environment updates latency, CPU, replicas, and reward.
- The agent learns a policy that maximizes long-term reward.

This is a useful internship project because it connects machine learning with cloud infrastructure, Kubernetes operations, time-series workloads, and performance/cost trade-offs.

## 3. Project Architecture

The project is organized into four main parts:

| Part | Main files | Role |
|---|---|---|
| Environment | `envs/autoscaling_env.py`, `envs/__init__.py` | Defines the autoscaling simulation and Gymnasium environments |
| Data loading | `envs/data_loader.py` | Converts real datasets into 200-step workload traces |
| Training | `agents/train_*.py` | Trains each RL algorithm on synthetic or real traces |
| Evaluation | `evaluation/evaluate*.py` | Runs fixed evaluation episodes and reports reward, SLA violation, cost, and churn |

The model outputs are stored in `models/`. TensorBoard logs are stored in `logs/`. The benchmark document is stored in `Synthetic_vs_RealData_Evaluation_Results.docx`.

## 4. Environment Design

The central environment is implemented in `envs/autoscaling_env.py`.

Each episode has 200 timesteps. At every timestep, the environment has:

- Current workload/load
- Current latency
- Current CPU utilization
- Current number of replicas
- Recent load history

The observation vector contains:

```text
[load_t-4, load_t-3, load_t-2, load_t-1, load_t, latency, cpu, replicas]
```

With the default `history_len=5`, the observation has 8 numeric values.

The environment supports three action designs.

### 4.1 Discrete Delta Actions

Used by DQN and PPO Discrete.

The action space has 5 choices:

| Action index | Meaning |
|---:|---|
| 0 | Scale down by 2 replicas |
| 1 | Scale down by 1 replica |
| 2 | Keep same replica count |
| 3 | Scale up by 1 replica |
| 4 | Scale up by 2 replicas |

This is close to real autoscaling because Kubernetes does not usually jump randomly to any replica count. It adjusts capacity over time.

### 4.2 Continuous Absolute Actions

Used by the original DDPG.

The action is a continuous number between `n_min` and `n_max`. The environment rounds it and sets the replica count directly.

Example:

```text
action = 2.1 -> replicas = 2
action = 9.7 -> replicas = 10
```

This design caused problems. The policy often learned to output boundary values such as minimum replicas or maximum replicas. That explains why original DDPG often has either very low cost with huge SLA violation or very high cost with low SLA violation.

### 4.3 Continuous Delta Actions

Used by Tuned DDPG, SAC, TD3, and PPO Delta.

The action is a continuous value between `-2` and `+2`. The environment rounds it and applies it as a replica change.

Example:

```text
action = -1.4 -> scale down by 1
action = 0.2 -> stay almost the same
action = 1.8 -> scale up by 2
```

This is the best action design in the project. It makes the agent learn controlled scaling behavior instead of absolute boundary decisions.

## 5. Capacity, Latency, and CPU Model

The environment uses a simple capacity model:

```text
capacity_per_pod = 150
total_capacity = replicas * capacity_per_pod
utilization = load / total_capacity
cpu = min(100, utilization * 100)
```

Latency increases sharply as utilization approaches 100 percent:

```text
latency = 50 / max(0.01, 1 - min(utilization, 0.99))
latency = min(latency, 1000)
```

This creates the expected autoscaling behavior:

- If replicas are too low, utilization becomes high and latency explodes.
- If replicas are high enough, latency stays lower.
- More replicas reduce SLA violation but increase cost.

The model is simplified, but it is useful for comparing scaling policies under the same conditions.

## 6. Reward Function

The reward function is:

```text
sla_violation = max(0, latency - 200)
cost = replicas
stability_penalty = abs(replicas - previous_replicas)

reward = -(alpha * sla_violation + beta * cost + gamma_stab * stability_penalty)
reward = clip(reward, -50, 50)
```

The default weights are:

```text
alpha = 1.0
beta = 0.5
gamma_stab = 0.1
```

The reward is negative because it is a penalty. A better policy has a reward closer to zero.

The three reward components have different meanings:

| Component | Meaning | Desired behavior |
|---|---|---|
| SLA violation | Latency above 200 ms | Lower is better |
| Cost | Number of running replicas | Lower is better |
| Stability penalty | Change in replica count | Lower is better |

The agent must learn a balance:

- Running too few replicas gives low cost but high SLA violation.
- Running too many replicas gives low SLA violation but high cost.
- Changing replicas too often creates high churn.

## 7. Important Clarification: gamma_stab Is Not Discount Gamma

In reinforcement learning, the word gamma often means the discount factor. The discount factor controls how much the agent cares about future rewards.

In this project, `gamma_stab` is different.

`gamma_stab` is the weight of the stability/churn penalty inside the reward function:

```text
gamma_stab * abs(replicas - previous_replicas)
```

So changing `gamma_stab` changes how much the agent is punished for scaling up or down.

### 7.1 Why Lowering gamma_stab Improved DQN

The original real-data DQN used a higher stability penalty, `gamma_stab=0.3`. That made the agent cautious. It preferred to avoid replica changes because every change was expensive in the reward.

On smooth synthetic traffic, this caution is not too harmful. The workload changes gradually, so the agent can keep churn low and still avoid many SLA violations.

On real Azure traffic, the workload is more bursty. A cautious DQN reacts too slowly. It saves cost and churn, but it allows latency spikes.

After lowering `gamma_stab` to `0.1`, DQN was punished less for changing replicas. It became more willing to scale. The result was:

| DQN version | Reward | SLA violation | Cost | Churn |
|---|---:|---:|---:|---:|
| DQN real data, gamma_stab=0.3 | -1,406.8 | 5,178.6 | 1,050.5 | 109.9 |
| DQN real data, gamma_stab=0.1 | -932.5 | 1,444.0 | 1,351.8 | 325.6 |

This confirms the trade-off:

- Reward improved strongly.
- SLA violation dropped strongly.
- Cost increased.
- Churn increased.

The agent became better at protecting performance, but less stable and more expensive.

## 8. Datasets Used

The project uses five workload sources.

### 8.1 Synthetic Workload

The synthetic workload is generated directly by the environment:

```text
load = 500 + 400 * sin(2 * pi * t / episode_length) + noise
```

This creates a smooth rise-and-fall workload. It is useful for initial experiments because the pattern is easy to learn and debug.

However, it is not enough for research conclusions because real cloud traffic is not always smooth.

### 8.2 Azure Functions Dataset

The Azure dataset uses invocation traces from serverless functions.

In `envs/data_loader.py`, `load_azure_day()`:

- Reads one daily CSV file.
- Sums invocation counts across all functions.
- Produces one aggregate per-minute load signal.
- Compresses the day into 200 bins.
- Rescales the load into the environment range.

Training uses the first 10 days. Evaluation uses held-out days 11 to 14. This matters because it tests generalization to days that the agent did not train on.

### 8.3 Alibaba Machine Usage Dataset

The Alibaba machine dataset contains machine-level resource utilization.

The project uses:

```text
data/alibaba/machine_usage.csv
```

Each machine trace is converted into one workload signal. The pressure score uses:

```text
65 percent CPU
20 percent memory
10 percent network
5 percent disk
```

This is appropriate for Kubernetes autoscaling because HPA is usually CPU-driven, but real pressure can also come from memory and I/O.

The train/test split is by machine ID, so evaluation uses machines not seen during training.

### 8.4 Alibaba Microservice Dataset

The Alibaba microservice dataset contains service-level behavior.

The project uses:

```text
data/alibaba_microservices/MSRTQps/MSRTQps_0.csv
data/alibaba_microservices/MSResource/MSResource_0.csv
```

The loader combines:

- QPS/request rate
- Response time
- Instance CPU usage
- Instance memory usage

This is very relevant to pod autoscaling because pods normally serve service traffic, and autoscaling decisions should react to service-level demand.

The split is by microservice, not by row. Evaluation uses held-out services.

### 8.5 GWA-T / Bitbrains VM Dataset

The GWA-T dataset contains virtual-machine traces with CPU, memory, disk, and network usage.

The project uses files under:

```text
data/gwat/
```

Each CSV represents one VM trace. The loader converts each VM trace into a 200-step workload signal using:

```text
60 percent CPU
25 percent memory
10 percent network
5 percent disk
```

This dataset is useful because it is resource-utilization based, like Alibaba machine data, but comes from a different cloud environment. It tests whether the learned autoscaling ideas generalize beyond one provider or one trace style.

## 9. Evaluation Method

Each evaluation script runs policies for 20 fixed-seed episodes and reports the mean result.

The metrics are:

| Metric | Meaning | Direction |
|---|---|---|
| Reward | Combined penalty from SLA, cost, and churn | Higher is better |
| SLA violation | Total latency above 200 ms | Lower is better |
| Cost | Sum of replicas over the episode | Lower is better |
| Churn | Total replica changes | Lower is better |

The evaluation files are:

| File | Dataset |
|---|---|
| `evaluation/evaluate.py` | Synthetic workload |
| `evaluation/evaluate_realdata.py` | Azure real data |
| `evaluation/evaluate_alibaba.py` | Alibaba machine data |
| `evaluation/evaluate_alibaba_microservices.py` | Alibaba microservice data |
| `evaluation/evaluate_gwat.py` | GWA-T / Bitbrains VM data |

## 10. Algorithms Used

### 10.1 Random Baseline

Random selects a valid action at every timestep without using the observation.

It is not supposed to perform well. Its purpose is to prove that the environment is meaningful. If a trained agent cannot beat Random, then the training setup is broken or the algorithm is not learning useful behavior.

Random performs badly in every dataset because it creates unnecessary scaling changes and often has the wrong number of replicas.

### 10.2 HPA Baseline

HPA is the Kubernetes Horizontal Pod Autoscaler style baseline.

It uses the current CPU utilization and current replica count to choose a desired replica count. HPA is strong because autoscaling is naturally related to CPU pressure.

However, HPA is rule-based. It does not learn from previous episodes. It does not optimize the exact reward function. It does not explicitly balance future SLA, cost, and churn. Because of that, learned methods can beat it on complex traces.

HPA is strongest on smoother or CPU-aligned workloads. It becomes weaker when the trace has bursty behavior or when the best policy requires a more careful trade-off than a fixed CPU rule.

### 10.3 DQN

DQN is a value-based reinforcement learning algorithm for discrete actions.

In this project, DQN chooses one of five delta actions:

```text
-2, -1, 0, +1, +2
```

DQN learns a Q-value for each action. A Q-value estimates how good an action is in the current state.

DQN is useful here because pod autoscaling can naturally be represented as small scale-up or scale-down decisions. It performed well on synthetic data and was competitive on Alibaba microservice data.

DQN struggled on some real workloads because:

- It has only five discrete actions.
- It can become conservative if the stability penalty is high.
- It may react too slowly to bursts.
- Its value estimates can generalize poorly when test traces are more irregular than training traces.

### 10.4 Original DDPG

DDPG is an off-policy actor-critic algorithm for continuous actions.

Original DDPG used the continuous absolute action environment:

```text
action = desired replica count
```

This sounds reasonable, but it caused a major problem. The actor often learned boundary outputs. It would repeatedly choose minimum replicas or maximum replicas.

That explains the original DDPG results:

- When it chooses minimum replicas, cost is very low and churn is zero, but SLA violation becomes huge.
- When it chooses maximum replicas, SLA violation can be low, but cost becomes very high.

The original DDPG results show that continuous RL is not automatically better. The action design matters.

### 10.5 Tuned DDPG

Tuned DDPG keeps the DDPG algorithm but fixes the training setup.

The main tuning changes were:

| Tuning change | Why it helped |
|---|---|
| Changed action from absolute replica count to bounded delta | Prevented direct collapse to min/max replicas |
| Used `AutoscalingContinuousDelta-v0` | Made the action closer to real autoscaling behavior |
| Added `VecNormalize` for observations and rewards | Stabilized learning across different workload scales |
| Lowered learning rate to `3e-4` | Reduced unstable actor/critic updates |
| Increased replay buffer to `100_000` | Improved experience diversity |
| Added `learning_starts=5_000` | Let the buffer collect experience before training |
| Used `batch_size=128` | Made gradient updates more stable |
| Used step-based training frequency | Updated more consistently than episode-only training |
| Used smaller exploration noise, `sigma=0.35` | Encouraged exploration without wild replica jumps |
| Used network architecture `[128, 128]` | Increased model capacity while staying small enough for this environment |

Tuned DDPG performed much better than original DDPG because it no longer treated autoscaling as "choose any replica count instantly." It learned "adjust the current replica count carefully."

### 10.6 SAC

SAC, Soft Actor-Critic, is an off-policy actor-critic algorithm for continuous actions.

Its key idea is entropy regularization. In simple terms, SAC is rewarded not only for high return but also for maintaining useful exploration. It does not become too deterministic too early.

This is very helpful for autoscaling because the best policy is a balance:

- Scale enough to avoid SLA violation.
- Do not over-scale too much.
- Do not churn too much.

SAC was often the best algorithm because it explored many capacity choices during training and found a strong balance between SLA, cost, and churn.

### 10.7 TD3

TD3, Twin Delayed DDPG, is an improved version of DDPG.

It fixes some common DDPG weaknesses:

- It uses two critics to reduce overestimation.
- It delays actor updates to make learning more stable.
- It adds target policy smoothing to reduce brittle action choices.

TD3 performed strongly because it keeps the benefits of continuous control while avoiding some instability of original DDPG. In the results, TD3 often has very low SLA violation, but sometimes uses more replicas than SAC, which slightly reduces its reward.

### 10.8 PPO Discrete

PPO is an on-policy policy-gradient algorithm. PPO Discrete uses the same five discrete delta actions as DQN.

PPO is stable because it uses clipped policy updates. This prevents the policy from changing too aggressively during training.

PPO Discrete performed reasonably well because it avoids the boundary collapse of original DDPG. However, it is often not the best because it can be conservative and expensive. It sometimes keeps extra replicas running to avoid SLA violations, which hurts cost.

### 10.9 PPO Delta

PPO Delta uses PPO with continuous delta actions.

This gives PPO a smoother action space than PPO Discrete. It can learn gradual scale-up and scale-down behavior.

However, PPO is on-policy, meaning it learns only from fresh experience collected by the current policy. Off-policy methods such as SAC, TD3, and DDPG can reuse replay-buffer experience more efficiently. In this project, PPO Delta is stable but not consistently better than SAC or TD3.

## 11. Benchmark Results

### 11.1 Synthetic Workload

| Policy | Reward | SLA violation | Cost | Churn |
|---|---:|---:|---:|---:|
| Random | -3,842.6 | 44,591.4 | 1,173.1 | 195.5 |
| DQN | -744.6 | 363.9 | 1,241.3 | 24.8 |
| HPA | -718.2 | 189.3 | 1,323.8 | 44.4 |
| DDPG | -7,369.0 | 111,228.7 | 400.0 | 0.0 |
| Tuned DDPG | -628.9 | 205.0 | 1,113.0 | 70.8 |

On synthetic traffic, Tuned DDPG wins by reward. HPA has slightly lower SLA violation, but Tuned DDPG uses fewer replicas, so its total reward is better.

Original DDPG fails completely by staying at minimum replicas. Its cost is only 400.0 and churn is 0.0, but SLA violation is extremely high.

### 11.2 Azure Real Data

| Policy | Reward | SLA violation | Cost | Churn |
|---|---:|---:|---:|---:|
| Random | -3,548.8 | 40,265.0 | 1,173.1 | 195.5 |
| DQN, gamma_stab=0.3 | -1,406.8 | 5,178.6 | 1,050.5 | 109.9 |
| HPA | -840.2 | 1,092.7 | 1,287.0 | 187.8 |
| DQN, gamma_stab=0.1 | -932.5 | 1,444.0 | 1,351.8 | 325.6 |
| DDPG | -1,000.8 | 0.0 | 2,000.0 | 8.0 |
| Tuned DDPG | -744.6 | 717.7 | 1,265.2 | 173.2 |
| SAC | -702.9 | 452.6 | 1,250.6 | 146.0 |
| TD3 | -733.2 | 598.8 | 1,259.5 | 168.2 |
| PPO Discrete | -739.9 | 527.9 | 1,296.5 | 215.2 |
| PPO Delta | -740.2 | 507.7 | 1,318.6 | 192.2 |

On Azure, SAC is best. TD3, PPO, and Tuned DDPG are also strong and all beat HPA by reward. This is important because it shows that learned autoscaling can outperform a Kubernetes-style rule baseline on held-out real traffic.

Original DDPG gets zero SLA violation, but at very high cost. It keeps too many replicas running. That is not a good autoscaling policy, even though the SLA number looks excellent.

### 11.3 Alibaba Machine Data

| Policy | Reward | SLA violation | Cost | Churn |
|---|---:|---:|---:|---:|
| Random | -3,414.2 | 37,659.9 | 1,173.1 | 195.5 |
| HPA | -1,329.8 | 5,965.5 | 1,258.7 | 192.1 |
| DQN | -1,297.4 | 5,484.2 | 1,209.7 | 234.8 |
| DDPG | -7,852.6 | 115,233.8 | 400.0 | 0.0 |
| Tuned DDPG | -779.3 | 442.9 | 1,388.4 | 143.8 |
| SAC | -770.5 | 144.5 | 1,467.2 | 114.3 |
| TD3 | -774.3 | 150.3 | 1,439.1 | 136.2 |
| PPO Discrete | -812.2 | 211.6 | 1,500.8 | 178.8 |
| PPO Delta | -812.1 | 147.3 | 1,554.3 | 145.4 |

On Alibaba machine traces, SAC is best, closely followed by TD3 and Tuned DDPG. These methods strongly reduce SLA violation compared with HPA and DQN.

The result is interesting because Alibaba machine data is resource-utilization based. SAC and TD3 handle this well because continuous delta control lets them adjust capacity smoothly.

### 11.4 Alibaba Microservice Data

| Policy | Reward | SLA violation | Cost | Churn |
|---|---:|---:|---:|---:|
| Random | -3,491.4 | 38,453.8 | 1,173.1 | 195.5 |
| HPA | -747.0 | 639.5 | 1,274.9 | 106.9 |
| DQN | -695.9 | 502.2 | 1,231.3 | 192.7 |
| DDPG | -1,000.8 | 0.0 | 2,000.0 | 8.0 |
| Tuned DDPG | -650.2 | 429.8 | 1,131.3 | 128.2 |
| SAC | -622.0 | 333.2 | 1,154.0 | 90.8 |
| TD3 | -651.6 | 384.5 | 1,140.3 | 138.2 |
| PPO Discrete | -662.9 | 527.6 | 1,150.8 | 101.8 |
| PPO Delta | -794.9 | 626.9 | 1,372.0 | 167.5 |

On Alibaba microservices, SAC is again best. This dataset favors policies that balance request-level performance with cost. SAC has the best overall trade-off: low SLA violation, low cost, and low churn.

DQN is also strong here and beats HPA. That suggests the microservice traces produce a pattern where discrete delta actions are enough to learn useful behavior.

### 11.5 GWA-T / Bitbrains VM Data

| Policy | Reward | SLA violation | Cost | Churn |
|---|---:|---:|---:|---:|
| Random | -3,361.9 | 38,469.7 | 1,173.1 | 195.5 |
| HPA | -1,674.5 | 12,554.1 | 1,173.4 | 180.2 |
| DQN | -1,681.6 | 12,947.9 | 1,182.1 | 159.8 |
| DDPG | -6,769.1 | 100,557.2 | 400.0 | 0.0 |
| Tuned DDPG | -907.3 | 550.4 | 1,621.2 | 102.7 |
| SAC | -893.9 | 810.0 | 1,525.5 | 73.3 |
| TD3 | -897.3 | 640.3 | 1,630.0 | 96.3 |
| PPO Discrete | -937.0 | 526.8 | 1,753.5 | 120.5 |
| PPO Delta | -937.4 | 584.2 | 1,736.0 | 108.3 |

GWA-T is harder for HPA and DQN. Both beat Random, but both have high SLA violation. The continuous delta methods are much stronger.

SAC wins by reward because it has the best overall balance. PPO has low SLA violation but uses many replicas, so cost hurts its reward. Tuned DDPG and TD3 are very close to SAC.

## 12. Dataset-by-Algorithm Reward Matrix

Higher reward is better because all rewards are negative penalties.

| Dataset | Random | HPA | DQN | DQN gamma_stab=0.1 | DDPG | Tuned DDPG | SAC | TD3 | PPO Discrete | PPO Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Synthetic | -3,842.6 | -718.2 | -744.6 | real-data only | -7,369.0 | -628.9 | not trained | not trained | not trained | not trained |
| Azure real data | -3,548.8 | -840.2 | -1,406.8 | -932.5 | -1,000.8 | -744.6 | -702.9 | -733.2 | -739.9 | -740.2 |
| Alibaba machine | -3,414.2 | -1,329.8 | -1,297.4 | not trained | -7,852.6 | -779.3 | -770.5 | -774.3 | -812.2 | -812.1 |
| Alibaba microservice | -3,491.4 | -747.0 | -695.9 | not trained | -1,000.8 | -650.2 | -622.0 | -651.6 | -662.9 | -794.9 |
| GWA-T / Bitbrains | -3,361.9 | -1,674.5 | -1,681.6 | not trained | -6,769.1 | -907.3 | -893.9 | -897.3 | -937.0 | -937.4 |

## 13. Why SAC, TD3, PPO, and Tuned DDPG Beat the Others

The best algorithms share one major advantage: they learn bounded scaling changes instead of absolute replica counts.

In autoscaling, the current replica count matters. If the system has 6 pods, the next action should usually be "add one", "remove one", or "stay", not "jump directly to 2 or 10". Delta actions make learning easier and safer.

### 13.1 Why SAC Often Wins

SAC wins often because it balances exploration and stability. Its entropy term prevents the policy from becoming too narrow too early. In autoscaling, this helps because there are many possible trade-offs between cost and SLA violation.

SAC usually finds policies that:

- Avoid severe SLA violations.
- Do not over-provision as much as the most conservative policies.
- Keep churn relatively controlled.

That is why SAC wins on Azure, Alibaba machine, Alibaba microservice, and GWA-T in the completed benchmark.

### 13.2 Why TD3 Is Strong

TD3 is strong because it fixes DDPG's critic instability. The two-critic design reduces overestimated Q-values, and delayed policy updates make the actor less likely to chase noisy critic estimates.

TD3 often gets very low SLA violation. It sometimes uses slightly more replicas than SAC, so SAC can still win by reward.

### 13.3 Why Tuned DDPG Is Strong

Tuned DDPG is strong because the tuning fixed the real problem: action design and normalization.

The original DDPG was not failing because actor-critic methods are useless. It was failing because the policy could output direct replica counts and collapse to boundaries.

After switching to delta actions and normalized training, DDPG learned useful autoscaling behavior.

### 13.4 Why PPO Is Good but Usually Not Best

PPO is stable and avoids catastrophic collapse. Its clipped policy update makes training safer.

However, PPO is on-policy and can be less sample-efficient than SAC and TD3. In this project, PPO often protects SLA by keeping more replicas running. This gives low SLA violation but high cost, so total reward is slightly worse.

PPO's result is still valuable: it is a stable method and a good comparison point, even when it does not win.

### 13.5 Why HPA Does Not Always Win

HPA is reactive and rule-based. It scales from current CPU, not from learned workload patterns or reward optimization.

HPA can perform well when CPU is a reliable signal and the workload changes smoothly. It struggles when the best policy requires learned trade-offs across SLA, cost, and churn.

### 13.6 Why DQN Is Mixed

DQN is useful and sometimes competitive, especially with discrete scaling actions. It beat HPA on Alibaba machine and Alibaba microservice data.

But DQN is more sensitive to reward weights and trace irregularity. On Azure and GWA-T, it suffered high SLA violation because it was not responsive enough compared with the continuous delta agents.

### 13.7 Why Original DDPG Fails

Original DDPG repeatedly shows boundary behavior.

Examples:

- Synthetic: cost 400.0, churn 0.0, SLA violation 111,228.7
- Alibaba machine: cost 400.0, churn 0.0, SLA violation 115,233.8
- GWA-T: cost 400.0, churn 0.0, SLA violation 100,557.2

This means the agent stayed at the minimum replica count. It was cheap and stable but did not protect the SLA.

On Azure and Alibaba microservices, DDPG instead moved toward maximum capacity:

- SLA violation 0.0
- Cost 2,000.0
- Very low churn

That protects latency but wastes resources. Both behaviors are boundary policies, not good autoscaling.

## 14. Main Research Findings

### Finding 1: Reward Design Controls Agent Personality

The reward function determines whether the agent becomes aggressive, conservative, cheap, or stable.

The DQN gamma_stab experiment proves this. Lowering the churn penalty made DQN more aggressive. SLA improved, but churn and cost increased.

### Finding 2: Action Space Matters as Much as Algorithm Choice

Original DDPG and Tuned DDPG use the same algorithm family, but their results are completely different.

The main difference is action space:

- Original DDPG: choose absolute replica count.
- Tuned DDPG: choose bounded replica delta.

The delta design is much better for autoscaling.

### Finding 3: Continuous Delta Actor-Critic Methods Are Best Overall

SAC, TD3, and Tuned DDPG are consistently strong because they combine:

- Continuous control
- Bounded scaling changes
- Normalized observations and rewards
- Stable training mechanisms

### Finding 4: HPA Is a Strong Baseline, Not a Weak One

HPA is simple but meaningful. It performs well on several datasets, especially synthetic and Azure.

For the project to be convincing, RL must beat HPA, not just Random. The final experiments do that on several real datasets.

### Finding 5: Generalization Needs Held-Out Traces

The project evaluates on held-out days, machines, services, or VM traces. This is important because it avoids testing only on data the model saw during training.

This makes the benchmark more credible.

## 15. Code Explanation

### 15.1 `envs/autoscaling_env.py`

This file defines the main `AutoscalingEnv` class.

Important parameters:

| Parameter | Meaning |
|---|---|
| `history_len` | Number of past load values in the observation |
| `n_min` | Minimum allowed replicas |
| `n_max` | Maximum allowed replicas |
| `episode_length` | Number of timesteps per episode |
| `continuous` | Whether the action space is continuous |
| `gamma_stab` | Weight of the scaling stability penalty |
| `load_traces` | Optional real workload traces |
| `continuous_action_mode` | Either `absolute` or `delta` |

Important methods:

| Method | Role |
|---|---|
| `reset()` | Starts a new episode and selects a workload trace |
| `_generate_load()` | Generates synthetic load or reads from a real trace |
| `_compute_latency_cpu()` | Converts load and replicas into latency and CPU |
| `_get_obs()` | Builds the observation vector |
| `step()` | Applies action, updates system state, calculates reward |

### 15.2 `envs/__init__.py`

This file registers the Gymnasium environments:

- `Autoscaling-v0`
- `AutoscalingContinuous-v0`
- `AutoscalingContinuousDelta-v0`

This allows training scripts to call:

```python
gym.make("Autoscaling-v0")
```

### 15.3 `envs/data_loader.py`

This file converts raw datasets into traces that the environment can use.

Important functions:

| Function | Role |
|---|---|
| `load_azure_day()` | Loads and bins one Azure day |
| `rescale_to_range()` | Min-max rescales a trace to the environment range |
| `robust_rescale_to_range()` | Percentile-based rescaling for noisy real traces |
| `build_alibaba_trace_cache()` | Builds machine-level Alibaba traces |
| `build_alibaba_microservice_trace_cache()` | Builds service-level Alibaba traces |
| `build_gwat_trace_cache()` | Builds GWA-T VM traces |

The cache files are important because they avoid reprocessing large raw datasets every time a model is trained.

### 15.4 Training Scripts

The `agents/` folder contains one training script per algorithm or dataset family.

Examples:

| File | Purpose |
|---|---|
| `agents/train_dqn.py` | Train DQN on synthetic data |
| `agents/train_dqn_realdata.py` | Train DQN on Azure real data |
| `agents/train_dqn_realdata_lowstab.py` | Train DQN on Azure with lower gamma_stab |
| `agents/train_ddpg.py` | Train original DDPG |
| `agents/train_ddpg_tuned.py` | Train tuned DDPG with delta actions |
| `agents/train_sac.py` | Train SAC |
| `agents/train_td3.py` | Train TD3 |
| `agents/train_ppo_discrete.py` | Train PPO with discrete actions |
| `agents/train_ppo_delta.py` | Train PPO with continuous delta actions |
| `agents/train_alibaba.py` | Train all algorithms on Alibaba machine traces |
| `agents/train_alibaba_microservices.py` | Train all algorithms on Alibaba microservice traces |
| `agents/train_gwat.py` | Train all algorithms on GWA-T traces |

The dataset-wide trainers support:

```text
--algo dqn
--algo ddpg
--algo tuned-ddpg
--algo sac
--algo td3
--algo ppo-discrete
--algo ppo-delta
--algo all
--prepare-only
```

### 15.5 Evaluation Scripts

The `evaluation/` folder compares trained models.

The core evaluation logic is in `evaluation/evaluate.py`:

- `run_episode()` runs one episode and accumulates metrics.
- `evaluate_policy()` averages results across 20 episodes.
- `_normalize_action()` ensures actions are valid for each action space.
- `make_vecnormalized_policy()` loads saved `VecNormalize` statistics for normalized models.

Dataset-specific evaluators load the correct held-out traces and models.

## 16. Limitations

The project is strong for an internship research prototype, but it still has limitations.

### 16.1 Simplified Latency Model

The latency model is mathematical, not from a real Kubernetes cluster. It captures the intuition that latency increases near saturation, but it does not model real networking, scheduling delays, cold starts, or pod startup time.

### 16.2 Manual Reward Weights

The reward weights were chosen manually. Different applications may prefer different trade-offs. For example, a banking app may prioritize SLA more, while a batch-processing app may prioritize cost.

### 16.3 Dataset Rescaling

All datasets are rescaled into the same load range. This makes comparison possible, but it also abstracts away some real units.

### 16.4 Limited Random Seeds

The results are promising, but stronger research evidence would require repeating each experiment with multiple random seeds and reporting mean plus standard deviation.

### 16.5 No Live Kubernetes Deployment Yet

The current project is a simulator. A next stage would connect the learned policy to a Kubernetes test cluster and compare it against the real HPA controller.

## 17. Recommended Future Work

The strongest future improvements are:

- Repeat SAC, TD3, Tuned DDPG, and PPO across multiple random seeds.
- Add confidence intervals to the benchmark tables.
- Test intermediate DQN `gamma_stab` values such as 0.15 and 0.2.
- Add pod startup delay to make the simulator more realistic.
- Add scale-down stabilization windows like Kubernetes HPA.
- Add multi-resource observations, especially memory and request rate.
- Compare against KEDA-style request-based autoscaling.
- Test transfer learning: train on Alibaba, evaluate on GWA-T, or train on multiple datasets together.
- Deploy the best policy in a local Kubernetes cluster using a simple autoscaling controller.

## 18. Final Conclusion

This project demonstrates that reinforcement learning can be useful for Kubernetes pod autoscaling, but the details matter. A naive continuous agent such as original DDPG can fail badly. A well-designed RL setup with bounded delta actions, normalization, and stable actor-critic learning can outperform HPA on several real workload datasets.

The best overall method is SAC because it gives the strongest balance between SLA protection, resource cost, and scaling stability. TD3 and Tuned DDPG are also very strong. PPO is stable and useful, but often slightly more expensive. DQN is a good discrete baseline, but it is more sensitive to reward tuning and workload burstiness.

The main lesson is that autoscaling is not only a prediction problem. It is a control problem with competing objectives. The best policies are the ones that learn when to spend more replicas, when to save cost, and when to avoid unnecessary scaling movement.
