import gymnasium as gym
from gymnasium import spaces
import numpy as np


class AutoscalingEnv(gym.Env):
    def __init__(self, history_len=5, n_min=2, n_max=10, episode_length=200,
                 continuous=False, alpha=1.0, beta=0.5, gamma_stab=0.1,
                 load_traces=None, continuous_action_mode="absolute", max_delta=2):
        super().__init__()
        self.history_len = history_len
        self.n_min, self.n_max = n_min, n_max
        self.episode_length = episode_length
        self.continuous = continuous
        self.continuous_action_mode = continuous_action_mode
        self.max_delta = max_delta
        self.alpha, self.beta, self.gamma_stab = alpha, beta, gamma_stab
        self.load_traces = load_traces      # None -> synthetic sine wave; list of arrays -> real days
        self.current_trace = None

        if self.continuous:
            if continuous_action_mode == "absolute":
                self.action_space = spaces.Box(
                    low=float(n_min), high=float(n_max), shape=(1,), dtype=np.float32
                )
            elif continuous_action_mode == "delta":
                self.action_space = spaces.Box(
                    low=-float(max_delta), high=float(max_delta), shape=(1,), dtype=np.float32
                )
            else:
                raise ValueError(f"Unknown continuous_action_mode: {continuous_action_mode}")
        else:
            # DQN's action space: 5 discrete choices
            self.action_space = spaces.Discrete(5)
            self.action_map = {0: -2, 1: -1, 2: 0, 3: 1, 4: 2}

        obs_dim = history_len + 1 + 1 + 1
        self.observation_space = spaces.Box(
            low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.action_space.seed(seed)

        self.t = 0
        self.replicas = self.n_min

        if self.load_traces is not None:
            idx = self.np_random.integers(0, len(self.load_traces))
            self.current_trace = self.load_traces[idx]     # pick one real day for this episode
        else:
            self.current_trace = None

        self.load_history = [self._generate_load(0) for _ in range(self.history_len)]
        self.latency, self.cpu = self._compute_latency_cpu(self.load_history[-1], self.replicas)
        return self._get_obs(), {}

    def _generate_load(self, t):
        if self.current_trace is not None:
            idx = min(t, len(self.current_trace) - 1)
            base = self.current_trace[idx]
        else:
            base = 500 + 400 * np.sin(2 * np.pi * t / self.episode_length)
        noise = self.np_random.normal(0, 20)
        return max(0, base + noise)
    
    def _compute_latency_cpu(self, load, replicas):
       
        capacity_per_pod = 150   # was 100
        total_capacity = replicas * capacity_per_pod
        utilization = load / total_capacity
        cpu = min(100, utilization * 100)
        # latency grows sharply as utilization approaches 100% (simple queueing intuition)
        latency = 50 / max(0.01, (1 - min(utilization, 0.99)))
        latency = min(latency, 1000)
        return latency, cpu
    
    

    def _get_obs(self):
        return np.array(
            self.load_history + [self.latency, self.cpu, self.replicas],
            dtype=np.float32
        )
    
    def step(self, action):
        prev_replicas = self.replicas

        if self.continuous:
            action_value = float(np.asarray(action).reshape(-1)[0])
            if self.continuous_action_mode == "delta":
                delta = int(np.round(action_value))
                self.replicas = int(np.clip(self.replicas + delta, self.n_min, self.n_max))
            else:
                desired = int(np.round(action_value))
                self.replicas = int(np.clip(desired, self.n_min, self.n_max))
        else:
            delta = self.action_map[action]
            self.replicas = int(np.clip(self.replicas + delta, self.n_min, self.n_max))

        # everything below this line is IDENTICAL for both DQN and DDPG —
        # load generation, latency/cpu calculation, and reward formula are unchanged
        self.t += 1
        new_load = self._generate_load(self.t)
        self.load_history.append(new_load)
        self.load_history.pop(0)

        self.latency, self.cpu = self._compute_latency_cpu(new_load, self.replicas)

        sla_target = 200
        sla_violation = max(0, self.latency - sla_target)
        cost = self.replicas
        stability_penalty = abs(self.replicas - prev_replicas)

        
        reward = -(self.alpha * sla_violation + self.beta * cost + self.gamma_stab * stability_penalty)
        reward = float(np.clip(reward, -50, 50))

        terminated = False
        truncated = self.t >= self.episode_length

        obs = self._get_obs()
        info = {"latency": self.latency, "cpu": self.cpu, "replicas": self.replicas}

        return obs, reward, terminated, truncated, info
    
    
    
    
if __name__ == "__main__":
    env = AutoscalingEnv()
    obs, info = env.reset()
    env.reset(seed=42)
    total_reward = 0
    
       
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 20 == 0:
            print(f"t={step:3d}  replicas={info['replicas']:2d}  latency={info['latency']:7.1f}ms  reward={reward:8.1f}")
        if terminated or truncated:
            break

    print(f"Episode finished. Total reward: {total_reward:.1f}")
    print(f"Final state — replicas: {info['replicas']}, latency: {info['latency']:.1f}ms")


            

