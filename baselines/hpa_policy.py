import numpy as np


class HPAPolicy:
    def __init__(self, target_utilization=0.5, n_min=2, n_max=10):
        self.target_utilization = target_utilization
        self.n_min = n_min
        self.n_max = n_max

    def act(self, current_replicas, current_cpu_pct):
        current_utilization = current_cpu_pct / 100.0
        desired = np.ceil(
            current_replicas * (current_utilization / self.target_utilization)
        )
        desired = int(np.clip(desired, self.n_min, self.n_max))
        return desired