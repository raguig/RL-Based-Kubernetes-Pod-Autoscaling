from envs.data_loader import load_azure_day, rescale_to_range
import matplotlib.pyplot as plt

binned = load_azure_day("data/workload_traces/invocations_per_function_md.anon.d01.csv")
rescaled = rescale_to_range(binned)

print("Binned (raw) shape:", binned.shape)
print("Binned (raw) min/max:", binned.min(), binned.max())
print("Rescaled shape:", rescaled.shape)
print("Rescaled min/max:", rescaled.min(), rescaled.max())

plt.figure(figsize=(10, 4))
plt.plot(rescaled)
plt.title("Day 1 - rescaled load")
plt.xlabel("Timestep (binned to 200)")
plt.ylabel("Load (rescaled to 100-900 range)")
plt.savefig("day01_check.png")
print("Saved plot to day01_check.png")