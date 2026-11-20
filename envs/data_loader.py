from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ALIBABA_MACHINE_USAGE_COLUMNS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "mem_gps",
    "mpki",
    "net_in",
    "net_out",
    "disk_io_percent",
]

ALIBABA_MACHINE_USAGE_USECOLS = [
    "machine_id",
    "time_stamp",
    "cpu_util_percent",
    "mem_util_percent",
    "net_in",
    "net_out",
    "disk_io_percent",
]

ALIBABA_DEFAULT_CACHE = "data/alibaba/alibaba_machine_traces.npz"
ALIBABA_MICROSERVICE_QPS_PATH = "data/alibaba_microservices/MSRTQps/MSRTQps_0.csv"
ALIBABA_MICROSERVICE_RESOURCE_PATH = "data/alibaba_microservices/MSResource/MSResource_0.csv"
ALIBABA_MICROSERVICE_DEFAULT_CACHE = "data/alibaba_microservices/alibaba_microservice_traces.npz"
ALIBABA_MICROSERVICE_QPS_METRIC_SUFFIX = "_MCR"
ALIBABA_MICROSERVICE_RT_METRIC_SUFFIX = "_RT"

GWAT_DEFAULT_ROOT = "data/gwat"
GWAT_DEFAULT_CACHE = "data/gwat/gwat_vm_traces.npz"
GWAT_COLUMNS = {
    "Timestamp [ms]": "timestamp",
    "CPU cores": "cpu_cores",
    "CPU capacity provisioned [MHZ]": "cpu_capacity_mhz",
    "CPU usage [MHZ]": "cpu_usage_mhz",
    "CPU usage [%]": "cpu_usage_percent",
    "Memory capacity provisioned [KB]": "memory_capacity_kb",
    "Memory usage [KB]": "memory_usage_kb",
    "Disk read throughput [KB/s]": "disk_read_kbps",
    "Disk write throughput [KB/s]": "disk_write_kbps",
    "Network received throughput [KB/s]": "network_received_kbps",
    "Network transmitted throughput [KB/s]": "network_transmitted_kbps",
}


def load_azure_day(csv_path, n_bins=200):
    """
    Load one day of Azure Functions invocation data and compress the
    1440 per-minute readings down to n_bins values (matching episode_length).
    """
    df = pd.read_csv(csv_path)
    minute_cols = [c for c in df.columns if c not in ("HashOwner", "HashApp", "HashFunction", "Trigger")]

    # sum invocations across every function, per minute -> one aggregate load signal
    total_per_minute = df[minute_cols].sum(axis=0).values.astype(float)  # length 1440

    # bin-average 1440 minutes down to n_bins steps, so it lines up with episode_length
    bin_size = len(total_per_minute) // n_bins
    trimmed = total_per_minute[: bin_size * n_bins]
    binned = trimmed.reshape(n_bins, bin_size).mean(axis=1)
    return binned


def rescale_to_range(load_array, low=100, high=900):
    """
    Min-max rescale a real trace into the operating range the environment's
    capacity model (capacity_per_pod=150, n_max=10) was designed around.
    """
    lo, hi = load_array.min(), load_array.max()
    if hi - lo < 1e-6:
        return np.full_like(load_array, (low + high) / 2)
    normalized = (load_array - lo) / (hi - lo)
    return low + normalized * (high - low)


def robust_rescale_to_range(load_array, low=100, high=900, p_low=5, p_high=95):
    """
    Rescale a trace with percentile clipping so occasional Alibaba outliers do
    not flatten the useful shape of the workload.
    """
    values = np.asarray(load_array, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.full_like(values, (low + high) / 2, dtype=np.float32)

    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi - lo < 1e-6:
        return np.full_like(values, (low + high) / 2, dtype=np.float32)

    clipped = np.clip(values, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return (low + normalized * (high - low)).astype(np.float32)


def _resample_to_bins(values, n_bins=200):
    values = np.asarray(values, dtype=np.float32)
    if values.size == n_bins:
        return values

    if values.size < n_bins:
        old_x = np.linspace(0, 1, values.size)
        new_x = np.linspace(0, 1, n_bins)
        return np.interp(new_x, old_x, values).astype(np.float32)

    edges = np.linspace(0, values.size, n_bins + 1).astype(int)
    binned = np.empty(n_bins, dtype=np.float32)
    for i in range(n_bins):
        start, end = edges[i], edges[i + 1]
        if start == end:
            binned[i] = values[min(start, values.size - 1)]
        else:
            binned[i] = values[start:end].mean()
    return binned


def _resample_timed_series(times, values, n_bins=200):
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    if values.size == 1:
        return np.full(n_bins, values[0], dtype=np.float32)

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    if times[-1] - times[0] < 1e-6:
        return _resample_to_bins(values, n_bins=n_bins)

    edges = np.linspace(times[0], times[-1], n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    indices = np.searchsorted(times, edges)
    binned = np.empty(n_bins, dtype=np.float32)

    for i in range(n_bins):
        start, end = indices[i], indices[i + 1]
        if end > start:
            binned[i] = values[start:end].mean()
        else:
            binned[i] = np.interp(centers[i], times, values)

    return binned


def _add_metric_values(store, grouped_values, metric_name):
    for (service_name, timestamp), value in grouped_values.items():
        service_name = str(service_name)
        timestamp = float(timestamp)
        metrics = store.setdefault(service_name, {})
        values = metrics.setdefault(metric_name, {})
        values[timestamp] = values.get(timestamp, 0.0) + float(value)


def _add_rt_values(store, grouped_values):
    for (service_name, timestamp), row in grouped_values.iterrows():
        service_name = str(service_name)
        timestamp = float(timestamp)
        metrics = store.setdefault(service_name, {})
        rt_sum = metrics.setdefault("rt_sum", {})
        rt_count = metrics.setdefault("rt_count", {})
        rt_sum[timestamp] = rt_sum.get(timestamp, 0.0) + float(row["sum"])
        rt_count[timestamp] = rt_count.get(timestamp, 0.0) + float(row["count"])


def _dict_to_sorted_arrays(values_by_time):
    if not values_by_time:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float32)

    times = np.asarray(sorted(values_by_time), dtype=np.float64)
    values = np.asarray([values_by_time[t] for t in times], dtype=np.float32)
    return times, values


def _mean_from_sum_count(sum_by_time, count_by_time):
    values = {}
    for timestamp, total in sum_by_time.items():
        count = count_by_time.get(timestamp, 0.0)
        if count > 0:
            values[timestamp] = total / count
    return values


def _complete_service_count(metric_store, service_names, metric_name, min_points):
    return sum(
        len(metric_store.get(service_name, {}).get(metric_name, {})) >= min_points
        for service_name in service_names
    )


def _alibaba_microservice_qps_mask(metric_series):
    return metric_series.astype(str).str.endswith(ALIBABA_MICROSERVICE_QPS_METRIC_SUFFIX)


def _alibaba_microservice_rt_mask(metric_series):
    return metric_series.astype(str).str.endswith(ALIBABA_MICROSERVICE_RT_METRIC_SUFFIX)


def _robust_unit_scale(values, p_low=5, p_high=95):
    values = np.asarray(values, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)

    lo, hi = np.percentile(finite, [p_low, p_high])
    if hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)

    scaled = (np.clip(values, lo, hi) - lo) / (hi - lo)
    return np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _alibaba_pressure_score(chunk):
    """
    Convert Alibaba machine utilization into one pod-autoscaling demand signal.

    CPU is dominant because HPA is CPU-oriented in Kubernetes. Memory and
    network add multi-resource pressure, while disk gets a small weight because
    it is noisier and may include abnormal values in the trace.
    """
    numeric_cols = [
        "cpu_util_percent",
        "mem_util_percent",
        "net_in",
        "net_out",
        "disk_io_percent",
    ]
    for col in numeric_cols:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")

    cpu = chunk["cpu_util_percent"].clip(0, 100).fillna(0)
    mem = chunk["mem_util_percent"].clip(0, 100).fillna(0)
    net_in = chunk["net_in"].clip(0, 100).fillna(0)
    net_out = chunk["net_out"].clip(0, 100).fillna(0)
    net = np.maximum(net_in, net_out)
    disk = chunk["disk_io_percent"].where(chunk["disk_io_percent"].between(0, 100))
    disk = disk.fillna(0)

    return (
        0.65 * cpu
        + 0.20 * mem
        + 0.10 * net
        + 0.05 * disk
    ).astype(np.float32)


def build_alibaba_trace_cache(
    csv_path="data/alibaba/machine_usage.csv",
    cache_path=ALIBABA_DEFAULT_CACHE,
    n_bins=200,
    max_machines=512,
    min_points=200,
    train_fraction=0.7,
    seed=42,
    chunksize=500_000,
    assume_sorted_by_machine=True,
    force=False,
):
    """
    Build a compact cache of Alibaba per-machine workload traces.

    Each selected machine becomes one 200-step workload trace. The train/test
    split is by machine ID, so evaluation measures generalization to machines
    the agent never saw during training.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return load_alibaba_train_test_traces(cache_path)

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Alibaba machine usage file not found: {csv_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    buffers = {}
    selected_machine_ids = []
    enough_selected = False

    reader = pd.read_csv(
        csv_path,
        header=None,
        names=ALIBABA_MACHINE_USAGE_COLUMNS,
        usecols=ALIBABA_MACHINE_USAGE_USECOLS,
        chunksize=chunksize,
    )

    for chunk in reader:
        chunk = chunk.dropna(subset=["machine_id", "time_stamp"])
        chunk["time_stamp"] = pd.to_numeric(chunk["time_stamp"], errors="coerce")
        chunk = chunk.dropna(subset=["time_stamp"])
        if chunk.empty:
            continue

        chunk["_pressure"] = _alibaba_pressure_score(chunk)

        for machine_id, group in chunk.groupby("machine_id", sort=False):
            machine_id = str(machine_id)
            if machine_id not in buffers:
                if len(selected_machine_ids) < max_machines:
                    selected_machine_ids.append(machine_id)
                    buffers[machine_id] = {"time": [], "pressure": []}
                elif assume_sorted_by_machine:
                    enough_selected = all(
                        sum(len(part) for part in data["pressure"]) >= min_points
                        for data in buffers.values()
                    )
                    if enough_selected:
                        break
                    continue
                else:
                    continue

            buffers[machine_id]["time"].append(group["time_stamp"].to_numpy(np.float64))
            buffers[machine_id]["pressure"].append(group["_pressure"].to_numpy(np.float32))

        if enough_selected:
            break

    traces = []
    machine_ids = []
    for machine_id in selected_machine_ids:
        data = buffers[machine_id]
        if not data["pressure"]:
            continue

        times = np.concatenate(data["time"])
        pressure = np.concatenate(data["pressure"])
        if pressure.size < min_points:
            continue

        order = np.argsort(times)
        pressure = pressure[order]
        trace = _resample_to_bins(pressure, n_bins=n_bins)
        trace = robust_rescale_to_range(trace, low=100, high=900)
        traces.append(trace)
        machine_ids.append(machine_id)

    if len(traces) < 2:
        raise ValueError(
            "Need at least two Alibaba machine traces for a train/test split. "
            "Increase max_machines or lower min_points."
        )

    traces = np.stack(traces).astype(np.float32)
    machine_ids = np.asarray(machine_ids)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(traces))
    n_train = int(round(len(traces) * train_fraction))
    n_train = min(max(1, n_train), len(traces) - 1)
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    np.savez_compressed(
        cache_path,
        train_traces=traces[train_idx],
        test_traces=traces[test_idx],
        train_machine_ids=machine_ids[train_idx],
        test_machine_ids=machine_ids[test_idx],
        all_machine_ids=machine_ids,
        n_bins=np.asarray([n_bins], dtype=np.int32),
    )

    return load_alibaba_train_test_traces(cache_path)


def load_alibaba_train_test_traces(cache_path=ALIBABA_DEFAULT_CACHE):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Alibaba trace cache not found: {cache_path}. "
            "Run build_alibaba_trace_cache() or agents/train_alibaba.py first."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "train_traces": data["train_traces"].astype(np.float32),
            "test_traces": data["test_traces"].astype(np.float32),
            "train_machine_ids": data["train_machine_ids"],
            "test_machine_ids": data["test_machine_ids"],
        }


def get_alibaba_train_test_traces(
    csv_path="data/alibaba/machine_usage.csv",
    cache_path=ALIBABA_DEFAULT_CACHE,
    prepare_if_missing=True,
    **build_kwargs,
):
    force = build_kwargs.get("force", False)
    if Path(cache_path).exists() and not force:
        return load_alibaba_train_test_traces(cache_path)

    if not prepare_if_missing:
        raise FileNotFoundError(
            f"Alibaba trace cache not found: {cache_path}. "
            "Run agents/train_alibaba.py --prepare-only first."
        )

    return build_alibaba_trace_cache(
        csv_path=csv_path,
        cache_path=cache_path,
        **build_kwargs,
    )


def build_alibaba_microservice_trace_cache(
    qps_path=ALIBABA_MICROSERVICE_QPS_PATH,
    resource_path=ALIBABA_MICROSERVICE_RESOURCE_PATH,
    cache_path=ALIBABA_MICROSERVICE_DEFAULT_CACHE,
    n_bins=200,
    max_services=256,
    min_points=20,
    train_fraction=0.7,
    seed=42,
    chunksize=500_000,
    completion_fraction=0.8,
    use_resource=True,
    force=False,
    verbose=False,
):
    """
    Build compact per-microservice demand traces from Alibaba 2021 data.

    The environment still receives the same single load signal and reward
    function. Here that signal is derived from microservice request rate, with
    latency and CPU/memory pressure used as shape modifiers.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return load_alibaba_microservice_train_test_traces(cache_path)

    qps_path = Path(qps_path)
    resource_path = Path(resource_path) if resource_path else None
    if not qps_path.exists():
        raise FileNotFoundError(f"Alibaba microservice QPS/RT file not found: {qps_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    service_counts = Counter()
    metric_store = {}

    qps_count_reader = pd.read_csv(
        qps_path,
        usecols=["msname", "metric"],
        chunksize=chunksize,
    )
    for chunk_number, chunk in enumerate(qps_count_reader, start=1):
        chunk = chunk.dropna(subset=["msname", "metric"])
        chunk = chunk[_alibaba_microservice_qps_mask(chunk["metric"])]
        if chunk.empty:
            continue

        service_counts.update(chunk["msname"].astype(str).tolist())
        if verbose and chunk_number % 5 == 0:
            print(
                "QPS density pass: "
                f"{len(service_counts)} services observed"
            )

    selected_services = [
        service_name for service_name, _ in service_counts.most_common(max_services)
    ]
    selected_set = set(selected_services)
    if len(selected_services) < 2:
        raise ValueError(
            "Need at least two Alibaba microservices with QPS data. "
            "Download a QPS shard or lower max_services."
        )

    qps_reader = pd.read_csv(
        qps_path,
        usecols=["timestamp", "msname", "metric", "value"],
        chunksize=chunksize,
    )
    target_complete = max(2, int(len(selected_services) * completion_fraction))

    for chunk_number, chunk in enumerate(qps_reader, start=1):
        chunk = chunk.dropna(subset=["timestamp", "msname", "metric", "value"])
        qps_mask = _alibaba_microservice_qps_mask(chunk["metric"])
        rt_mask = _alibaba_microservice_rt_mask(chunk["metric"])
        chunk = chunk[qps_mask | rt_mask]
        if chunk.empty:
            continue

        chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
        chunk["value"] = pd.to_numeric(chunk["value"], errors="coerce")
        chunk = chunk.dropna(subset=["timestamp", "value"])
        if chunk.empty:
            continue

        tracked = chunk[chunk["msname"].isin(selected_set)]
        if tracked.empty:
            continue

        qps_rows = tracked[_alibaba_microservice_qps_mask(tracked["metric"])]
        if not qps_rows.empty:
            grouped_qps = qps_rows.groupby(["msname", "timestamp"])["value"].sum()
            _add_metric_values(metric_store, grouped_qps, "qps")

        rt_rows = tracked[_alibaba_microservice_rt_mask(tracked["metric"])]
        if not rt_rows.empty:
            grouped_rt = rt_rows.groupby(["msname", "timestamp"])["value"].agg(["sum", "count"])
            _add_rt_values(metric_store, grouped_rt)

        complete = _complete_service_count(metric_store, selected_services, "qps", min_points)
        if verbose and chunk_number % 5 == 0:
            print(
                "QPS pass: "
                f"{len(selected_services)} selected services, {complete} with >= {min_points} points"
            )

        if complete >= target_complete:
            break

    eligible_services = [
        service_name for service_name in selected_services
        if len(metric_store.get(service_name, {}).get("qps", {})) >= min_points
    ]
    if len(eligible_services) < 2:
        raise ValueError(
            "Need at least two Alibaba microservice traces for a train/test split. "
            "Increase max_services, lower min_points, or download more shards."
        )

    resource_store = {}
    if use_resource and resource_path and resource_path.exists():
        resource_reader = pd.read_csv(
            resource_path,
            usecols=["msname", "timestamp", "instance_cpu_usage", "instance_memory_usage"],
            chunksize=chunksize,
        )
        eligible_set = set(eligible_services)
        resource_target = max(2, int(len(eligible_services) * completion_fraction))

        for chunk_number, chunk in enumerate(resource_reader, start=1):
            chunk = chunk.dropna(subset=["timestamp", "msname"])
            chunk = chunk[chunk["msname"].isin(eligible_set)]
            if chunk.empty:
                continue

            chunk["timestamp"] = pd.to_numeric(chunk["timestamp"], errors="coerce")
            chunk["instance_cpu_usage"] = pd.to_numeric(
                chunk["instance_cpu_usage"], errors="coerce"
            )
            chunk["instance_memory_usage"] = pd.to_numeric(
                chunk["instance_memory_usage"], errors="coerce"
            )
            chunk = chunk.dropna(subset=["timestamp", "instance_cpu_usage", "instance_memory_usage"])
            if chunk.empty:
                continue

            grouped = chunk.groupby(["msname", "timestamp"])[
                ["instance_cpu_usage", "instance_memory_usage"]
            ].mean()

            for (service_name, timestamp), row in grouped.iterrows():
                service_name = str(service_name)
                timestamp = float(timestamp)
                metrics = resource_store.setdefault(service_name, {"cpu": {}, "memory": {}})
                metrics["cpu"][timestamp] = float(row["instance_cpu_usage"])
                metrics["memory"][timestamp] = float(row["instance_memory_usage"])

            complete = _complete_service_count(resource_store, eligible_services, "cpu", min_points)
            if verbose and chunk_number % 5 == 0:
                print(
                    "Resource pass: "
                    f"{complete} services with >= {min_points} resource points"
                )

            if complete >= resource_target:
                break
    elif verbose:
        print("Resource file not found; using QPS/RT only.")

    traces = []
    service_ids = []
    for service_name in eligible_services:
        metrics = metric_store[service_name]
        qps_times, qps_values = _dict_to_sorted_arrays(metrics["qps"])
        if qps_values.size < min_points:
            continue

        qps_bins = _resample_timed_series(qps_times, qps_values, n_bins=n_bins)
        qps_scaled = robust_rescale_to_range(qps_bins, low=100, high=850)

        rt_values_by_time = _mean_from_sum_count(
            metrics.get("rt_sum", {}),
            metrics.get("rt_count", {}),
        )
        rt_times, rt_values = _dict_to_sorted_arrays(rt_values_by_time)
        if rt_values.size > 1:
            rt_bins = _resample_timed_series(rt_times, rt_values, n_bins=n_bins)
            rt_pressure = robust_rescale_to_range(rt_bins, low=0, high=1)
        else:
            rt_pressure = np.zeros(n_bins, dtype=np.float32)

        resource_pressure = np.zeros(n_bins, dtype=np.float32)
        if service_name in resource_store:
            cpu_times, cpu_values = _dict_to_sorted_arrays(resource_store[service_name]["cpu"])
            mem_times, mem_values = _dict_to_sorted_arrays(resource_store[service_name]["memory"])
            if cpu_values.size > 1 and mem_values.size > 1:
                cpu_bins = np.clip(_resample_timed_series(cpu_times, cpu_values, n_bins=n_bins), 0, 1.5)
                mem_bins = np.clip(_resample_timed_series(mem_times, mem_values, n_bins=n_bins), 0, 1.5)
                resource_pressure = np.clip(0.7 * cpu_bins + 0.3 * mem_bins, 0, 1)

        combined = qps_scaled * (1.0 + 0.20 * rt_pressure + 0.20 * resource_pressure)
        trace = robust_rescale_to_range(combined, low=100, high=900)
        traces.append(trace)
        service_ids.append(service_name)

    if len(traces) < 2:
        raise ValueError(
            "Need at least two usable Alibaba microservice traces after preprocessing. "
            "Increase max_services, lower min_points, or download more shards."
        )

    traces = np.stack(traces).astype(np.float32)
    service_ids = np.asarray(service_ids)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(traces))
    n_train = int(round(len(traces) * train_fraction))
    n_train = min(max(1, n_train), len(traces) - 1)
    train_idx = order[:n_train]
    test_idx = order[n_train:]

    np.savez_compressed(
        cache_path,
        train_traces=traces[train_idx],
        test_traces=traces[test_idx],
        train_service_ids=service_ids[train_idx],
        test_service_ids=service_ids[test_idx],
        all_service_ids=service_ids,
        n_bins=np.asarray([n_bins], dtype=np.int32),
    )

    return load_alibaba_microservice_train_test_traces(cache_path)


def load_alibaba_microservice_train_test_traces(
    cache_path=ALIBABA_MICROSERVICE_DEFAULT_CACHE,
):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Alibaba microservice trace cache not found: {cache_path}. "
            "Run agents/train_alibaba_microservices.py --prepare-only first."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "train_traces": data["train_traces"].astype(np.float32),
            "test_traces": data["test_traces"].astype(np.float32),
            "train_service_ids": data["train_service_ids"],
            "test_service_ids": data["test_service_ids"],
        }


def get_alibaba_microservice_train_test_traces(
    qps_path=ALIBABA_MICROSERVICE_QPS_PATH,
    resource_path=ALIBABA_MICROSERVICE_RESOURCE_PATH,
    cache_path=ALIBABA_MICROSERVICE_DEFAULT_CACHE,
    prepare_if_missing=True,
    **build_kwargs,
):
    force = build_kwargs.get("force", False)
    if Path(cache_path).exists() and not force:
        return load_alibaba_microservice_train_test_traces(cache_path)

    if not prepare_if_missing:
        raise FileNotFoundError(
            f"Alibaba microservice trace cache not found: {cache_path}. "
            "Run agents/train_alibaba_microservices.py --prepare-only first."
        )

    return build_alibaba_microservice_trace_cache(
        qps_path=qps_path,
        resource_path=resource_path,
        cache_path=cache_path,
        **build_kwargs,
    )


def _read_gwat_trace(csv_path, n_bins=200, min_points=200):
    df = pd.read_csv(csv_path, sep=r";\s*", engine="python")
    df.columns = [str(col).strip() for col in df.columns]
    df = df.rename(columns=GWAT_COLUMNS)

    required_cols = [
        "timestamp",
        "cpu_usage_percent",
        "cpu_usage_mhz",
        "cpu_capacity_mhz",
        "memory_capacity_kb",
        "memory_usage_kb",
        "disk_read_kbps",
        "disk_write_kbps",
        "network_received_kbps",
        "network_transmitted_kbps",
    ]
    for col in required_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "timestamp" not in df.columns:
        return None

    df = df.dropna(subset=["timestamp"])
    if len(df) < min_points:
        return None

    if "cpu_usage_percent" in df.columns:
        cpu = df["cpu_usage_percent"]
    else:
        cpu = pd.Series(np.nan, index=df.index, dtype=np.float32)

    if cpu.isna().all() and {"cpu_usage_mhz", "cpu_capacity_mhz"}.issubset(df.columns):
        capacity = df["cpu_capacity_mhz"].replace(0, np.nan)
        cpu = 100.0 * df["cpu_usage_mhz"] / capacity
    cpu = cpu.clip(0, 100).fillna(0).to_numpy(np.float32)

    if {"memory_usage_kb", "memory_capacity_kb"}.issubset(df.columns):
        mem_capacity = df["memory_capacity_kb"].replace(0, np.nan)
        mem = 100.0 * df["memory_usage_kb"] / mem_capacity
    else:
        mem = pd.Series(0.0, index=df.index, dtype=np.float32)
    mem = mem.clip(0, 100).fillna(0).to_numpy(np.float32)

    disk_read = df.get("disk_read_kbps", pd.Series(0.0, index=df.index)).fillna(0)
    disk_write = df.get("disk_write_kbps", pd.Series(0.0, index=df.index)).fillna(0)
    net_recv = df.get("network_received_kbps", pd.Series(0.0, index=df.index)).fillna(0)
    net_tx = df.get("network_transmitted_kbps", pd.Series(0.0, index=df.index)).fillna(0)

    disk = 100.0 * _robust_unit_scale(disk_read.to_numpy(np.float32) + disk_write.to_numpy(np.float32))
    network = 100.0 * _robust_unit_scale(net_recv.to_numpy(np.float32) + net_tx.to_numpy(np.float32))

    pressure = (
        0.60 * cpu
        + 0.25 * mem
        + 0.10 * network
        + 0.05 * disk
    ).astype(np.float32)
    pressure = np.nan_to_num(pressure, nan=0.0, posinf=100.0, neginf=0.0)

    binned = _resample_timed_series(
        df["timestamp"].to_numpy(np.float64),
        pressure,
        n_bins=n_bins,
    )
    return robust_rescale_to_range(binned, low=100, high=900)


def build_gwat_trace_cache(
    root_path=GWAT_DEFAULT_ROOT,
    cache_path=GWAT_DEFAULT_CACHE,
    n_bins=200,
    max_traces=512,
    min_points=200,
    train_fraction=0.7,
    seed=42,
    force=False,
    verbose=False,
):
    """
    Build compact per-VM workload traces from GWA-T/Bitbrains data.

    Each VM-month CSV becomes one 200-step autoscaling workload. The single
    demand signal is CPU-heavy, with memory, network, and disk utilization
    included as secondary resource pressure.
    """
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        return load_gwat_train_test_traces(cache_path)

    root_path = Path(root_path)
    if not root_path.exists():
        raise FileNotFoundError(f"GWA-T root folder not found: {root_path}")

    csv_paths = sorted(root_path.rglob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No GWA-T CSV files found under: {root_path}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(csv_paths))

    traces = []
    trace_ids = []
    for index in order:
        csv_path = csv_paths[index]
        try:
            trace = _read_gwat_trace(csv_path, n_bins=n_bins, min_points=min_points)
        except Exception as exc:
            if verbose:
                print(f"Skipping {csv_path}: {exc}")
            continue

        if trace is None:
            continue

        traces.append(trace)
        trace_ids.append(str(csv_path.relative_to(root_path)).replace("\\", "/"))

        if verbose and len(traces) % 50 == 0:
            print(f"GWA-T preprocessing: {len(traces)} usable traces")

        if len(traces) >= max_traces:
            break

    if len(traces) < 2:
        raise ValueError(
            "Need at least two usable GWA-T traces for a train/test split. "
            "Increase max_traces or lower min_points."
        )

    traces = np.stack(traces).astype(np.float32)
    trace_ids = np.asarray(trace_ids)

    split_order = rng.permutation(len(traces))
    n_train = int(round(len(traces) * train_fraction))
    n_train = min(max(1, n_train), len(traces) - 1)
    train_idx = split_order[:n_train]
    test_idx = split_order[n_train:]

    np.savez_compressed(
        cache_path,
        train_traces=traces[train_idx],
        test_traces=traces[test_idx],
        train_trace_ids=trace_ids[train_idx],
        test_trace_ids=trace_ids[test_idx],
        all_trace_ids=trace_ids,
        n_bins=np.asarray([n_bins], dtype=np.int32),
    )

    return load_gwat_train_test_traces(cache_path)


def load_gwat_train_test_traces(cache_path=GWAT_DEFAULT_CACHE):
    cache_path = Path(cache_path)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"GWA-T trace cache not found: {cache_path}. "
            "Run agents/train_gwat.py --prepare-only first."
        )

    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "train_traces": data["train_traces"].astype(np.float32),
            "test_traces": data["test_traces"].astype(np.float32),
            "train_trace_ids": data["train_trace_ids"],
            "test_trace_ids": data["test_trace_ids"],
        }


def get_gwat_train_test_traces(
    root_path=GWAT_DEFAULT_ROOT,
    cache_path=GWAT_DEFAULT_CACHE,
    prepare_if_missing=True,
    **build_kwargs,
):
    force = build_kwargs.get("force", False)
    if Path(cache_path).exists() and not force:
        return load_gwat_train_test_traces(cache_path)

    if not prepare_if_missing:
        raise FileNotFoundError(
            f"GWA-T trace cache not found: {cache_path}. "
            "Run agents/train_gwat.py --prepare-only first."
        )

    return build_gwat_trace_cache(
        root_path=root_path,
        cache_path=cache_path,
        **build_kwargs,
    )
