import json
import sys
from collections import defaultdict
from typing import DefaultDict, List, TypedDict

import matplotlib.pyplot as plt
import numpy as np


class KernelStat(TypedDict):
    durations: List[float]
    count: int


def classify_event(_event):
    """
    Classify an Nsight Systems event based on its structure.
    Returns one of: 'kernel', 'memcpy', 'memset', 'cuda_api', 'other'
    """
    if "CudaEvent" in _event:
        cuda_event = _event["CudaEvent"]
        if "memcpy" in cuda_event:
            return "memcpy"  # EventClass 1
        if "kernel" in cuda_event:
            return "kernel"  # EventClass 3
        if "sync" in cuda_event:
            return "sync"  # EventClass 5
        if "cudaEventRecord" in cuda_event:
            return "cudaER"  # EventClass 6

    if "TraceProcessEvent" in _event or "CudaApiEvent" in _event:
        return "cuda_api"  # host-side CUDA API call

    return "other"


if len(sys.argv) < 2:
    print("Usage: python get-stats.py <trace.json>")
    sys.exit(1)
FILE_NAME = sys.argv[1]

KERNEL_TIME = []
KERNEL_LAUNCHES = 0
MEMCPY_TIME = []
MEMCPY_LAUNCHES = 0
SYNC_TIME = []
SYNC_LAUNCHES = 0
CUDA_API_TIME = float(0)
APP_START = float("inf")
APP_END = float("-inf")


kernel_stats: DefaultDict[str, KernelStat] = defaultdict(
    lambda: {"durations": [], "count": 0}
)

with open(FILE_NAME, "r", encoding="utf-8") as f:
    for line in f:
        try:
            event_obj = json.loads(line)
        except json.JSONDecodeError:
            continue

        event = (
            event_obj.get("CudaEvent")
            or event_obj.get("TraceProcessEvent")
            or event_obj.get("CudaApiEvent")
        )
        if not event:
            continue

        start = int(event.get("startNs", 0))
        end = int(event.get("endNs", 0))
        duration = (end - start) / 1e6  # convert duration in ns to ms

        APP_START = min(APP_START, start)
        APP_END = max(APP_END, end)

        EVENT_TYPE = classify_event(event_obj)
        if EVENT_TYPE == "kernel":
            KERNEL_LAUNCHES += 1
            KERNEL_TIME.append(duration)
            name = event["kernel"].get("demangledName", "unknown")
            kernel_stats[name]["durations"].append(duration)
            kernel_stats[name]["count"] += 1
        elif EVENT_TYPE == "memcpy":
            MEMCPY_LAUNCHES += 1
            MEMCPY_TIME.append(duration)
        elif EVENT_TYPE == "sync":
            SYNC_LAUNCHES += 1
            SYNC_TIME.append(duration)
        elif EVENT_TYPE == "cuda_api":
            CUDA_API_TIME += duration

top_total = sorted(
    kernel_stats.items(),
    key=lambda item: sum(item[1]["durations"]),  # sort by total duration
    reverse=True,
)[:10]
top_avg = sorted(
    kernel_stats.items(),
    key=lambda item: sum(item[1]["durations"])
    / item[1]["count"],  # sort by total duration
    reverse=True,
)[:10]
top_max = sorted(
    kernel_stats.items(),
    key=lambda item: max(item[1]["durations"]),  # sort by max single duration
    reverse=True,
)[:10]


kernel_time = np.array(KERNEL_TIME)
memcpy_time = np.array(MEMCPY_TIME)
sync_time = np.array(SYNC_TIME)

app_time = (APP_END - APP_START) / 1e6
total_kernel_time = kernel_time.sum()
mean_kernel_time = kernel_time.mean()
stddev_kernel_time = kernel_time.std()
total_memcpy_time = memcpy_time.sum()
mean_memcpy_time = memcpy_time.mean()
stddev_memcpy_time = memcpy_time.std()
total_sync_time = sync_time.sum()
cuda_api_time = CUDA_API_TIME

gpu_fraction = total_kernel_time / app_time
serial_fraction = 1.0 - gpu_fraction
gpu_speedups = [2, 5, 10, 20, 50, 100]

print("Some global statistics")
print("----------------------")
print(f"Total execution time (TET)  :   {app_time:.2f} ms")

print(f"Total kernel launches       :   {kernel_time.size}")
print(f"Number of unique kernels    :   {len(kernel_stats)}")
print(
    f"Total kernel time           :   {total_kernel_time:.2f} ms ({gpu_fraction * 100:.2f}% of TET)"
)
print(f"Average kernel time         :   {mean_kernel_time:.2f} ms")
print(f"StdDev kernel time          :   {stddev_kernel_time:.2f} ms")
print(f"Median kernel time          :   {np.median(kernel_time)} ms")
print(f"90th percentile kernel time :   {np.percentile(kernel_time, 90)} ms")
print(f"99th percentile kernel time :   {np.percentile(kernel_time, 99)} ms")
print(f"Largest kernel duration     :   {np.max(kernel_time):.2f} ms")

print(f"Total memcpy events         :   {memcpy_time.size}")
print(
    f"Total memcpy time           :   {total_memcpy_time:.2f} ms ({total_memcpy_time / app_time * 100:.2f}% of TET)"
)
print(f"Largest memcpy duration     :   {np.max(memcpy_time):.2f} ms")
print(
    f"Total CUDA API time         :   {cuda_api_time:.2f} ms ({cuda_api_time / app_time * 100:.2f}% of TET)"
)

print(
    f"\nPredicted global speedups for a {gpu_fraction * 100:.2f}% GPU fraction\n"
    "and a few typical local GPU speedups (Amdahl’s Law)"
)
print("------------------------------------------------------------")
for S_gpu in gpu_speedups:
    total_speedup = 1.0 / (serial_fraction + (gpu_fraction / S_gpu))
    print(f"GPU speedup: {S_gpu:>3}x → Global speedup: {total_speedup:.2f}x")

print("\nTop-10 kernels, sorted by largest total execution time")
print("------------------------------------------------------")
for kname, stats in top_total:
    total_dur = sum(stats["durations"])
    count = stats["count"]
    avg_dur = total_dur / count if count > 0 else 0
    print(f"Kernel: {kname}")
    print(
        f"  Total duration: {total_dur:.2f} ms ({total_dur / app_time * 100:.2f}% of TET)"
    )
    print(f"  Number of calls: {count}")
    print(f"  Average duration: {avg_dur:.2f} ms\n")

print("\nTop-10 kernels, sorted by average execution time")
print("------------------------------------------------")
for kname, stats in top_avg:
    total_dur = sum(stats["durations"])
    count = stats["count"]
    avg_dur = total_dur / count if count > 0 else 0
    print(f"Kernel: {kname}")
    print(f"  Total duration: {total_dur:.2f} ms")
    print(f"  Number of calls: {count}")
    print(f"  Average duration: {avg_dur:.2f} ms\n")


# Plot of the distribution of kernel durations
plt.figure(figsize=(10, 6))
plt.hist(kernel_time, bins=500, log=True)
plt.xlabel("Kernel duration (ms)")
plt.ylabel("Frequency (log scale)")
plt.title("F128_01")
plt.xscale("log")
# Fix for clipping
plt.ylim(top=plt.ylim()[1] * 1.2)
# Save image
plt.tight_layout()
plt.savefig("kernel_durations.png", dpi=600, bbox_inches="tight")
plt.close()
