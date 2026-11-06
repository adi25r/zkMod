#!/usr/bin/env python3
"""NVIDIA GPU poller via NVML (pip install nvidia-ml-py).  Usage: sampler_nvidia.py out.csv [interval_s] [gpu_index]
Same CSV as sampler_amd.py:  epoch_s, mem_busy_pct, gpu_busy_pct, vram_used_bytes   (absolute-schedule loop, runs until SIGTERM)

  mem_busy_pct = NVML utilization.memory = % of the sample period in which device memory was being read or written.
  It is an ACTIVITY gauge, not bandwidth, and NVML only refreshes it every ~1/6 s - 1 s (driver dependent), so polling at
  10 ms mostly returns repeated values. Use nsys --gpu-metrics-devices (README, 'Real DRAM bandwidth') for true bandwidth.
"""
import sys, time, signal
import pynvml
iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
pynvml.nvmlInit(); h = pynvml.nvmlDeviceGetHandleByIndex(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
run = [True]; signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))
off = time.time() - time.perf_counter(); nxt = time.perf_counter(); out = open(sys.argv[1], "w", buffering=1 << 16)
while run[0]:
    u = pynvml.nvmlDeviceGetUtilizationRates(h); m = pynvml.nvmlDeviceGetMemoryInfo(h)
    out.write("%.4f,%d,%d,%d\n" % (time.perf_counter() + off, u.memory, u.gpu, m.used))
    nxt += iv; d = nxt - time.perf_counter()
    if d > 0: time.sleep(d)
    else: nxt = time.perf_counter()
out.close()
