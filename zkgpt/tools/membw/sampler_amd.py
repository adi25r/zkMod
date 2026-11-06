#!/usr/bin/env python3
"""AMD MI-series poller (sysfs). 10 ms poller of mem_busy_percent / gpu_busy_percent / mem_info_vram_used. Usage: sampler_amd.py out.csv [interval_s]; edit D (PCI address of the card)
Absolute-schedule loop (no drift); rows: epoch_s,mem_busy,gpu_busy,vram_used_bytes. Runs until SIGTERM."""
import os, sys, time, signal
D = "/sys/bus/pci/devices/0000:43:00.0/"
iv = float(sys.argv[2]) if len(sys.argv) > 2 else 0.01
fm, fg, fv = (os.open(D + n, os.O_RDONLY) for n in ("mem_busy_percent", "gpu_busy_percent", "mem_info_vram_used"))
run = [True]; signal.signal(signal.SIGTERM, lambda *a: run.__setitem__(0, False))
off = time.time() - time.perf_counter(); nxt = time.perf_counter(); out = open(sys.argv[1], "w", buffering=1 << 16)
while run[0]:
    out.write("%.4f,%s,%s,%s\n" % (time.perf_counter() + off, os.pread(fm, 8, 0).strip().decode(), os.pread(fg, 8, 0).strip().decode(), os.pread(fv, 24, 0).strip().decode()))
    nxt += iv; d = nxt - time.perf_counter()
    if d > 0: time.sleep(d)
    else: nxt = time.perf_counter()
out.close()
