# Build and run

Everything goes through the driver in the monorepo root, `../zkmod`. Commands below run from that root unless stated.

## Dependencies

| Need | Detail |
|---|---|
| CMake | 3.18 or later |
| Host compiler | C++20 |
| AMD toolchain | ROCm with HIP. The tested box has ROCm 7 in `/opt/rocm-scale` and 6.4.3 in `/opt/rocm-6.4.3`, architecture `gfx90a` (MI210). |
| `mcl/` | Sibling directory, `https://github.com/herumi/mcl.git` at `d1a53d4`, built with `MCL_FP_BIT=384`. |
| `zkmod-gpu/` | Sibling directory, `https://github.com/pawks/zkmod-gpu.git` at `6c6614c`. |
| Plotting | Python 3 with matplotlib and numpy |

CMake fixes the sibling paths as `${CMAKE_CURRENT_SOURCE_DIR}/..`, so the layout cannot be changed without editing `CMakeLists.txt`. The binary's run path holds the absolute `mcl/lib` location, so moving `mcl/` breaks built binaries.

**Provenance.** `./zkmod deps` pins zkmod-gpu at remote commit `6c6614c`. The existing figure logs were built against `966c05b`. A fresh build therefore does not use the same device library as the logged numbers.

## Driver commands

| Command | Effect |
|---|---|
| `./zkmod check` | Report toolchains, GPUs, dependencies, binaries, free VRAM and host memory |
| `./zkmod deps` | Clone and build mcl, clone zkmod-gpu, at the pinned commits |
| `./zkmod build zkgpt` | Configure into `build-scale`, build `bin/main_gkr_split` |
| `./zkmod prove zkgpt --model M [--mode mono\|cpu\|gpu] [--len L] [--splits N] [--gpu G]` | One run. Mode defaults to `gpu`, `L` to 64. The log lands in `tools/results/figures/logs/`, or `LOGS`. `mono` accepts only gpt2 and opt350m. |
| `./zkmod figures zkgpt --smoke [--gpu G]` | gpt2 `gpu` mode, then plot into `tools/results/figures/smoke/` |
| `./zkmod figures zkgpt --full [--gpu G]` | Run `tools/results/figures/run_all.sh` |
| `./zkmod figures zkgpt --plot [--logs DIR] [--out DIR] [fig ...]` | Plot only |
| `./zkmod all` | `deps`, `build zkgpt`, `figures zkgpt --smoke` |

Build environment: `JOBS` (default 32), `ZKGPT_CMAKE_ARGS` (extra configure arguments), `ZKGPT_HIP_ARCH` (default `gfx90a`).

## Configure recipe

The verified configure line is `rocm_cmake_args()` in `../zkmod`, which applies when `/opt/rocm-scale/lib/llvm/bin/clang++` exists. Prefer `./zkmod build zkgpt`. The equivalent is:

```bash
cmake -S . -B build-scale \
  -DCMAKE_HIP_COMPILER=/opt/rocm-scale/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm-scale \
  -DCMAKE_PREFIX_PATH=/opt/rocm-scale \
  -DHIP_DIR=/opt/rocm-scale/lib/cmake/hip \
  -Dhip_DIR=/opt/rocm-scale/lib/cmake/hip \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  "-DCMAKE_HIP_FLAGS=--rocm-path=/opt/rocm-scale -I/opt/rocm-6.4.3/lib/llvm/include" \
  "-DZKGPT_HIP_EXTRA_FLAGS=--rocm-path=/opt/rocm-scale;-std=gnu++20;-I/opt/rocm-6.4.3/lib/llvm/include" \
  "-DCMAKE_EXE_LINKER_FLAGS=-L/opt/rocm-scale/lib -L/opt/rocm-6.4.3/lib/llvm/lib -Wl,-rpath,/opt/rocm-scale/lib:/opt/rocm-6.4.3/lib/llvm/lib"
cmake --build build-scale --target main_gkr_split -j32
```

Why each part is there:

| Setting | Reason |
|---|---|
| `HIP_DIR`, `hip_DIR` | Without them CMake finds HIP under `/opt/rocm`, and the link fails. |
| `CMAKE_HIP_FLAGS` | Reaches CMake's compiler check during `project()`. `CMakeLists.txt` overwrites this variable right after `project()`. |
| `ZKGPT_HIP_EXTRA_FLAGS` | Compile-only flags that survive that overwrite. They also reach the zkmod-gpu sub-build. |
| `-I/opt/rocm-6.4.3/lib/llvm/include` | The ROCm 7 clang has no `omp.h`. |
| `CMAKE_EXE_LINKER_FLAGS` | Finds `libomp` from ROCm 6.4.3 and records run paths. |

Do not add `-x hip` to these flags. An older recipe did, and it does not build from scratch.

What `CMakeLists.txt` does on its own:

- Passes `--gcc-install-dir` to ROCm clang based on the host `g++`, because clang otherwise picks the highest-numbered GCC directory even with no libstdc++ headers there (`cmath not found`). This has to be set before `project()`.
- Compiles every source as HIP with `-fgpu-rdc`, and `-rdc=true` on nvcc.
- Passes `-fopenmp` at compile and link. Without it every `#pragma omp` is silently a no-op.
- Defines `MCL_FP_BIT=384` on all targets.
- Writes binaries to `bin/`.

At run time, use the `LD_LIBRARY_PATH` that `rocm_env()` in `../zkmod` and the queue scripts set:

```bash
export LD_LIBRARY_PATH=/opt/rocm-scale/lib:/opt/rocm/lib:/opt/rocm/llvm/lib
```

`lld` "occupancy target" warnings at link time are harmless.

## Binary targets

| Target | Extra defines | Purpose |
|---|---|---|
| `main_gkr_split` | | The split and monolith prover. Built by the driver. |

## CLI

### `main_gkr_split`

```
main_gkr_split <depth> <embed> <num_splits> [len] [threads] [lin_dim] [flags]
```

Defaults: depth 12, embed 768, splits 2, **len 30**, threads 32, `lin_dim = 3*embed`. Always pass len 64 for gpt2, and the real `lin_dim` (3072 for gpt2). An unknown flag exits with failure.

| Flag | Effect |
|---|---|
| `--c N`, `--cs a,b,c` | Range chunk bits, global or per partition. Must divide 54, at most 30. |
| `--matmul-k N`, `--matmul-ks` | Matmul stages |
| `--range-k N`, `--range-ks` | Range sumcheck stages |
| `--lasso-k N`, `--lasso-ks` | Lasso stages |
| `--lasso-l N`, `--lasso-ls` | Override `l_stage` directly. `k` takes priority. |
| `--sc-tile-bits N` | Extension table tile bits |
| `--splits a,b,c` | Explicit layer composition summing to depth. Overrides `num_splits`. |
| `--cpu`, `--cpu-deg3`, `--cpu-open`, `--cpu-commit` | CPU reference paths: range deg1, range deg3, range open, Hyrax commit |
| `--concurrent [W]` | Run mode |
| `--batched [B]`, `--pipeline [B]` | Run mode |
| `--overlapped [B]`, `--witgen-chain [B]` | Run mode. `-1` is memory-gated. |
| `--witgen-barrier` | With `--overlapped`: all witgen first |
| `--unique-parts` | One partition per role. Measurement only, proof not valid. |
| `--witgen-only` | Build every partition, no proving |
| `--threads-total N` | Overrides the positional thread budget |
| `--partitions N`, `--vram-gb N` | Range prover partitions and VRAM budget. Keep partitions at 1. |
| `--vram-csv PATH`, `--vram-interval S` | Device-wide sampler output and period, default 0.001 s |
| `--uvm-prefetch` | UVM binary only |

The binary's own usage text lists `--range-size`, which is no longer accepted. `range_size` is fixed at 54 here.

A run passes when `status=OK` (forked modes) or `SPLIT_SUMMARY` (sequential) is printed **and** the number of `All verification passed` lines equals the partition count. See the caveat in [ARCHITECTURE.md](ARCHITECTURE.md).

```bash
HIP_VISIBLE_DEVICES=1 ./bin/main_gkr_split 12 768 12 64 192 3072 --overlapped -1 \
  --c 18 --matmul-k 4 --range-k 8 --lasso-k 8 --sc-tile-bits 18
```

Check free VRAM first. A nearly full GPU produces an out-of-memory error at the first allocation. Most real runs should go through `./zkmod prove`, because the arm environment matters.

## Environment gates

Derived from a grep of `getenv` in `src/`, `gpu_provers/` and `witgen_gpu/`. Regenerate with:

```bash
grep -rhoE 'getenv\("[A-Za-z_0-9]+"\)' src gpu_provers witgen_gpu | sed 's/getenv("//;s/")//' | sort -u
```

**Most gates test presence, not value: `GKR_LASSO_CPU=0` still forces the CPU.** The ones that parse a value are noted.

### Device selection

| Variable | Effect |
|---|---|
| `GKR_MATMUL_CPU` | Matmul product sumcheck on host |
| `GKR_LASSO_CPU` | Lasso sumcheck on host |
| `GKR_SC_CPU` | Generic GKR sumcheck on host |
| `GKR_PRED_CPU` | GKR predicate on host |
| `GKR_OPEN_GPU` | Range open on GPU, unless `--cpu-open` |
| `GKR_BFEVAL_GPU` | `brute_force_compute_eval` on GPU |
| `GKR_WITGEN_GPU` | FC witgen on GPU |
| `GKR_ROUND_GPU` | Round layer witgen on GPU |
| `GKR_GATES_GPU` | Gate evaluator on GPU |
| `GKR_WITGEN_FULL`, `GKR_GELU_GPU` | All builders on GPU. The second is an alias. |
| `HIP_VISIBLE_DEVICES` | Device selection. The gate also reads it for the VRAM check. |
| `GKR_SPLIT_DEVICES=a,b` | Overlapped: round-robin children across devices |
| `GKR_WITGEN_DEVICE=W` | Overlapped: controller owns witgen on device W |

### Memory and execution

| Variable | Effect |
|---|---|
| `GKR_SC_TILE_BITS=t` | Overrides `--sc-tile-bits` |
| `GKR_SC_STREAM_BITS=b` | Stage-1 source streaming window. Unset means dense. |
| `GKR_SC_SPARSE=0` | Disable the default sparse Lasso source upload |
| `GKR_PTABLE_BLOCKS=n` | Overrides the ptable block count, default 2 per compute unit |
| `GKR_GATE_CHUNKS=n` | u-partition chunks, default 256 |
| `GKR_NO_GATE_PARALLEL` | Disable threaded gate scatter |
| `GKR_FC_ROUND_FUSE=0` | Disable fusing round onto FC output |
| `GKR_NO_ZERO_BETA_SKIP` | Disable the all-zero beta skip |
| `GKR_EARLY_CARRY` | Publish the carry early |
| `GKR_PREBUILD=0` | Disable prebuild |
| `GKR_TPC=n` | Threads per child |
| `GKR_PIN_CORES` | Pin children to physical core blocks |
| `GKR_MIN_FREE_MIB=n` | Host margin, default 40960 |
| `GKR_PART_PEAK_MIB`, `GKR_PART_BUILT_MIB` | Host seeds |
| `GKR_PART_VRAM_MIB`, `GKR_PART_WG_VRAM_MIB` | VRAM seeds |
| `GKR_VRAM_MARGIN_MIB=n` | VRAM margin, default 2048 |
| `GKR_VRAM_GATE=0` | Disable the VRAM gate |
| `GKR_GPU_PROCS=K` | Cap on children past their build grant |
| `GKR_PROBE_CACHE=dir` | Probe cache directory, `0` disables. Default `$HOME/.cache/zkml_probe` (`HOME`). |
| `ZKGPT_WITNESS_CACHE=dir` | Cache directory, `0` disables. **On by default**, at `<repo>/.witness_cache`. Benchmarks must set `0`. |
| `ZKGPT_WEIGHT_SEED=n` | Weights as a function of partition index |
| `GKR_UVM_ZEROCOPY_GB` | UVM threshold, default 4 |
| `ZKGPT_UVM_LOCK`, `ZKGPT_UVM_NO_LOCK` | UVM single-instance lock path and bypass |

### Tests and consistency checks

| Variable | Effect |
|---|---|
| `GKR_TEST_INPUT=seed` | Nonzero prompt. Required for boundary tests. |
| `GKR_TAMPER_CARRY=i` | Perturb partition i's carry, which must REJECT at `stage=boundary_link` |
| `GKR_WITGEN_CHECK`, `GKR_ROUND_CHECK`, `GKR_GATES_CHECK` | Run both paths and assert equality |
| `GKR_WITGEN_FULL_CHECK`, `GKR_GELU_CHECK` | Same for builders |
| `GKR_PRED_CHECK` | Both predicate paths |
| `GKR_BFEVAL_CHECK` | Compare `bf_eval` paths |
| `GKR_COMMIT_FR_VERIFY` | Recompute the commit on the CPU and compare |
| `GKR_UPART_CHECK` | Compare the u-partition with a plain loop |
| `GKR_ASSERT_ZERO_BETA` | Verify the beta skip |
| `GKR_CHECK_RESERVE` | Check the witness reservation |
| `GKR_SC_STREAM_PAIRS`, `GKR_SC_FORCE_STREAM` | Force streamed GKR rounds on small models |

`GKR_LASSO_SPARSE_CHECK` appears in a comment in `gpu_lasso_sparse.hpp`, and no code reads it.

### Profiling and tracing

| Variable | Output |
|---|---|
| `GKR_PHASE_LOG=0` | Silence `PHASE` lines |
| `GKR_PHASE_XFER` | Add cumulative `h2d_gib` to `PHASE` lines |
| `GKR_ACCT` | `[acct]` breakdown, including Lagrange and marshalling time |
| `GKR_MM_PROF` | `MATMUL_BREAKDOWN`. Inserts syncs, so it perturbs timing. |
| `GKR_GEN_PROF` | `initPhase1/2` and predicate sub-steps |
| `GKR_MSM_PROF` | MSM sub-phases |
| `GKR_WITGEN_PROF` | Per-builder witgen breakdown |
| `GKR_XFER_TIME` | Witgen transfer time |
| `GKR_MEMACCT` | Per-partition `mallinfo2` accounting |
| `GKR_ALLOCTRACE`, `GKR_ALLOCTRACE_GIB` | Name large allocations |
| `GKR_DBG_MEM`, `GKR_DBG_LASSO_SIZES`, `GKR_DEG3_DBG`, `GKR_DUMP_SIZES`, `GKR_LASSO_STATS` | Sizes and sparsity |
| `GKR_DUMP_CARRY` | Fingerprint carries and witnesses. Costly, so leave it off in timed runs. |
| `GKR_TRACE_SEGMENTS` | Name each segment before opening it |
| `ZKGPT_SAMPLER_PCI`, `ZKGPT_SAMPLER_KFD_GPU_ID` | Limit the sampler to one card |

`GPU_MAX_HW_QUEUES` is read by the HIP runtime, not this code. `GLIBC_TUNABLES=glibc.malloc.hugetlb=1` is a glibc setting. No queue sets it.

## Log format

### `PHASE` lines

Written to stderr, one per transition, in one write. Time is `gettimeofday` in microseconds, which is comparable across forked processes and with `PARTITION` and the `--vram-csv` stamps.

```
PHASE part=3 pid=48211 ev=begin phase=range_prove            d=1 t_us=1756812345678901
PHASE part=3 pid=48211 ev=end   phase=range_prove            d=1 t_us=1756812365163402 dur_s=19.4845
PHASE part=-1 pid=48200 ev=mark  phase=release_wave0         d=0 t_us=1756812301004112
```

`part=-1` is the parent. `d` is nesting depth.

| Phase | Meaning |
|---|---|
| `probe_witgen` | Boundary-size probe |
| `build` containing `witgen` | Circuit and witness |
| `export_carry`, `carry_wait`, `carry_import`, `carry_publish`, `chain_wait`, `handoff_wait` | Carry chain |
| `build_wait`, `token_wait`, `gate_wait`, `barrier_wait` | Parked waiting for a slot or gate |
| `prove` | Whole prove side, one per partition |
| `standalone_input_commit`, `output_commit`, `execute_commit` | Commits |
| `range_build`, `range_prove` | Range prover |
| `boundary_link` | Boundary publication and check |
| `gkr_prove` containing `verify_gkr`, `gkr_layer<i>[_fc]`, `verify_lasso`, `past_peak_free`, `open_commit` | `verifier::prove` |
| `released`, `release_wave<g>`, `release_w<g>`, `release_batch<g>`, `joined_wave<g>`, `joined_batch<g>` | Parent marks and events |

```bash
grep '^PHASE part=3 ' run.log                   # one partition's lifetime
grep -E '^PHASE .*ev=end .*phase=(witgen|prove) ' run.log
```

### `PARTITION` line

One per partition, on stdout.

| Fields | Meaning |
|---|---|
| `idx`, `len`, `layers`, `c`, `matmul_k`, `range_k`, `lasso_k`, `lasso_l` | Identity and knobs. Older logs have `chunk_width=`. |
| `t0_us`, `t1_us`, `time_s` | Prove wall time |
| `circuit_s` | Overlaps `time_s`. Not witgen time. |
| `prover_s`, `commit_s`, `standalone_commit_s`, `range_s`, `range_build_s`, `exec_commit_s` | Component timers |
| `prover_zkgpt_s`, `prover_core_s` | `standalone_commit + prover_time + output_commit` |
| `prover_total_s` | Adds range. Excludes witness evaluation and Lagrange work. |
| `verifier_s`, `proof_kb` | This partition's verifier time and proof size |
| `ph_gkr_generic`, `ph_gkr_matmul`, `ph_lasso`, `ph_hyrax_commit`, `ph_hyrax_open` | Prover buckets |
| `gpu_peak_mib`, `host_peak_mib` | Prove-phase peaks. The host peak is VmHWM, reset at the build/prove seam. |
| `prove_h2d_gib`, `prove_d2h_gib` | Transfer volume |
| `witgen_s`, `witgen_fc_s`, `witgen_gates_s` | Real witgen time |
| `witgen_gpu_peak_mib`, `witgen_host_peak_mib`, `witgen_resident_mib`, `witgen_h2d_gib`, `witgen_d2h_gib` | Witgen peaks and transfers |

`grep -o host_peak_mib=` also matches `witgen_host_peak_mib`, which is a much smaller number.

### Other lines

| Line | Content |
|---|---|
| `TIMING_SPLIT idx=` | Four-way split: `wg_*` (witness, input init, circuit, carry, other) and `pf_*` (commit, lasso, sumcheck, input check, range, other, verify), plus components. Used by the flame chart. |
| `SPLIT_OVERLAPPED` | `parts batch mem_gated threads_per_child threads_total witgen_gpu round_gpu gates_gpu witgen_full witgen_device carry_mib barrier prep_s barrier_makespan_s overlapped_total_s status` |
| `SPLIT_CONCURRENT` | `parts width waves threads_per_child threads_total prep_s concurrent_makespan_s status` |
| `SPLIT_BATCHED` | `parts batch batches threads_per_child threads_total build_sum_s exposed_build_s batched_total_s status` |
| `SPLIT_SUMMARY` | `makespan_s seq_sum_s output_commit_s parts len verifier_s proof_kb` |
| `OVERLAPPED_BUILT`, `EARLY_CARRY` | Build completion and carry publication times |
| `BOUNDARY_LINK`, `BOUNDARY_LINK_SUMMARY` | Per-boundary result and total |
| `RESULT: ACCEPT`, `RESULT: REJECT` | One per proof |
| `All verification passed!!` | One per partition whose Lasso equality held |
| `PROVER_BREAKDOWN`, `PROVER_TOTALS`, `RANGE_BREAKDOWN`, `LASSO_INIT`, `MATMUL_BREAKDOWN`, `OPEN_SEG`, `WITGEN_BREAKDOWN` | Detailed attribution |
| `GPU_SAMPLE`, `GPU_SAMPLE_MAX` | Device-wide sampler, with `--vram-csv` |

## Checking soundness by hand

Test boundary linking with `GKR_TAMPER_CARRY` and `GKR_TEST_INPUT`.

## Operational hazards

- **Shared box.** Check `rocm-smi --showmeminfo vram` and `MemAvailable` before any run.
- **Wedged driver.** A process in D state at 0% CPU right after its first HIP call, with `kfd_create_process` in `/proc/<pid>/wchan`, means the KFD driver is blocked box-wide. Choosing another GPU does not help.
- **UVM.** Do not launch new `HSA_XNACK=1` or managed sweeps, because they wedge the driver. Never `SIGKILL` a page-faulting managed run. It leaves the GPU reporting free VRAM while `hipMalloc` fails, and recovery needs `rocm-smi --gpureset`.
- **Long queues** belong in `screen` or `nohup`.
- **`pgrep -f 'bin/main_gkr_split'`** in `run_zkgptbaselineprofile.sh` matches any command line with that string, including a monitoring shell, and stalls the queue.
- **`kill`.** A killed run leaves `.bin.tmp<pid>` files in the witness cache directory.
