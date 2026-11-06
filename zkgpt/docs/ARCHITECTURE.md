# Architecture

This is a GPU prover for transformer inference, derived from zkGPT. It keeps zkGPT's protocol and offloads the expensive work to HIP kernels. `main_gkr_split` adds zkMod layer partitioning, which proves a model as a chain of independently proved blocks.

## Proof system

The statement is that a model with committed weights produced a given output on a given input. Field arithmetic is over the BN254 scalar field (`mcl::bn::Fr`). A proof has three parts.

| Part | Role | Code |
|---|---|---|
| GKR | Layer-by-layer sumcheck reduction over the layered circuit. Fully connected layers use a dedicated matmul product sumcheck. | `verifier::verifyGKR` in `src/verifier.cpp`, `src/prover.cpp`, `src/neuralNetwork.cpp` |
| Lasso reduction | Combines the per-layer claims on `val[0]` into one claim on the input layer with a product sumcheck. | `verifier::verifyLasso` |
| Hyrax | Polynomial commitment. Witnesses are committed row by row with MSMs and opened with an inner-product argument. | `src/hyrax.cpp`, `verifier::openCommit` |
| Range check | Chunked lookup (logup) that proves values lie in `range_size` bits, separate from GKR. Stages: setup, invert, commit, deg3, deg1, open. | `range_prover::range_prove` in `src/range_prover.cpp`, `src/hyrax_rp.cpp` |

The name "Lasso" in this code base means the input-claim reduction above, which is a different thing from the Lasso lookup argument. The lookup is the logup range check.

`verifier::prove` runs `verifyGKR`, then `verifyLasso`, then `openCommit`. The prover and verifier run in one process and call each other directly. Proof size is an accumulated byte count (`prover::proof_size`), and no proof is serialized.

### Knobs

| Knob | Flag | Meaning |
|---|---|---|
| `c` | `--c`, default 9 | Range chunk bits. See below. |
| `matmul_k` | `--matmul-k` | Blendy stage count for the matmul sumcheck. `l_stage = ceil(n/k)` per call. |
| `lasso_k` | `--lasso-k` | Blendy stage count for the Lasso product sumcheck. |
| `range_k` | `--range-k`, default 1 | Stage count for the range sumchecks. |
| `t` | `--sc-tile-bits`, default -1 | Caps the extension table's future-variable window at `2^t`. Footprint is `2*3^l*2^min(n-l,t)`. `-1` disables tiling. |
| stream bits | `GKR_SC_STREAM_BITS`, unset | Window for stage-1 source streaming. Unset means dense. Independent of `t`: tiling sizes the extension table and streaming sizes the source. |

**`c` must divide `range_size`, which is 54 in `main_gkr_split`.** `range_prover::push_back` and `range_prover::range_prove` throw `std::invalid_argument` otherwise, and `kChunkBitsCfg` also rejects `c > 30` because the domain is built as `1 << c` in an `int`. The legal values are 1, 2, 3, 6, 9, 18 and 27. The comment in `src/config.hpp` that also lists 54 is wrong on that value.

## What runs on the GPU

GPU code is always compiled in. Paths are selected at run time. `GKR_GPU_ENABLED` is not defined by any CMake target.

| Phase | Kernels | Default | Move it with |
|---|---|---|---|
| Matmul product sumcheck | `gpu_provers/gpu_matmul_prover.hpp`, `blendysc.hpp` | GPU | `GKR_MATMUL_CPU` |
| Lasso product sumcheck | `gpu_matmul_prover.hpp`, `gpu_range_prover.hpp`, `gpu_lasso_sparse.hpp` | GPU | `GKR_LASSO_CPU` |
| Generic GKR sumcheck | `src/gkr_gpu.cpp` | GPU | `GKR_SC_CPU` |
| GKR predicate reduction | `src/gkr_pred_gpu.cpp` | GPU | `GKR_PRED_CPU` |
| Hyrax commit | `gpu_hyrax_commit.hpp`, `gpu_fr_scalars.hpp`, zkmod-gpu MSM | GPU | `--cpu-commit` |
| Range deg3 sumcheck | `gpu_deg3_sumcheck.hpp`, `gpu_deg3_staged.hpp` | GPU | `--cpu-deg3` |
| Range deg1 sumcheck | `gpu_range_prover.hpp`, `blendy_linear.hpp` | GPU | `--cpu` |
| Range open | `gpu_hyrax_open.hpp` | **CPU** | `GKR_OPEN_GPU` opts in, `--cpu-open` wins over it |
| Input evaluation in `openCommit` | `gpu_bfeval.hpp` | **CPU**, OpenMP | `GKR_BFEVAL_GPU` opts in |
| Fully connected witgen | `witgen_gpu/gpu_witgen.hpp` | **CPU** | `GKR_WITGEN_GPU` opts in |
| Round layer witgen | same | **CPU** | `GKR_ROUND_GPU` opts in, needs resident FC output |
| Gate evaluator witgen | same | **CPU** | `GKR_GATES_GPU` opts in |
| Layer builders witgen | same | **CPU** | `GKR_WITGEN_FULL` opts in |

The GPU witness generation runs set `GKR_WITGEN_GPU` and `GKR_ROUND_GPU` only, as `gates_gpu=0 witgen_full=0` on their `SPLIT_OVERLAPPED` lines shows.

There is no silent GPU to CPU fallback for device allocation: `HIP_CHECK` in `gpu_field.hpp` exits with status 12 on `hipErrorOutOfMemory`. One conditional path does exist: `roundLayer` runs its CPU body when the FC output is not resident.

## Field bridge

Host arithmetic uses `mcl::bn::Fr` (`FieldElement`). Device kernels use `blstrs__scalar__Scalar` (`gpu_fe`) from zkmod-gpu's `bls12_381_core` library, which despite the name is configured for BN254 here. `fe_to_gpu` and `fe_from_gpu` in `gpu_provers/gpu_field.hpp` are a `memcpy`: both sides use the same modulus and Montgomery form with `R = 2^256`, so the limbs are identical. If the device field is reparameterized, the bridge becomes wrong.

## The split prover

`src/main_gkr_split.cpp` is the main entry point. It proves modular-zkp's arithmetization and nothing else, at every split count including the `splits=1` monolith: one `LLMSplitBlock` per partition, a probed half/half input layout with the weights at the midpoint, and `kMerge = 1` for both the builder and the range prover. Partitions are layer-granular. Without `--splits a,b,c`, layers are divided evenly with the remainder on the first partitions.

### Execution of one run

1. **Probe.** For each distinct split depth, a forked child (phase `probe_witgen`) builds the circuit to obtain `getTotalInputSize`. The result is cached on disk in `~/.cache/zkml_probe` (`GKR_PROBE_CACHE`). The probe also runs at `splits=1`. Shared boundary sizes come from `next_pow2_size`.
2. **Build** (`build_partition`). Calls `neuralNetwork::create`, phase `witgen`. Non-last partitions pad the output layer.
3. **Prove** (`prove_built_partition`), in this order: construct the verifier, then `standalone_input_commit`, `output_commit`, `range_build`, `range_prove`, `execute_commit`, `boundary_link`, then `gkr_prove` containing `verify_gkr`, `verify_lasso`, `past_peak_free` and `open_commit`.

### Run modes

| Mode | Function | Witgen and proving | Summary line |
|---|---|---|---|
| no flag | `run_split_sequential` | Build then prove each partition in turn, in one process. The only mode at `splits=1`. | `SPLIT_SUMMARY` |
| `--concurrent [W]` | `run_split_concurrent` | Parent builds all, forks a child per partition, releases in waves of W. Reports witgen and proving disjointly. | `SPLIT_CONCURRENT` |
| `--batched [B]`, alias `--pipeline` | `run_split_batched` | Parent builds a batch while the previous one proves. Refuses `GKR_WITGEN_GPU`, `GKR_GATES_GPU` and `GKR_WITGEN_FULL`, because a parent HIP context does not survive `fork`. | `SPLIT_BATCHED` |
| `--overlapped [B]`, alias `--witgen-chain` | `run_split_overlapped` | Children fork before any HIP call. Child i+1 builds while child i proves. **The default for measurements.** | `SPLIT_OVERLAPPED` |
| `--overlapped -1` | same | As above, with the memory gate. The mode the figures use. | same |
| `--overlapped N --witgen-barrier` | same | All witgen, then all proving. Reports `prep_s` and `barrier_makespan_s`. | same |

In this repo "serial" describes the relation of witgen to proving: all witgen first, then all proving, which may itself be concurrent. `--concurrent` and `--overlapped --witgen-barrier` are serial in that sense. The no-flag mode is not: it interleaves, and it must not be used as a baseline.

`--unique-parts` proves one partition per structural role. It is for cost measurement only, and the resulting proof is not valid end to end. `--witgen-only` builds without proving.

### Carry and early carry

The carry is the activation window partition i+1 reads as input. `exportSplitBlockCarry` extracts it from partition i's `val[0]`, a window of `len * feature_dim` values. In overlapped mode it passes through a `MAP_SHARED` region (`ChainShm`), with pipe `gates[i]` signalling that it is ready. This makes witgen a strictly serial chain.

`GKR_EARLY_CARRY=1` publishes the carry from the `on_values_ready` hook in `create()`, right after the builder loop, instead of after merge, `initSubset` and input copy. The child re-exports at the end of its build and exits non-zero if the two differ. The figure queues set it for all partitioned runs. `GKR_EARLY_CARRY` has no effect in controller mode or on the last partition.

Prebuild (`GKR_PREBUILD`, on by default, off whenever the witness cache is enabled) lets child i+1 take a build slot and do everything that does not read the carry, such as layout, weights and the input layer. The `on_before_values` hook then waits for the real carry. Slots go out in chain order through pipes `started[i]`, which avoids deadlock.

`GKR_WITGEN_DEVICE=W` is a controller mode that runs all witgen on one device and hands witnesses over through the witness cache in `/dev/shm`. It is not supported with `--overlapped -1`.

### Boundary link

`src/boundary_link.hpp` ties adjacent partitions together. It is needed because the output layer of every non-last partition is all zeros: each layer of this arithmetization is a checker whose outputs must vanish, so `commit_private_output_for_next` commits zeros and binds nothing. The carry lives in partition i's input witness and again as the prefix of partition i+1's input.

Once both commitments exist, a random `z` is drawn and the check is

    sum_k W_i[q_off + k] eq(z, k) == sum_k P_{i+1}[k] eq(z, k)

with both sides proven against Hyrax row commitments. The window can sit at an unaligned offset, so each segment needs at most two dot-product proofs. Each side publishes into a `MAP_SHARED` slot, and whichever arrives second runs the check, so no child blocks while it holds a proving slot. The parent requires every slot to report success before printing `status=OK`, and prints `BOUNDARY_LINK_SUMMARY`. `GKR_TAMPER_CARRY=<i>` is the negative test.

### Thread pool

`cfg.num_threads` is one pool. `main` calls `omp_set_num_threads` with it for the whole process, including the monolith. A forked child gets `tpc = num_threads / b`, or `GKR_TPC`, clamped to the pool. `child_threads_set` applies `tpc` to OpenMP and to the prover's `max_threads`. The gate grants a new build only while `(alive + 1) * tpc <= num_threads`.

`GKR_PIN_CORES=1` pins each child to a block of `tpc` physical cores, claimed from a shared table when work starts. Cores come from the inherited affinity mask and the sibling lists in sysfs.

### Memory gate

`--overlapped -1` runs a release loop in the parent, in `run_split_overlapped`, that decides when each child may build and prove. Children report transitions over a pipe.

| State | Meaning | Host charge |
|---|---|---|
| `F` | Forked, idle | none |
| `w` | Waiting for a build slot | none |
| `b` | Building | built estimate, and the full prove peak when deciding on another build |
| `B`, `W` | Built, waiting to prove | full peak estimate |
| `P` | Proving | full peak estimate |
| `R` | Proving, past its peak | max RSS seen while in `R`, times 1.1 |
| `X` | Exited | none |

A proving slot is granted when `MemAvailable - owed - GKR_MIN_FREE_MIB >= 0` and the VRAM check passes. A build slot additionally requires the witgen cap, `GKR_GPU_PROCS` and the thread pool to allow it. The witgen cap keeps the number of building, built and waiting partitions below the number of provers that fit.

- **Host memory** comes from `MemAvailable`, bounded by the cgroup limit (`cgroup_available_kib`), and VmHWM per child.
- **VRAM** comes from KFD sysfs per process. The VRAM gate needs a single device and is disabled by `GKR_VRAM_GATE=0`. It charges each slot's remaining growth to its estimate.
- **Seeds.** `GKR_PART_PEAK_MIB`, `GKR_PART_BUILT_MIB`, `GKR_PART_VRAM_MIB` and `GKR_PART_WG_VRAM_MIB` seed the estimates. Live children correct them upward. Once partitions finish, their observed peaks times 1.1 replace the seeds.
- **Liveness escape.** If nothing is proving, the head waiter is released regardless of room. The kernel or driver may then refuse the allocation, and the row fails.
- **`GKR_GPU_PROCS=K`** caps children past their build grant, which keeps the MI210 hardware scheduler within its 8-process limit when combined with `GPU_MAX_HW_QUEUES=1`.

Each child has its own prove-token pipe (`ptok`), and children do not return tokens in gated mode.

### Peak-release semaphore

Host and VRAM peaks both land in `verify_lasso`. Right after it, in `verifier::prove`, phase `past_peak_free` calls `prover::release_lasso_state`, drops the verifier's beta tables, calls `malloc_trim` and fires `g_past_peak_hook`. The child posts `R`, and the parent charges measured RSS instead of a full peak, so the next prover can start.

## Run modes

Set by `tools/results/figures/queue/run_zkmod.sh` and `run_monolith.sh`. Every run sets `ZKGPT_WITNESS_CACHE=0`, and partitioned runs set `GKR_EARLY_CARRY=1`.

| Mode | Witgen | Prove | Partitions | Flags | Environment |
|---|---|---|---|---|---|
| `mono` | CPU | CPU | 1 | `--c 9 --cpu --cpu-deg3 --cpu-open --cpu-commit` | `GKR_LASSO_CPU GKR_MATMUL_CPU GKR_PRED_CPU GKR_SC_CPU` |
| `cpu` | CPU | CPU | default | as `mono`, plus `--overlapped -1` | as `mono` |
| `gpu` | GPU | GPU | default | `--overlapped -1 --c 18 --matmul-k 4 --range-k 8 --lasso-k 8 --sc-tile-bits 18` | `GPU_MAX_HW_QUEUES=1 GKR_GPU_PROCS=12 GKR_PIN_CORES=1 GKR_BFEVAL_GPU=1 GKR_SC_STREAM_BITS=22 GKR_WITGEN_GPU GKR_ROUND_GPU` |

Default partitions are one layer each. `TPC` defaults to `192 / partitions`, and to `192 / 3 = 64` for opt2b7 at its default split.

| Model | Layers | Embed | Lin | Partitions |
|---|---|---|---|---|
| gpt2 | 12 | 768 | 3072 | 12 |
| opt350m | 24 | 1024 | 4096 | 24 |
| opt1b3 | 24 | 2048 | 8192 | 24 |
| opt2b7 | 32 | 2560 | 10240 | 32 |

`mono` has `SPEC` entries for gpt2 and opt350m only. The opt1b3 and opt2b7 monoliths do not fit the 723 GiB box.

## Invariants

- **Construct the verifier before any commit.** Its constructor calls `prover::init`, which resets `cc`, `cc_input`, `cc_output` and `input_commit_segments`.
- **`executeCommit` is unconditional.** Skipping it leaves `comm.w` null and `openCommit` crashes.
- **Commit the full padded range.** `new Fr[n]()` does not zero, because `mcl::FpT` has a user-provided empty constructor. `new ll[n]()` does zero. Filling only part of `w` makes honest proofs reject intermittently on non-power-of-two lengths. `GKR_TRACE_SEGMENTS=1` names the failing segment.
- **`merge` is `kMerge` for both the builder and the range prover.** They must agree, or the range set covers a different circuit.
- **`range_partitions` stays 1.** Auto-splitting registers P constraints, and `sum(next_pow2) != next_pow2(sum)`, which changes the statement.
- **The circuit width is `next_pow2(len)`.** The binary rounds it.
- **Thread counts come off the members** `p.max_threads` and `v.max_threads`. Leaving them 0 divides by zero.
- **The parent reaches `fork` with no HIP context.** Only the sequential mode enables GPU witgen in the main process. Gates for GPU witgen are stored in `want_*` and applied after `fork`.
- **Timed GPU regions must sync first.** Kernels launch asynchronously. `range_prover.cpp` calls `gkr_phase_barrier`. `gkr_prover_barrier` in `verifier.cpp` is defined only under `GKR_GPU_ENABLED`, which CMake never sets, so it is a no-op in normal builds and `ph_gkr_matmul`, `ph_gkr_generic` and the Lasso timer can leak kernel time into the next phase.
- **Do not guard GPU paths on `#ifdef GKR_GPU_ENABLED`.** The path would compile out with no diagnostic.
- **The Lagrange basis is not counted.** `brute_force_compute_LR` runs before the timed `hyrax::open`, matching the modular-zkp baselines.
- **Declare regions with `ZKGPT_PHASE_SCOPE`.** It emits a `PHASE` line and a ROCTx range.
- **`circuit_s` is not witgen time.** Its timer starts in `create()` and stops at the top of `verifier::prove`. Use `witgen_s`.
- **`timer::elapse_sec()` is cumulative**, not the last interval. Accumulating it per layer overstates the sum.

