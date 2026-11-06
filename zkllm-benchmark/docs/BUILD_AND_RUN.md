# Build and run

## Known build problems

| Problem | Evidence | What to do |
|---------|----------|------------|
| Bare `make` builds only `bls12-381.o`. The first rule in the file is the header dependency rule, not `all`. | `make -n` | Name targets, or use `make all`. |
| `ffn`, `rmsnorm`, `self-attn`, `skip-connection` and `main` are not in `TARGETS`. `make ffn` falls to make's built-in rule and runs `cc ffn.o -o ffn`, without the other objects. `llama-ffn.py` calls `make ffn`. | `make -n ffn` | Treat the upstream per-op LLaMa pipeline as not buildable from this Makefile. |
| `ARCH` is not part of the object dependencies, so a changed `ARCH` reuses old objects. The default is `sm_86`. | `Makefile` | `make clean` when changing it. |
| `./zkmod build zkllm` accepts an `nvcc` on `PATH`, but the Makefile uses only `$(CUDA_HOME)/bin/nvcc` or `$(CONDA_PREFIX)/bin/nvcc`. | `zkmod`, `Makefile` | Set `CUDA_HOME` or activate the conda env. |

## Toolchain

| Item | Requirement |
|------|-------------|
| Compiler | `nvcc` from **CUDA 12.9**. |
| Why 12.9 | `ptxas` from CUDA 12.6 and 12.8 miscompiles the PTX carry-chain field multiply at `-O2` and above in some kernels. It produced wrong Lagrange fold weights in `blendy.cu` and failed staged sumchecks with `claim != p(0) + p(1)`. `ptxas -O1` and 12.9 are correct, according to the comments in `bls12-381.cu` and the sbatch scripts. The Makefile warns on 12.6 and 12.8. |
| Stopgap | `ZKLLM_PORTABLE_MUL` selects the slower portable multiply. The Makefile has no variable for it, so override all flags: `make NVCC_FLAGS="-arch=sm_90 -std=c++17 -DZKLLM_PORTABLE_MUL" ...`. |
| Selecting the toolkit | `CUDA_HOME=<toolkit>`: uses that `nvcc`, and links cudart statically, so no module is needed at run time. Unset: uses `$CONDA_PREFIX/bin/nvcc` with `-lcudart` from the env. |
| Standard | `-std=c++17`, device LTO (`-dlto`) |
| MPS | `nvidia-cuda-mps-control` for any pipelined run. Without it provers time-slice rather than co-execute. The `run_*.sh` experiment scripts start it, except `run_experiments.sh`. |

Upstream recommended CUDA 12.1.0 in conda. Its status with respect to this miscompile is unknown.

### `ARCH` values

| `ARCH` | GPU |
|--------|-----|
| `sm_86` (default) | A6000, RTX 3090 |
| `sm_89` | RTX 4090, L40S |
| `sm_80` | A100 |
| `sm_90` | H100 |


### Make targets

`matmul-bm hadamard-bm com-bm tlookup-bm ppgen commit-param partition`, plus `all` and `clean`.

## Python setup

There is no lockfile. Install into any Python 3.11 environment, with `python` on `PATH`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch transformers numpy matplotlib
```

`acc/` also needs `datasets` and `tqdm`.

## Prepare a model

OPT is not gated, so the token argument can be a placeholder. `run_e2e.sh` and `run_sm_scaling.sh` do these steps themselves when a model is missing.

```bash
python download-models.py facebook/opt-125m x     # writes ./model-storage
python opt-commit.py 125m                         # needs a GPU; writes zkllm-workdir/opt-125m
touch zkllm-workdir/opt-125m/.complete            # see below
python opt-recommit.py 125m --dst zkllm-workdir-sqrt/opt-125m   # square-root commitments
```

| Script | Arguments |
|--------|-----------|
| `opt-commit.py` | `model_size`, `--log_scaling_factor` (16), `--log_off_factor` (5) |
| `opt-recommit.py` | `model` (one of `125m 1.3b 2.7b 6.7b 13b`), `--dst` (required), `--src`, `--log-gens-offset` (0), `--device` |
| `llama-commit.py`, `llama-ppgen.py` | LLaMa-2 7B/13B. Gated models: pass a real token to `download-models.py`. |

`opt-commit.py` does not write the `.complete` marker that the experiment scripts check, so touch it after committing by hand.

Both `model-storage` and `zkllm-workdir` may be symlinks to scratch storage. `opt-recommit.py` symlinks integer weights to absolute paths, so a sqrt workdir made on one machine is not portable.

## `opt-partition.py`

```bash
python opt-partition.py <model_size> <seq_len> [flags]
```

`seq_len` must be a power of two. Run from the repository root. The driver runs `make partition ppgen` itself on every invocation.

| Flag | Default | Meaning |
|------|---------|---------|
| `--splits` | 1 | Contiguous layer partitions; remainder goes to the first ones. Above the layer count is an error. |
| `--layers` | none | Explicit sizes, e.g. `8,8,8,8`. Overrides `--splits`. |
| `--num_layers` | all | Decoder layers to prove |
| `--devices` | `0` | Comma-separated CUDA ids; partition i uses `devices[i % n]` |
| `--concurrent` | all | Wave width, or maximum provers in pipeline mode |
| `--sequential` | off | Wave width 1. Ignored when `--pipeline` is set. |
| `--pipeline` | off | Start each prover as soon as its input exists |
| `--input_file` | `layer_input.bin` | Input activation. Random if missing; dimensions of an existing file are not checked against `seq_len`. |
| `--workdir` | `./zkllm-workdir/opt-<size>` | Committed weights. The `--help` text says `Llama-2-{N}b`, which is stale. |
| `--out-json` | `opt-partition-summary.json` | JSON summary |
| `--skip-witgen` | off | Reuse `part*_in.bin` (wave mode) |
| `--mem-per-proc-mb` | 4500 | Pipeline gate: free memory needed to launch a prover |
| `--witgen-margin-mb` | 1500 | Pipeline gate: memory kept free for the witgen chain |
| `--prover-mode` | `layer` | `layer` or `op` |
| `--early-carry` | off | One witgen process for the chain. Default comes from `ZKLLM_EARLY_CARRY=1`. |
| `--witgen-first` | off | Witness-first; needs `--splits 1 --prover-mode layer`, no `--pipeline` |

The exit status is 0 only when `status=OK`, which needs each partition OK, `chain_ok` and `carry_ok`. `cudaMalloc` is unchecked in the prover, so an out-of-memory condition can appear as a corrupted proof, not a clean error. That is why the harness measures the memory gate first.

### Output lines

| Line | Fields |
|------|--------|
| `SPLIT_SUMMARY` | `parts prep_s makespan_s wall_makespan_s e2e_s seq_sum_s witgen_first early_carry chain_ok carry_ok status`, plus `wf_*` with `--witgen-first` |
| `PIPELINE_SUMMARY` | `parts concurrent witgen_chain_s e2e_s prover_sum_s prover_max_s early_carry chain_ok carry_ok status` |
| `PARTITION` | Per process, from `partition.cu`: `load_s witgen_s input_commit_s prove_s output_commit_s open_flush_s squash_s prover_total_s wall_s alloc_peak_mb status` and others |
| `LAYER` (stderr) | `load_s compute_s prove_s wall_s gpu_used_mb` |

### Environment variables

From `getenv` and `os.environ` in the code:

| Variable | Default | Effect |
|----------|---------|--------|
| `ZKLLM_BLENDY` | on | `0` selects the legacy sumcheck |
| `ZKLLM_BLENDY_L` | 2 | Rounds per stage |
| `ZKLLM_SC_TILE_BITS` | 16 | Tile bits `t` |
| `ZKLLM_WSTREAM` | on | `0` keeps weights resident on the GPU |
| `ZKLLM_WSTREAM_CHUNK_MB` | 64 | Weight chunk size |
| `ZKLLM_ASTREAM` | on | `0` keeps activations on the GPU |
| `ZKLLM_ACT_CHUNK_MB` | 256 | Activation chunk size |
| `ZKLLM_INT_MATMUL` | on | `0` uses the field kernel everywhere |
| `ZKLLM_COMMIT_CHUNK_MB` | 256 | G1 temporaries per commit chunk |
| `ZKLLM_BATCH_OPEN` | off | `1` batches openings |
| `ZKLLM_BATCH_OPEN_MIN` | 16384 | Batching threshold |
| `ZKLLM_BATCH_OPEN_MB` | 256 | Batch cap |
| `ZKLLM_SQUASH` | off | `1` squashes lookups |
| `ZKLLM_SQUASH_MAX_LOG` | 24 | Maximum squashed elements, log2 |
| `ZKLLM_EARLY_CARRY` | off | `1` sets the default of `--early-carry` |
| `CUDA_VISIBLE_DEVICES` | | Set by the driver per partition from `--devices` |

Harness variables: `ZKMOD_PROVER_MODE`, `PIPE_SPLITS`, `WORKDIR_ROOT`, `SKIP_SEQ`, `HF_TOKEN` (`run_experiments.sh`).

### Other drivers

| Script | Purpose |
|--------|---------|
| `llama-partition.py` | LLaMa-2 counterpart of `opt-partition.py`, with fewer modes |
| `acc/` | Perplexity tests for LLaMa-2 (`acc/download-data.py`, `acc/main.py <7\|13> <seq_len> <nsamples>`) |

## `run_experiments.sh`

```bash
./run_experiments.sh [seq_len] [out_dir] [models...]    # defaults: 2048, results, 125m 1.3b 2.7b 6.7b 13b
```

For each model it downloads and commits if needed, warms the page cache, then runs:

| Tag | Command |
|-----|---------|
| `zkllm` | `--splits 1 --sequential` (layer path) |
| `zkmod2_seq`, `zkmod4_seq` | `--splits N --sequential --prover-mode $ZKMOD_PROVER_MODE` |
| `zkmod<P>_pipe` | `--splits P --pipeline`, for each P in `PIPE_SPLITS` (default `2 4 8`) |

The memory gate derives from the model's own `zkmod2_seq` run: `--mem-per-proc-mb` = peak + 1024, `--witgen-margin-mb` = witgen footprint + 1024. `SKIP_SEQ=1` skips the three baselines and expects `opt-<M>_zkmod2_seq.{mem,log}` in the output directory.

Outputs per run in `out_dir`: `opt-<M>_<tag>.{log,json,mem}`, plus one `RESULT` line per run in `summary.log`. `summary.log` is append-only across sweeps.

## Experiment scripts

Run from `zkllm-benchmark/`. All need `CUDA_HOME` pointing at CUDA 12.9 and one NVIDIA GPU, and start MPS themselves where needed. Each writes a new results directory, named with `RUN_ID` (default: a timestamp).

| Script | Purpose | Results directory | Key env |
|--------|---------|-------------------|---------|
| `run_e2e.sh [models...]` | zkLLM vs zkMod sweep with square-root commitments. Wraps `run_experiments.sh`. | `results-h100-e2e-<seq>-<id>` | `SEQ_LEN` (2048), `PIPE_SPLITS` (`4 8 12 16`), `ZKMOD_PROVER_MODE` (`op`) |
| `run_sm_scaling.sh` | Baseline with SMs capped by `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`. A probe records the SM count seen. | `results-sm-scaling-<id>` | `MODELS`, `SEQ_LENS`, `SM_COUNTS`, `REPS` |

## `./zkmod` commands

From the monorepo root:

| Command | Effect |
|---------|--------|
| `./zkmod check` | Reports toolchains, including whether `nvcc` exists |
| `./zkmod build zkllm [--arch sm_90]` | `make partition ppgen commit-param`. `ARCH` from the env also works. |
| `./zkmod prove zkllm [--seq L] [--splits "4 8"] [models...]` | Runs `run_e2e.sh`. Default seq 2048. |
| `./zkmod figures zkllm` | Runs the plot scripts in place and lists the ones that exited nonzero. |

## Quick proving-time estimates

Kept from the upstream README. Upstream measures proving time as sumcheck plus commitment opening, excluding compilation and loading. These benchmarks use random tensors of the given sizes. `tlookup-bm`, `matmul-bm` and `hadamard-bm` measure sumchecks only, and `com-bm` measures commitments. Each Python wrapper builds its target with `make`.

The bottleneck is `zkAttn` in LLaMa-2, with input size `num_layer * num_head * seq_len * seq_len`: `32 * 32 * 2048 * 2048` (7B) or `40 * 40 * 2048 * 2048` (13B). `zkAttn` involves 5 (or 2) `tlookup`s and two Hadamard products.

```bash
# Sumcheck of tlookup: 40*40 = 1600 instances, 2048*2048 = 4194304 each, range [0, 65536).
# Multiply by the number of tlookups (2 or 5).
python tlookup-bm.py 1600 4194304 0 65536    # around 80 s

# Sumcheck of Hadamard: 1600 instances of 4194304
python hadamard-bm.py 1600 4194304           # around 90 s

# Openings: 40*40*2 = 3200 instances (input and multiplicative inverse per tlookup).
# 65536 is the generator dim. Multiply by the number of tlookups.
python com-bm.py 3200 4194304 65536          # around 15 s
```

`matmul-bm.py` takes `num dim0 dim1 dim2`.

Upstream's numbers are from an A6000, which it reports as slower than the A100 used for its experiments, and it reports up to 2x fluctuation on one GPU model. The bottleneck costs `5 * 80 + 90 + 5 * 15 = 565` s or `2 * 80 + 90 + 2 * 15 = 280` s for 13B models. These upstream figures predate this fork's Blendy, streaming and commitment changes, so treat them only as an order of magnitude. `zkattn-bench.py` automates the recipe.
