# zkMod

[![AI-DECLARATION: copilot](https://img.shields.io/badge/䷼%20AI--DECLARATION-copilot-fee2e2?labelColor=fee2e2)](https://ai-declaration.md)

zkMod splits a transformer into partitions and proves them concurrently, with witness generation overlapping proving. This artifact contains the two GPU provers it is evaluated on, and one driver, `./zkmod`, to build them, run the experiments and plot the figures.

## Repository structure

```
zkmod              driver: build, prove and plot
zkgpt/             GKR prover for AMD GPUs (HIP, ROCm)
  src/               prover, verifier, circuit, entry points
  gpu_provers/       sumcheck, range and Hyrax kernels
  witgen_gpu/        GPU witness generation
  tools/             experiment queues and plotting
  docs/              architecture, CLI and environment reference
zkllm-benchmark/   per-operation prover for NVIDIA GPUs (CUDA)
  *.cu, *.cuh        prover sources
  opt-partition.py   runs one partitioned proof
  run_*.sh           experiment scripts
  plots/             plotting
  docs/              architecture, CLI and environment reference
mcl/, zkmod-gpu/   zkgpt dependencies, fetched by ./zkmod deps at pinned commits
```

| | `zkgpt/` | `zkllm-benchmark/` |
|---|---|---|
| Proof system | zkGPT: one GKR circuit, logup range check, Hyrax commitments | zkLLM: separate per-operation provers, tlookup, no GKR circuit |
| GPU | AMD MI210 | NVIDIA H100 |
| Toolchain | ROCm 7 at `/opt/rocm-scale` | CUDA 12.9 at `CUDA_HOME` |
| Field | BN254 | BLS12-381 |
| Models | gpt2, opt350m, opt1b3, opt2b7 | opt-125m, 1.3b, 2.7b, 6.7b, 13b |

The provers share no code. Each compares zkMod against its own monolithic baseline.

## Getting started

```bash
./zkmod check          # toolchains, GPUs and what is built
./zkmod all            # zkgpt: dependencies, build, one gpt2 proof and a figure (about 3 min)
```

## Running the zkgpt experiments

Needs an AMD MI210 (64 GiB) and about 700 GiB of host memory; the evaluation machine has 2x EPYC 9654.

```bash
./zkmod deps
./zkmod build zkgpt

./zkmod prove zkgpt --model gpt2 --mode gpu       # one run
./zkmod figures zkgpt --full                      # every run behind the figures, then plot
./zkmod figures zkgpt --plot --logs DIR --out DIR # plot existing logs only
```

`--full` takes about 8 hours of proving, plus any time spent waiting for memory, so run it under `screen` or `nohup`. It skips runs whose log already shows a verified proof, so it can be restarted. Logs go to `zkgpt/tools/results/figures/logs/` and figures to `zkgpt/tools/results/figures/figs/`.

Each run is one of three modes, chosen with `--mode`:

| Mode | Runs |
|---|---|
| `mono` | Monolithic CPU baseline |
| `cpu` | zkMod partitions, CPU witness generation and CPU proving |
| `gpu` | zkMod partitions, GPU witness generation and GPU proving |

To regenerate one figure, run its queue from `zkgpt/`, then plot it with `python3 tools/plot_figures.py <figure>`:

| Figure | Shows | Queue in `tools/results/figures/queue/` |
|---|---|---|
| `zkgptbaselineprofile` | CPU monolith latency against host threads | `run_zkgptbaselineprofile.sh` |
| `seqlen64allcpu` | Monolith against `cpu`, all models at length 64 | `run_seqlen64.sh` |
| `seqlen64allgpu` | Monolith against `gpu`, all models at length 64 | `run_seqlen64.sh` |
| `cpuhostthreadscaling` | `cpu` speedup over the monolith against host threads | `run_zkgptbaselineprofile.sh`, `run_cpuhostthreadscaling.sh` |
| `zkgptseqlenscaling` | `gpu` speedup at lengths 64, 128 and 256 | `run_seqlen64.sh`, `run_zkgpt_scaling.sh` |
| `zkgptpartitionscaling` | `gpu` speedup against partition count | `run_zkgpt_scaling.sh` |
| `zkgpt_proof_overheads` | Table: proof size, transmission and verifier time | `run_seqlen64.sh` |

A run counts as valid only if every partition logs `All verification passed`. The reported time is `overlapped_total_s`, and gpt2 always uses length 64.

## Running the zkllm experiments

Needs an NVIDIA H100 and up to 512 GiB of host memory.

```bash
export CUDA_HOME=/path/to/cuda-12.9
python -m venv .venv && source .venv/bin/activate
pip install torch transformers numpy matplotlib

./zkmod build zkllm --arch sm_90
./zkmod prove zkllm --seq 64 125m 1.3b 2.7b 6.7b 13b
./zkmod figures zkllm
```

Models are downloaded and committed on first use. Run these from `zkllm-benchmark/` to produce every figure's data; `./zkmod figures zkllm` then writes the figures to `plots/figs/`:

| Figure | Command |
|---|---|
| `zkllm_seq64`, `zkllm_partitions_scaling`, `zkllm_proof_overheads` | `SEQ_LEN=64 ./run_e2e.sh` |
| `zkllm_seqlen_scaling` | The above, then `SEQ_LEN=128` and `SEQ_LEN=256` with `PIPE_SPLITS="8 16"` |
| `sm_scaling_64` | `SEQ_LENS=64 ./run_sm_scaling.sh` |

## More detail

`docs/ARCHITECTURE.md` and `docs/BUILD_AND_RUN.md` in each prover cover the implementation, command-line flags, environment variables and log format.

## AI disclosure

This project declares AI involvement at level `copilot`. See [AI-DECLARATION.md](AI-DECLARATION.md).
