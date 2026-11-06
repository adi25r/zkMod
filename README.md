# zkMod

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

./zkmod prove zkgpt --model gpt2 --arm 4          # one run
./zkmod figures zkgpt --full                      # every run behind the figures, then plot
./zkmod figures zkgpt --plot --logs DIR --out DIR # plot existing logs only
```

`--full` takes at least 13 hours, so run it under `screen` or `nohup`. It skips runs whose log already shows a verified proof, so it can be restarted. Logs go to `zkgpt/tools/results/figures/logs/` and figures to `zkgpt/tools/results/figures/figs/`.

Arms select where each step runs. Arm 0 is the monolithic CPU baseline; arms 1-4 are partitioned:

| Arm | Witness generation | Proving |
|---|---|---|
| 1 | CPU | CPU |
| 2 | CPU | GPU |
| 3 | GPU | CPU |
| 4 | GPU | GPU |

To run one figure's experiments, call its queue from `zkgpt/`, then plot it with `python3 tools/plot_figures.py <key>`:

| Figure | Plot key | Queue in `tools/results/figures/queue/` |
|---|---|---|
| Baseline breakdown | `3` | `run_fig3.sh` |
| End-to-end time | `5`, `5ab` | `run_fig5_fig6.sh` |
| Thread scaling | `5c`, `threads` | `run_fig3.sh`, `run_fig5_threads_gated.sh` |
| Memory | `6` | `run_fig5_fig6.sh`, `run_fig7_fig8.sh` |
| Sequence length | `7`, `7b` | `run_fig5_fig6.sh`, `run_fig7_fig8.sh` |
| Partition count | `8`, `partitions` | `run_fig7_fig8.sh` |
| Timing breakdown | `flame` | `run_fig5_fig6.sh` |

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

Models are downloaded and committed on first use. Run these from `zkllm-benchmark/` for the remaining experiments:

| Experiment | Command | Plot script in `plots/` |
|---|---|---|
| zkLLM vs zkMod, by sequence length | `SEQ_LEN=64 ./run_e2e.sh`, then `SEQ_LEN=128` and `256` with `PIPE_SPLITS="8 16"` | `plot_paper.py` |
| GPU core scaling | `./run_sm_scaling.sh` | `plot_sm_scaling.py` |
| Witness-first baseline | `./run_witgen_first.sh` | `plot_witgen_first.py`, `plot_h100_e2e_wf.py` |
| Early carry | `MODEL=2.7b ./run_early_carry_ab.sh` | `plot_early_carry.py` |

## More detail

`docs/ARCHITECTURE.md` and `docs/BUILD_AND_RUN.md` in each prover cover the implementation, command-line flags, environment variables and log format.
