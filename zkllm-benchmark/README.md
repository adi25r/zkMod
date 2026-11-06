# zkllm-benchmark

The NVIDIA/CUDA implementation in the zkMod monorepo: zkLLM plus zkMod partitioning, proving OPT models on one GPU. The sibling AMD/HIP GKR prover is `../zkgpt`. The monorepo driver is [`../zkmod`](../zkmod).

## Running

Use the driver from the repository root. See the [top-level README](../README.md).

## Layout

| Path | Contents |
|---|---|
| `*.cu`, `*.cuh` | Prover sources |
| `opt-partition.py` | Runs one partitioned proof |
| `opt-commit.py`, `opt-recommit.py` | Commit model weights |
| `run_e2e.sh`, `run_experiments.sh`, `run_sm_scaling.sh` | Experiment scripts |
| `plots/` | Plot scripts |
| `docs/` | [Architecture](docs/ARCHITECTURE.md), [build and run reference](docs/BUILD_AND_RUN.md) |

## Accuracy

`./acc` measures perplexity. Upstream reduced the number of bits to use fewer `tlookup`s in `zkAttn`, following feedback that accuracy preservation was overemphasised.

## Origin and authorship

zkLLM is the work of Haochen Sun, Jason Li and Hongyang Zhang (University of Waterloo): *zkLLM: Zero Knowledge Proofs for Large Language Models*, ACM CCS 2024, [arXiv:2404.16109](https://arxiv.org/abs/2404.16109). Upstream's code is under the MIT License, Copyright (c) 2024 Haochen Sun, Jason Li, and Hongyang Zhang.

This directory derives from the benchmark fork `adi25r/zkllm-benchmark`, which adds zkMod partitioning and the changes listed in the architecture document. Upstream warns that it has not undergone security auditing and is not ready for industrial applications; the same applies here.
