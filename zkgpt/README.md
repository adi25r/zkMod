# zkgpt

A GPU prover for transformer inference, derived from zkGPT: GKR sumcheck, a chunked-lookup (logup) range check and the Hyrax polynomial commitment, with the expensive phases in HIP kernels. `bin/main_gkr_split` adds zkMod partitioning, which proves a model as a chain of independently proved blocks tied together by a boundary link, with witness generation overlapping proof generation.

This is a research implementation. Its verifier challenges are a Fiat-Shamir placeholder, and other soundness caveats apply. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Running

Use the driver from the repository root. See the [top-level README](../README.md).

## Layout

| Path | Contents |
|---|---|
| `src/` | Prover, verifier, circuit, entry points (`main_gkr_split.cpp`, `main_gkr.cpp`, `main_demo_llm_split.cpp`) |
| `gpu_provers/` | Sumcheck, range and Hyrax kernels, field bridge |
| `witgen_gpu/` | GPU witness generation |
| `tools/` | Experiment queues (`tools/results/figures/`), sweep drivers, plotting |
| `docs/` | [Architecture](docs/ARCHITECTURE.md), [build and run reference](docs/BUILD_AND_RUN.md) |

`../mcl` and `../zkmod-gpu` are required siblings. CMake resolves them by relative path.
