# Architecture

## What zkLLM is

zkLLM (Sun, Li, Zhang; ACM CCS 2024) proves LLM inference with CUDA-native tensor proofs. There is no circuit and no GKR: no source file here builds a layered circuit or mentions GKR. Each tensor operation calls its own sumcheck, and a layer's proof is the sequence of those calls.

| Building block | Statement | Code |
|----------------|-----------|------|
| Matmul | Matrix product reduced to an inner product over the row variables | `zkip`, `zkip_stacked`, `zkFC::prove` in `zkfc.cu`; `inner_product_sumcheck` in `proof.cu` |
| Hadamard | Element-wise product, 2 factors or several | `hadamard_product_sumcheck`, `Fr_hp_sc`, `multi_hadamard_sumchecks` in `proof.cu` |
| tlookup | Lookup argument with multiplicities `m`, challenges `alpha`, `beta`, and an inverse vector. It is LogUp-style. | `tLookup`, `tLookupRange`, `tLookupRangeMapping` in `tlookup.cu` |
| Rescaling | `X = X_ * s + rem`, with `rem` range-checked by tlookup | `rescaling.cu` |
| zkSoftmax / zkAttn | Digit-decomposes the shifted input, looks up each segment's `exp` value, and multiplies segments by Hadamard | `zksoftmax.cu` |
| Commitments | Pedersen over BLS12-381 with row commitments, opened by folding (`Commitment::me_open`, `open_check`) | `commitment.cu`, `g1-tensor.cu`, `bls12-381.cu` |

`tLookupRangeMapping` serves both purposes of the lookup: a range check, and a function evaluation (index to precomputed value). Lookups are proved in the same run as the operator they constrain. In OPT, ReLU uses this mapping with `relu-table.bin`; `zkReLU::prove` in `zkrelu.cu` is an empty stub.

### Limits of the statement

These are properties of the code, and they affect how any result may be described.

- **No proof object or separate verifier.** The prover checks each sub-proof inline and never serialises one (`plots/proof_size.py` docstring). Upstream describes zkLLM as an interactive proof with prover and verifier work implemented side by side.
- **No Fiat-Shamir.** Challenges come from `random_vec` in `fr-tensor.cu` (`std::random_device` seeding `mt19937`), not from a transcript.
- **Proof size is analytic.** `plots/proof_size.py` counts messages from layer dimensions. Commitments to intermediate activations, lookup multiplicities and inverse vectors are not counted because the code does not produce them; the zkLLM paper commits them.
- **Unconstrained host-side or float values.** The softmax shift comes from a double-precision kernel (`zksoftmax_shift`), and `1/rms` from `llama_rms_inv`. Only downstream use is checked. A project note from 2026-09-17 says RoPE in the LLaMa path is likewise unconstrained; that was not rechecked here.
- **OPT statement.** One pre-LN decoder layer, attention single-head over `embed_dim`, `out_proj` not proved (`opt-layer.cuh`). Post-LN `opt-350m` is rejected by `opt-commit.py` and `opt-partition.py`.
- **Random inputs.** `opt-partition.py` writes a random activation when the input file is missing, and that is what the sweeps prove. There is no tokenised prompt.

### Upstream layout and what remains of it

Upstream built `main`, `ppgen`, `commit-param`, `self-attn`, `ffn`, `rmsnorm`, `skip-connection` and `partition`, passing plaintext `.bin` files between stages. The sources `self-attn.cu`, `ffn.cu`, `rmsnorm.cu`, `skip-connection.cu` and `main.cu` remain here, but they are no longer Makefile targets. See [BUILD_AND_RUN.md](BUILD_AND_RUN.md#known-build-problems).

## What this fork adds

| Feature | What it does | Where | Switch (default) |
|---------|--------------|-------|------------------|
| zkMod partitioning | Splits the model into contiguous layer partitions, each proved by its own `partition` process. A partition commits its output, and the next opens the same commitment at its input claim. The driver compares commitment bytes (`chain_ok`). | `partition.cu`, `opt-partition.py` | `--splits` (1) |
| Wave proving | Runs a serial witgen chain first, then proves in waves of W processes. | `opt-partition.py` | `--sequential`, `--concurrent` |
| Pipelined proving | Witgen runs in its own thread, and partition i's prover launches when its input exists and the GPU memory gate allows. | `opt-partition.py` | `--pipeline` (off) |
| Early carry | One witgen process for the whole chain publishes `carry_<cut>` with a `.ready` marker as soon as each input is final. Each carry is checked byte for byte against the producing prover's output (`carry_ok`). | `partition.cu` `--dump-cuts`, `opt-partition.py` | `--early-carry` or `ZKLLM_EARLY_CARRY=1` (off) |
| Witness-first | zkLLM's own method. A record pass stores every tensor a proof needs in host memory, then a replay pass proves from the stored copies. Needs `--splits 1`, the layer prover and no `--pipeline`. | `witness-store.cu`, `partition.cu` | `--witgen-first` (off) |
| Blendy tiling | Staged sumcheck with tiled extension tables. Variables are bound in stages of l rounds; tables take `B^l * 2^t` cells instead of `(B/2)^l * N`. Ported from `gpu_provers/blendysc.hpp` in the zkgpt lineage. The transcript is unchanged. | `blendy.cu`, `blendy.cuh` | `ZKLLM_BLENDY` (on), `ZKLLM_BLENDY_L` (2), `ZKLLM_SC_TILE_BITS` (16) |
| Weight streaming | Keeps int32 weights in host memory and copies them to the GPU in row chunks for each pass. Results are exact field sums, bit-identical to the resident path. | `weight-stream.cu` | `ZKLLM_WSTREAM` (on), `ZKLLM_WSTREAM_CHUNK_MB` (64) |
| Activation streaming | Op-state tensors live in pinned host buffers between uses (pool cache capped at 2 GiB). | `opt-layer.cu`, `partition.cu` | `ZKLLM_ASTREAM` (on), `ZKLLM_ACT_CHUNK_MB` (256) |
| Integer matmul | Forward matmuls run in int64 when `max\|A\| * max\|B\| * K < 2^63`, and map back to the same field element. | `int-matmul.cu` | `ZKLLM_INT_MATMUL` (on) |
| Square-root commitments | Re-commits each parameter with `2^(floor(log2 N / 2) + offset)` generators (Hyrax shape), which reduces opening group work. Integer weights are symlinked, not copied. | `opt-recommit.py` | `--log-gens-offset` (0) |
| Op prover | Partitions at tensor-operation cuts: 31 ops per OPT layer (`OPT_OPS_PER_LAYER`). Each live boundary tensor is committed and opened by name. | `partition.cu --op-range`, `OptOpRunner` in `opt-layer.cu` | `--prover-mode op` |
| Batched openings | Batches qualifying openings, flushed at the end of a partition. | `commitment.cu` | `ZKLLM_BATCH_OPEN` (off), `ZKLLM_BATCH_OPEN_MIN` (16384), `ZKLLM_BATCH_OPEN_MB` (256) |
| Lookup squashing | Proves same-table lookups as one LogUp over concatenated inputs. | `squash.cu` | `ZKLLM_SQUASH` (off), `ZKLLM_SQUASH_MAX_LOG` (24) |

### Op prover and layer prover

`opt-partition.py` defaults to `--prover-mode layer`. `run_experiments.sh` and the sweep scripts pass `ZKMOD_PROVER_MODE` (default `op`) to the partitioned runs, and always run the monolithic baseline on the layer path. The driver describes the op path as giving the same proofs and boundary commitments in about a quarter of the GPU memory. Witness-first exists only on the layer path, because `partition.cu` rejects `--witgen-first` with `--op-range`.

`--splits` always cuts at layers. `op-partition.py` is a separate driver that partitions at K operations per partition for what one measured sweep showed.

### Boundary files

Run from the repository root, since these paths are relative and not namespaced per run. Two drivers in one directory overwrite each other.

| File | Meaning |
|------|---------|
| `act-pp.bin` | Shared generators for boundary commitments. Must not be regenerated between witgen and prove. Regenerated when its size differs from `pow2(embed_dim) * 144`. |
| `relu-table.bin` | ReLU mapping table |
| `part<i>_in.bin`, `carry_<cut>` | Partition input from the witgen chain (normal, early carry) |
| `part<i>_prove_out.bin`, `final_out.bin` | Prover's recomputed output; final witgen output |
| `part<i>-{in,out}-commitment.bin` | Boundary commitments (layer path) |
| `part<i>-{in,out}-x-commitment.bin` | Boundary commitments (op path) |

## Source groups

| Group | Files | Role |
|-------|-------|------|
| Field and curve | `bls12-381.cu/.cuh`, `fr-tensor.cu/.cuh`, `g1-tensor.cu/.cuh`, `ioutils.cu` | Fr/Fp arithmetic and field multiply selection; tensors on the GPU; file I/O; device allocation stats |
| Commitments | `commitment.cu/.cuh`, `commit-param.cu`, `ppgen.cu` | Commit and open; weight commit tool; generator tool |
| Sumchecks | `proof.cu/.cuh`, `polynomial.cu/.cuh`, `blendy.cu/.cuh` | Original sumchecks, polynomial helper, staged engine |
| Operators | `zkfc.cu`, `int-matmul.cu`, `tlookup.cu`, `zksoftmax.cu`, `rescaling.cu`, `zkrelu.cu`, `squash.cu` | Tensor operators and their proofs |
| Model layers | `llama-layer.cu`, `opt-layer.cu`, `weight-stream.cu`, `witness-store.cu` | Per-layer statement (`run_opt_layer`), op state (`opt_live_at`), streaming, witness store |
| Entry point | `partition.cu` | Single process for one partition: `witgen` or `prove` |
| Upstream per-op programs | `rmsnorm.cu`, `self-attn.cu`, `ffn.cu`, `skip-connection.cu`, `main.cu` | Original LLaMa pipeline, no longer in `TARGETS` |
| Micro-benchmarks | `tlookup-bm.cu`, `matmul-bm.cu`, `hadamard-bm.cu`, `com-bm.cu`, `bench-mul.cu` | Per-operation timing and multiply throughput |
| Tests | `test-prims.cu`, `test-blendy.cu`, `test-blendy-diag.cu`, `check_prims.py` | Primitive checks against Python bignums; staged vs legacy sumcheck |
| Timer | `timer.cpp`, `timer.hpp` | Wall-clock timer used by all targets |
| Compatibility | `zkml_nvcc_compat.hpp` | Force-included when the other prover is built with nvcc. Not used by zkLLM. |

## How it differs from the zkgpt prover

`../zkgpt-gpu-rebase-zkmod` is the sibling implementation. Its numbers are not directly comparable with these: the models, circuits, proof systems and GPUs all differ.

| | zkllm-benchmark | zkgpt-gpu-rebase-zkmod |
|---|-----------------|------------------------|
| Proof model | Independent per-operation sumchecks, no circuit | Layered circuit proved by GKR sumcheck |
| Commitment | Pedersen over BLS12-381, row commitments opened by folding | Hyrax over BN254 via mcl |
| Lookup | tlookup fused into the operator proof | Separate chunked-lookup range proof |
| Models | OPT (LLaMa path also present) | GPT-2 and OPT |
| GPU API | CUDA | HIP (AMD) |
| Default metric | Driver `e2e_s` | `overlapped_total_s` |
| Partition link | Byte-compared boundary commitments, each opened at a claim | See that repository's docs |

The project note of 2026-09-17 says both provers mislabel their lookup as Lasso. That is wrong for zkLLM: no source file in this tree or in the upstream checkout mentions Lasso.
