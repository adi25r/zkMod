"""Analytic zkLLM / zkMod proof size, from the layer dimensions alone (as zkGPT's proof_size counter).

The prover here checks each sub-proof inline and never serialises one, so the size is counted from the
protocol instead: every message the prover would send for run_opt_layer (opt-layer.cu), op by op.

  sumcheck round        one polynomial: degree+1 Fr        zkip/hadamard deg 2, tLookup deg 3,
                                                            softmax multi-hadamard (3 segments) deg 4
  sumcheck end          the final evaluations of its operands (the verifier's oracle claims)
  tLookup on D values   log2(max(pow2(D), table)) rounds    (tlookup.cu pads D up to the table)
  weight opening        me_open over log2(G) rounds: 3 G1 each + 1 final G1, G = generator count
                        (opt-commit.py: G = pow2(out_features << 5) for a matrix, pow2(dim) for a vector).
                        Weight commitments are fixed per model and held by the verifier: not counted.
  check_add             Z(u), Y(u), B(u)
  zkMod boundary        partition i < P-1 sends its output commitment (pow2(seq) row commitments over
                        pow2(embed) generators, partition.cu) and opens it; partition i+1 opens the same
                        commitment at its input claim.

Not counted, as the code does not produce them: commitments to intermediate activations, lookup
multiplicities and inverse vectors (zkLLM's paper commits these). squash and opening batching are off
in the e2e runs (squashed_lookups = openings_batched = 0), so every lookup and opening is separate.
"""
FR = 32          # BLS12-381 scalar
G1 = 48          # compressed BLS12-381 G1 point

# OPT: (embed_dim, ffn_dim, layers)
OPT = {"125m": (768, 3072, 12), "1.3b": (2048, 8192, 24), "2.7b": (2560, 10240, 32),
       "6.7b": (4096, 16384, 32), "13b": (5120, 20480, 40)}


def lg(n):
    return (n - 1).bit_length()


def p2(n):
    return 1 << lg(n)


def zkip(n_rounds):                       # inner-product sumcheck: deg-2 rounds, a(v), b(v)
    return (3 * n_rounds + 2) * FR


def opening(gens):                        # me_open of one committed vector/row block, plus the opened value
    return (3 * lg(gens) + 1) * G1 + FR


def lookup(D, table, mapped):             # tLookup / tLookupRangeMapping: deg-3 rounds, then A, S(, S_out), m
    return (4 * lg(max(p2(D), table)) + (4 if mapped else 3)) * FR


def rescale(D, bits):                     # Rescaling: X(u) = X_(u) s + rem(u), then the rem range lookup
    return 3 * FR + lookup(D, 1 << bits, False)


def bias(dim):                            # check_add + open_vector
    return 3 * FR + opening(p2(dim))


def linear(s, d_in, d_out, bits):         # opt_linear: zkFC + weight opening, bias, rescale
    return zkip(lg(d_in)) + opening(p2(d_out << 5)) + bias(d_out) + rescale(s * d_out, bits)


def layernorm(s, E):                      # opt_layernorm -> llama_rmsnorm: g fc, 2 rescales, hadamard, bias
    return (zkip(0) + opening(p2(E)) + 2 * rescale(s * E, 16)
            + (3 * lg(s * E) + 2) * FR + bias(E))


def softmax(s):                           # zkSoftmax(bs = 2^8, 2^20, 2^20, 2^15; L = 1): 3-segment product, 4 lookups
    N = s * s
    return ((5 * lg(N) + 3) * FR + lookup(N, 1 << 8, False)
            + sum(lookup(N, t, True) for t in (1 << 20, 1 << 20, 1 << 15)))


def layer(s, E, H):
    attn = (3 * linear(s, E, E, 16) + zkip(lg(E)) + softmax(s) + zkip(lg(s))
            + rescale(s * E, 20) + rescale(s * E, 20))
    ffn = linear(s, E, H, 20) + lookup(s * H, 1 << 22, True) + linear(s, H, E, 16)
    return 2 * layernorm(s, E) + attn + ffn


def boundary(s, E):                       # one zkMod cut: output commitment + its two openings
    return p2(s) * G1 + 2 * opening(p2(E))


def proof_bytes(model, seq=64, partitions=1):
    E, H, L = OPT[model]
    return L * layer(seq, E, H) + (partitions - 1) * boundary(seq, E)


if __name__ == "__main__":
    for m in OPT:
        print(m, " ".join(f"P={p}:{proof_bytes(m, 64, p) / 1024:.1f}KB" for p in (1, 4, 8, 12)))
