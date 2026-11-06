"""Analytic proof size for the zkLLM figures, counted from the layer dimensions."""
FR = 32          # BLS12-381 scalar
G1 = 48          # compressed BLS12-381 G1 point

# OPT: (embed_dim, ffn_dim, layers)
OPT = {"125m": (768, 3072, 12), "350m": (1024, 4096, 24), "1.3b": (2048, 8192, 24), "2.7b": (2560, 10240, 32),
       "6.7b": (4096, 16384, 32), "13b": (5120, 20480, 40)}


def lg(n):
    return (n - 1).bit_length()


def p2(n):
    return 1 << lg(n)


def zkip(n_rounds, B=1):                  # inner-product sumcheck: deg-2 rounds, a(v), b(v)
    return (3 * (n_rounds + lg(B)) + 2) * FR


def opening(gens):                        # me_open of one committed vector/row block, plus the opened value
    return (3 * lg(gens) + 1) * G1 + FR


def lookup(D, table, mapped, B=1):        # tLookup / tLookupRangeMapping: deg-3 rounds, then A, S(, S_out), m
    return (4 * lg(max(p2(D) * p2(B), table)) + (4 if mapped else 3)) * FR


def rescale(D, bits, B=1):                # Rescaling: X(u) = X_(u) s + rem(u), then the rem range lookup
    return 3 * FR + lookup(D, 1 << bits, False, B)


def bias(dim):                            # check_add + open_vector
    return 3 * FR + opening(p2(dim))


def linear(s, d_in, d_out, bits, B=1):    # opt_linear: zkFC + weight opening, bias, rescale
    return zkip(lg(d_in), B) + opening(p2(d_out << 5)) + bias(d_out) + rescale(s * d_out, bits, B)


def layernorm(s, E, B=1):                 # opt_layernorm -> llama_rmsnorm: g fc, 2 rescales, hadamard, bias
    return (zkip(0, B) + opening(p2(E)) + 2 * rescale(s * E, 16, B)
            + (3 * lg(s * E * p2(B)) + 2) * FR + bias(E))


def softmax(s, B=1):                      # zkSoftmax(bs = 2^8, 2^20, 2^20, 2^15; L = 1): 3-segment product, 4 lookups
    N = s * s
    return ((5 * lg(N * p2(B)) + 3) * FR + lookup(N, 1 << 8, False, B)
            + sum(lookup(N, t, True, B) for t in (1 << 20, 1 << 20, 1 << 15)))


def layer(s, E, H, B=1):                  # one layer, or with B > 1 one proof covering B stacked layers
    attn = (3 * linear(s, E, E, 16, B) + zkip(lg(E), B) + softmax(s, B) + zkip(lg(s), B)
            + rescale(s * E, 20, B) + rescale(s * E, 20, B))
    ffn = linear(s, E, H, 20, B) + lookup(s * H, 1 << 22, True, B) + linear(s, H, E, 16, B)
    return 2 * layernorm(s, E, B) + attn + ffn


def boundary_commit(s, E, point=G1):      # one zkMod cut: commitment to x, the intermediate value live there
    return p2(s * E) // p2(E) * point


def boundary(s, E):                       # one zkMod cut: the producer's and the consumer's opening of it
    return 2 * opening(p2(E))


def commitment_bytes(model, seq=64, partitions=1, point=G1):
    """Weight commitments (opt-commit.py), published once and the same for zkMod, plus zkMod's commitment to
    the intermediate value at each of its partitions - 1 cuts. Each (out, in) weight is committed transposed
    as pow2(in) rows over pow2(out << 5) generators, i.e. pow2(in) / 32 points; each of the 10 bias /
    layernorm vectors is one point. out_proj is committed too. opt-commit.py writes Jacobian points
    (144 B); pass point=144 to match the files on disk."""
    E, H, L = OPT[model]
    points = 5 * p2(E) // 32 + p2(H) // 32 + 10   # q, k, v, out_proj, fc1 take E inputs; fc2 takes H
    return L * points * point + (partitions - 1) * boundary_commit(seq, E, point)


def proof_bytes(model, seq=64, partitions=1, batched=False):
    E, H, L = OPT[model]
    if batched:                           # each partition batches its own share of the layers
        per = [L // partitions + (i < L % partitions) for i in range(partitions)]
        return sum(layer(seq, E, H, b) for b in per) + (partitions - 1) * boundary(seq, E)
    return L * layer(seq, E, H) + (partitions - 1) * boundary(seq, E)


if __name__ == "__main__":
    for batched in (False, True):
        print("batched" if batched else "per-layer")
        for m in OPT:
            print(" ", m, " ".join(f"P={p}:{proof_bytes(m, 64, p, batched) / 1024:.1f}KB" for p in (1, 4, 8, 12)))
