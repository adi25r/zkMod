#ifndef BLENDY_CUH
#define BLENDY_CUH

// Staged ("Blendy") sumcheck with t-tiling, ported from zkML gpu_provers/blendysc.hpp and
// gpu_provers/gpu_deg3_staged.hpp.
//
// The n variables are bound in stages of l rounds. At the start of a stage every factor
// tensor is extended from {0,1}^l to {0..B-1}^l over the stage variables (B = max
// factors per term + 1), and one small product table ("ptable", B^l cells per term) is
// reduced over the remaining (future) variables. The l rounds of the stage only read and
// Lagrange-fold that ptable. The big arrays are folded once per stage, at the boundary,
// with a 2^l eq table. The extension tables are tiled over the future variables (t bits
// per tile, ZKLLM_SC_TILE_BITS), so they take B^l * 2^t cells instead of (B/2)^l * N.
//
// Unlike zkML, the eq(u, x) weight of zkLLM's sumchecks is never a factor tensor. It
// splits as eq(u_stage, x_stage) * eq(u_future, x_future). The future part weights the
// ptable reduction, and the stage part weights the round sum, so an eq-weighted product
// of d tensors keeps a degree-d table. round() returns, per group of terms, the
// coefficients of
//     R(X) = sum_{tail in {0,1}^rest} [eq(u_rest, tail)] * prod_j (F_j(X, tail) + shift_j)
// That is the current round's polynomial WITHOUT the current and previously bound eq
// factors, which is exactly what zkLLM's callers push or combine. Their transcripts are
// therefore unchanged.
//
// Bit convention: u[i] is the eq challenge for bit i of the ORIGINAL index (as in Fr_me).
// MSB_FIRST binds bit n-1 first (zkip, multi_hadamard_sumchecks, tLookup). LSB_FIRST
// binds bit 0 first (Fr_hp_sc).

#include "fr-tensor.cuh"
#include "polynomial.cuh"
#include <vector>

namespace blendy {

constexpr int MAX_TENSORS = 5;
constexpr int MAX_FACTORS = 3;
constexpr int MAX_TERMS = 4;
constexpr int MAX_GROUPS = 3;

enum class Order { MSB_FIRST, LSB_FIRST };

struct Term {
    std::vector<int> factors;      // indices into the engine's tensor list
    std::vector<Fr_t> shifts;      // additive constant per factor; empty = all zero
    bool eq_weighted;
    int group;                     // terms of one group are summed into one polynomial
};

struct TermDesc {
    int nf;
    int f[MAX_FACTORS];
    Fr_t shift[MAX_FACTORS];
    int eq;
    int group;
    int scale;                     // number of R^-1 factors the raw Montgomery products leave
};

struct Desc {
    int n_terms;
    int n_groups;
    TermDesc t[MAX_TERMS];
    int group_deg[MAX_GROUPS];
    int group_adjust[MAX_GROUPS];  // extra power of R applied to the group's output (e.g. -1)
};

struct Scalars { Fr_t v[MAX_TERMS]; };

bool enabled();                    // ZKLLM_BLENDY, default on
int default_stage_len();           // ZKLLM_BLENDY_L, default 2
int default_tile_bits();           // ZKLLM_SC_TILE_BITS, default 16; negative = untiled

class Sumcheck {
public:
    // tensors: (device pointer, length). Length may be < 2^num_vars (zero padded).
    // u: eq challenges, u[i] for original bit i; may be empty if no term is eq-weighted.
    // rounds: how many of the num_vars variables to bind (default all); the rest stay free.
    Sumcheck(const std::vector<std::pair<const Fr_t*, uint>>& tensors, const std::vector<Term>& terms,
        const std::vector<Fr_t>& u, uint num_vars, Order order, int rounds = -1,
        int l_stage = default_stage_len(), int tile_bits = default_tile_bits());
    ~Sumcheck();
    Sumcheck(const Sumcheck&) = delete;
    Sumcheck& operator=(const Sumcheck&) = delete;

    // Coefficients (low degree first) of each group's round polynomial. term_coefs, if
    // given, multiplies each term (e.g. tLookup's running alpha).
    std::vector<std::vector<Fr_t>> round(const std::vector<Fr_t>& term_coefs = {});
    void bind(const Fr_t& v);

    void set_group_adjust(int group, int adjust);
    uint rounds_left() const { return nr - done; }
    // The folded array of a tensor; valid at a stage boundary (in particular after the last round).
    const Fr_t* data(int tensor) const;
    uint size() const { return 1u << (n - done); }
    Fr_t value(int tensor) const;  // element 0 of the folded array, after the last round

private:
    void begin_stage();
    void end_stage();
    uint bit_of_round(uint r) const { return order == Order::MSB_FIRST ? n - 1 - r : r; }

    uint n, nr, done = 0;
    Order order;
    int L, T, B;
    std::vector<Fr_t> u, stage_v;
    Desc desc;
    int n_tensors;
    std::vector<const Fr_t*> src;      // current arrays
    std::vector<uint> src_size;
    std::vector<Fr_t*> buf[2];         // fold destinations, N>>L and N>>2L
    int next_buf = 0;
    Fr_t *ext[MAX_TENSORS] = {};
    Fr_t *ptable = nullptr, *partials = nullptr, *eq_tab = nullptr, *u_dev = nullptr, *coef_out = nullptr;
    long ext_cap = 0, partial_cap = 0, eq_cap = 0;
    // stage state
    bool in_stage = false;
    uint stage_first = 0;              // global index of the stage's first round
    int l_cur = 0, rem = 0;
};

}  // namespace blendy

#endif
