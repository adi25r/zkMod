#ifndef BLENDY_CUH
#define BLENDY_CUH

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
    std::vector<int> factors;  // indices into the engine's tensor list
    std::vector<Fr_t> shifts;  // additive constant per factor
    bool eq_weighted;
    int group;  // terms of one group are summed into one polynomial
};

struct TermDesc {
    int nf;
    int f[MAX_FACTORS];
    Fr_t shift[MAX_FACTORS];
    int eq;
    int group;
    int scale;  // number of R^-1 factors the raw Montgomery products leave
};

struct Desc {
    int n_terms;
    int n_groups;
    TermDesc t[MAX_TERMS];
    int group_deg[MAX_GROUPS];
    int group_adjust[MAX_GROUPS];  // extra power of R applied to the group's output (e.g. -1)
};

struct Scalars {
    Fr_t v[MAX_TERMS];
};

bool enabled();
int default_stage_len();  // ZKLLM_BLENDY_L, default 2
int default_tile_bits();  // ZKLLM_SC_TILE_BITS, default 16

class Sumcheck {
public:
    // tensors: (device pointer, length)
    Sumcheck(const std::vector<std::pair<const Fr_t*, uint>>& tensors, const std::vector<Term>& terms,
             const std::vector<Fr_t>& u, uint num_vars, Order order, int rounds = -1,
             int l_stage = default_stage_len(), int tile_bits = default_tile_bits());
    ~Sumcheck();
    Sumcheck(const Sumcheck&) = delete;
    Sumcheck& operator=(const Sumcheck&) = delete;

    // Coefficients
    std::vector<std::vector<Fr_t>> round(const std::vector<Fr_t>& term_coefs = {});
    void bind(const Fr_t& v);

    void set_group_adjust(int group, int adjust);
    uint rounds_left() const { return nr - done; }
    // The folded array of a tensor
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
    std::vector<const Fr_t*> src;  // current arrays
    std::vector<uint> src_size;
    std::vector<Fr_t*> buf[2];  // fold destinations, N>>L and N>>2L
    int next_buf = 0;
    Fr_t* ext[MAX_TENSORS] = {};
    Fr_t *ptable = nullptr, *partials = nullptr, *eq_tab = nullptr, *u_dev = nullptr, *coef_out = nullptr;
    long ext_cap = 0, partial_cap = 0, eq_cap = 0;
    // stage state
    bool in_stage = false;
    uint stage_first = 0;  // global index of the stage's first round
    int l_cur = 0, rem = 0;
};

}  // namespace blendy

#endif
