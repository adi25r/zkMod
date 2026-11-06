#include "common.hpp"
#include <random>

static std::mt19937_64 g_rng(std::random_device{}());

void seed_rng(uint64_t seed) {
    g_rng.seed(seed);
}

FieldElement get_random_element() {
    static std::uniform_int_distribution<uint64_t> dist64(0, std::numeric_limits<uint64_t>::max());
    uint64_t limbs[4] = { dist64(g_rng), dist64(g_rng), dist64(g_rng), dist64(g_rng) };

    FieldElement res;
    res.setArrayMask(limbs, sizeof(limbs)); // reduces mod r internally
    return res;
}
// langrange coefficients
std::vector<FieldElement> build_eq_table(const std::vector<FieldElement>& r) {
    int n = r.size();
    int table_size = 1 << n;
    std::vector<FieldElement> eq_table(table_size, 1);
    const FieldElement one = FieldElement(1);
    for (int i = 0; i < n; ++i) {
        int stride = 1 << i;
        for (int j = 0; j < stride; ++j) {
            FieldElement v = eq_table[j];
            FieldElement r_comp = one;
            r_comp -= r[i];
            eq_table[j] = v * r_comp;
            eq_table[j + stride] = v * r[i];
        }
    }
    return eq_table;
}
