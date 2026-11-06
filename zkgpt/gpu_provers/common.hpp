#ifndef COMMON_HPP
#define COMMON_HPP

#include <vector>
#include <random>
#include <mcl/bn.hpp>

typedef mcl::bn::Fr FieldElement;

#ifndef TO_FE_DEFINED
#define TO_FE_DEFINED
inline FieldElement to_FE(uint32_t val) {
    FieldElement r;
    r = val;
    return r;
}
#endif

FieldElement get_random_element();
void seed_rng(uint64_t seed);

std::vector<FieldElement> build_eq_table(const std::vector<FieldElement>& r);

inline FieldElement mle_eval(const std::vector<FieldElement>& v, const std::vector<FieldElement>& r) {
    auto eq = build_eq_table(r);
    FieldElement acc;
    acc.clear();
    for (int i = 0; i < (int)v.size(); ++i)
        acc += eq[i] * v[i];
    return acc;
}

#endif  // COMMON_HPP
