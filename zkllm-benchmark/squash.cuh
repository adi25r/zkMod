#ifndef SQUASH_CUH
#define SQUASH_CUH

// Circuit squashing for lookups

#include "fr-tensor.cuh"
#include "tlookup.cuh"

namespace squash {

bool enabled();  // ZKLLM_SQUASH
void begin();    // starts a squash scope (no-op unless enabled)
bool active();

// Rescaling remainder `rem`
void add_range(uint scaling_factor, const FrTensor& rem);
// ReLU (or any range mapping) input `in`
void add_mapping(tLookupRangeMapping& lookup, const FrTensor& in);

// Proves every queued table and ends the scope
size_t flush();

}  // namespace squash

#endif  // SQUASH_CUH
