#ifndef SQUASH_CUH
#define SQUASH_CUH

// Circuit squashing for lookups: inside a partition, the lookup arguments of ops that use the same
// table (the rescaling remainder ranges, the ReLU mapping) are proved as one LogUp over the
// concatenated inputs, like zkGPT's merged range/nonlinear layers, instead of one per op. Each op
// still checks its own identity (X = X_ * s + rem at a random point for rescaling) when it runs;
// only the table-membership argument is deferred and shared.
//
// Queued inputs are kept on the GPU as int32 (every remainder and ReLU input fits). A table's
// queue is proved when adding to it would pass 2^ZKLLM_SQUASH_MAX_LOG elements (default 24; this
// bounds the combined proof's GPU memory), and at flush().
//   ZKLLM_SQUASH=1 enables it (default off).

#include "fr-tensor.cuh"
#include "tlookup.cuh"

namespace squash {

bool enabled();                 // ZKLLM_SQUASH
void begin();                   // starts a squash scope (no-op unless enabled)
bool active();

// Rescaling remainder `rem` (unpadded) against the range table [-sf/2, sf/2).
void add_range(uint scaling_factor, const FrTensor& rem);
// ReLU (or any range mapping) input `in`; the mapping's outputs are recomputed at the flush.
void add_mapping(tLookupRangeMapping& lookup, const FrTensor& in);

// Proves every queued table and ends the scope; returns the number of combined lookup proofs.
size_t flush();

} // namespace squash

#endif // SQUASH_CUH
