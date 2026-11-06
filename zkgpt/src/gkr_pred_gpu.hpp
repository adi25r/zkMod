#pragma once

#include <vector>
#include "global_var.hpp"
#include "circuit.h"

// GPU predicate reduction (generic-GKR finalize buckets)
//
// verifier::predicatePhase1/2 walk a layer's whole gate list and reduce into two
// (resp. three) field accumulators:
//
//   uni_value[gate.lu] += beta_g[g] * beta_u[u]              * sc
//   bin_value[gate.l]  += beta_g[g] * beta_u[u] * beta_v[v]  * sc
//
// There is no output scatter -- every gate lands in one of 2/3 scalars -- so this
// parallelises without the disjointness machinery build_u_partition needs.
//
// The beta tables are uploaded once per layer and read by both phases; the gate
// arrays are streamed per call. GKR_PRED_CPU=1 reverts to the host path, which
// stays in place as the correctness reference.

// Is the GPU predicate path active? (false when GKR_PRED_CPU is set.)
bool gkr_pred_gpu_enabled();

// Upload a host beta table to its persistent device slot.
// which: 0 = beta_g, 1 = beta_u, 2 = beta_v.
void gkr_pred_upload_beta(int which, const std::vector<F> &tbl);

// out[0..1] / out[0..2] receive the reduced accumulators. The corresponding beta
// tables must already have been uploaded for this layer.
void gkr_predicate1_gpu(const std::vector<uniGate> &gates, F out[2]);
void gkr_predicate2_gpu(const std::vector<binGate> &gates, F out[3]);

// Release device buffers. Call once at proof end.
void gkr_pred_teardown();
// Free only the per-call gate arrays (re-staged each call); see gkr_pred_gpu.cpp.
void gkr_pred_release_gates();

// GKR_GEN_PROF=1 sub-timings for the predicate path.
void gkr_pred_report();
