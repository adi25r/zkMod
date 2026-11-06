#pragma once

#include <vector>
#include "global_var.hpp"
#include "circuit.h"

// GPU predicate reduction

// Is the GPU predicate path active
bool gkr_pred_gpu_enabled();

void gkr_pred_upload_beta(int which, const std::vector<F>& tbl);

// out[0..1] / out[0..2] receive the reduced accumulators
void gkr_predicate1_gpu(const std::vector<uniGate>& gates, F out[2]);
void gkr_predicate2_gpu(const std::vector<binGate>& gates, F out[3]);

// Release device buffers
void gkr_pred_teardown();
// Free only the per-call gate arrays (re-staged each call)
void gkr_pred_release_gates();

// GKR_GEN_PROF=1 sub-timings for the predicate path
void gkr_pred_report();
