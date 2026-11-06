#pragma once

#include <vector>
#include "polynomial.h"
#include "global_var.hpp"

// Sumcheck state management

// Upload V_mult[0/1] and mult_array[0/1] to GPU for the current sumcheck phase
void gkr_sc_begin(linear_poly* v0, linear_poly* m0, uint n0, linear_poly* v1, linear_poly* m1, uint n1);

// Returns true if GPU state is active for group b
bool gkr_sc_active_b(int b);
// True when the GPU sumcheck fold is the active path (GKR_SC_CPU unset)
bool gkr_sc_gpu_enabled();
void gkr_sc_report();

// GPU sumcheck fold for group b
quadratic_poly gkr_sc_update_b(int b, const F& r, uint n, uint total_sz);

// Handle the degenerate single-element case
void gkr_sc_handle_single(int b, F& add_term, const F& r);

// Download first element of each group (after all rounds complete)
void gkr_sc_end(linear_poly& v0, linear_poly& m0, linear_poly& v1, linear_poly& m1);

// Free the sumcheck device buffers
void gkr_sc_teardown();
// Free the buffers if no group is active (between phases)
bool gkr_sc_release_idle();
