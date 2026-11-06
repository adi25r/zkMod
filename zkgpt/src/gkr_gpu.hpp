#pragma once

#include <vector>
#include "polynomial.h"
#include "global_var.hpp"

// Sumcheck state management

// Upload V_mult[0/1] and mult_array[0/1] to GPU for the current sumcheck phase.
// Call once at the end of sumcheckInitPhase1 or sumcheckInitPhase2.
// n0, n1 are the initial total[0] and total[1].
void gkr_sc_begin(
    linear_poly* v0, linear_poly* m0, uint n0,
    linear_poly* v1, linear_poly* m1, uint n1);

// Returns true if GPU state is active for group b.
bool gkr_sc_active_b(int b);
// True when the GPU sumcheck fold is the active path (GKR_SC_CPU unset).
bool gkr_sc_gpu_enabled();
void gkr_sc_report();

// GPU sumcheck fold for group b:
//   tmp_v[i]    = interpolate(tmp_v[g0].eval(r), tmp_v[g1].eval(r))
//   tmp_mult[i] = interpolate(tmp_mult[g0].eval(r), tmp_mult[g1].eval(r))
//   ret        += tmp_mult[i] * tmp_v[i]   (quadratic)
// n = current total[b] (before halving), total_sz = total_size[b].
// GPU arrays are updated in-place (only [0..n/2) are valid afterwards).
quadratic_poly gkr_sc_update_b(int b, const F& r, uint n, uint total_sz);

// Handle the degenerate single-element case (total[b] == 1):
//   eval element 0 at r, add to add_term, write back constant poly.
void gkr_sc_handle_single(int b, F& add_term, const F& r);

// Download first element of each group (after all rounds complete).
// Called from sumcheckFinalize. Marks the state inactive but KEEPS the device
// buffers, so gkr_sc_begin's grow-only path can reuse them on the next layer.
void gkr_sc_end(
    linear_poly& v0, linear_poly& m0,
    linear_poly& v1, linear_poly& m1);

// Free the sumcheck device buffers. Call once at proof end (the entry points do).
void gkr_sc_teardown();
// Free the buffers if no group is active (between phases); false if one is.
bool gkr_sc_release_idle();
