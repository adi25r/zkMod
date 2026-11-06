#ifndef WEIGHT_STREAM_CUH
#define WEIGHT_STREAM_CUH

// Streamed linear-layer weights: the int32 matrix stays in host memory and is copied to the
// GPU in row chunks for each use, instead of being held on the GPU as 32-byte field
// elements for the whole layer.
//
// Every use of W in a linear layer is a pass over its rows:
//   forward      Y = X W                              Y += X[:, rows] W[rows, :] per chunk
//   sumcheck     W_red[i] = sum_j eq(u_out, j) W[i, j] per row, independent across rows
//   opening      v[k] = sum_r eq(u_rows, r) W_pad[r*G + k]   over the committed rows of G
// All are exact field sums, so the results (and so the proof) equal the resident path's
// bit for bit. ZKLLM_WSTREAM=0 selects the resident path; ZKLLM_WSTREAM_CHUNK_MB sets the
// int32 chunk size (default 64).

#include "fr-tensor.cuh"
#include "commitment.cuh"
#include "polynomial.cuh"
#include <vector>

struct StreamedWeight {
    std::vector<int> host;         // in_dim x out_dim, row-major, as committed
    Commitment generator;
    G1TensorJacobian com;
    uint in_dim, out_dim;
};

bool weight_streaming_enabled();   // ZKLLM_WSTREAM, default on

StreamedWeight load_streamed_weight(const string& generator_filename, const string& weight_filename, const string& com_filename,
    uint in_dim, uint out_dim);

// chunk_rows = 0: rows per chunk from ZKLLM_WSTREAM_CHUNK_MB.
FrTensor streamed_matmul(const FrTensor& X, const StreamedWeight& w, uint chunk_rows = 0);

struct StreamedLinearProof {
    std::vector<Polynomial> proof; // zkip transcript
    Fr_t claim_W;                  // W at (u_input, u_output), checked against the commitment opening
};

// zkFC::prove followed by verifyWeightClaim, with W streamed. Throws on any failed check.
StreamedLinearProof streamed_prove_linear(const FrTensor& X, const FrTensor& Y, const StreamedWeight& w,
    const std::vector<Fr_t>& u_batch, const std::vector<Fr_t>& u_input, const std::vector<Fr_t>& u_output, uint chunk_rows = 0);

// The same proof from the reductions of X and Y instead of the tensors, for callers that stream
// X and Y: claim = Y(u_batch, u_output), X_reduced = X folded over the batch by u_batch,
// claim_X = X(u_batch, u_input).
StreamedLinearProof streamed_prove_linear_reduced(const Fr_t& claim, const FrTensor& X_reduced, const Fr_t& claim_X,
    const StreamedWeight& w, const std::vector<Fr_t>& u_input, const std::vector<Fr_t>& u_output, uint chunk_rows = 0);

// Same, drawing the challenges like zkFC::prove.
void streamed_prove_linear(const FrTensor& X, const FrTensor& Y, const StreamedWeight& w);

#endif // WEIGHT_STREAM_CUH
