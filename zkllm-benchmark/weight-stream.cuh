#ifndef WEIGHT_STREAM_CUH
#define WEIGHT_STREAM_CUH

// Streamed linear-layer weights

#include "fr-tensor.cuh"
#include "commitment.cuh"
#include "polynomial.cuh"
#include <vector>

struct StreamedWeight {
    std::vector<int> host;  // in_dim x out_dim, row-major, as committed
    Commitment generator;
    G1TensorJacobian com;
    uint in_dim, out_dim;
};

bool weight_streaming_enabled();

StreamedWeight load_streamed_weight(const string& generator_filename, const string& weight_filename,
                                    const string& com_filename, uint in_dim, uint out_dim);

// chunk_rows = 0: rows per chunk from ZKLLM_WSTREAM_CHUNK_MB
FrTensor streamed_matmul(const FrTensor& X, const StreamedWeight& w, uint chunk_rows = 0);

struct StreamedLinearProof {
    std::vector<Polynomial> proof;  // zkip transcript
    Fr_t claim_W;                   // W at (u_input, u_output), checked against the commitment opening
};

// zkFC::prove followed by verifyWeightClaim, with W streamed
StreamedLinearProof streamed_prove_linear(const FrTensor& X, const FrTensor& Y, const StreamedWeight& w,
                                          const std::vector<Fr_t>& u_batch, const std::vector<Fr_t>& u_input,
                                          const std::vector<Fr_t>& u_output, uint chunk_rows = 0);

// The same proof from the reductions of X and Y instead of the tensors
StreamedLinearProof streamed_prove_linear_reduced(const Fr_t& claim, const FrTensor& X_reduced,
                                                  const Fr_t& claim_X, const StreamedWeight& w,
                                                  const std::vector<Fr_t>& u_input,
                                                  const std::vector<Fr_t>& u_output, uint chunk_rows = 0);

// Same, drawing the challenges like zkFC::prove
void streamed_prove_linear(const FrTensor& X, const FrTensor& Y, const StreamedWeight& w);

#endif  // WEIGHT_STREAM_CUH
