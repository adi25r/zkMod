#ifndef ZKFC_CUH
#define ZKFC_CUH

#include <cstddef>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "bls12-381.cuh"  // adjust this to point to the blstrs header file
#include "fr-tensor.cuh"
#include "proof.cuh"
#include "commitment.cuh"
#include "polynomial.cuh"

// This is for computing
class zkFC {
public:
    const uint inputSize, outputSize;
    bool has_bias;
    FrTensor weights, bias;

    zkFC(uint input_size, uint output_size, const FrTensor& weight);
    zkFC(uint input_size, uint output_size, const FrTensor& weight, const FrTensor& bias);
    FrTensor operator()(const FrTensor& X) const;

    vector<Claim> prove(const FrTensor& X, const FrTensor& Y) const;
    // With given challenges
    vector<Claim> prove(const FrTensor& X, const FrTensor& Y, const vector<Fr_t>& u_batch,
                        const vector<Fr_t>& u_input, const vector<Fr_t>& u_output,
                        vector<Polynomial>* proof_out = nullptr) const;

    static zkFC from_float_gpu_ptr(uint input_size, uint output_size, unsigned long scaling_factor,
                                   float* weight_ptr, float* bias_ptr);
    static zkFC from_float_gpu_ptr(uint input_size, uint output_size, unsigned long scaling_factor,
                                   float* weight_ptr);
    static FrTensor load_float_gpu_input(uint batch_size, uint input_size, unsigned long scaling_factor,
                                         float* input_ptr);
};

// This is for proving
class zkFCStacked {
public:
    bool has_bias;
    const uint num;
    const uint batchSize;
    const uint inputSize;
    const uint outputSize;

    FrTensor X, W, b, Y;  // num * batchSize * inputSize

    zkFCStacked(bool has_bias, uint num, uint batch_size, uint input_size, uint output_size,
                const vector<zkFC>& layers, const vector<FrTensor>& Xs, const vector<FrTensor>& Ys);

    void prove(vector<Polynomial>& proof) const;

    void prove(const vector<Fr_t>& u_num, const vector<Fr_t>& v_num, const vector<Fr_t>& u_batch,
               const vector<Fr_t>& u_input, const vector<Fr_t>& u_output, vector<Polynomial>& proof) const;
};

Fr_t zkip(const Fr_t& claim, const FrTensor& a, const FrTensor& b, const vector<Fr_t>& u,
          vector<Polynomial>& proof);

Fr_t zkip_stacked(const Fr_t& claim, const FrTensor& A, const FrTensor& B, const vector<Fr_t>& uN,
                  const vector<Fr_t>& uD, const vector<Fr_t> vN, uint N, uint D, vector<Polynomial>& proof);

#endif  // ZKFC_CUH
