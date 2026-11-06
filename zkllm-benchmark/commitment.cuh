#ifndef COMMITMENT_CUH
#define COMMITMENT_CUH

#include "fr-tensor.cuh"
#include "g1-tensor.cuh"
#include "proof.cuh"

class Commitment : public G1TensorJacobian {
public:
    using G1TensorJacobian::G1TensorJacobian;

    using G1TensorJacobian::operator+;
    using G1TensorJacobian::operator-;
    using G1TensorJacobian::operator*;
    using G1TensorJacobian::operator*=;

    G1TensorJacobian commit(const FrTensor& t) const;
    G1TensorJacobian commit_int(const FrTensor& t) const;
    G1TensorJacobian commit_int_multi(const vector<FrTensor>& t) const;
    // For tensors of int64-range values
    G1TensorJacobian commit_long(const FrTensor& t) const;
    // commit_int when every value fits int32, commit_long when it fits int64, commit otherwise
    G1TensorJacobian commit_auto(const FrTensor& t) const;
    // 0 if every value of t fits int32, 1 if int64, 2 otherwise
    static int value_class(const FrTensor& t);
    G1TensorJacobian commit_as(const FrTensor& t, int value_class) const;

    Fr_t open(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u) const;

    static Commitment random(uint size);
    static Fr_t me_open(const FrTensor& t, const Commitment& generators, vector<Fr_t>::const_iterator begin,
                        vector<Fr_t>::const_iterator end, vector<G1Jacobian_t>& proof);

    // Opening batching
    static void begin_open_batch();
    // Ends the batch
    static size_t flush_openings(vector<vector<G1Jacobian_t>>* proofs = nullptr);
    static void me_open_check(const FrTensor& t, const Commitment& generators,
                              vector<Fr_t>::const_iterator begin, vector<Fr_t>::const_iterator end,
                              const Fr_t& expected, const string& what);
    void open_check(const FrTensor& t, const G1TensorJacobian& c, const vector<Fr_t>& u, const Fr_t& expected,
                    const string& what) const;
};

struct Weight {
    Commitment generator;
    FrTensor weight;
    G1TensorJacobian com;
    uint in_dim;
    uint out_dim;
};

Weight create_weight(string generator_filename, string weight_filename, string com_filename, uint in_dim,
                     uint out_dim);

KERNEL void me_open_step(GLOBAL Fr_t* scalars, GLOBAL G1Jacobian_t* generators,
                         Fr_t u,  // always assume that scalars and u is in mont form
                         GLOBAL Fr_t* new_scalars, GLOBAL G1Jacobian_t* new_generators,
                         GLOBAL G1Jacobian_t* temp_out, GLOBAL G1Jacobian_t* temp_out0,
                         GLOBAL G1Jacobian_t* temp_out1, uint old_size, uint new_size);

#endif