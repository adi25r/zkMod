#include "rescaling.cuh"
#include "squash.cuh"

Rescaling::Rescaling(uint scaling_factor)
    : scaling_factor(scaling_factor),
      tl_rem(-static_cast<int>(scaling_factor >> 1), scaling_factor),
      rem_tensor_ptr(nullptr) {}

KERNEL void rescaling_kernel(Fr_t* in_ptr, Fr_t* out_ptr, Fr_t* rem_ptr, long scaling_factor, uint N) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        // Pre-rescale activations can sit at scale 2^56
        const __int128 sf = scaling_factor;
        const __int128 hsf = sf >> 1;
        const __int128 x = scalar_to_i128(in_ptr[tid]);
        const __int128 temp = (x + hsf) % sf;
        const __int128 x_rem = (temp < 0 ? temp + sf : temp) - hsf;
        const __int128 x_rescaled = (x - x_rem) / sf;
        out_ptr[tid] = i128_to_scalar(x_rescaled);
        rem_ptr[tid] = i128_to_scalar(x_rem);
    }
}

FrTensor Rescaling::operator()(const FrTensor& X) {
    if (rem_tensor_ptr)
        delete rem_tensor_ptr;
    rem_tensor_ptr = new FrTensor(X.size);

    FrTensor out(X.size);
    uint block_size = 256;
    rescaling_kernel<<<(X.size + block_size - 1) / block_size, block_size>>>(
        X.gpu_data, out.gpu_data, rem_tensor_ptr->gpu_data, scaling_factor, X.size);
    cudaDeviceSynchronize();

    return out;
}

Rescaling::~Rescaling() {
    if (rem_tensor_ptr)
        delete rem_tensor_ptr;
}

vector<Claim> Rescaling::prove(const FrTensor& X, const FrTensor& X_) {
    if (X.size != X_.size) {
        throw std::runtime_error("Error: the size of X and X_ should be the same.");
    }

    auto u = random_vec(ceilLog2(X.size));
    auto v = random_vec(ceilLog2(X.size));

    auto rand_temp = random_vec(2);
    vector<Polynomial> proof;

    auto rem = rem_tensor_ptr->pad({rem_tensor_ptr->size});

    if (X(u) != X_(u) * Fr_t({scaling_factor, 0, 0, 0, 0, 0, 0, 0}) + rem(u)) {
        throw std::runtime_error("Error: the rem is not correct.");
    }
    if (squash::active()) {  // the range argument goes into the partition's combined lookup
        squash::add_range(scaling_factor, *rem_tensor_ptr);
        return {};
    }
    auto m = tl_rem.prep(rem);
    tl_rem.prove(rem, m, rand_temp[0], rand_temp[1], u, v, proof);

    cout << "Rescaling proof complete." << endl;
    return {};
}