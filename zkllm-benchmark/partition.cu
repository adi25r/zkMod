// ./partition <input.bin> <seq_len> <embed_dim> <hidden_dim> <workdir> <first_layer> <num_layers> <output.bin> <witgen|prove>
//             [--part-idx N] [--act-pp act-pp.bin] [--commit-input 0|1] [--commit-output 0|1] [--eps 1e-5]
//             [--prefix partN] [--swiglu-table swiglu-table.bin] [--tamper-after-commit 0|1]
//             [--op-range A N] [--dump-cuts c1,c2,... --dump-prefix P]
// --op-range (OPT only) proves the N tensor operations A..A+N-1 of the model instead of whole layers
// (first_layer/num_layers are then ignored). The input/output is the state at the cut: at a layer
// boundary the usual int32 activation file, otherwise one int64 file per live tensor,
// <file>.<name>.bin. Every boundary tensor is committed and opened. In witgen mode, --dump-cuts
// saves the state at those cuts to <P><cut>, each followed by an empty <P><cut>.ready marker; with
// --dump-max-pending M it first waits while M earlier markers still exist (the consumer deletes them).
// --witgen-first (prove, layer path only): witness-first proving, see witness-store.cuh. A record pass runs witgen for
// every layer and keeps every tensor the proofs need in host memory; a replay pass then proves from the stored tensors.
// Without --op-range the cuts are layer indices: witgen saves x (int32) at the start of each listed layer,
// so one witgen process publishes every partition's input ("early carry") the moment its forward values
// are final, instead of a chain of per-partition processes each waiting for the previous one to exit.

#include "squash.cuh"
#include "llama-layer.cuh"
#include "opt-layer.cuh"
#include "ioutils.cuh"
#include "witness-store.cuh"
#include <cstdio>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

namespace {

struct Args {
    string input_file, workdir, output_file, mode;
    uint seq_len = 0, embed_dim = 0, hidden_dim = 0, first_layer = 0, num_layers = 0;
    int part_idx = 0;
    string act_pp = "act-pp.bin";
    bool commit_input = false, commit_output = false, tamper_after_commit = false;
    double eps = 1e-5;
    string prefix, swiglu_table = "swiglu-table.bin", arch = "llama";
    int op_first = -1;
    uint op_count = 0;
    string dump_cuts, dump_prefix;
    uint dump_max_pending = 0;   // 0 = unlimited
    bool witgen_first = false;
};

[[noreturn]] void usage(const char* argv0)
{
    fprintf(stderr, "usage: %s <input.bin> <seq_len> <embed_dim> <hidden_dim> <workdir> <first_layer> <num_layers> <output.bin> <witgen|prove>"
        " [--part-idx N] [--act-pp file] [--commit-input 0|1] [--commit-output 0|1] [--eps x] [--prefix p]"
        " [--swiglu-table file] [--arch llama|opt] [--tamper-after-commit 0|1] [--op-range A N]"
        " [--dump-cuts c1,c2,... --dump-prefix P] [--witgen-first 0|1]\n", argv0);
    exit(2);
}

Args parse_args(int argc, char* argv[])
{
    if (argc < 10) usage(argv[0]);
    Args a;
    a.input_file = argv[1];
    a.seq_len = std::stoul(argv[2]);
    a.embed_dim = std::stoul(argv[3]);
    a.hidden_dim = std::stoul(argv[4]);
    a.workdir = argv[5];
    a.first_layer = std::stoul(argv[6]);
    a.num_layers = std::stoul(argv[7]);
    a.output_file = argv[8];
    a.mode = argv[9];
    if (a.mode != "witgen" && a.mode != "prove") usage(argv[0]);

    for (int i = 10; i + 1 < argc; i += 2) {
        const string key = argv[i], val = argv[i + 1];
        if (key == "--part-idx") a.part_idx = std::stoi(val);
        else if (key == "--act-pp") a.act_pp = val;
        else if (key == "--commit-input") a.commit_input = std::stoi(val) != 0;
        else if (key == "--commit-output") a.commit_output = std::stoi(val) != 0;
        else if (key == "--eps") a.eps = std::stod(val);
        else if (key == "--prefix") a.prefix = val;
        else if (key == "--swiglu-table") a.swiglu_table = val;
        else if (key == "--arch") a.arch = val;
        else if (key == "--tamper-after-commit") a.tamper_after_commit = std::stoi(val) != 0;
        else if (key == "--op-range" && i + 2 < argc) { a.op_first = std::stoi(val); a.op_count = std::stoul(argv[i + 2]); ++ i; }
        else if (key == "--dump-cuts") a.dump_cuts = val;
        else if (key == "--dump-prefix") a.dump_prefix = val;
        else if (key == "--dump-max-pending") a.dump_max_pending = std::stoul(val);
        else if (key == "--witgen-first") a.witgen_first = std::stoi(val);
        else usage(argv[0]);
    }
    if (a.prefix.empty()) a.prefix = "part" + std::to_string(a.part_idx);
    if (a.witgen_first && a.mode != "prove") {
        fprintf(stderr, "--witgen-first needs prove mode\n");
        exit(2);
    }
    return a;
}

// Elements per chunk when a boundary tensor is moved between file, host and GPU (ZKLLM_ACT_CHUNK_MB).
size_t io_chunk_elems()
{
    const char* s = std::getenv("ZKLLM_ACT_CHUNK_MB");
    const long mb = s ? std::atol(s) : 256;
    return (static_cast<size_t>(mb > 0 ? mb : 256) << 20) / sizeof(Fr_t);
}

// An int64 state file streamed into host memory, converted to field elements on the GPU chunk by chunk.
std::shared_ptr<Activation> load_long_activation(const string& file)
{
    const size_t n = findsize(file) / sizeof(long), C = io_chunk_elems();
    PinnedBuffer host(n);
    std::vector<long> buf(std::min(n, C));
    FILE* f = fopen(file.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + file);
    long* dl;
    cudaMalloc((void**)&dl, sizeof(long) * buf.size());
    for (size_t i = 0; i < n; i += C) {
        const uint m = static_cast<uint>(std::min(C, n - i));
        if (fread(buf.data(), sizeof(long), m, f) != m) throw std::runtime_error("short read " + file);
        cudaMemcpy(dl, buf.data(), sizeof(long) * m, cudaMemcpyHostToDevice);
        FrTensor t(m);
        long_to_scalar_kernel<<<(m + FrNumThread - 1) / FrNumThread, FrNumThread>>>(dl, t.gpu_data, m);
        cudaDeviceSynchronize();
        host.write(i, t);
    }
    cudaFree(dl);
    fclose(f);
    return std::make_shared<Activation>(std::move(host));
}

void save_long_activation(const Activation& a, const string& file)
{
    const size_t n = a.size(), C = io_chunk_elems();
    std::vector<long> buf(std::min(n, C));
    FILE* f = fopen(file.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + file);
    long* dl;
    cudaMalloc((void**)&dl, sizeof(long) * buf.size());
    for (size_t i = 0; i < n; i += C) {
        const uint m = static_cast<uint>(std::min(C, n - i));
        FrTensor t = a.slice(i, m);
        scalar_to_long_kernel<<<(m + FrNumThread - 1) / FrNumThread, FrNumThread>>>(t.gpu_data, dl, m);
        cudaDeviceSynchronize();
        cudaMemcpy(buf.data(), dl, sizeof(long) * m, cudaMemcpyDeviceToHost);
        fwrite(buf.data(), sizeof(long), m, f);
    }
    cudaFree(dl);
    fclose(f);
}

// State at model cut `cut`: the int32 activation file at a layer boundary, else <base>.<name>.bin (int64).
OpState load_state(const string& base, uint cut, const LayerDims& d)
{
    OpState st;
    if (cut % OPT_OPS_PER_LAYER == 0) {
        auto x = std::make_shared<Activation>(FrTensor::from_int_bin(base));
        if (x->size() != d.seq_len * d.embed_dim) throw std::runtime_error("partition: input size does not match seq_len*embed_dim");
        x->offload();
        st["x"] = x;
    } else {
        for (auto& name : opt_live_at(cut % OPT_OPS_PER_LAYER)) {
            if (activation_streaming_enabled()) st[name] = load_long_activation(base + "." + name + ".bin");
            else st[name] = std::make_shared<Activation>(FrTensor::from_long_bin(base + "." + name + ".bin"));
        }
    }
    return st;
}

// Written under temporary names and renamed, so a reader never sees a partial file.
void save_state(const string& base, uint cut, const OpState& st)
{
    if (cut % OPT_OPS_PER_LAYER == 0) {
        st.at("x")->gpu().save_int(base + ".tmp");
        st.at("x")->offload();
        std::rename((base + ".tmp").c_str(), base.c_str());
        return;
    }
    for (auto& kv : st) {
        const string f = base + "." + kv.first + ".bin";
        if (activation_streaming_enabled()) save_long_activation(*kv.second, f + ".tmp");
        else kv.second->gpu().save_long(f + ".tmp");
        std::rename((f + ".tmp").c_str(), f.c_str());
    }
}

// Witgen-side cut dumps (--dump-cuts/--dump-prefix/--dump-max-pending): the state at each listed cut is
// saved to <prefix><cut> and followed by an empty <prefix><cut>.ready marker.
struct CutDumper {
    std::vector<uint> cuts;
    string prefix;
    uint max_pending = 0;
    std::vector<string> dumped;
    Timer t_backpressure;

    explicit CutDumper(const Args& args) : prefix(args.dump_prefix), max_pending(args.dump_max_pending)
    {
        for (size_t p = 0; !args.dump_cuts.empty() && p != string::npos; ) {
            size_t q = args.dump_cuts.find(',', p);
            cuts.push_back(std::stoul(args.dump_cuts.substr(p, q == string::npos ? string::npos : q - p)));
            p = q == string::npos ? q : q + 1;
        }
    }

    bool wants(uint cut) const { return std::find(cuts.begin(), cuts.end(), cut) != cuts.end(); }

    // Waits for backpressure, then hands the cut's base name to save (which must write atomically) and
    // marks it ready.
    template <class Save> void dump(uint cut, Save save)
    {
        const string base = prefix + std::to_string(cut);
        if (max_pending) {
            auto pending = [&] {
                uint n = 0;
                for (auto& b : dumped) if (FILE* f = fopen((b + ".ready").c_str(), "r")) { fclose(f); ++ n; }
                return n;
            };
            t_backpressure.start();
            while (pending() >= max_pending) std::this_thread::sleep_for(std::chrono::milliseconds(50));
            t_backpressure.stop();
        }
        dumped.push_back(base);
        save(base);
        FILE* f = fopen((base + ".ready").c_str(), "w");
        if (f) fclose(f);
    }
};

// Commit every tensor of a boundary state (flattened, power-of-two padded) and open it at a random
// point. A tensor larger than one chunk never sits on the GPU whole: rows of commitment are computed
// chunk by chunk, and the opening's fold over the rows and the evaluation at the point are
// accumulated the same way (same folds as Commitment::open and Fr_me, so the same values).
void commit_state(const Commitment& gen, const OpState& st, const string& prefix, const string& side)
{
    for (auto& kv : st) {
        Activation& a = *kv.second;
        const uint n = a.size(), P = 1u << ceilLog2(n), G = gen.size, rows = P / G;
        uint R = rows;
        while (R > 1 && static_cast<size_t>(R) * G > io_chunk_elems()) R >>= 1;
        const string where = side + " boundary opening mismatch for " + kv.first;
        if (!activation_streaming_enabled() || R == rows) {
            const FrTensor& t = a.gpu();
            std::unique_ptr<FrTensor> padded;
            if (t.size != P) padded = std::make_unique<FrTensor>(t.pad({t.size}));
            const FrTensor& flat = padded ? *padded : t;
            G1TensorJacobian com = gen.commit_auto(flat);
            auto r = random_vec(ceilLog2(flat.size));
            gen.open_check(flat, com, r, flat(r), "partition: " + where);
            com.save(prefix + "-" + side + "-" + kv.first + "-commitment.bin");
            padded.reset();
            a.offload();
            continue;
        }
        const uint nc = rows / R, CE = R * G;
        auto chunk = [&](uint c) { return a.slice(static_cast<size_t>(c) * CE, CE); };
        int cls = 0;
        for (uint c = 0; c < nc && cls < 2; ++ c) cls = std::max(cls, Commitment::value_class(chunk(c)));
        auto r = random_vec(ceilLog2(P));
        const uint lr = ceilLog2(rows), lR = ceilLog2(R);
        const vector<Fr_t> u_in(r.begin(), r.end() - lr), u_rows(r.end() - lr, r.end());
        const vector<Fr_t> ur_lo(u_rows.begin(), u_rows.begin() + lR), ur_hi(u_rows.begin() + lR, u_rows.end());
        const vector<Fr_t> r_lo(r.begin(), r.begin() + ceilLog2(CE)), r_hi(r.begin() + ceilLog2(CE), r.end());
        G1TensorJacobian com(rows);
        FrTensor parts(nc * G);
        vector<Fr_t> vals(nc);
        for (uint c = 0; c < nc; ++ c) {
            FrTensor t = chunk(c);
            G1TensorJacobian cc = gen.commit_as(t, cls);
            cudaMemcpy(com.gpu_data + static_cast<size_t>(c) * R, cc.gpu_data, sizeof(G1Jacobian_t) * R, cudaMemcpyDeviceToDevice);
            FrTensor pc = lR ? t.partial_me(ur_lo, G) : FrTensor(t);
            cudaMemcpy(parts.gpu_data + static_cast<size_t>(c) * G, pc.gpu_data, sizeof(Fr_t) * G, cudaMemcpyDeviceToDevice);
            vals[c] = t(r_lo);
        }
        auto g_temp = com(u_rows);   // as in Commitment::open
        (void)g_temp;
        FrTensor folded = ur_hi.empty() ? FrTensor(parts) : parts.partial_me(ur_hi, G);
        Fr_t value = FrTensor(nc, vals.data())(r_hi);
        Commitment::me_open_check(folded, gen, u_in.begin(), u_in.end(), value, "partition: " + where);
        com.save(prefix + "-" + side + "-" + kv.first + "-commitment.bin");
    }
}

int run_ops(const Args& args)
{
    if (args.arch != "opt") throw std::runtime_error("partition: --op-range is only implemented for --arch opt");
    const bool prove = args.mode == "prove";
    const LayerDims dims{args.seq_len, args.embed_dim, args.hidden_dim, args.eps};
    const uint A = static_cast<uint>(args.op_first), B = A + args.op_count;
    Timer wall, t_load, t_in, t_out;
    wall.start();

    t_load.start();
    OpState st = load_state(args.input_file, A, dims);
    t_load.stop();

    std::unique_ptr<Commitment> act_gen;
    if (prove && (args.commit_input || args.commit_output)) {
        act_gen = std::make_unique<Commitment>(args.act_pp);
        if (act_gen->size != 1u << ceilLog2(args.embed_dim))
            throw std::runtime_error("partition: act-pp has " + std::to_string(act_gen->size) + " generators, expected pow2(embed_dim)");
    }
    if (prove) { Commitment::begin_open_batch(); squash::begin(); }
    if (prove && args.commit_input) {
        t_in.start();
        commit_state(*act_gen, st, args.prefix, "in");
        t_in.stop();
    }

    LayerTimers t;
    OptOpRunner runner(args.workdir, dims, args.swiglu_table, prove, t);
    CutDumper dumper(args);
    for (uint g = A; g < B; ++ g) {
        runner.run(g, st);
        if (!prove && dumper.wants(g + 1)) dumper.dump(g + 1, [&](const string& base) { save_state(base, g + 1, st); });
    }

    if (prove && args.commit_output) {
        t_out.start();
        commit_state(*act_gen, st, args.prefix, "out");
        t_out.stop();
    }
    Timer t_open, t_squash;
    size_t n_open = 0, n_squash = 0;
    if (prove) {
        t_squash.start();
        n_squash = squash::flush();
        t_squash.stop();
        t_open.start();
        n_open = Commitment::flush_openings();
        t_open.stop();
    }
    t_load.start();
    save_state(args.output_file, B, st);
    t_load.stop();
    wall.stop();

    const double in_s = t_in.getTotalTime(), out_s = t_out.getTotalTime();
    const double open_s = t_open.getTotalTime(), squash_s = t_squash.getTotalTime();
    printf("PARTITION idx=%d op_first=%u op_count=%u load_s=%.6f weight_load_s=%.6f witgen_s=%.6f input_commit_s=%.6f prove_s=%.6f output_commit_s=%.6f "
        "open_flush_s=%.6f openings_batched=%zu squash_s=%.6f squashed_lookups=%zu prover_total_s=%.6f wall_s=%.6f dump_wait_s=%.6f alloc_peak_mb=%zu status=OK\n",
        args.part_idx, A, args.op_count, t_load.getTotalTime(), runner.load_s, t.compute_s, in_s, t.prove_s, out_s, open_s, n_open, squash_s, n_squash,
        in_s + t.prove_s + out_s + open_s + squash_s, wall.getTotalTime(), dumper.t_backpressure.getTotalTime(), device_alloc_stats().peak >> 20);
    return 0;
}

int run(const Args& args)
{
    if (args.op_first >= 0) {
        if (args.witgen_first) throw std::runtime_error("--witgen-first needs the layer path (no --op-range)");
        return run_ops(args);
    }
    const bool prove = args.mode == "prove";
    const LayerDims dims{args.seq_len, args.embed_dim, args.hidden_dim, args.eps};

    Timer wall;
    wall.start();

    FrTensor X_in = FrTensor::from_int_bin(args.input_file);
    if (X_in.size != args.seq_len * args.embed_dim)
        throw std::runtime_error("partition: input size " + std::to_string(X_in.size) + " != seq_len*embed_dim");

    // Boundary commitments share one public basis of embed_dim generators, so the tensor is
    // committed as seq_len rows and the producer's and consumer's commitments are identical.
    const bool do_commit_input = prove && args.commit_input;
    const bool do_commit_output = prove && args.commit_output;
    std::unique_ptr<Commitment> act_gen;
    if (do_commit_input || do_commit_output) {
        act_gen = std::make_unique<Commitment>(args.act_pp);
        if (act_gen->size != 1u << ceilLog2(args.embed_dim))
            throw std::runtime_error("partition: act-pp has " + std::to_string(act_gen->size) + " generators, expected pow2(embed_dim)");
    }

    if (prove) { Commitment::begin_open_batch(); squash::begin(); }
    Timer t_input_commit, t_output_commit;
    std::unique_ptr<G1TensorJacobian> C_in;
    if (do_commit_input) {
        t_input_commit.start();
        C_in = std::make_unique<G1TensorJacobian>(act_gen->commit_int(X_in.pad({X_in.size})));
        t_input_commit.stop();
        C_in->save(args.prefix + "-in-commitment.bin");
        if (args.tamper_after_commit) X_in += {1, 0, 0, 0, 0, 0, 0, 0};
    }

    FrTensor swiglu_values = FrTensor::from_int_bin(args.swiglu_table);
    tLookupRangeMapping swiglu(-(1 << 21), 1 << 22, swiglu_values);
    // Logit gaps below the row log-sum-exp are decomposed into these segments (at scale 2^32*sqrt(d)).
    // The trailing 2^15 segment with theta=1 only widens the range from 48 to 63 bits (its table is 1 at
    // 0 and rounds to 0 elsewhere), which the single-head-over-embed_dim attention needs on larger models.
    zkSoftmax softmax({1 << 8, 1 << 20, 1 << 20, 1 << 15}, 1, 0, 1UL << 32, {1 << 18, 1 << 22, 1.0}, args.seq_len, args.seq_len, args.embed_dim, 1);

    BoundaryClaim first;
    LayerTimers t;
    double load_s = 0;
    CutDumper dumper(args);
    Timer t_carry, t_record;
    // Witness-first: the record pass (witgen only) stores every layer's tensors in host memory; the pass below
    // then proves from them.
    if (args.witgen_first) {
        t_record.start();
        wstore::set_mode(wstore::RECORD);
        FrTensor xr(X_in);
        for (uint l = args.first_layer; l < args.first_layer + args.num_layers; ++l) {
            const double c0 = t.compute_s, s0 = wstore::stats().store_s;
            Timer t_load;
            t_load.start();
            wstore::begin_layer(l);
            if (args.arch == "opt") {
                OPTLayerWeights w(args.workdir, l, dims);
                t_load.stop();
                xr = run_opt_layer(xr, dims, w, swiglu, softmax, false, t, nullptr);
            } else {
                LayerWeights w(args.workdir, l, dims);
                t_load.stop();
                xr = run_layer(xr, dims, w, swiglu, softmax, false, t, nullptr);
            }
            wstore::end_layer();
            load_s += t_load.getTotalTime();
            fprintf(stderr, "RECORD %u load_s=%.2f compute_s=%.2f store_s=%.2f host_gb=%.2f\n", l, t_load.getTotalTime(),
                t.compute_s - c0, wstore::stats().store_s - s0, wstore::stats().bytes / 1e9);
        }
        wstore::set_mode(wstore::REPLAY);
        t_record.stop();
    }
    FrTensor x(X_in);
    for (uint l = args.first_layer; l < args.first_layer + args.num_layers; ++l) {
        BoundaryClaim* claim = l == args.first_layer ? &first : nullptr;
        const double c0 = t.compute_s, p0 = t.prove_s;
        Timer t_layer, t_load;
        t_layer.start();
        t_load.start();
        wstore::begin_layer(l);
        if (args.arch == "opt") {
            OPTLayerWeights w(args.workdir, l, dims);
            t_load.stop();
            x = run_opt_layer(x, dims, w, swiglu, softmax, prove, t, claim);
        } else {
            LayerWeights w(args.workdir, l, dims);
            t_load.stop();
            x = run_layer(x, dims, w, swiglu, softmax, prove, t, claim);
        }
        wstore::end_layer();
        t_layer.stop();
        load_s += t_load.getTotalTime();
        size_t mem_free = 0, mem_total = 0;
        cudaMemGetInfo(&mem_free, &mem_total);
        fprintf(stderr, "LAYER %u load_s=%.2f compute_s=%.2f prove_s=%.2f wall_s=%.2f gpu_used_mb=%zu\n", l, t_load.getTotalTime(),
            t.compute_s - c0, t.prove_s - p0, t_layer.getTotalTime(), (mem_total - mem_free) >> 20);
        // Early carry: x is now the next partition's input, final before the rest of this witgen run.
        if (!prove && dumper.wants(l + 1)) {
            t_carry.start();
            dumper.dump(l + 1, [&](const string& base) {
                x.save_int(base + ".tmp");
                std::rename((base + ".tmp").c_str(), base.c_str());
            });
            t_carry.stop();
            fprintf(stderr, "EARLY_CARRY layer=%u file=%s%u\n", l + 1, args.dump_prefix.c_str(), l + 1);
        }
    }

    if (do_commit_input) {
        t_input_commit.start();
        if (!first.valid) throw std::runtime_error("partition: no input claim produced");
        act_gen->open_check(X_in.pad({X_in.size}), *C_in, first.v, first.value, "partition: input boundary opening mismatch");
        t_input_commit.stop();
        cout << "Input boundary opening complete" << endl;
    }

    if (do_commit_output) {
        t_output_commit.start();
        // Boundary tensors are committed flat and power-of-two padded, matching the sumcheck's multilinear extension.
        FrTensor x_flat = x.pad({x.size});
        G1TensorJacobian C_out = act_gen->commit_int(x_flat);
        auto r = random_vec(ceilLog2(x_flat.size));
        act_gen->open_check(x_flat, C_out, r, x_flat(r), "partition: output boundary opening mismatch");
        t_output_commit.stop();
        C_out.save(args.prefix + "-out-commitment.bin");
        cout << "Output boundary opening complete" << endl;
    }
    Timer t_open, t_squash;
    size_t n_open = 0, n_squash = 0;
    if (prove) {
        t_squash.start();
        n_squash = squash::flush();
        t_squash.stop();
        t_open.start();
        n_open = Commitment::flush_openings();
        t_open.stop();
    }

    x.save_int(args.output_file);
    wstore::set_mode(wstore::OFF);
    wall.stop();

    const double input_commit_s = t_input_commit.getTotalTime();
    const double output_commit_s = t_output_commit.getTotalTime();
    const double open_s = t_open.getTotalTime(), squash_s = t_squash.getTotalTime();
    const double prover_total_s = input_commit_s + t.prove_s + output_commit_s + open_s + squash_s;
    printf("PARTITION idx=%d first_layer=%u num_layers=%u load_s=%.6f witgen_s=%.6f input_commit_s=%.6f prove_s=%.6f output_commit_s=%.6f open_flush_s=%.6f openings_batched=%zu squash_s=%.6f squashed_lookups=%zu prover_total_s=%.6f wall_s=%.6f carry_s=%.6f dump_wait_s=%.6f "
        "witgen_first=%d record_s=%.6f ws_store_s=%.6f ws_load_s=%.6f ws_peak_gb=%.3f ws_tensors=%zu ws_raw_tensors=%zu ws_pageable_tensors=%zu alloc_peak_mb=%zu status=OK\n",
        args.part_idx, args.first_layer, args.num_layers, load_s, t.compute_s, input_commit_s, t.prove_s, output_commit_s, open_s, n_open, squash_s, n_squash, prover_total_s, wall.getTotalTime(),
        t_carry.getTotalTime(), dumper.t_backpressure.getTotalTime(),
        int(args.witgen_first), t_record.getTotalTime(), wstore::stats().store_s, wstore::stats().load_s, wstore::stats().peak_bytes / 1e9,
        wstore::stats().tensors, wstore::stats().raw_tensors, wstore::stats().pageable_tensors, device_alloc_stats().peak >> 20);
    return 0;
}

} // namespace

int main(int argc, char* argv[])
{
    const Args args = parse_args(argc, argv);
    try {
        return run(args);
    } catch (const std::exception& e) {
        fprintf(stderr, "partition: %s\n", e.what());
        printf("PARTITION idx=%d first_layer=%u num_layers=%u status=FAIL\n", args.part_idx, args.first_layer, args.num_layers);
        return 1;
    }
}
