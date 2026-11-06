// Allocation tracer for the memory plots (analysis build only; see plots/memtrace/build.sh).
// At exit, writes to $MEMTRACE_OUT (default "memtrace"):
//   <out>_peak.csv      bytes,frames   -- every allocation live at the tracked peak
//   <out>_timeline.csv  t_ms,live_bytes -- live device bytes after every malloc/free
#pragma once
#include <cuda_runtime.h>
#include <execinfo.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <vector>

struct MtAlloc { size_t n; std::vector<void*> bt; };
struct MtState {
    std::map<void*, MtAlloc> live;
    size_t cur = 0, peak = 0;
    std::vector<MtAlloc> snap;
    std::vector<std::pair<double, size_t>> timeline;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    std::mutex mu;
    double now_ms() const { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }
    ~MtState() {
        const char* o = getenv("MEMTRACE_OUT");
        std::string out = o ? o : "memtrace";
        FILE* f = fopen((out + "_peak.csv").c_str(), "w");
        fprintf(f, "bytes,frames\n");
        for (auto& a : snap) {
            char** s = backtrace_symbols(a.bt.data(), a.bt.size());
            std::string fr;
            for (size_t i = 1; i < a.bt.size(); ++i) { if (i > 1) fr += ";"; fr += s[i]; }
            free(s);
            for (auto& c : fr) if (c == ',' || c == '"') c = ' ';
            fprintf(f, "%zu,%s\n", a.n, fr.c_str());
        }
        fclose(f);
        f = fopen((out + "_timeline.csv").c_str(), "w");
        fprintf(f, "t_ms,live_bytes\n");
        for (auto& [t, b] : timeline) fprintf(f, "%.3f,%zu\n", t, b);
        fclose(f);
        fprintf(stderr, "MEMTRACE peak_MiB=%.1f live_allocs_at_peak=%zu -> %s_*.csv\n", peak / 1048576.0, snap.size(), out.c_str());
    }
};
inline MtState& mt() { static MtState s; return s; }
inline void mt_on_alloc(void* p, size_t n) {
    auto& s = mt(); std::lock_guard<std::mutex> g(s.mu);
    MtAlloc a{n, std::vector<void*>(16)};
    a.bt.resize(backtrace(a.bt.data(), 16));
    s.live[p] = a; s.cur += n;
    s.timeline.push_back({s.now_ms(), s.cur});
    if (s.cur > s.peak) {                       // exact: re-snapshot on every new peak
        s.snap.clear();
        for (auto& [q, x] : s.live) s.snap.push_back(x);
        s.peak = s.cur;
    }
}
inline void mt_on_free(void* p) {
    auto& s = mt(); std::lock_guard<std::mutex> g(s.mu);
    auto it = s.live.find(p); if (it == s.live.end()) return;
    s.cur -= it->second.n; s.live.erase(it);
    s.timeline.push_back({s.now_ms(), s.cur});
}
inline cudaError_t cuda_free_tracked(void* p) { mt_on_free(p); return cudaFree(p); }
