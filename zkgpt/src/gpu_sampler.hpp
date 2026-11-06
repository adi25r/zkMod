#pragma once
// gpu_sampler — device-wide VRAM sampling from a separate process
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <dirent.h>
#include <signal.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <sys/wait.h>

namespace zkgpt_sampler {

struct Card {
    std::string name, pci, used_path, total_path, gtt_path, busy_path, membusy_path;
};

inline std::vector<Card> enumerate() {
    std::vector<Card> out;
    DIR* d = opendir("/sys/class/drm");
    if (!d)
        return out;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.rfind("card", 0) != 0)
            continue;
        if (n.find('-') != std::string::npos)
            continue;  // connector nodes
        const std::string dev = std::string("/sys/class/drm/") + n + "/device";
        Card c;
        c.used_path = dev + "/mem_info_vram_used";
        c.total_path = dev + "/mem_info_vram_total";
        c.gtt_path = dev + "/mem_info_gtt_used";
        c.busy_path = dev + "/gpu_busy_percent";
        c.membusy_path = dev + "/mem_busy_percent";
        if (FILE* f = fopen(c.used_path.c_str(), "r"))
            fclose(f);
        else
            continue;
        char buf[512];
        ssize_t k = readlink(dev.c_str(), buf, sizeof(buf) - 1);
        if (k > 0) {
            buf[k] = 0;
            const char* s = strrchr(buf, '/');
            c.pci = s ? s + 1 : buf;
        }
        c.name = n;
        out.push_back(c);
    }
    closedir(d);
    return out;
}

struct ProcStat {
    long ppid = 0;
    unsigned long long utime = 0, stime = 0;  // clock ticks, whole process
    long threads = 0;
    unsigned long long rss_pages = 0;
    char comm[64] = {0};
    bool ok = false;
};

inline ProcStat read_proc_stat(long pid) {
    ProcStat ps;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%ld/stat", pid);
    FILE* f = fopen(path, "r");
    if (!f)
        return ps;
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (!n)
        return ps;
    buf[n] = 0;
    char* close = strrchr(buf, ')');
    if (!close)
        return ps;
    // comm, minus the parens, truncated
    const char* open_p = strchr(buf, '(');
    if (open_p && close > open_p) {
        size_t len = (size_t)(close - open_p - 1);
        if (len >= sizeof(ps.comm))
            len = sizeof(ps.comm) - 1;
        memcpy(ps.comm, open_p + 1, len);
        ps.comm[len] = 0;
        for (char* c = ps.comm; *c; ++c)
            if (*c == ',')
                *c = '_';
    }
    char* p = close + 2;
    char state = 0;
    long ppid = 0, threads = 0;
    unsigned long long ut = 0, st = 0, rss = 0;
    int got = sscanf(p,
                     "%c %ld %*d %*d %*d %*d %*u %*u %*u %*u %*u %llu %llu "
                     "%*d %*d %*d %*d %ld %*d %*u %*u %llu",
                     &state, &ppid, &ut, &st, &threads, &rss);
    if (got < 6)
        return ps;
    ps.ppid = ppid;
    ps.utime = ut;
    ps.stime = st;
    ps.threads = threads;
    ps.rss_pages = rss;
    ps.ok = true;
    return ps;
}

// Descendant closure of `root`, by one pass over /proc
inline void discover_descendants(long root, long exclude, std::vector<long>& out) {
    std::vector<std::pair<long, long> > edges;  // (pid, ppid)
    DIR* d = opendir("/proc");
    if (!d)
        return;
    while (dirent* e = readdir(d)) {
        char* end = nullptr;
        long pid = strtol(e->d_name, &end, 10);
        if (!end || *end || pid <= 0)
            continue;
        ProcStat ps = read_proc_stat(pid);
        if (ps.ok)
            edges.push_back(std::make_pair(pid, ps.ppid));
    }
    closedir(d);
    std::vector<long> keep;
    keep.push_back(root);
    // Repeated relaxation rather than building a tree
    bool grew = true;
    while (grew) {
        grew = false;
        for (size_t i = 0; i < edges.size(); ++i) {
            long pid = edges[i].first, ppid = edges[i].second;
            bool have = false, parent_in = false;
            for (size_t k = 0; k < keep.size(); ++k) {
                if (keep[k] == pid)
                    have = true;
                if (keep[k] == ppid)
                    parent_in = true;
            }
            if (pid == exclude)
                continue;  // the sampler is a descendant of root
            if (parent_in && !have) {
                keep.push_back(pid);
                grew = true;
            }
        }
    }
    out = keep;
}

// System-wide jiffies from /proc/stat's first line
inline void read_system_jiffies(unsigned long long* busy, unsigned long long* idle) {
    *busy = 0;
    *idle = 0;
    FILE* f = fopen("/proc/stat", "r");
    if (!f)
        return;
    unsigned long long v[10] = {0};
    if (fscanf(f, "cpu %llu %llu %llu %llu %llu %llu %llu %llu %llu %llu", &v[0], &v[1], &v[2], &v[3], &v[4],
               &v[5], &v[6], &v[7], &v[8], &v[9]) >= 4) {
        *idle = v[3] + v[4];                              // idle + iowait
        *busy = v[0] + v[1] + v[2] + v[5] + v[6] + v[7];  // user nice sys irq softirq steal
    }
    fclose(f);
}

inline unsigned long long read_u64(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f)
        return 0;
    unsigned long long v = 0;
    if (fscanf(f, "%llu", &v) != 1)
        v = 0;
    fclose(f);
    return v;
}

// Forks the sampler
inline unsigned long long proc_vram_b(long pid, const char* kfd_id) {
    const std::string dir = "/sys/class/kfd/kfd/proc/" + std::to_string(pid) + "/";
    if (kfd_id && *kfd_id)
        return read_u64(dir + "vram_" + kfd_id);
    unsigned long long sum = 0;
    if (DIR* d = opendir(dir.c_str())) {
        while (dirent* e = readdir(d))
            if (!strncmp(e->d_name, "vram_", 5))
                sum += read_u64(dir + e->d_name);
        closedir(d);
    }
    return sum;
}

inline pid_t start(const std::string& csv_path, double interval_s, double (*now_us)()) {
    std::vector<Card> cards = enumerate();
    if (cards.empty())
        return -1;

    // Sibling CSV for the CPU rows, derived from the VRAM CSV path
    std::string cpu_path = csv_path;
    const size_t dot = cpu_path.rfind(".csv");
    if (dot != std::string::npos && dot + 4 == cpu_path.size()) {
        const size_t v = cpu_path.rfind("_vram");
        if (v != std::string::npos && v + 5 == dot)
            cpu_path.erase(v, 5);
        cpu_path.insert(cpu_path.size() - 4, "_cpu");
    } else {
        cpu_path += "_cpu.csv";
    }

    pid_t pid = fork();
    if (pid != 0)
        return pid;  // parent (or fork failure)

    // Die with the parent
    prctl(PR_SET_PDEATHSIG, SIGTERM);
    if (getppid() == 1)
        _exit(0);

    FILE* out = fopen(csv_path.c_str(), "w");
    if (!out)
        _exit(1);
    fprintf(out,
            "t_us,card,pci,vram_used_b,vram_total_b,gtt_used_b,"
            "gpu_busy_pct,mem_busy_pct\n");

    // vram_total never changes
    std::vector<unsigned long long> totals(cards.size());
    for (size_t i = 0; i < cards.size(); ++i)
        totals[i] = read_u64(cards[i].total_path);

    long tick = 0;
    const char* only_pci = getenv("ZKGPT_SAMPLER_PCI");
    std::vector<size_t> fast;
    for (size_t i = 0; i < cards.size(); ++i)
        if (!only_pci || !*only_pci || cards[i].pci == only_pci)
            fast.push_back(i);
    if (fast.empty())  // bus id did not match anything: sample all
        for (size_t i = 0; i < cards.size(); ++i)
            fast.push_back(i);
    const int slow_every = interval_s > 0 ? (int)(0.1 / interval_s + 0.5) : 1;

    FILE* cout_ = fopen(cpu_path.c_str(), "w");
    if (cout_)
        fprintf(cout_,
                "t_us,pid,ppid,utime_ticks,stime_ticks,num_threads,rss_pages,"
                "clk_tck,ncpu,comm,vram_b\n");
    // Per-process VRAM, from /sys/class/kfd/kfd/proc/<pid>/vram_<gpu_id>
    const char* kfd_id = getenv("ZKGPT_SAMPLER_KFD_GPU_ID");
    const long root = getppid();  // the run itself
    const long clk = sysconf(_SC_CLK_TCK);
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    std::vector<long> tracked;
    tracked.push_back(root);
    // Rediscover roughly once a second
    const int rescan_every = interval_s > 0 ? (int)(1.0 / interval_s + 0.5) : 1;

    // Default SIGTERM disposition ends the loop
    double next = now_us();
    for (;;) {
        const double t = now_us();
        const bool sweep = (slow_every <= 1) || (tick % slow_every == 0);
        for (size_t i = 0; i < cards.size(); ++i) {
            bool in_fast = false;
            for (size_t k = 0; k < fast.size(); ++k)
                if (fast[k] == i)
                    in_fast = true;
            if (!in_fast && !sweep)
                continue;
            const Card& c = cards[i];
            fprintf(out, "%.0f,%s,%s,%llu,%llu,%llu,%llu,%llu\n", t, c.name.c_str(), c.pci.c_str(),
                    read_u64(c.used_path), totals[i], read_u64(c.gtt_path), read_u64(c.busy_path),
                    read_u64(c.membusy_path));
        }
        fflush(out);

        if (cout_) {
            if (tick % (rescan_every > 0 ? rescan_every : 1) == 0)
                discover_descendants(root, (long)getpid(), tracked);
            for (size_t i = 0; i < tracked.size(); ++i) {
                ProcStat ps = read_proc_stat(tracked[i]);
                if (!ps.ok)
                    continue;  // exited between scan and read
                fprintf(cout_, "%.0f,%ld,%ld,%llu,%llu,%ld,%llu,%ld,%ld,%s,%llu\n", t, tracked[i], ps.ppid,
                        ps.utime, ps.stime, ps.threads, ps.rss_pages, clk, ncpu, ps.comm,
                        proc_vram_b(tracked[i], kfd_id));
            }
            unsigned long long sbusy = 0, sidle = 0;
            read_system_jiffies(&sbusy, &sidle);
            // pid 0 is the reserved system row
            fprintf(cout_, "%.0f,0,0,%llu,%llu,0,0,%ld,%ld,SYSTEM,0\n", t, sbusy, sidle, clk, ncpu);
            fflush(cout_);
        }
        ++tick;
        next += interval_s * 1e6;
        const double slack = next - now_us();
        if (slack > 0)
            usleep((useconds_t)slack);
        else
            next = now_us();
    }
}

inline void stop(pid_t pid) {
    if (pid <= 0)
        return;
    kill(pid, SIGTERM);
    int st = 0;
    waitpid(pid, &st, 0);
}

struct Stat {
    unsigned long long vram_peak_b = 0;
    unsigned long long busy_max = 0;
    double busy_sum = 0;  // for the mean over the window
    unsigned long long samples = 0;
    double busy_mean() const { return samples ? busy_sum / samples : 0.0; }
};

// Per-PCI-id stats over [t0_us, t1_us]
inline std::map<std::string, Stat> peak_in_window(const std::string& csv_path, double t0_us, double t1_us) {
    std::map<std::string, Stat> out;
    FILE* f = fopen(csv_path.c_str(), "r");
    if (!f)
        return out;
    char line[512];
    if (!fgets(line, sizeof(line), f)) {
        fclose(f);
        return out;
    }  // header
    while (fgets(line, sizeof(line), f)) {
        double t = 0;
        char card[64], pci[64];
        unsigned long long used = 0, tot = 0, gtt = 0, busy = 0, membusy = 0;
        if (sscanf(line, "%lf,%63[^,],%63[^,],%llu,%llu,%llu,%llu,%llu", &t, card, pci, &used, &tot, &gtt,
                   &busy, &membusy) != 8)
            continue;
        if (t < t0_us)
            continue;
        if (t1_us > 0 && t > t1_us)
            continue;
        Stat& s = out[pci];
        if (used > s.vram_peak_b)
            s.vram_peak_b = used;
        if (busy > s.busy_max)
            s.busy_max = busy;
        s.busy_sum += (double)busy;
        ++s.samples;
    }
    fclose(f);
    return out;
}

}  // namespace zkgpt_sampler
