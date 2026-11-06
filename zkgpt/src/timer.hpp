// Created by 69029 on 3/25/2021

#ifndef ZKCNN_TIMER_HPP
#define ZKCNN_TIMER_HPP

#include <chrono>
#include <assert.h>
#include <vector>
#include <string>
#include <initializer_list>
#include <iomanip>

struct timer_range {
    bool ready = false;
    double start_epoch_sec = 0.0;
    double end_epoch_sec = 0.0;
    double duration_sec = 0.0;
};

class timer {
public:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;

    explicit timer(bool accumulate_ = true) {
        (void)accumulate_;
        total_time_sec = 0;
        status = false;
        has_first_start = false;
        has_last_stop = false;
        merge_valid = false;
    }
    void start();
    void stop(const char* out, bool total = true, bool restart = true);
    void stop();
    void clear() {
        total_time_sec = 0;
        status = false;
        has_first_start = false;
        has_last_stop = false;
        merge_valid = false;
        merged_range = {};
    }
    double elapse_sec() const {
        assert(status == false);
        return total_time_sec;
    }
    bool hasStartTimestamp() const { return has_first_start; }
    bool hasStopTimestamp() const { return has_last_stop; }
    time_point firstStartTime() const;
    time_point lastStopTime() const;
    double firstStartEpochSec() const;
    double lastStopEpochSec() const;
    timer_range range() const;
    void merge_timer(const timer_range);

private:
    static double toEpochSeconds(const time_point& tp);

    timer_range merged_range;
    bool merge_valid;
    time_point t0;
    time_point first_start_time;
    time_point last_stop_time;
    double total_time_sec;
    bool status;
    bool has_first_start;
    bool has_last_stop;
};

timer_range combine_ranges(const std::vector<timer_range>& ranges);
timer_range combine_ranges(std::initializer_list<timer_range> ranges);
std::string format_range(const timer_range& range);
double range_duration(const timer_range& range);

#endif  // ZKCNN_TIMER_HPP
