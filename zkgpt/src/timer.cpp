#include "timer.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace std;

void timer::start() 
{
    assert(status == false);
    t0 = clock::now();
    if (!has_first_start) {
        first_start_time = t0;
        has_first_start = true;
    }
    status = true;
}

void timer::merge_timer(timer_range val){
    if (val.ready) {
        if (merge_valid)
            merged_range = combine_ranges({val, merged_range});
        else {
            merge_valid = true;
            merged_range = val;
        }
    }
}

void timer::stop(const char* output,bool total,bool restart) 
{
    assert(status == true);
    auto t1 = clock::now();
    auto time_span_sec = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0);
    total_time_sec += time_span_sec.count();
    last_stop_time = t1;
    has_last_stop = true;
    status = false;
    if(strlen(output)>0)
    {
        if(total)
            cout<<output<<" time: "<<total_time_sec<<endl;
        else
            cout<<output<<" time: "<<time_span_sec.count()<<endl;
    }
    if(restart)
        start();
}
void timer::stop()
{
    stop("",true,false);
} 

timer::time_point timer::firstStartTime() const {
    assert(has_first_start);
    return first_start_time;
}

timer::time_point timer::lastStopTime() const {
    assert(has_last_stop);
    return last_stop_time;
}

double timer::firstStartEpochSec() const {
    assert(has_first_start);
    return toEpochSeconds(first_start_time);
}

double timer::lastStopEpochSec() const {
    assert(has_last_stop);
    return toEpochSeconds(last_stop_time);
}

timer_range timer::range() const {
    timer_range result;
    result.duration_sec = total_time_sec;
    if (has_first_start && has_last_stop) {
        result.ready = true;
        result.start_epoch_sec = toEpochSeconds(first_start_time);
        result.end_epoch_sec = toEpochSeconds(last_stop_time);
    }
    if(merge_valid)
        return combine_ranges({result,merged_range});
    else
        return result;
}

double timer::toEpochSeconds(const time_point &tp) {
    return std::chrono::duration_cast<std::chrono::duration<double>>(tp.time_since_epoch()).count();
}

timer_range combine_ranges(const std::vector<timer_range> &ranges) {
    timer_range combined;
    bool has_value = false;
    for (const auto &range : ranges) {
        if (!range.ready) continue;
        combined.duration_sec += range.duration_sec;
        has_value = true;
    }
    if (has_value) {
        combined.ready = true;
        combined.start_epoch_sec = 0.0;
        combined.end_epoch_sec = combined.duration_sec;
    }
    return combined;
}

timer_range combine_ranges(std::initializer_list<timer_range> ranges) {
    return combine_ranges(std::vector<timer_range>(ranges));
}

std::string format_range(const timer_range &range) {
    if (!range.ready) return "";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6)
        << range.start_epoch_sec << "," << range.end_epoch_sec;
    return oss.str();
}

double range_duration(const timer_range &range) {
    return range.ready ? range.duration_sec : 0.0;
}
