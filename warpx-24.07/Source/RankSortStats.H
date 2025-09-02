#ifndef RANK_SORT_STATS_h_
#define RANK_SORT_STATS_h_

struct RankSortStats {
    int rebuild_count = 0;
    long total_empty_slots = 0;
    long total_capacity = 0;
    double local_time = 0.0;

    RankSortStats(int count, long slots, long capacity, double time = 0.0)
    : rebuild_count(count),
    total_empty_slots(slots),
    total_capacity(capacity),
    local_time(time)
    {}

    RankSortStats() = default;

    RankSortStats& operator+=(const RankSortStats& rhs) {
        rebuild_count += rhs.rebuild_count;
        total_empty_slots += rhs.total_empty_slots;
        total_capacity += rhs.total_capacity;
        local_time += rhs.local_time;
        return *this;
    }
};

#endif