#pragma once
#include "../../config.h"
#include "../include/pch.h"
#include "../networking/buffers.h"

#if PRINT_TIMINGS == 1
std::chrono::high_resolution_clock::time_point time_start;
std::chrono::high_resolution_clock::time_point time_stop;
std::chrono::microseconds time_duration;
#define start_timer()                                           \
    if (current_phase != PHASE_INIT)                            \
    {                                                           \
        time_start = std::chrono::high_resolution_clock::now(); \
    }
#define stop_timer(FUNCTION_NAME)                                                                         \
    if (current_phase != PHASE_INIT)                                                                      \
    {                                                                                                     \
        time_stop = std::chrono::high_resolution_clock::now();                                            \
        time_duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start);    \
        double time_duration_sec = time_duration.count() / 1000000.0;                                     \
        std::cout << "P" << PARTY << ": " << FUNCTION_NAME << " took " << time_duration_sec << " seconds" \
                  << std::endl;                                                                           \
    }
/* printf("P%i: %s took %li seconds\n", PARTY, FUNCTION_NAME, time_duration.count()); } */
#else
#define start_timer()
#define stop_timer(FUNCTION_NAME)
#endif

#define print_online(x)                                                                   \
    if (current_phase == PHASE_LIVE && PRINT_IMPORTANT)                                   \
    {                                                                                     \
        std::cout << "P" << PARTY << ", PID" << process_offset << ": " << x << std::endl; \
    }
#define print_init(x)                                                                     \
    if (current_phase == PHASE_INIT && PRINT_IMPORTANT)                                   \
    {                                                                                     \
        std::cout << "P" << PARTY << ", PID" << process_offset << ": " << x << std::endl; \
    }

void print(const char* format, ...)
{
    va_list args;
    va_start(args, format);

    printf("P%i, PID%i: ", PARTY, process_offset);
    vprintf(format, args);

    va_end(args);
}

template <typename T>
void print_result(T* var)
{
    printf("P%i: Result: ", PARTY);
    uint8_t v8val[sizeof(T)];
    std::memcpy(v8val, var, sizeof(v8val));
    for (uint i = sizeof(T); i > 0; i--)
        std::cout << std::bitset<sizeof(uint8_t) * 8>(v8val[i - 1]);
    printf("\n");
}

#if FUNCTION_IDENTIFIER >= 35
struct Layer_Timing
{
    int layer_id;
    std::string layer_name;
    std::chrono::microseconds pre_time_duration;
    std::chrono::microseconds live_time_duration;
    std::chrono::high_resolution_clock::time_point timer;
    uint64_t elements_sent_live;
    uint64_t elements_sent_pre;
    uint64_t elements_received_live;
    uint64_t elements_received_pre;
    uint64_t elements_sent_live_bytes;
};

std::vector<Layer_Timing> layer_stats;

void start_layer_stats(std::string layer_name, int layer_id)
{
    if (current_phase == PHASE_INIT || PROTOCOL == 13)
    {
        layer_stats.push_back(Layer_Timing());
        layer_stats.back().layer_id = layer_id;
        layer_stats.back().layer_name = layer_name;
#if num_players == 3
#if PRE == 1
        layer_stats.back().elements_sent_pre = total_send_pre[0] + total_send_pre[1];
        layer_stats.back().elements_received_pre = total_recv_pre[0] + total_recv_pre[1];
#endif
        layer_stats.back().elements_sent_live = total_send[0] + total_send[1];
        layer_stats.back().elements_received_live = total_recv[0] + total_recv[1];
#elif num_players == 4
#if PRE == 1
        layer_stats.back().elements_sent_pre = total_send_pre[0] + total_send_pre[1] + total_send_pre[2];
        layer_stats.back().elements_received_pre = total_recv_pre[0] + total_recv_pre[1] + total_recv_pre[2];
#endif
        layer_stats.back().elements_sent_live = total_send[0] + total_send[1] + total_send[2];
        layer_stats.back().elements_received_live = total_recv[0] + total_recv[1] + total_recv[2];
#endif
    }
    else if (current_phase == PHASE_PRE)
    {
        layer_stats[layer_id].timer = std::chrono::high_resolution_clock::now();
    }
    else if (current_phase == PHASE_LIVE)
    {
        layer_stats[layer_id].timer = std::chrono::high_resolution_clock::now();
    }
}

void stop_layer_stats(int layer_id)
{
    if (current_phase == PHASE_INIT)
    {
#if num_players == 3
#if PRE == 1
        layer_stats.back().elements_sent_pre =
            total_send_pre[0] + total_send_pre[1] - layer_stats.back().elements_sent_pre;
        layer_stats.back().elements_received_pre =
            total_recv_pre[0] + total_recv_pre[1] - layer_stats.back().elements_received_pre;
#endif
        layer_stats.back().elements_sent_live = total_send[0] + total_send[1] - layer_stats.back().elements_sent_live;
        layer_stats.back().elements_received_live =
            total_recv[0] + total_recv[1] - layer_stats.back().elements_received_live;
#elif num_players == 4
#if PRE == 1
        layer_stats.back().elements_sent_pre =
            total_send_pre[0] + total_send_pre[1] + total_send_pre[2] - layer_stats.back().elements_sent_pre;
        layer_stats.back().elements_received_pre =
            total_recv_pre[0] + total_recv_pre[1] + total_recv_pre[2] - layer_stats.back().elements_received_pre;
#endif
        layer_stats.back().elements_sent_live =
            total_send[0] + total_send[1] + total_send[2] - layer_stats.back().elements_sent_live;
        layer_stats.back().elements_received_live =
            total_recv[0] + total_recv[1] + total_recv[2] - layer_stats.back().elements_received_live;
#endif
    }
    else if (current_phase == PHASE_PRE)
    {
        layer_stats[layer_id].pre_time_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - layer_stats[layer_id].timer);
    }
    else if (current_phase == PHASE_LIVE)
    {
        layer_stats[layer_id].live_time_duration = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - layer_stats[layer_id].timer);
    }
}

void print_layer_stats()
{
    std::cout.precision(4);
#if PRE == 1
    for (auto& layer : layer_stats)
    {
        std::cout << "P" << PARTY << ": --NN_STATS (Individual)-- ID: " << layer.layer_id << " " << layer.layer_name
                  << "    MB SENT:" << layer.elements_sent_live * (double(DATTYPE) / (8000 * 1000))
                  << "   MB RECEIVED:" << layer.elements_received_live * (double(DATTYPE) / (8000 * 1000))
                  << "   MB SENT PRE:" << layer.elements_sent_pre * (double(DATTYPE) / (8000 * 1000))
                  << "   MB RECEIVED PRE: " << layer.elements_received_pre * (double(DATTYPE) / (8000 * 1000))
                  << "    ms LIVE: " << double(layer.live_time_duration.count()) / 1000
                  << "    ms PRE: " << double(layer.pre_time_duration.count()) / 1000 << std::endl;
    }
#else
    for (auto& layer : layer_stats)
    {
        std::cout << "P" << PARTY << ": --NN_STATS (Individual)-- ID: " << layer.layer_id << " " << layer.layer_name
                  << "    MB SENT:" << layer.elements_sent_live * (double(DATTYPE) / (8000 * 1000))
                  << "   MB RECEIVED:" << layer.elements_received_live * (double(DATTYPE) / (8000 * 1000))
                  << "    ms LIVE: " << double(layer.live_time_duration.count()) / 1000 << std::endl;
    }
#endif
    // define hashmap with all layer types
    std::unordered_map<std::string, std::vector<Layer_Timing>> layer_stats_map;
    for (auto& layer : layer_stats)
    {
        layer_stats_map[layer.layer_name].push_back(layer);
    }

    for (auto& layer_pair : layer_stats_map)
    {
        uint64_t total_elements_sent_live = 0;
        uint64_t total_elements_sent_pre = 0;
        uint64_t total_elements_received_live = 0;
        uint64_t total_elements_received_pre = 0;
        double total_live_time_duration = 0;
        double total_pre_time_duration = 0;
        for (auto& layer_stats : layer_pair.second)
        {
            total_elements_sent_live += layer_stats.elements_sent_live;
            total_elements_sent_pre += layer_stats.elements_sent_pre;
            total_elements_received_live += layer_stats.elements_received_live;
            total_elements_received_pre += layer_stats.elements_received_pre;
            total_live_time_duration += layer_stats.live_time_duration.count();
            total_pre_time_duration += layer_stats.pre_time_duration.count();
        }
#if PRE == 1
        std::cout << "P" << PARTY << ": --NN_STATS (Aggregated)-- " << layer_pair.first
                  << "    MB SENT:" << total_elements_sent_live * (double(DATTYPE) / (8000 * 1000))
                  << "   MB RECEIVED: " << total_elements_received_live * (double(DATTYPE) / (8000 * 1000))
                  << "   MB SENT PRE:" << total_elements_sent_pre * (double(DATTYPE) / (8000 * 1000))
                  << "   MB RECEIVED PRE: " << total_elements_received_pre * (double(DATTYPE) / (8000 * 1000))
                  << "    ms LIVE: " << double(total_live_time_duration) / 1000
                  << "    ms PRE: " << double(total_pre_time_duration) / 1000 << std::endl;
#else
        std::cout << "P" << PARTY << ": --NN_STATS (Aggregated)-- " << layer_pair.first
                  << "    MB SENT:" << total_elements_sent_live * (double(DATTYPE) / (8000 * 1000))
                  << "   MB RECEIVED:" << total_elements_received_live * (double(DATTYPE) / (8000 * 1000))
                  << "    ms LIVE: " << double(total_live_time_duration) / 1000 << std::endl;
#endif
    }
}
#endif

void print_communication()
{
#if PRINT_IMPORTANT == 1
    // set decimal precision
    std::cout.precision(4);
#if PRE == 1
#if num_players == 2
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Sending to other players:" << total_send_pre[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB "
              << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Receiving from other players:" << total_recv_pre[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB "
              << std::endl;
#elif num_players == 3
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Sending to other players:" << total_send_pre[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_send_pre[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Receiving from other players:" << total_recv_pre[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_recv_pre[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
#elif num_players == 4
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Sending to other players:" << total_send_pre[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_send_pre[PMIDDLE] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_send_pre[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Receiving from other players:" << total_recv_pre[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_recv_pre[PMIDDLE] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_recv_pre[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
#endif
#endif
#if num_players == 2
    std::cout << "P" << PARTY << ", ONLINE, PID" << process_offset << ": "
              << "Sending to other players:" << total_send[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB "
              << std::endl;
    std::cout << "P" << PARTY << ", ONLINE PID" << process_offset << ": "
              << "Receiving from other players:" << total_recv[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB "
              << std::endl;
#elif num_players == 3
    std::cout << "P" << PARTY << ", ONLINE, PID" << process_offset << ": "
              << "Sending to other players:" << total_send[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_send[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
    std::cout << "P" << PARTY << ", ONLINE PID" << process_offset << ": "
              << "Receiving from other players:" << total_recv[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_recv[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
#elif num_players == 4
    std::cout << "P" << PARTY << ", ONLINE, PID" << process_offset << ": "
              << "Sending to other players:" << total_send[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_send[PMIDDLE] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_send[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
    std::cout << "P" << PARTY << ", ONLINE, PID" << process_offset << ": "
              << "Receiving from other players:" << total_recv[PPREV] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_recv[PMIDDLE] * (float(DATTYPE) / (8000 * 1000)) << "MB, "
              << total_recv[PNEXT] * (float(DATTYPE) / (8000 * 1000)) << "MB " << std::endl;
#endif

#endif
}
