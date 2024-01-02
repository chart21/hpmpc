#pragma once
#include <bitset>
#include <cstring>
#include <iostream>
#include <stdint.h>
#include <stdarg.h>
#include <stdio.h>
#include <chrono>
#include "../config.h"

#if PRINT_TIMINGS == 1
std::chrono::high_resolution_clock::time_point time_start;
std::chrono::high_resolution_clock::time_point time_stop;
std::chrono::microseconds time_duration;
#define start_timer() \
    if(current_phase == 1 ) { \
    time_start = std::chrono::high_resolution_clock::now(); \
    } 
#define stop_timer(FUNCTION_NAME) \
    if(current_phase == 1 ) { \
    time_stop = std::chrono::high_resolution_clock::now(); \
    time_duration = std::chrono::duration_cast<std::chrono::microseconds>(time_stop - time_start); \
        double time_duration_sec = time_duration.count() / 1000000.0; \
        std::cout << "P" << PARTY << ": " << FUNCTION_NAME << " took " << time_duration_sec << " seconds" << std::endl; }
    /* printf("P%i: %s took %li seconds\n", PARTY, FUNCTION_NAME, time_duration.count()); } */
#else
#define start_timer()
#define stop_timer(FUNCTION_NAME)
#endif

#define print_online(x) \
    if(current_phase == 1) { \
        std::cout << "P" << PARTY << ", PID" << process_offset << ": " << x << std::endl; \
    }

void print(const char* format, ...) {
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
        std::cout << std::bitset<sizeof(uint8_t)*8>(v8val[i-1]);
    printf("\n");
}


