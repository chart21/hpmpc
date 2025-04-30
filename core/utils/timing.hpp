#pragma once
#include "../include/pch.h"
template <typename F, typename... Args>
auto funcTime(std::string printText, F func, Args&&... args)
{
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    double time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - t1).count();
    std::cout << "---TIMING--- " << time / (1000000) << "s " << printText << '\n';
    return time;
}
