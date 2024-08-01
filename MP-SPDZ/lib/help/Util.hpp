#pragma once

#include <cstdint>
#include <sstream>

#include "../../core/utils/print.hpp"

namespace IR {

enum class Level {
    WARNING,
    INFO,
    DEBUG,
    ERROR,
};

template <class... Args>
void log(Level level, Args... args) {
    std::ostringstream oss;
    switch (level) {
    case Level::DEBUG:
        oss << "\033[35mDEBUG\033[0m: ";
        ((oss << args), ...);
        oss << "\n";
        break;
    case Level::WARNING:
        oss << "\033[33mWARNING\033[0m: ";
        ((oss << args), ...);
        oss << "\n";
        break;
    case Level::INFO:
        oss << "\033[36mINFO\033[0m: ";
        ((oss << args), ...);
        oss << "\n";
        break;
    case Level::ERROR:
        oss << "\033[31mERROR\033[0m: ";
        ((oss << args), ...);
        oss << "\n";
        print(oss.str().c_str());
        exit(EXIT_FAILURE);
        break;
    }
    print(oss.str().c_str());
}

/**
 * return first <bits> bits from num
 * @param num
 * @param bits number of bits to keep
 * @return returns first `bits` bits from `num`
 */
inline uint64_t mask(const uint64_t& num, const uint64_t& bits) {
    return bits < (64) ? (num & ((1lu << bits) - 1)) : num;
}

inline uint64_t div_ceil(const uint64_t& a, const uint64_t& b) {
    return a == 0lu ? 0lu : 1lu + ((a - 1lu) / b);
}

} // namespace IR
