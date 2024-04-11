#include "Util.hpp"

#include <netinet/in.h>
#include <unistd.h>

#include <cassert>
#include <cstring>

namespace IR {

int64_t read_long(std::istream& fd) {
    int64_t res = 0;
    fd.read((char*)&res, 8);
    return be64toh(res);
}

int32_t read_int(std::istream& fd) {
    int32_t res = 0;
    fd.read((char*)&res, 4);
    return be32toh(res);
}

uint64_t mask(const uint64_t& num, const uint64_t& bits) {
    return bits < (64) ? (num & ((1lu << bits) - 1)) : num;
}

} // namespace IR
