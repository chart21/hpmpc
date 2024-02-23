#include "Util.hpp"

#include <netinet/in.h>
#include <unistd.h>

#include <cassert>
#include <cstring>

namespace IR {

uint64_t read_next_int(const int& fd, unsigned char* buf, const size_t& size) {
    assert(size <= 8);
    int ans = read(fd, buf, size);

    if (ans <= 0) {
        log(Level::ERROR, "no number read");
    }

    return to_int_n(buf, size);
}

uint64_t to_int_n(const unsigned char* buf, const size_t& size) {
    assert(size <= 8);
    uint64_t cur = 0;

    for (size_t i = 0; i < size; ++i) {
        cur |= static_cast<uint64_t>(buf[i]) << (i * 8ul);
    }

    if (size == 8)
        return be64toh(cur);
    return int(be32toh(cur));
}

} // namespace IR
