#ifndef INPUT_H
#define INPUT_H

#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../Constants.hpp"
#include "Util.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace IR {

/**
 * @class Input represents the input for a party
 * @note only for a single number. For SIMD use multiple Input classes and use <vec> to specify the
 * position
 */
class Input {
  public:
    explicit Input(const std::string& path, const int& player_num, const int& cur_vec)
        : player_num(player_num), vec(cur_vec), cur(0) {
        open_input_file(path);
    }

    Input() : player_num(0), vec(1), file(nullptr), cur(0){};

    Input(const Input& other) = delete;
    Input(Input&& other) noexcept
        : player_num(std::move(other.player_num)), file(std::move(other.file)),
          size(std::move(other.size)), cur(std::move(other.cur)) {
        other.file = nullptr;
    }

    Input& operator=(const Input& other) = delete;
    Input& operator=(Input&& other) noexcept {
        if (&other == this)
            return *this;

        file = std::move(other.file);
        size = std::move(other.size);
        cur = std::move(other.cur);
        player_num = std::move(other.player_num);

        other.file = nullptr;
        return *this;
    }

    /**
     * creates a memory mapping of the file at <path> which is used to read input
     */
    void open_input_file(const std::string& path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            log(Level::ERROR, "couldn 't open file \"", path, "\"");
        }

        struct stat stats;
        if (fstat(fd, &stats) == -1) {
            log(Level::ERROR, "couldn't get stats for \"", path, "\"");
        }

        size = stats.st_size;
        file = static_cast<char*>(mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0));

        close(fd);

        if (file == MAP_FAILED) {
            log(Level::ERROR, "map failed for \"", path, "\"");
        }
    }

    ~Input() {
        if (file) {
            munmap(file, size);
        }
    }

    /**
     * reads the next characters from the file until there is whitespace and converts the string to
     * type T using @param convert a number of type <T> (integer/float) from the input file
     *
     * @param convert takes a string and returns an object of type <T>
     * @note T should be a number type (int/float/...)
     */
    template <class T>
    T next(std::function<T(const std::string&)> convert);

    inline int get_player() const { return player_num; };
    inline int get_vec_num() const { return vec; };
    inline bool is_open() const { return file; }

  private:
    int player_num; // used to identify the party that is associated with this file
    unsigned vec;   // position for SIMD values
    char* file;     // mappping of the file
    size_t size;    // number of bytes in the file
    size_t cur;     // current position in the file
};

template <class T>
T Input::next(std::function<T(const std::string&)> convert) {
    if (!is_open())
        return 0;
    std::string_view content(file + cur, size - cur);
    size_t index = content.find_first_of(' ');

    if (index == std::string::npos) {
        cur = 0;
        return convert(std::string(content));
    }

    cur += index + 1;
    T res = convert(std::string(content.substr(0, index)));
    return res;
}

static std::vector<Input> inputs;

std::vector<Input>::iterator get_input_from(const int& player_num, const size_t& cur_vec) {
    auto it = inputs.begin();
    for (; it != inputs.end(); ++it) {
        if (it->get_player() == player_num && it->get_vec_num() == cur_vec)
            return it;
    }

    return it;
}

/**
 * read VEC_SIZE integers from the <input-file>
 * @param player_num the party who privately wants to share their input
 * @param thread_id the thread trying to read the input (0/1)
 * @return VEC_SIZE integers from party @param player_id
 */
template <size_t VEC_SIZE>
std::array<int, VEC_SIZE> next_input(const int& player_num, const int& thread_id) {
    std::array<int, VEC_SIZE> res{};
    if (current_phase == PHASE_INIT || player_num != PARTY)
        return res;

    for (size_t cur = 0; cur < VEC_SIZE; cur++) {
        auto in = get_input_from(player_num, cur);
        if (in == inputs.end()) {
            inputs.emplace_back(INPUT_PATH + "Input-P" + std::to_string(player_num) + "-" +
                                    std::to_string(thread_id) + "-" + std::to_string(cur),
                                player_num, cur);
            in = inputs.end() - 1;
        }

        int a = in->template next<int>(
            [](const std::string& s) -> int { return std::stoi(s.c_str(), nullptr, 10); });

        res[cur] = a;
    }
    return res;
}

/**
 * read VEC_SIZE floats from <input-file>
 * @param player_num the party who privately wants to share their input
 * @param thread_id the thread trying to read the input (0/1)
 * @return VEC_SIZE floats from party @param player_num
 */
template <size_t VEC_SIZE>
std::array<float, VEC_SIZE> next_input_f(const int& player_num, const int& thread_id) {
    std::array<float, VEC_SIZE> res{};
    if (current_phase == PHASE_INIT || player_num != PARTY)
        return res;

    for (size_t cur = 0; cur < VEC_SIZE; ++cur) {
        auto in = get_input_from(player_num, cur);
        if (in == inputs.end()) {
            inputs.emplace_back(INPUT_PATH + "Input-P" + std::to_string(player_num) + "-" +
                                    std::to_string(thread_id) + "-" + std::to_string(cur),
                                player_num, 0);
            in = inputs.end() - 1;
        }

        float a = in->template next<float>(
            [](const std::string& s) -> float { return std::stof(s.c_str(), nullptr); });
        res[cur] = a;
    }
    return res;
}

static std::queue<DATATYPE> bit_queue;

/**
 * @if <bit_queue> empty -> Reads a single float from <input-file> and stores BITENGTH bits in a
 * queue
 * @else pops queue and returns VEC_SIZE bits from party @param player_id
 * @param player_id party to share their private input
 * @return VEC_SIZE bits from party @param player_id
 */
template <size_t VEC_SIZE>
DATATYPE get_next_bit(const int& player_id) {
    if (current_phase == PHASE_INIT || player_id != PARTY)
        return ZERO;

    alignas(DATATYPE) UINT_TYPE tmp[VEC_SIZE];
    if (bit_queue.empty()) {
        auto input = next_input_f<VEC_SIZE>(player_id, 0);

        for (size_t i = 0; i < BITLENGTH; i++) {
            for (size_t j = 0; j < VEC_SIZE; ++j) {
                tmp[j] = (UINT_TYPE(std::round(input[j])) >> i) & 1u;
            }
            DATATYPE dat;
            orthogonalize_arithmetic(tmp, &dat, 1);
            bit_queue.push(dat);
        }
    }

    auto res = bit_queue.front();
    bit_queue.pop();
    return res;
}

} // namespace IR

#endif
