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

    template <class T>
    T next(std::function<T(const std::string&)> convert);

    inline int get_player() const { return player_num; };
    inline int get_vec_num() const { return vec; };
    inline bool is_open() const { return file; }

  private:
    int player_num;
    unsigned vec;
    char* file;
    size_t size;
    size_t cur;
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

template <size_t VEC_SIZE>
std::array<float, VEC_SIZE> next_input_f(const int& player_num, const int& thread_id) {
    std::array<float, VEC_SIZE> res{};
    if (current_phase == PHASE_INIT || player_num != PARTY)
        return res;

    for (size_t cur = 0; cur < VEC_SIZE; ++cur) {
        auto in = get_input_from(player_num, 0);
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

DATATYPE get_next_bit(const int& player_id) {
    if (current_phase == PHASE_INIT || player_id != PARTY)
        return ZERO;

    if (bit_queue.empty()) {
        auto input = next_input_f<SIZE_VEC>(player_id, 0);
        std::vector<UINT_TYPE> tmp;
        tmp.reserve(DATTYPE / BITLENGTH);

        for (size_t i = 0; i < BITLENGTH; i++) {
            for (const auto& ele : input) {
                tmp.push_back((UINT_TYPE(std::round(ele)) >> i) & 1u);
            }
            DATATYPE dat;
            orthogonalize_arithmetic((UINT_TYPE*)tmp.data(), &dat, 1);
            bit_queue.push(dat);
            tmp.clear();
        }
    }

    auto res = bit_queue.front();
    bit_queue.pop();
    return res;
}

} // namespace IR

#endif
