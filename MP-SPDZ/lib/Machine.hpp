#ifndef MACHINE_H
#define MACHINE_H

#include <chrono>
#include <fstream>  // ifstream for schedule file
#include <iostream> // to print results to STDOUT
#include <map>
#include <sstream> // to seperate bytecode files
#include <string>  // file path
#include <thread>  // for the tapes
#include <vector>  // register

#include "Constants.hpp"
#include "Program.hpp"
#include "Shares/CInteger.hpp"
#include "Shares/Integer.hpp"
#include "help/Util.hpp"

using std::thread;
using std::vector;

namespace IR {

struct Timer {
    void start() {
        started = true;
        cur = std::chrono::high_resolution_clock::now();
    }
    double stop() {
        double res = get_time();
        started = false;
        return res;
    }
    double get_time() const {
        if (started)
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cur)
                .count();
        else
            return 0;
    }

  private:
    bool started = false;

    std::chrono::time_point<std::chrono::high_resolution_clock> cur;
};

template <class sint, template <int, class> class sbit, class BitShare, int N = 64>
class Machine {
  public:
    explicit Machine(std::string&& path);

    void load_schedule(std::string&& path); // parse schedule file
    void setup();                           // allocate registers/memory
    void run();                             // starts main thread and init memory

    thread& run_tape(const unsigned& tape, const int& arg);

    /* main method for each thread */
    inline static void execute(Machine& m, Program<sint, sbit, BitShare, N>& p, int arg) {
        p.run(m, arg);
    }

    void update_max_mem(const Type& type,
                        const unsigned& addr); // required while parsing to get memory size

    vector<sint> s_mem;
    vector<CInteger<INT_TYPE, UINT_TYPE>> c_mem;
    vector<Integer<int64_t, uint64_t>> ci_mem;
    vector<sbit<N, BitShare>> sb_mem;
#if BITLENGTH == 64
    vector<CInteger<INT_TYPE, UINT_TYPE>> cb_mem;
#else
    vector<Integer<int64_t, uint64_t>> cb_mem;
#endif

    std::ostream& get_out() const { return std::cout; }

    void start(const int& index);
    void stop(const int& index);
    void time() const;

    Input public_input;

  private:
    vector<Program<sint, sbit, BitShare, N>> progs; // all bytecode-files
    vector<thread> tapes;                           // all threads running the VM

    unsigned max_mem[REG_TYPES]; // save max value -> size of memory
    bool loaded;
    std::map<int, Timer> timer;
};

template <class sint, template <int, class> class sbit, class BitShare, int N>
Machine<sint, sbit, BitShare, N>::Machine(std::string&& path) : max_mem(), loaded(false) {
    load_schedule(std::move(path));
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Machine<sint, sbit, BitShare, N>::load_schedule(std::string&& path) {
    std::ifstream file(SCHEDULES_PATH + path);

    if (not file.is_open()) {
        log(Level::ERROR, "couldn't open file: \"", SCHEDULES_PATH + path, "\"");
        return;
    }

    std::string line;

    std::getline(file, line);
    tapes.reserve(std::stol(line));

    std::getline(file, line);
    size_t files = std::stol(line);

    progs.reserve(files);
    std::getline(file, line);

    std::stringstream s(line);
    std::string cur;

    while (s >> cur) {
        size_t i = cur.find(':');
        if (i < cur.length())
            progs.emplace_back(BYTECODE_PATH + cur.substr(0, i) + ".bc", progs.size());
    }

    // legacy lines
    std::getline(file, line);
    std::getline(file, line);

    std::getline(file, line); // compile command

    std::getline(file, line); // domain requirements

    // TODO: some options
    file.close();

    loaded = true;
    for (auto& prog : progs) {
        if (loaded)
            loaded = prog.load_program(*this);
        else
            break;
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Machine<sint, sbit, BitShare, N>::setup() {
    c_mem.resize(max_mem[static_cast<unsigned>(Type::CINT)]);
    s_mem.resize(max_mem[static_cast<unsigned>(Type::SINT)]);
    ci_mem.resize(max_mem[static_cast<unsigned>(Type::INT)]);
    cb_mem.resize(max_mem[static_cast<unsigned>(Type::CBIT)]);
    sb_mem.resize(max_mem[static_cast<unsigned>(Type::SBIT)]);

    for (auto& prog : progs)
        prog.setup();
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Machine<sint, sbit, BitShare, N>::run() {
    if (progs.size() == 0 || not loaded) {
        return;
    }

    setup();

    timer[0].start();
    thread& cur = run_tape(0, 0);
    cur.join();
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
thread& Machine<sint, sbit, BitShare, N>::run_tape(const unsigned& tape, const int& arg) {
    if (progs.size() <= tape)
        log(Level::ERROR, "can't run tape: ", tape);

    return tapes.emplace_back(execute, std::ref(*this), std::ref(progs[tape]), arg);
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Machine<sint, sbit, BitShare, N>::update_max_mem(const Type& type, const unsigned& addr) {
    size_t i = static_cast<size_t>(type);

    if (max_mem[i] < addr) {
        max_mem[i] = addr;
    }
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Machine<sint, sbit, BitShare, N>::start(const int& index) {
    timer[index].start();
    get_out() << "starting timer " << index << " at 0\n";
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Machine<sint, sbit, BitShare, N>::stop(const int& index) {
    auto res = timer[index].stop();
    get_out() << "stopping timer " << index << " at " << res << "\n";
}

template <class sint, template <int, class> class sbit, class BitShare, int N>
void Machine<sint, sbit, BitShare, N>::time() const {
    auto res = timer.at(0).get_time();
    get_out() << "Elapsed time: " << res << "\n";
}

} // namespace IR

#endif
