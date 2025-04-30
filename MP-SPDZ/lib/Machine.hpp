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

    /**
     * The time passed since this timer started
     *
     * @return
     * - if started -> time passed since timer started (in seconds)
     * @return
     * - else 0
     */
    double get_time() const {
        if (started)
            return std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - cur)
                .count();
        else
            return 0;
    }

  private:
    bool started = false; // true if the timer is measuring time

    std::chrono::time_point<std::chrono::high_resolution_clock>
        cur; // time passed since timer started
};

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N = 64>
class Machine {
  public:
    /**
     * Loads schedule-file and calls `load_schedule` to initialize programs
     * @param path path to schedule-file
     */
    explicit Machine(std::string&& path);

    /**
     * Reads schedule-file and bytcode-files -> initializes `progs`
     */
    void load_schedule(std::string&& path);
    void setup(); // Allocates registers/memory
    void run();   // Starts main thread and inits memory

    /**
     * Start tape without copying the whole program/tape as this is a single thread program
     *
     * @param tape tape number defined by MP-SPDZ compiler
     * @param arg thread argument as each thread has an argument
     */
    void run_tape_no_thread(const unsigned& tape, const int& arg);

    /**
     * Starts tape in a new thread. Is NOT used at the moment
     *
     * @warning
     * - Should only be used for the main thread (tape == 0).
     * @warning
     * - Running the same tape twice directly after one another will fail because the program is
     * not copied therefore both threads would operate on the same registers
     */
    thread& run_tape(const unsigned& tape, const int& arg);

    /* Main method for each thread */
    inline static void execute(Machine& m, Program<int_t, cint, Share, sint, sbit, BitShare, N>& p,
                               int arg) {
        p.run(m, arg, 0); // only 1st(0) thread is started here
    }

    /**
     * Update greatest required memory address required for computation/storage
     *
     * @param type type of the memory cell addressed
     * @param addr new address the MP-SPDZ-compiler tries to access
     */
    void update_max_mem(const Type& type, const unsigned& addr);

    vector<sint> s_mem;               // secret share memory (`Additive_Shares`)
    vector<cint> c_mem;               // clear int memory
    vector<int_t> ci_mem;             // 64-bit int memory
    vector<sbit<N, BitShare>> sb_mem; // sbit memory (`XOR_Shares`)
#if BITLENGTH == 64
    vector<cint> cb_mem; // clear bit memory
#else
    vector<int_t> cb_mem; // clear bit memory
#endif

    /**
     * @return Output stream used for printing results etc.
     */
    std::ostream& get_out() { return out_stream; }

    /**
     * Start timer at `index`
     */
    void start(const int& index);

    /**
     * Stop timer at `index` and print the final time
     */
    void stop(const int& index);

    /**
     * Print the time that passed since starting the main tape
     */
    void time();

    Input public_input; // Used to read public-input from file

    size_t get_random() { return rand_engine(); } // same for all parties (very insecure ^^)
    size_t get_random_diff() { return rand_engine_diff(); } // different for all parties

  private:
    vector<Program<int_t, cint, Share, sint, sbit, BitShare, N>> progs; // all bytecode-files
    vector<thread> tapes; // all threads running in the VM (should only have one element)

    unsigned max_mem[REG_TYPES];   // save max value -> size of memory for all types
    bool loaded;                   // true if schedule file has been loaded successfully
    std::map<int, Timer> timer;    // map of timers (should not be used for benchmarks)
    std::mt19937 rand_engine;      // creates the same random numbers for every party (testing)
    std::mt19937 rand_engine_diff; // creates random numbers (different for every party)

    std::ostream out_stream;
};

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
Machine<int_t, cint, Share, sint, sbit, BitShare, N>::Machine(std::string&& path)
    : max_mem(), loaded(false), rand_engine(21), rand_engine_diff(std::random_device()()),
      out_stream(std::cout.rdbuf()) {
    load_schedule(std::move(path));
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::load_schedule(std::string&& path) {
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

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::setup() {
    c_mem.resize(max_mem[static_cast<unsigned>(Type::CINT)]);
    s_mem.resize(max_mem[static_cast<unsigned>(Type::SINT)]);
    ci_mem.resize(max_mem[static_cast<unsigned>(Type::INT)]);
    cb_mem.resize(max_mem[static_cast<unsigned>(Type::CBIT)]);
    sb_mem.resize(max_mem[static_cast<unsigned>(Type::SBIT)]);

    for (auto& prog : progs)
        prog.setup();
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::run() {
    if (progs.size() == 0 || not loaded) {
        return;
    }

    setup();

    timer[0].start();
    progs[0].run(*this, 0, 0);
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
thread& Machine<int_t, cint, Share, sint, sbit, BitShare, N>::run_tape(const unsigned& tape,
                                                                       const int& arg) {
    if (progs.size() <= tape)
        log(Level::ERROR, "can't run tape: ", tape);

    return tapes.emplace_back(execute, std::ref(*this), std::ref(progs[tape]), arg);
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::update_max_mem(const Type& type,
                                                                          const unsigned& addr) {
    size_t i = static_cast<size_t>(type);

    if (max_mem[i] < addr) {
        max_mem[i] = addr;
    }
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::start(const int& index) {
    if (IS_ONLINE) {
        timer[index].start();
        get_out() << "starting timer " << index << " at 0\n";
    }
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::stop(const int& index) {
    if (IS_ONLINE) {
        auto res = timer[index].stop();
        get_out() << "stopping timer " << index << " at " << res << "\n";
    }
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::time() {
    if (IS_ONLINE) {
        auto res = timer.at(0).get_time();
        get_out() << "Elapsed time: " << res << "s\n";
    }
}

template <class int_t, class cint, class Share, class sint, template <int, class> class sbit,
          class BitShare, int N>
void Machine<int_t, cint, Share, sint, sbit, BitShare, N>::run_tape_no_thread(const unsigned& tape,
                                                                              const int& arg) {
    auto& prog = progs[tape]; // no copy needed
    prog.run(*this, arg, 1);  // should always be thread 1
}

} // namespace IR

#endif
