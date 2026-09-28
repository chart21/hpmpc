#pragma once
// ADDITIONAL_RELU_THREADS: element-parallel circuit levels for single-batch inference.
//
// hpmpc vectorizes only across DATTYPE / BITLENGTH independent inputs, so one image runs every level of
// the ReLU's MSB adder, A2B and bit injection element by element on one thread. Within a level all
// elements run the same code: each consumes the same number of words from the protocol's streams
// (sends to and receives from the other party, preprocessed outputs, its own randomness), in element
// order. stream_parallel_for measures that per-element consumption on the first element, then runs the
// remaining elements on the worker pool, each worker with cursors into the streams at exactly the
// positions the serial loop would have used. The messages, their order and their split into
// SEND_BUFFER / RECV_BUFFER chunks stay the same, and so do all values.
#include "../../config.h"
#include "../../protocols/Protocols.h"
#include "worker_pool.hpp"
#include <array>
#include <vector>

#if ADDITIONAL_RELU_THREADS > 0
namespace stream_parallel
{
// positions in the streams a level element may use, and the ones it must not
struct Counters
{
    int64_t send, recv, pre, pre_bool, pre_arith, rnd, btriple;
    int send_round, recv_round;
    std::array<int64_t, 5> fixed;  // other parties, triples: must not move inside a parallel segment
};

inline Counters snapshot()
{
    Counters c;
    c.send = send_count[PNEXT];
    c.recv = share_buffer[PNEXT];
    c.pre = (int64_t)preprocessed_outputs_index;
#if BEAVER == 1 && PRE == 1
    c.pre_bool = preprocessed_outputs_bool_index ? (int64_t)preprocessed_outputs_bool_index[0] : 0;
    c.pre_arith = preprocessed_outputs_arithmetic_index ? (int64_t)preprocessed_outputs_arithmetic_index[0] : 0;
#else
    c.pre_bool = c.pre_arith = 0;
#endif
    c.rnd = (int64_t)rnd_calls_self;
#if BEAVER == 1 && PRE == 1
    c.btriple = (int64_t)curr_boolean_triple_index;
#else
    c.btriple = 0;
#endif
    c.send_round = sending_rounds;
    c.recv_round = rounds;
    c.fixed = {send_count[PSELF], share_buffer[PSELF],
#if BEAVER == 1 && PRE == 1
               0, (int64_t)curr_arithmetic_triple_index,
#else
               0, 0,
#endif
               (int64_t)preprocessed_outputs_input_index};
    return c;
}

// below this, a segment runs serially (the pool hand-off costs a few us); an element of a wider DATATYPE
// carries DATTYPE / 32 times the work
constexpr int kMinSegment = DATTYPE >= 256 ? 64 : 512;
}  // namespace stream_parallel
#endif

// f(i) for i = 0 .. len - 1 with the side effects of the serial loop, on the worker pool when possible
template <typename F>
void stream_parallel_for(int len, F&& f)
{
#if ADDITIONAL_RELU_THREADS > 0
    using namespace stream_parallel;
    if (current_phase != PHASE_LIVE || len < 2 * kMinSegment || tl_stream)
    {
        for (int i = 0; i < len; i++)
            f(i);
        return;
    }
    // per-element consumption, from an element that did not cross a buffer boundary
    int i = 0;
    Counters d{};
    for (;;)
    {
        if (i == len)
            return;
        const Counters a = snapshot();
        f(i++);
        const Counters b = snapshot();
        if (a.send_round == b.send_round && a.recv_round == b.recv_round && b.send >= a.send && b.recv >= a.recv)
        {
            d = {b.send - a.send, b.recv - a.recv, b.pre - a.pre, b.pre_bool - a.pre_bool,
                 b.pre_arith - a.pre_arith, b.rnd - a.rnd, b.btriple - a.btriple, 0, 0, {}};
            break;
        }
    }
    constexpr int T = WORKER_POOL_THREADS + 1;
    std::vector<DATATYPE> rnd;
    while (i < len)
    {
        int64_t k = len - i;
#if SEND_BUFFER > 0
        if (d.send)
            k = std::min<int64_t>(k, (SEND_BUFFER - send_count[PNEXT]) / d.send);
#endif
#if RECV_BUFFER > 0
        if (d.recv)
            k = std::min<int64_t>(k, (RECV_BUFFER - share_buffer[PNEXT]) / d.recv);
#endif
        if (k < kMinSegment)
        {
            // a buffer boundary is near: the serial element sends / refills as usual
            f(i++);
            continue;
        }
        const Counters a = snapshot();
        rnd.resize(k * d.rnd);
        getRandomVals(PSELF, rnd.data(), rnd.size());
        rnd_calls_self -= rnd.size();  // counted again when consumed through the cursors below
        StreamCursor base{&sending_args[PNEXT].sent_elements[sending_rounds][a.send],
                          d.recv ? &receiving_args[PNEXT].received_elements[rounds - 1][a.recv] : nullptr,
                          preprocessed_outputs ? preprocessed_outputs + a.pre : nullptr,
#if BEAVER == 1 && PRE == 1
                          d.pre_bool ? preprocessed_outputs_bool[0] + a.pre_bool : nullptr,
                          d.pre_arith ? preprocessed_outputs_arithmetic[0] + a.pre_arith : nullptr,
#else
                          nullptr, nullptr,
#endif
                          rnd.data(), (uint64_t)a.btriple};
        std::array<int, T> bad{};
        const int start = i;
        GemmPool::get().run([&](int t) {
            const int64_t lo = k * t / T, hi = k * (t + 1) / T;
            StreamCursor c{base.send + lo * d.send, base.recv ? base.recv + lo * d.recv : nullptr,
                           base.pre ? base.pre + lo * d.pre : nullptr,
                           base.pre_bool ? base.pre_bool + lo * d.pre_bool : nullptr,
                           base.pre_arith ? base.pre_arith + lo * d.pre_arith : nullptr, base.rnd + lo * d.rnd,
                           base.btriple + lo * d.btriple};
            tl_stream = &c;
            for (int64_t e = lo; e < hi; e++)
                f(start + (int)e);
            tl_stream = nullptr;
            // every element must have consumed exactly the measured amount
            bad[t] = c.send != base.send + hi * d.send || c.rnd != base.rnd + hi * d.rnd ||
                     (base.recv && c.recv != base.recv + hi * d.recv) ||
                     (base.pre && c.pre != base.pre + hi * d.pre) ||
                     (base.pre_bool && c.pre_bool != base.pre_bool + hi * d.pre_bool) ||
                     (base.pre_arith && c.pre_arith != base.pre_arith + hi * d.pre_arith) ||
                     c.btriple != base.btriple + hi * d.btriple;
        });
        const Counters b = snapshot();
        for (int t = 0; t < T; t++)
            if (bad[t])
                stream_cursor_misuse("an element whose stream use differs from the first one");
        if (b.fixed != a.fixed || b.send != a.send || b.recv != a.recv || b.pre != a.pre ||
            b.pre_bool != a.pre_bool || b.pre_arith != a.pre_arith || b.btriple != a.btriple || b.send_round != a.send_round ||
            b.recv_round != a.recv_round)
            stream_cursor_misuse("a stream without cursor");
        send_count[PNEXT] += k * d.send;
        share_buffer[PNEXT] += k * d.recv;
        preprocessed_outputs_index += k * d.pre;
#if BEAVER == 1 && PRE == 1
        if (d.pre_bool)
            preprocessed_outputs_bool_index[0] += k * d.pre_bool;
        if (d.pre_arith)
            preprocessed_outputs_arithmetic_index[0] += k * d.pre_arith;
#endif
        rnd_calls_self += rnd.size();
#if BEAVER == 1 && PRE == 1
        curr_boolean_triple_index += k * d.btriple;
#endif
        i += k;
    }
#else
    for (int i = 0; i < len; i++)
        f(i);
#endif
}
