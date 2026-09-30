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
#include <cstdio>
#include <source_location>
#include <vector>

#if ADDITIONAL_RELU_THREADS > 0
namespace stream_parallel
{
// the global indices of the index-addressed preprocessing streams (StreamIndex order)
inline std::array<uint64_t*, IDX_COUNT> index_globals()
{
#if BEAVER == 1 && PRE == 1
    return {&curr_boolean_triple_index, &curr_arithmetic_triple_index, &curr_beaver_3_triple_index,
            &curr_beaver_4_triple_index, &curr_random_multiplication_index, &curr_arithmetic_ab2_triple_index,
            &curr_boolean_ab2_triple_index, &g_a2b_s1_pending};
#else
    return {};
#endif
}

// positions in the streams a level element may use, and the ones it must not
struct Counters
{
    int64_t send, recv, pre, pre_bool, pre_arith, rnd;
    std::array<int64_t, IDX_COUNT> idx;
    int send_round, recv_round;
    std::array<int64_t, 3> fixed;  // the other party's counters, preprocessing stores: must not move in a segment
};

inline Counters snapshot()
{
    Counters c{};
    c.send = send_count[PNEXT];
    c.recv = share_buffer[PNEXT];
    c.pre = (int64_t)preprocessed_outputs_index;
#if BEAVER == 1 && PRE == 1
    c.pre_bool = preprocessed_outputs_bool_index ? (int64_t)preprocessed_outputs_bool_index[0] : 0;
    c.pre_arith = preprocessed_outputs_arithmetic_index ? (int64_t)preprocessed_outputs_arithmetic_index[0] : 0;
#endif
    c.rnd = (int64_t)rnd_calls_self;
    const auto g = index_globals();
    for (int k = 0; k < IDX_COUNT; k++)
        c.idx[k] = g[k] ? (int64_t)*g[k] : 0;
    c.send_round = sending_rounds;
    c.recv_round = rounds;
    c.fixed = {send_count[PSELF], share_buffer[PSELF], (int64_t)preprocessed_outputs_input_index};
    return c;
}

inline Counters diff(const Counters& b, const Counters& a)
{
    Counters d{};
    d.send = b.send - a.send, d.recv = b.recv - a.recv, d.pre = b.pre - a.pre;
    d.pre_bool = b.pre_bool - a.pre_bool, d.pre_arith = b.pre_arith - a.pre_arith, d.rnd = b.rnd - a.rnd;
    for (int k = 0; k < IDX_COUNT; k++)
        d.idx[k] = b.idx[k] - a.idx[k];
    return d;
}

inline bool same_positions(const Counters& a, const Counters& b)
{
    return a.fixed == b.fixed && a.send == b.send && a.recv == b.recv && a.pre == b.pre && a.pre_bool == b.pre_bool &&
           a.pre_arith == b.pre_arith && a.idx == b.idx && a.send_round == b.send_round && a.recv_round == b.recv_round;
}

// below this, a segment runs serially (the pool hand-off costs a few us); an element of a wider DATATYPE
// carries DATTYPE / 32 times the work
constexpr int kMinSegment = DATTYPE >= 256 ? 64 : 512;
}  // namespace stream_parallel
#endif

#ifndef STREAM_PARALLEL_GEMM
#define STREAM_PARALLEL_GEMM 1  // GEMM mask-and-send / completion on the pool (debugging switch)
#endif
#ifndef STREAM_PARALLEL_CTOR
#define STREAM_PARALLEL_CTOR 1  // MSB adders constructed on the pool (debugging switch)
#endif
#ifndef STREAM_PARALLEL_RELU
#define STREAM_PARALLEL_RELU 1  // adder steps, A2B and bit injection on the pool (debugging switch)
#endif

#ifndef STREAM_PARALLEL_PRE
#define STREAM_PARALLEL_PRE 1  // the ReLU levels of the ABY2 preprocessing pass on the pool too (stream_parallel_pre_for)
#endif
#define STREAM_PARALLEL_PRE_ACTIVE (STREAM_PARALLEL_PRE == 1 && ADDITIONAL_RELU_THREADS > 0 && BEAVER == 1 && PRE == 1 && \
                                    PROTOCOL == 4)

#if STREAM_PARALLEL_PRE_ACTIVE
namespace stream_parallel
{
// positions in the preprocessing pass's append-only streams (PreCursor) and in the streams it reads
struct PreCounters
{
    int64_t send, type[2], ab_bool, ab_arith, ab2_bool, ab2_arith, bool_add, mux_arith, mux_bool, cot_arith, out,
        out_bool[2], out_arith[2];
    int64_t rnd, pre, pre_bool, pre_arith;
    std::array<int64_t, IDX_COUNT> idx;
    int send_round;
    std::array<int64_t, 4> fixed;  // online streams, which a preprocessing element must not touch
};

inline PreCounters pre_snapshot()
{
    PreCounters c{};
    c.send = send_count_pre[PNEXT];
    for (int r = 0; r < 2; r++)
        c.type[r] = triple_type_index.size() > (size_t) r ? (int64_t) triple_type_index[r] : 0;
    c.ab_bool = (int64_t) boolean_triple_index, c.ab_arith = (int64_t) arithmetic_triple_index;
    c.ab2_bool = (int64_t) boolean_ab2_triple_index, c.ab2_arith = (int64_t) arithmetic_ab2_triple_index;
    c.bool_add = (int64_t) boolean_addition_triple_index;
    c.mux_arith = (int64_t) arithmetic_multiplexer_triple_index, c.mux_bool = (int64_t) boolean_multiplexer_triple_index;
    c.cot_arith = (int64_t) arithmetic_cot_triple_index;
    c.out = (int64_t) preprocessed_outputs_input_index;
    for (int r = 0; r < 2; r++)
    {
        c.out_bool[r] = preprocessed_outputs_bool_input_index ? (int64_t) preprocessed_outputs_bool_input_index[r] : 0;
        c.out_arith[r] = preprocessed_outputs_arithmetic_input_index ? (int64_t) preprocessed_outputs_arithmetic_input_index[r] : 0;
    }
    c.rnd = (int64_t) rnd_calls_self;
    c.pre = (int64_t) preprocessed_outputs_index;
    c.pre_bool = preprocessed_outputs_bool_index ? (int64_t) preprocessed_outputs_bool_index[0] : 0;
    c.pre_arith = preprocessed_outputs_arithmetic_index ? (int64_t) preprocessed_outputs_arithmetic_index[0] : 0;
    const auto g = index_globals();
    for (int k = 0; k < IDX_COUNT; k++)
        c.idx[k] = g[k] ? (int64_t) *g[k] : 0;
    c.send_round = sending_rounds;
    c.fixed = {send_count[PNEXT], share_buffer[PNEXT], send_count_pre[PSELF], (int64_t) rounds};
    return c;
}

// the counters as one flat list, for differences and checks
inline std::vector<int64_t> pre_flat(const PreCounters& c)
{
    std::vector<int64_t> v{c.send, c.type[0], c.type[1], c.ab_bool, c.ab_arith, c.ab2_bool, c.ab2_arith, c.bool_add,
                           c.mux_arith, c.mux_bool, c.cot_arith, c.out, c.out_bool[0], c.out_bool[1], c.out_arith[0],
                           c.out_arith[1], c.rnd, c.pre, c.pre_bool, c.pre_arith};
    v.insert(v.end(), c.idx.begin(), c.idx.end());
    return v;
}
}  // namespace stream_parallel

// stream_parallel_for for a level of the preprocessing pass (PHASE_PRE): as online, the first element measures what
// each element consumes, and the rest of the level runs on the worker pool with cursors at the positions the serial
// loop would have used. The writes go through tl_pre (the pass's append-only streams: triple types and triple inputs,
// the pre-send buffer, stored outputs), the reads through tl_stream (randomness, retrieved triples and outputs).
template <typename F>
void stream_parallel_pre_for(int len, F&& f, std::source_location loc)
{
    using namespace stream_parallel;
    if (len < 2 * kMinSegment || tl_stream || tl_pre)
    {
        for (int i = 0; i < len; i++)
            f(i);
        return;
    }
    int i = 0;
    std::vector<int64_t> d;
    PreCounters a0{};
    for (;;)
    {
        if (i == len)
            return;
        a0 = pre_snapshot();
        f(i++);
        const PreCounters b0 = pre_snapshot();
        if (a0.send_round == b0.send_round && b0.send >= a0.send)
        {
            const auto fa = pre_flat(a0), fb = pre_flat(b0);
            d.resize(fa.size());
            for (size_t x = 0; x < fa.size(); x++)
                d[x] = fb[x] - fa[x];
            break;
        }
    }
    constexpr int T = WORKER_POOL_THREADS + 1;
    std::vector<DATATYPE> rnd;
    while (i < len)
    {
        int64_t k = len - i;
#if SEND_BUFFER > 0
        if (d[0])
            k = std::min<int64_t>(k, (SEND_BUFFER - send_count_pre[PNEXT]) / d[0]);
#endif
        if (k < kMinSegment)
        {
            f(i++);  // a send-buffer boundary is near: the serial element flushes as usual
            continue;
        }
        const PreCounters a = pre_snapshot();
        rnd.resize(k * d[16]);
        getRandomVals(PSELF, rnd.data(), rnd.size());
        rnd_calls_self -= rnd.size();  // counted again when consumed through the cursors below
        auto at = [](DATATYPE* base, int64_t pos) { return base ? base + pos : nullptr; };
        PreCursor wbase{};
        wbase.send = &sending_args_pre[PNEXT].sent_elements[sending_rounds][a.send];
        for (int r = 0; r < 2; r++)
            wbase.type[r] = triple_type.size() > (size_t) r && triple_type[r] ? triple_type[r] + a.type[r] : nullptr;
        wbase.ab_bool_a = at(boolean_triple_a, a.ab_bool), wbase.ab_bool_b = at(boolean_triple_b, a.ab_bool);
        wbase.ab_arith_a = at(arithmetic_triple_a, a.ab_arith), wbase.ab_arith_b = at(arithmetic_triple_b, a.ab_arith);
        wbase.ab2_bool = at(PARTY == 0 ? boolean_ab2_triple_a : boolean_ab2_triple_b, a.ab2_bool);
        wbase.ab2_arith = at(PARTY == 0 ? arithmetic_ab2_triple_a : arithmetic_ab2_triple_b, a.ab2_arith);
        wbase.bool_add = at(PARTY == 0 ? boolean_addition_triple_a : boolean_addition_triple_b, a.bool_add);
        wbase.mux_arith = at(multiplexer_triple_a, a.mux_arith), wbase.mux_bool = at(multiplexer_triple_b, a.mux_bool);
        wbase.cot_arith = at(cot_triple_a, a.cot_arith);
        wbase.out = at(preprocessed_outputs, a.out);
        for (int r = 0; r < 2; r++)
        {
            wbase.out_bool[r] = preprocessed_outputs_bool ? at(preprocessed_outputs_bool[r], a.out_bool[r]) : nullptr;
            wbase.out_arith[r] = preprocessed_outputs_arithmetic ? at(preprocessed_outputs_arithmetic[r], a.out_arith[r]) : nullptr;
        }
        // a stream an element uses must exist
        {
            const auto used = [&](int x) { return d[x] != 0; };
            const bool ok = (!used(1) || wbase.type[0]) && (!used(2) || wbase.type[1]) && (!used(3) || (wbase.ab_bool_a && wbase.ab_bool_b)) &&
                            (!used(4) || (wbase.ab_arith_a && wbase.ab_arith_b)) && (!used(5) || wbase.ab2_bool) &&
                            (!used(6) || wbase.ab2_arith) && (!used(7) || wbase.bool_add) && (!used(8) || wbase.mux_arith) &&
                            (!used(9) || wbase.mux_bool) && (!used(10) || wbase.cot_arith) && (!used(11) || wbase.out) &&
                            (!used(12) || wbase.out_bool[0]) && (!used(13) || wbase.out_bool[1]) &&
                            (!used(14) || wbase.out_arith[0]) && (!used(15) || wbase.out_arith[1]);
            if (!ok)
                stream_cursor_misuse("a preprocessing stream without a buffer");
        }
        StreamCursor rbase{nullptr, nullptr, at(preprocessed_outputs, a.pre),
                           d[18] ? preprocessed_outputs_bool[0] + a.pre_bool : nullptr,
                           d[19] ? preprocessed_outputs_arithmetic[0] + a.pre_arith : nullptr, rnd.data(), {}};
        for (int x = 0; x < IDX_COUNT; x++)
            rbase.idx[x] = (uint64_t) a.idx[x];
        std::array<int, T> bad{};
        const int start = i;
        GemmPool::get().run([&](int t) {
            const int64_t lo = k * t / T, hi = k * (t + 1) / T;
            auto adv = [](auto* p, int64_t n) { return p ? p + n : p; };
            PreCursor w{adv(wbase.send, lo * d[0]), {adv(wbase.type[0], lo * d[1]), adv(wbase.type[1], lo * d[2])},
                        adv(wbase.ab_bool_a, lo * d[3]), adv(wbase.ab_bool_b, lo * d[3]),
                        adv(wbase.ab_arith_a, lo * d[4]), adv(wbase.ab_arith_b, lo * d[4]),
                        adv(wbase.ab2_bool, lo * d[5]), adv(wbase.ab2_arith, lo * d[6]), adv(wbase.bool_add, lo * d[7]),
                        adv(wbase.mux_arith, lo * d[8]), adv(wbase.mux_bool, lo * d[9]), adv(wbase.cot_arith, lo * d[10]),
                        nullptr, adv(wbase.out, lo * d[11]),
                        {adv(wbase.out_bool[0], lo * d[12]), adv(wbase.out_bool[1], lo * d[13])},
                        {adv(wbase.out_arith[0], lo * d[14]), adv(wbase.out_arith[1], lo * d[15])}};
            StreamCursor c{nullptr, nullptr, adv(rbase.pre, lo * d[17]), adv(rbase.pre_bool, lo * d[18]),
                           adv(rbase.pre_arith, lo * d[19]), rbase.rnd + lo * d[16], {}};
            for (int x = 0; x < IDX_COUNT; x++)
                c.idx[x] = rbase.idx[x] + lo * d[20 + x];
            tl_pre = &w;
            tl_stream = &c;
            for (int64_t e = lo; e < hi; e++)
                f(start + (int) e);
            tl_stream = nullptr;
            tl_pre = nullptr;
            // every element must have used exactly the measured amount of every stream
            const auto end = [&](auto* p, auto* b, int64_t per) { return p == (b ? b + hi * per : b); };
            bool ok = end(w.send, wbase.send, d[0]) && end(w.type[0], wbase.type[0], d[1]) && end(w.type[1], wbase.type[1], d[2]) &&
                      end(w.ab_bool_a, wbase.ab_bool_a, d[3]) && end(w.ab_arith_a, wbase.ab_arith_a, d[4]) &&
                      end(w.ab2_bool, wbase.ab2_bool, d[5]) && end(w.ab2_arith, wbase.ab2_arith, d[6]) &&
                      end(w.bool_add, wbase.bool_add, d[7]) && end(w.mux_arith, wbase.mux_arith, d[8]) &&
                      end(w.mux_bool, wbase.mux_bool, d[9]) && end(w.cot_arith, wbase.cot_arith, d[10]) &&
                      end(w.out, wbase.out, d[11]) && end(w.out_bool[0], wbase.out_bool[0], d[12]) &&
                      end(w.out_bool[1], wbase.out_bool[1], d[13]) && end(w.out_arith[0], wbase.out_arith[0], d[14]) &&
                      end(w.out_arith[1], wbase.out_arith[1], d[15]) && c.rnd == rbase.rnd + hi * d[16] &&
                      end(c.pre, rbase.pre, d[17]) && end(c.pre_bool, rbase.pre_bool, d[18]) &&
                      end(c.pre_arith, rbase.pre_arith, d[19]);
            for (int x = 0; x < IDX_COUNT; x++)
                ok = ok && c.idx[x] == rbase.idx[x] + hi * d[20 + x];
            bad[t] = !ok;
        });
        const PreCounters b = pre_snapshot();
        for (int t = 0; t < T; t++)
            if (bad[t])
            {
                fprintf(stderr, "stream_parallel_pre_for at %s:%u: an element used the streams differently\n",
                        loc.file_name(), (unsigned) loc.line());
                stream_cursor_misuse("an element whose stream use differs from the first one");
            }
        if (pre_flat(a) != pre_flat(b) || a.fixed != b.fixed || a.send_round != b.send_round)
            stream_cursor_misuse("a preprocessing stream without cursor");
        // advance the global positions past the level's segment
        send_count_pre[PNEXT] += k * d[0];
        for (int r = 0; r < 2; r++)
            if (d[1 + r])
                triple_type_index[r] += k * d[1 + r];
        boolean_triple_index += k * d[3], arithmetic_triple_index += k * d[4];
        boolean_ab2_triple_index += k * d[5], arithmetic_ab2_triple_index += k * d[6];
        boolean_addition_triple_index += k * d[7];
        arithmetic_multiplexer_triple_index += k * d[8], boolean_multiplexer_triple_index += k * d[9];
        arithmetic_cot_triple_index += k * d[10];
        preprocessed_outputs_input_index += k * d[11];
        for (int r = 0; r < 2; r++)
        {
            if (d[12 + r])
                preprocessed_outputs_bool_input_index[r] += k * d[12 + r];
            if (d[14 + r])
                preprocessed_outputs_arithmetic_input_index[r] += k * d[14 + r];
        }
        rnd_calls_self += rnd.size();
        preprocessed_outputs_index += k * d[17];
        if (d[18])
            preprocessed_outputs_bool_index[0] += k * d[18];
        if (d[19])
            preprocessed_outputs_arithmetic_index[0] += k * d[19];
        const auto g = index_globals();
        for (int x = 0; x < IDX_COUNT; x++)
            if (g[x])
                *g[x] += k * d[20 + x];
        i += k;
    }
}
#endif

// f(i) for i = 0 .. len - 1 with the side effects of the serial loop, on the worker pool when possible. pre: the call
// site's elements may also run in parallel in the preprocessing pass (stream_parallel_pre_for)
template <bool enabled = true, bool pre = false, typename F>
void stream_parallel_for(int len, F&& f, std::source_location loc = std::source_location::current())
{
    if constexpr (!enabled)
    {
        for (int i = 0; i < len; i++)
            f(i);
        return;
    }
#if STREAM_PARALLEL_PRE_ACTIVE
    if constexpr (pre)
        if (current_phase == PHASE_PRE)
        {
            stream_parallel_pre_for(len, f, loc);
            return;
        }
#endif
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
            d = diff(b, a);
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
                          rnd.data(), {}};
        for (int x = 0; x < IDX_COUNT; x++)
            base.idx[x] = (uint64_t)a.idx[x];
        std::array<int, T> bad{};
        const int start = i;
        GemmPool::get().run([&](int t) {
            const int64_t lo = k * t / T, hi = k * (t + 1) / T;
            StreamCursor c{base.send + lo * d.send, base.recv ? base.recv + lo * d.recv : nullptr,
                           base.pre ? base.pre + lo * d.pre : nullptr,
                           base.pre_bool ? base.pre_bool + lo * d.pre_bool : nullptr,
                           base.pre_arith ? base.pre_arith + lo * d.pre_arith : nullptr, base.rnd + lo * d.rnd, {}};
            for (int x = 0; x < IDX_COUNT; x++)
                c.idx[x] = base.idx[x] + lo * d.idx[x];
            tl_stream = &c;
            for (int64_t e = lo; e < hi; e++)
                f(start + (int)e);
            tl_stream = nullptr;
            // every element must have consumed exactly the measured amount
            // bit set of the streams whose use differs: 1 send, 2 rnd, 4 recv, 8 pre, 16 pre_bool, 32 pre_arith,
            // 64 << x index stream x
            int b = (c.send != base.send + hi * d.send) | (c.rnd != base.rnd + hi * d.rnd) << 1 |
                    (base.recv && c.recv != base.recv + hi * d.recv) << 2 |
                    (base.pre && c.pre != base.pre + hi * d.pre) << 3 |
                    (base.pre_bool && c.pre_bool != base.pre_bool + hi * d.pre_bool) << 4 |
                    (base.pre_arith && c.pre_arith != base.pre_arith + hi * d.pre_arith) << 5;
            for (int x = 0; x < IDX_COUNT; x++)
                b |= (c.idx[x] != base.idx[x] + hi * d.idx[x]) << (6 + x);
            bad[t] = b;
        });
        const Counters b = snapshot();
        for (int t = 0; t < T; t++)
            if (bad[t])
            {
                fprintf(stderr, "stream_parallel_for at %s:%u: streams 0x%x differ (1 send, 2 rnd, 4 recv, 8 pre, "
                        "16 pre_bool, 32 pre_arith, 64<<x index x)\n", loc.file_name(), (unsigned) loc.line(), bad[t]);
                stream_cursor_misuse("an element whose stream use differs from the first one");
            }
        if (!same_positions(a, b))
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
        const auto g = index_globals();
        for (int x = 0; x < IDX_COUNT; x++)
            if (g[x])
                *g[x] += k * d.idx[x];
        i += k;
    }
#else
    for (int i = 0; i < len; i++)
        f(i);
#endif
}
