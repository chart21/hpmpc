#pragma once
#include "include/pch.h"

// Port stride between the CHEETAH channels of one process. Process p's CHEETAH base port is
// base_port + process_offset = BASE_PORT + num_players * (num_players - 1) * p + p (+ CHEETAH_PORT_OFFSET),
// so the bases span (num_players * (num_players - 1) + 1) * PROCESS_NUM ports. A stride of PROCESS_NUM put
// channel 1 of process p on channel 0 of process p + PROCESS_NUM / 3 as soon as CHEETAH_THREADS > 1
// (hangs / wrong triples with 24 processes and 2 threads).
#define CHEETAH_IO_OFFSET ((num_players * (num_players - 1) + 1) * PROCESS_NUM)
#include "arch/DATATYPE.h"

#ifndef FAKE_TRIPLES
#define FAKE_TRIPLES 0
#endif

// const ConvolutionParameter param(batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
struct ConvolutionParameter
{
    int batchSize;
    int inh;
    int inw;
    int din; // channel
    int dout; // n filters
    int wh;
    int ww;
    int padding;
    int stride;
    int out_h;
    int out_w;
    int dilation;
    int x_size_per_batch;
    int w_size_per_batch;
    int y_size_per_batch;


    ConvolutionParameter(int batchSize,
                         int inh,
                         int inw,
                         int din,
                         int dout,
                         int wh,
                         int ww,
                         int padding,
                         int stride,
                         int oh,
                         int ow,
                         int dilation)
        : batchSize(batchSize),
          inh(inh),
          inw(inw),
          din(din),
          dout(dout),
          wh(wh),
          ww(ww),
          padding(padding),
          stride(stride),
          out_h(oh),
          out_w(ow),
          dilation(dilation)
    {
        x_size_per_batch = inh * inw * din;
        w_size_per_batch = wh * ww * dout * din;
        y_size_per_batch = out_h * out_w * dout;
    }
};

struct BatchNorm2DParameter
{
		int batchSize;
		int ch;
		int h;
		int w;
		int hw;
    int x_size_per_batch;
    int w_size_per_batch;
    int y_size_per_batch;
    BatchNorm2DParameter(int batchSize,
                         int ch,
                         int h,
                         int w)
        : batchSize(batchSize),
          ch(ch),
          h(h),
          w(w)
    {
        hw = h * w;
        x_size_per_batch = ch * h * w;
        w_size_per_batch = ch;
        y_size_per_batch = x_size_per_batch;
    }
};



struct FullyConnectedParameter
{
    int batchSize;
    int in_feat;
    int out_feat;
    int x_size_per_batch;
    int w_size_per_batch;
    int y_size_per_batch;
    FullyConnectedParameter(int batch,
                            int in_feat,
                            int out_feat)
        : batchSize(batch),
          in_feat(in_feat),
          out_feat(out_feat)
    {
        x_size_per_batch = in_feat;
        w_size_per_batch = in_feat * out_feat;
        y_size_per_batch = out_feat;
    }
};

// CUT_FRACTIONAL_BITS_OPT eligibility of the 2PC ROT circuits (docs/CUT_FRACTIONAL_BITS_OPT.md; see beaver_triples.hpp)
#define CUT_FRAC_ELIGIBLE \
    (CUT_FRACTIONAL_BITS_OPT == 1 && TRUNC_DELAYED == 0 && FRACTIONAL >= 1 && FRACTIONAL <= BITLENGTH - 3 && \
     ROT_PREPROCESSING_OPT == 1 && BITLENGTH == 32 && \
     (RCA_MSB == 1 || PPA_MSB == 1 || PPA4_MSB == 1))
// Some A2B conversion (INIT pass) runs without CUT_FRACTIONAL_BITS_OPT: the Boolean addition must produce all slices
inline bool g_a2b_full_width = false;

#if FAKE_TRIPLES == 0
#define generateArithmeticTriples generateArithmeticDummyTriples
#define generateBooleanTriples generateBooleanDummyTriples
#define generateArithmeticAB2Triples generateArithmeticAB2DummyTriples
#define generateBooleanAB2Triples generateBooleanAB2DummyTriples
#define generateConvTriples generateLayerDummyTriples
#define generateFCTriples generateLayerDummyTriples
#define generateBatchNorm2DTriples generateLayerDummyTriples
#define generateBooleanAdditionTriples generateBooleanAdditionDummyTriples
#define generateMultiplexerTriples generateMultiplexerDummyTriples
#define generateCOTTriples generateCOTDummyTriples
#define generateBooleanCOTMultTriples generateBooleanCOTMultiplyDummyTriples
#define generateRandomMultiplications generateRandomMultiplicationDummyTriples

#include <core/hpmpc_interface.hpp>
#include <core/keys.hpp>

#define CHEETAH_PARTY (PARTY+1)

// CHEETAH_CONV_ASYNC: the batched conv triples run on their own thread during the ABY2 preprocessing pass
// (conv_async_start in protocols/beaver_triples.hpp). The pass records each layer's masks (SetupConv2dTriples) and
// then calls mark_ready; the HE pipeline waits for each layer in ready(); complete_preprocessing joins before the
// next generator uses the CHEETAH channels, and the CONV generation there only finishes up (MWK share corrections).
#if CHEETAH_CONV_REPACK == 1 && CHEETAH_CONV_PACKED == 1 && CHEETAH_CONV_TYPE == 0
// the packed convs in ConvTriple's repacking ring: set before the first Keys::instance (which sets it up and
// exchanges the Galois keys: both parties' for AB triples, A_KNOWN = 0)
inline const bool g_conv_repack_set = (Iface::conv_repack() = true, Iface::conv_repack_ab() = (A_KNOWN == 0), true);
#endif
#define CHEETAH_CONV_ASYNC_ACTIVE (CHEETAH_CONV_ASYNC == 1 && CHEETAH_WAN_OPT == 0 && PROTOCOL == 4 && DATTYPE == BITLENGTH && \
                                   CHEETAH_CONV_TYPE == 0 && CHEETAH_CONV_PACKED == 1 && CHEETAH_CONV_PIPELINE == 1)
// CHEETAH_CONV_SIDE: the lanes' conv triples (one pipelined product, CHEETAH_CONV_LANES) on four channels of their own
// (Keys::get_side_ios), started by complete_preprocessing on a thread of their own (tl_conv_side) while the next
// generators use the regular channels. Same calls, same PRNG streams: the same triples.
#if defined(CHEETAH_CONV_LANES_CHECK)
#define CONV_LANES_CHECK_ON 1
#else
#define CONV_LANES_CHECK_ON 0
#endif
#define CHEETAH_CONV_SIDE_ACTIVE (CHEETAH_CONV_SIDE == 1 && CHEETAH_CONV_LANES_ACTIVE && CHEETAH_WAN_OPT == 0 && \
                                  BIT_INJECTION_PREPROCESSING_OPT == 1)
inline thread_local bool tl_conv_side = false;
#if CHEETAH_CONV_ASYNC_ACTIVE
#include <condition_variable>
#include <mutex>
#include <thread>
namespace conv_async
{
inline std::thread worker;
inline std::mutex mutex;
inline std::condition_variable cv;
inline size_t ready_layers = 0;  // layers whose masks the preprocessing pass has recorded
inline bool launched = false, done = false;
inline void mark_ready(size_t n)
{
    {
        std::lock_guard<std::mutex> lock(mutex);
        ready_layers = n;
    }
    cv.notify_all();
}
inline void wait_ready(size_t i)
{
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [i] { return ready_layers > i; });
}
inline void join()
{
    if (worker.joinable())
        worker.join();
}
}  // namespace conv_async
#endif

#if CHEETAH_WAN_OPT == 1
inline void sync_cheetah_wan_barrier(Iface::Keys<IO::NetIO>& keys)
{
    auto** ios = keys.get_ios(CHEETAH_THREADS);
    for (int thread = 0; thread < CHEETAH_THREADS; ++thread)
    {
        ios[thread]->sync();
    }
}
#endif


// Input: arrays of arithmetic triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateArithmeticDummyTriples(type a[],
                                    type b[],
                                    type c[],
                                    int bitlength,
                                    uint64_t num_triples,
                                    std::string ip,
                                    int port)
{
    std::cout << "ARITH AB\n";

    if (num_triples == 0)
        return;

    port += CHEETAH_PORT_OFFSET;

    //convert SIMD variables to regular uints
    const int vectorization_factor = DATTYPE / bitlength;

    UINT_TYPE* uint_a = (UINT_TYPE*)std::aligned_alloc(alignof(DATATYPE), num_triples * sizeof(UINT_TYPE));
    unorthogonalize_arithmetic(a, uint_a, num_triples / (DATTYPE / bitlength));
    UINT_TYPE* uint_b = (UINT_TYPE*)std::aligned_alloc(alignof(DATATYPE), num_triples * sizeof(UINT_TYPE));
    unorthogonalize_arithmetic(b, uint_b, num_triples / (DATTYPE / bitlength));
    UINT_TYPE* uint_c = (UINT_TYPE*)std::aligned_alloc(alignof(DATATYPE), num_triples * sizeof(UINT_TYPE));

    Iface::generateArithTriplesCheetah(uint_a, uint_b, uint_c, bitlength, num_triples, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, Utils::PROTO::AB, CHEETAH_IO_OFFSET);

    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
    std::free(uint_a);
    std::free(uint_b);
    std::free(uint_c);
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateBooleanDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port,
                                 int cheetah_ot_type = CHEETAH_BOOL_OT_TYPE, bool disconnect = true)
{
    std::cout << "BOOL AB\n";

    if(num_triples == 0) return;

    port += CHEETAH_PORT_OFFSET;

    //reinterpret SIMD bitstream as uint8 bitstream
    uint8_t* uint_a = (uint8_t*) a;
    uint8_t* uint_b = (uint8_t*) b;
    uint8_t* uint_c = (uint8_t*) c;

    auto ot = _16KKOT_to_4OT;
    switch (cheetah_ot_type) {
        case 0: {
            ot = _2ROT;
            break;
        }
        case 1: {
            ot = _8KKOT;
            break;
        }
        case 2: {
            ot = _16KKOT_to_4OT;
            break;
        }
        case 3: {
            ot = _2COT;
            break;
        }
        default: ot = _2ROT;
    };

    Iface::generateBoolTriplesCheetah(
            uint_a, uint_b, uint_c,
            bitlength, num_triples ,ip, port, CHEETAH_PARTY,
            CHEETAH_THREADS,
            ot, CHEETAH_IO_OFFSET);

#if CHEETAH_DISCONNECT == 1
    if (disconnect)
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
#endif
}

template <typename type>
void generateBooleanCOTMultiplyDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port,
                                 bool disconnect = true)
{
    std::cout << "BOOL COT MULT\n";

    if(num_triples == 0) return;

    port += CHEETAH_PORT_OFFSET;

    //reinterpret SIMD bitstream as uint8 bitstream
    uint8_t* uint_a = (uint8_t*) a;
    uint8_t* uint_b = (uint8_t*) b;
    uint8_t* uint_c = (uint8_t*) c;

    Iface::generateBoolCOTMultTriplesCheetah(
            uint_a, uint_b, uint_c,
            bitlength, num_triples ,ip, port, CHEETAH_PARTY,
            CHEETAH_THREADS, CHEETAH_IO_OFFSET);

#if CHEETAH_DISCONNECT == 1
    if (disconnect)
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
#endif
}


#if BEAVER_N_TUPLES == 1
template <typename Datatype>
void generateBeaverNDummyTuples(Beaver3TuplesD<Datatype> &beaver_3_tuples, Beaver4TuplesD<Datatype> &beaver_4_tuples, uint64_t num_beaver_3_tuples, uint64_t num_beaver_4_tuples, const std::string& ip, int port)
{
    std::cout << "BEAVER N TUPLES\n";

    if(num_beaver_3_tuples == 0 && num_beaver_4_tuples == 0) return;
    port += CHEETAH_PORT_OFFSET;
    Beaver3Tuples l_beaver_3_tuples{
        (uint8_t*) beaver_3_tuples.a,
        (uint8_t*) beaver_3_tuples.b,
        (uint8_t*) beaver_3_tuples.c,
        (uint8_t*) beaver_3_tuples.ab,
        (uint8_t*) beaver_3_tuples.ac,
        (uint8_t*) beaver_3_tuples.bc,
        (uint8_t*) beaver_3_tuples.abc
    };
    Beaver4Tuples l_beaver_4_tuples{
        (uint8_t*) beaver_4_tuples.a,
        (uint8_t*) beaver_4_tuples.b,
        (uint8_t*) beaver_4_tuples.c,
        (uint8_t*) beaver_4_tuples.d,
        (uint8_t*) beaver_4_tuples.ab,
        (uint8_t*) beaver_4_tuples.ac,
        (uint8_t*) beaver_4_tuples.ad,
        (uint8_t*) beaver_4_tuples.bc,
        (uint8_t*) beaver_4_tuples.bd,
        (uint8_t*) beaver_4_tuples.cd,
        (uint8_t*) beaver_4_tuples.abc,
        (uint8_t*) beaver_4_tuples.abd,
        (uint8_t*) beaver_4_tuples.acd,
        (uint8_t*) beaver_4_tuples.bcd,
        (uint8_t*) beaver_4_tuples.abcd
    };

#if RESHARE_OPT == 1 && RESHARE_OPT_SIM == 1 && PPA4_MSB == 1
    // PPA4 SIM=1 skips the input-wire zero_adds, which is exact only if the 3-tuple mask fields are
    // party-local: .b P0-only (masks the P0-known s1 slices), .c P1-only (masks P1's s2 slices).
    constexpr bool b3_party_local_bc = true;
#else
    constexpr bool b3_party_local_bc = false;
#endif
    Iface::generateBool3TupleCheetah(l_beaver_3_tuples, num_beaver_3_tuples , ip, port, CHEETAH_PARTY, CHEETAH_THREADS, CHEETAH_IO_OFFSET, b3_party_local_bc);
#if CHEETAH_WAN_OPT == 1
    if (num_beaver_3_tuples > 0 && num_beaver_4_tuples > 0) {
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).get_ios(CHEETAH_THREADS)[0]->sync();
    }
#endif
    Iface::generateBool4TupleCheetah(l_beaver_4_tuples, num_beaver_4_tuples , ip, port, CHEETAH_PARTY, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
    #endif
}
#endif


// Input: arrays of arithmetic triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateArithmeticAB2DummyTriples(type a[],
                                    type b[],
                                    type c[],
                                    int bitlength,
                                    uint64_t num_triples,
                                    std::string ip,
                                    int port)
{
    std::cout << "ARITH AB2\n";

    if(num_triples == 0) return;

    port += CHEETAH_PORT_OFFSET;

    //convert SIMD variables to regular uints
    const int vectorization_factor = DATTYPE / bitlength;

    UINT_TYPE* uint_a;
    UINT_TYPE* uint_b;
    UINT_TYPE* uint_c;

    if(vectorization_factor == 1) // No need to unvectorize
    {
        uint_a = (UINT_TYPE*) a;
        uint_b = (UINT_TYPE*) b;
        uint_c = (UINT_TYPE*) c;
#if PARTY == 1
        uint_a = nullptr;
#else
        uint_b = nullptr;
#endif
    } else  {
#if PARTY == 0
        uint_a = (UINT_TYPE*)std::aligned_alloc(alignof(DATATYPE), num_triples * sizeof(UINT_TYPE));
        unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor));
#else
        uint_a = nullptr;
#endif
#if PARTY == 1
        uint_b = (UINT_TYPE*)std::aligned_alloc(alignof(DATATYPE), num_triples * sizeof(UINT_TYPE));
        unorthogonalize_arithmetic(b, uint_b, num_triples / (vectorization_factor));
#else
        uint_b = nullptr;
#endif // P_0 doesn't need b for AB2
        uint_c = (UINT_TYPE*)std::aligned_alloc(alignof(DATATYPE), num_triples * sizeof(UINT_TYPE));
    }

    Iface::generateArithTriplesCheetah(uint_a, uint_b, uint_c, bitlength, num_triples, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, Utils::PROTO::AB2, CHEETAH_IO_OFFSET);

    // convert UINT triple to SIMD type
    if (vectorization_factor != 1) {
        orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
#if PARTY == 0
        std::free(uint_a);
#endif
#if PARTY == 1
        std::free(uint_b);
#endif
        std::free(uint_c);
    }
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateBooleanAB2DummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port,
                                 int cheetah_ot_type = CHEETAH_BOOL_OT_TYPE, bool disconnect = true)
{
    std::cout << "BOOL AB2\n";

    if (num_triples == 0) return;

    port += CHEETAH_PORT_OFFSET;

    //reinterpret SIMD bitstream as uint8 bitstream
#if PARTY == 0
    uint8_t* uint_a = (uint8_t*) a;
#else
    std::vector<uint8_t> zerosa(num_triples, 0);
    uint8_t* uint_a = zerosa.data();
#endif
#if PARTY == 1
    uint8_t* uint_b = (uint8_t*) b;
#else
    std::vector<uint8_t> zeros(num_triples, 0);
    uint8_t* uint_b = zeros.data();
#endif // P_0 doesn't need b for AB2
    uint8_t* uint_c = (uint8_t*) c;

    auto ot = _16KKOT_to_4OT;
    switch (cheetah_ot_type) {
        case 0: {
            ot = _2ROT;
            break;
        }
        case 1: {
            ot = _8KKOT;
            break;
        }
        case 2: {
            ot = _16KKOT_to_4OT;
            break;
        }
        case 3: {
            ot = _2COT;
            break;
        }
        default: ot = _2ROT;
    };

    Iface::generateBoolTriplesCheetah(uint_a, uint_b, uint_c, bitlength,
            num_triples, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, ot, CHEETAH_IO_OFFSET);

#if CHEETAH_DISCONNECT == 1
    if (disconnect)
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
#endif
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with shares of a + b
template <typename type>
void generateBooleanAdditionDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port,
                                 int cheetah_ot_type = CHEETAH_BOOL_OT_TYPE)
{
    constexpr int num_bits_per_input = REDUCED_BITLENGTH_k - REDUCED_BITLENGTH_m;
    if(num_triples == 0) return;
    if(num_bits_per_input <= 0) return;
#if CHEETAH_WAN_OPT == 1
    auto& keys = Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port + CHEETAH_PORT_OFFSET, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
#endif
    //reinterpret SIMD bitstream as uint8 bitstream
#if PARTY == 0
    auto av = reinterpret_cast<type (*)[num_bits_per_input]> (a);
#else
    auto bv = reinterpret_cast<type (*)[num_bits_per_input]> (b);
#endif
    auto cv = reinterpret_cast<type (*)[num_bits_per_input]> (c);
    num_triples = num_triples / (num_bits_per_input * DATTYPE);
#if A2B_ADDER_BATCH == 1 && ROT_PREPROCESSING_OPT == 1 && CHEETAH_WAN_OPT == 0
    {
        // (untested) ripple-carry rounds from r = k - 1 (numeric LSB) down to lo; with A2B_ADDER_CUT the top FRACTIONAL
        // sum bits, which every cut consumer replaces by 0, are not computed (k - FRACTIONAL - 1 AND rounds, not k - 1)
        const int k = num_bits_per_input;
        int lo = 0;
#if A2B_ADDER_CUT == 1 && CUT_FRAC_ELIGIBLE
        if (!g_a2b_full_width && k == BITLENGTH)
            lo = FRACTIONAL;
#endif
        const int and_rounds = k - 1 - lo;
        const uint64_t n = num_triples;
        std::vector<type> carry(n), ot_a(n), ot_b(n), prod(n);
        auto* rounds = Iface::boolCOTMultRoundsBegin(n * DATTYPE, and_rounds, ip, port + CHEETAH_PORT_OFFSET, CHEETAH_PARTY,
                                                     CHEETAH_THREADS, CHEETAH_IO_OFFSET);
        for (int r = k - 1, round = 0; r >= lo; --r)
        {
            for (uint64_t i = 0; i < n; i++)
            {
#if PARTY == 0
                const type x = av[i][r];
#else
                const type x = bv[i][r];
#endif
                const type cr = (r == k - 1) ? SET_ALL_ZERO() : carry[i];
                cv[i][r] = x ^ cr;  // sum share
#if PARTY == 0
                ot_a[i] = x ^ cr, ot_b[i] = cr;  // the AND (a ^ c)(b ^ c) of the carry trick (LSB: a & b, c = 0)
#else
                ot_b[i] = x ^ cr, ot_a[i] = cr;
#endif
            }
            if (r == lo)
                break;
            Iface::boolCOTMultRound(rounds, round++, (const uint8_t*) ot_a.data(), (const uint8_t*) ot_b.data(),
                                    (uint8_t*) prod.data());
            for (uint64_t i = 0; i < n; i++)
                carry[i] = (r == k - 1) ? prod[i] : (prod[i] ^ carry[i]);  // c' = c ^ (a ^ c)(b ^ c)
        }
        for (int r = 0; r < lo; r++)
            for (uint64_t i = 0; i < n; i++) cv[i][r] = SET_ALL_ZERO();  // cut slices: never read
        Iface::boolCOTMultRoundsEnd(rounds);
#if CHEETAH_DISCONNECT == 1
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port + CHEETAH_PORT_OFFSET, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
#endif
        return;
    }
#endif
    type* carry_last = new type[num_triples];
    type* carry_this = new type[num_triples];
    type* ot_a = new type[num_triples];
    type* ot_b = new type[num_triples];
    int r = num_bits_per_input;
    const int k = num_bits_per_input;
    while(r > 0)
    {
        r--;
#if CHEETAH_WAN_OPT == 1
        sync_cheetah_wan_barrier(keys);
#endif
        switch(r)
        {
            case k - 1:
                for (uint64_t i = 0; i < num_triples ; i++)
                {
#if PARTY == 0
                    cv[i][r] = av[i][r];
                    ot_a[i] = av[i][r];
                    ot_b[i] = SET_ALL_ZERO();
#else
                    cv[i][r] = bv[i][r];
                    ot_b[i] = bv[i][r];
                    ot_a[i] = SET_ALL_ZERO();
#endif
                }
                #if ROT_PREPROCESSING_OPT == 1
                generateBooleanCOTMultTriples(ot_a, ot_b, carry_last, bitlength, num_triples * DATTYPE, ip, port, false);
                #else
                generateBooleanAB2Triples(ot_a, ot_b, carry_last, bitlength, num_triples * DATTYPE, ip, port, cheetah_ot_type, false);
                #endif
                break;
            case k - 2:
                for (uint64_t i = 0; i < num_triples ; i++)
                {
                    //update
#if PARTY == 0
                    cv[i][r] = av[i][r] ^ carry_last[i];
#else
                    cv[i][r] = bv[i][r] ^ carry_last[i];
#endif
                    //prepare
#if PARTY == 0
                    ot_a[i] = av[i][r] ^ carry_last[i];
                    ot_b[i] = carry_last[i];
#else
                    ot_b[i] = bv[i][r] ^ carry_last[i];
                    ot_a[i] = carry_last[i];
#endif
                }
                #if ROT_PREPROCESSING_OPT == 1
                generateBooleanCOTMultTriples(ot_a, ot_b, carry_this, bitlength, num_triples * DATTYPE, ip, port, false);
                #else
                generateBooleanTriples(ot_a, ot_b, carry_this, bitlength, num_triples * DATTYPE, ip, port, cheetah_ot_type, false);
                #endif

                break;
            default:
                // complete_carry
                for (uint64_t i = 0; i < num_triples ; i++)
                {
                    carry_this[i] = carry_this[i] ^ carry_last[i];
                    carry_last[i] = carry_this[i];
                }
                // update result
                for (uint64_t i = 0; i < num_triples ; i++)
#if PARTY == 0
                        cv[i][r] = av[i][r] ^ carry_last[i];
#else
                        cv[i][r] = bv[i][r] ^ carry_last[i];
#endif

                // prepare_carry
                for (uint64_t i = 0; i < num_triples ; i++)
                {
#if PARTY == 0
                    ot_a[i] = av[i][r] ^ carry_last[i];
                    ot_b[i] = carry_last[i];
#else
                    ot_b[i] = bv[i][r] ^ carry_last[i];
                    ot_a[i] = carry_last[i];
#endif
                }
                #if ROT_PREPROCESSING_OPT == 1
                generateBooleanCOTMultTriples(ot_a, ot_b, carry_this, bitlength, num_triples * DATTYPE, ip, port, false);
                #else
                generateBooleanTriples(ot_a, ot_b, carry_this, bitlength, num_triples * DATTYPE, ip, port, cheetah_ot_type, false);
                #endif
                break;
            case 0:
                // complete_carry
                for (uint64_t i = 0; i < num_triples ; i++)
                {
                    carry_this[i] = carry_this[i] ^ carry_last[i];
                    carry_last[i] = carry_this[i];
                }
                // update result
                for (uint64_t i = 0; i < num_triples ; i++)
#if PARTY == 0
                    cv[i][r] = av[i][r] ^ carry_last[i];
#else
                    cv[i][r] = bv[i][r] ^ carry_last[i];
#endif
                delete[] carry_last;
                delete[] carry_this;
                delete[] ot_a;
                delete[] ot_b;
#if CHEETAH_DISCONNECT == 1
                Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
#endif
                return;

        }
    }
}

// Input: For Party 0: Array of messages m0 stored in a[]
// Input: For Party 1: Array of selection bits stored in a[]
// Output: [c] will be filled with shares of mb, i.e. P0 holds -r and P1 holds m0 * b + r
template <typename type>
void generateCOTDummyTriples(type a[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port)
{
    if(num_triples == 0) return;

    port += CHEETAH_PORT_OFFSET;

    const int vectorization_factor = DATTYPE / bitlength;

    if(vectorization_factor == 1) // No need to unvectorize
        {
#if PARTY == 0
            UINT_TYPE* uint_a = (UINT_TYPE*) a;
#else
            uint8_t* uint_a = (uint8_t*) a;
#endif
            UINT_TYPE* uint_c = (UINT_TYPE*) c;

#if PARTY == 0
            Iface::generateCOT(CHEETAH_PARTY, uint_a, nullptr, uint_c, num_triples,
                ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
#else
            Iface::generateCOT(CHEETAH_PARTY, nullptr, uint_a, uint_c, num_triples,
                ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
#endif
            return;
        }

#if PARTY == 0
    UINT_TYPE* uint_a = NEW(UINT_TYPE[num_triples]); //stores m0
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor));

#else  //PARTY 1
    uint8_t* uint_a = (uint8_t*) a; //stores choice bit (packed)
#endif


    UINT_TYPE* uint_c = NEW(UINT_TYPE[num_triples]);

#if PARTY == 0
    Iface::generateCOT(CHEETAH_PARTY, uint_a, nullptr, uint_c, num_triples,
        ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
#else
    Iface::generateCOT(CHEETAH_PARTY, nullptr, uint_a, uint_c, num_triples,
        ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
#endif

    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
#if PARTY == 0
    DELETEARR(uint_a);
#endif
    DELETEARR(uint_c);

    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
    #endif
}

// Input: Arithmetic shares stored in a[] and shared bits stored in b[]
// Output: [c] will be filled with shares ab
template <typename type>
void generateMultiplexerDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port)
{
    if(num_triples == 0) return;

    port += CHEETAH_PORT_OFFSET;
    const int vectorization_factor = DATTYPE / bitlength;

    uint8_t* uint_b = (uint8_t*) b; //stores choice bit (packed)

    if(vectorization_factor == 1) // No need to unvectorize
        {
            UINT_TYPE* uint_a = (UINT_TYPE*) a;
            UINT_TYPE* uint_c = (UINT_TYPE*) c;

            Iface::do_multiplex(num_triples, uint_a, uint_b, uint_c, CHEETAH_PARTY,
                    ip, port, CHEETAH_IO_OFFSET, CHEETAH_THREADS);

            return;
        }

    UINT_TYPE* uint_a = NEW(UINT_TYPE[num_triples]); //stores arithmetic share
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor));

    UINT_TYPE* uint_c = NEW(UINT_TYPE[num_triples]);

    Iface::do_multiplex(num_triples, uint_a, uint_b, uint_c, CHEETAH_PARTY,
            ip, port, CHEETAH_IO_OFFSET, CHEETAH_THREADS);

    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
    DELETEARR(uint_a);
    DELETEARR(uint_c);

    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
    #endif

}





#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
// MODELWEIGHTS_KNOWN: P1's conv/FC triple share must equal the r1 it committed in aby2_pre (its online mask is
// l_P1 = TRUNC(-r1)). After the normal generation P1 sets its share to r1 and sends delta = r1 - c1, and P0
// subtracts it: reconstruction is unchanged, and delta is uniform to P0 because r1 is P1's private randomness.
// One message per layer. c holds `factor` lanes of `cnt` outputs ([lane][batch-major output]). The r1 were
// recorded in GEMM call order: a conv records its output index per batch element (one GEMM per element), an FC
// records the linear-order sentinel.
template <typename LayerParams, typename Keys>
static void mwk_fix_p1_share(Keys& keys, UINT_TYPE* c, uint64_t cnt, [[maybe_unused]] uint64_t per_batch,
                             IO::NetIO* own_io = nullptr)
{
    constexpr int factor = DATTYPE / BITLENGTH;
    constexpr bool is_conv = std::is_same_v<LayerParams, ConvolutionParameter>;
    auto& consume = is_conv ? g_mwk_p1_masks_consume : g_mwk_p1_fc_masks_consume;  // separate vectors, see buffers.h
    const uint64_t n = cnt * factor;
    std::vector<UINT_TYPE> delta(n);
    auto* io = own_io ? own_io : keys.get_ios(CHEETAH_THREADS)[0];
    constexpr uint64_t CHUNK = (uint64_t) 1 << 28;  // send_data takes an int length
#if PARTY == 1
    const auto& masks = is_conv ? g_mwk_p1_masks : g_mwk_p1_fc_masks;
    const auto& indices = is_conv ? g_mwk_p1_indices : g_mwk_p1_fc_indices;
    const uint64_t stride = is_conv ? per_batch : cnt;
    for (uint64_t k = 0; k < cnt; k++)
    {
        const uint64_t raw = indices[consume + k];
        const uint64_t idx = raw == G_MWK_LINEAR_SENTINEL ? k : (k / stride) * stride + raw;
        alignas(sizeof(DATATYPE)) UINT_TYPE r1[factor];
        unorthogonalize_arithmetic(&masks[consume + k], r1, 1);
        for (int j = 0; j < factor; j++)
        {
            delta[j * cnt + idx] = r1[j] - c[j * cnt + idx];
            c[j * cnt + idx] = r1[j];
        }
    }
    for (uint64_t off = 0; off < n; off += CHUNK)
        io->send_data(delta.data() + off, (int) (std::min(CHUNK, n - off) * sizeof(UINT_TYPE)));
    io->flush();
#else
    for (uint64_t off = 0; off < n; off += CHUNK)
        io->recv_data(delta.data() + off, (int) (std::min(CHUNK, n - off) * sizeof(UINT_TYPE)));
    for (uint64_t k = 0; k < n; k++)
        c[k] -= delta[k];
#endif
    consume += cnt;
}
#endif

//Input: arrays of layer triple shares [a], [b] with sizes predefined by convolution/Fc/Batchnorm params
//Output: Contigious array of clayer triple shares [c] storing the output
template <typename type, typename LayerParams>
void generateLayerDummyTriples(type** a,
                              type** b,
                              type c[],
                              int bitlength,
                              std::vector<LayerParams> params,
                              std::string ip,
                              int port)
{
    port += CHEETAH_PORT_OFFSET;
    if constexpr (std::is_same_v<LayerParams, ConvolutionParameter>) {
        std::cout << "CONVOLUTION ";
    } else if constexpr (std::is_same_v<LayerParams, FullyConnectedParameter>) {
        std::cout << "FC ";
    } else if constexpr (std::is_same_v<LayerParams, BatchNorm2DParameter>) {
        std::cout << "BATCHNORM ";
    }


#if CHEETAH_CONV_SIDE_ACTIVE
    // on the side thread the regular channels belong to the generators running meanwhile: leave them alone
    IO::NetIO** side_ios = nullptr;
    if (tl_conv_side)
        side_ios = Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).get_side_ios(4);
    else
#else
    IO::NetIO** side_ios = nullptr;
#endif
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
    auto& keys = Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
    const int factor = DATTYPE/BITLENGTH;

    if(factor == 1) { // No need to unvectorize
#if CHEETAH_CONV_TYPE == 1
        std::vector<Utils::ConvParm> parms(params.size());
        size_t total_batches = 0;
#endif
        // packed convs pipelined across layers: collected here, generated after the loop
        constexpr bool conv_batched = CHEETAH_CONV_TYPE == 0 && CHEETAH_CONV_PACKED == 1 && CHEETAH_CONV_PIPELINE == 1 &&
                                      std::is_same_v<LayerParams, ConvolutionParameter>;
        std::vector<Utils::ConvParm> batched_parms;
        std::vector<Iface::BNTripleLayer> bn_layers;  // CHEETAH_BN_BATCHED: generated after the loop, in one product
        UINT_TYPE** uint_w = (UINT_TYPE**) a;
        UINT_TYPE** uint_x = (UINT_TYPE**) b;
        UINT_TYPE* uint_y = (UINT_TYPE*) c;
        uint64_t y_index_counter = 0;

        for(size_t n = 0; n < params.size(); n++) {
            auto p = params[n];

            if constexpr (std::is_same_v<LayerParams, ConvolutionParameter>) {
                if (p.dilation != 1) {
                    std::cerr << "DILATION != 1 is not supported\n";
                }
                Utils::ConvParm conv{
                    .batchsize = p.batchSize,
                    .ic = p.din,
                    .iw = p.inw,
                    .ih = p.inh,
                    .fc = p.din,
                    .fw = p.ww,
                    .fh = p.wh,
                    .n_filters = p.dout,
                    .stride = p.stride,
                    .padding = p.padding,
                };

#if CHEETAH_CONV_TYPE == 0 && CHEETAH_CONV_PACKED == 1 && CHEETAH_CONV_PIPELINE == 1
                batched_parms.push_back(conv);
#elif CHEETAH_CONV_TYPE == 0 && CHEETAH_CONV_PACKED == 1
                Iface::generateConvTriplesPacked(keys,
                        A_KNOWN == 0 || PARTY == 1 ? uint_x[n] : nullptr,
                        A_KNOWN == 0 || PARTY == 0 ? uint_w[n] : nullptr,
                        uint_y + y_index_counter,
                        conv,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2,
                        factor
                );
#elif CHEETAH_CONV_TYPE == 0
                Iface::generateConvTriplesCheetahWrapper(keys,
                        A_KNOWN == 0 || PARTY == 1 ? uint_x[n] : nullptr,
                        A_KNOWN == 0 || PARTY == 0 ? uint_w[n] : nullptr,
                        uint_y + y_index_counter,
                        conv,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2,
                        factor, A_KNOWN == 0
                );
#else
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
                static_assert(MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 0,
                              "MODELWEIGHTS_KNOWN_DURING_PREPROCESSING requires CHEETAH_CONV_TYPE == 0");
#endif
                parms[n] = conv;
                total_batches += conv.batchsize;
#endif
            } else if constexpr (std::is_same_v<LayerParams, FullyConnectedParameter>) {
                Iface::generateFCTriplesCheetah(keys,
                        A_KNOWN == 0 || PARTY == 1 ? uint_x[n] : nullptr,
                        A_KNOWN == 0 || PARTY == 0 ? uint_w[n] : nullptr,
                        uint_y + y_index_counter,
                        p.batchSize, p.in_feat, p.out_feat,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2,
                        factor
                );
            } else if constexpr (std::is_same_v<LayerParams, BatchNorm2DParameter>) {
#if CHEETAH_BN_BATCHED == 1
                bn_layers.push_back({A_KNOWN == 0 || PARTY == 1 ? uint_x[n] : nullptr,
                                     A_KNOWN == 0 || PARTY == 0 ? uint_w[n] : nullptr,
                                     uint_y + y_index_counter, p.batchSize, (size_t) p.ch, (size_t) p.h,
                                     (size_t) p.w});
#else
                Iface::generateBNTriplesCheetah(keys,
                        A_KNOWN == 0 || PARTY == 1 ? uint_x[n] : nullptr,
                        A_KNOWN == 0 || PARTY == 0 ? uint_w[n] : nullptr,
                        uint_y + y_index_counter,
                        p.batchSize, p.ch, p.h, p.w,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2,
                        factor
                );
#endif
            } else {
                std::cerr << "Unsupported Param type\n";
            }
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
            if constexpr ((std::is_same_v<LayerParams, ConvolutionParameter> && !conv_batched) ||
                          std::is_same_v<LayerParams, FullyConnectedParameter>)
                mwk_fix_p1_share<LayerParams>(keys, uint_y + y_index_counter,
                                              (uint64_t) p.y_size_per_batch * p.batchSize, p.y_size_per_batch);
#endif
            y_index_counter += p.y_size_per_batch * p.batchSize;
#if CHEETAH_WAN_OPT == 1
            if (n + 1 < params.size()) {
                keys.get_ios(CHEETAH_THREADS)[0]->sync();
            }
#endif
        }
        if constexpr (conv_batched) {
#if CHEETAH_CONV_ASYNC_ACTIVE
            if (!conv_async::done)  // else generated during the preprocessing pass (conv_async)
#endif
            Iface::generateConvTriplesPackedBatch(keys, batched_parms,
                    A_KNOWN == 0 || PARTY == 1 ? uint_x : nullptr,
                    A_KNOWN == 0 || PARTY == 0 ? uint_w : nullptr,
                    uint_y, CHEETAH_PARTY, CHEETAH_THREADS,
                    A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2);
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
            uint64_t y_offset = 0;
            for (auto& p : params) {  // in layer order, as the masks were recorded
                mwk_fix_p1_share<LayerParams>(keys, uint_y + y_offset, (uint64_t) p.y_size_per_batch * p.batchSize,
                                              p.y_size_per_batch);
                y_offset += p.y_size_per_batch * p.batchSize;
            }
#endif
        }
#if CHEETAH_BN_BATCHED == 1
        if constexpr (std::is_same_v<LayerParams, BatchNorm2DParameter>)
            Iface::generateBNTriplesBatched(keys, bn_layers, CHEETAH_PARTY, CHEETAH_THREADS,
                                            A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2, factor);
#endif
#if CHEETAH_CONV_TYPE == 1
        if constexpr (std::is_same_v<LayerParams, ConvolutionParameter>) {
            Iface::generateConvTriplesCheetah2(keys, total_batches, parms,
                    A_KNOWN == 0 || PARTY == 1 ? uint_x : nullptr,
                    A_KNOWN == 0 || PARTY == 0 ? uint_w : nullptr,
                    uint_y,
                    A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2,
                    CHEETAH_PARTY, CHEETAH_THREADS, factor, A_KNOWN == 0
            );
        }
#endif
    } else {
        uint64_t c_index = 0;
        struct Deferred { UINT_TYPE *x, *w, *y; uint64_t y_size, c_index, per_batch = 0; };  // converted back after the call
#if CHEETAH_BN_BATCHED == 1
        std::vector<Deferred> deferred_bn;
        std::vector<Iface::BNTripleLayer> bn_layers;
#endif
#if CHEETAH_CONV_LANES_ACTIVE
        std::vector<Utils::ConvParm> conv_parms;
        std::vector<Deferred> conv_layers;
#endif
        for(size_t n = 0; n < params.size(); n++) {
            auto p = params[n];
            const uint64_t x_size = p.x_size_per_batch * p.batchSize;
            const uint64_t w_size = p.w_size_per_batch;
            const uint64_t y_size = p.y_size_per_batch * p.batchSize;

#if A_KNOWN == 0 || PARTY == 1
            UINT_TYPE* x = new UINT_TYPE[factor * x_size]; // Party1 holds X2 in plain in AB2 setting
#else
            UINT_TYPE* x = nullptr;
#endif
#if A_KNOWN == 0 || PARTY == 0
            UINT_TYPE* w = new UINT_TYPE[w_size * factor];  // W is always constant
#else
            UINT_TYPE* w = nullptr;
#endif
            UINT_TYPE* y = new UINT_TYPE[factor * y_size];

#if A_KNOWN == 0 || PARTY == 1
            for (uint64_t i = 0; i < x_size; i++) {
                alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&b[n][i], temp, 1);
                for (int j = 0; j < factor; j++)
                    x[j * x_size + i] = temp[j];
            }
#endif
#if A_KNOWN == 0 || PARTY == 0
            // the lanes' conv is one convolution with lane 0's weights: the other lanes' copies only for the premise check
            constexpr bool lane0_w = CHEETAH_CONV_LANES_ACTIVE && !CONV_LANES_CHECK_ON &&
                                     std::is_same_v<LayerParams, ConvolutionParameter>;
            for (uint64_t i = 0; i < w_size; i++) {
                alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&a[n][i], temp, 1);
                for (int j = 0; j < (lane0_w ? 1 : factor); ++j) {
                    w[j * w_size + i] = temp[j];
                }
            }
#endif

            p.batchSize *= factor;

            if constexpr (std::is_same_v<LayerParams, ConvolutionParameter>) {
                if (p.dilation != 1) {
                    std::cerr << "DILATION != 1 is not supported\n";
                }
                Utils::ConvParm conv{
                    .batchsize = p.batchSize,
                    .ic = p.din,
                    .iw = p.inw,
                    .ih = p.inh,
                    .fc = p.din,
                    .fw = p.ww,
                    .fh = p.wh,
                    .n_filters = p.dout,
                    .stride = p.stride,
                    .padding = p.padding,
                };

#if CHEETAH_CONV_LANES_ACTIVE
                // The weight masks are the same in every lane (SHARE_PREP: the model owner's share is -w and the other
                // party's 0, and the model's weights are the same for all lanes' images), so the lanes' convolutions are ONE
                // convolution over all their images with lane 0's weights: full ciphertexts and one set of weight
                // transforms instead of one single-image convolution per lane. All layers go to one pipelined call
                // after the loop (see conv_layers).
#if defined(CHEETAH_CONV_LANES_CHECK) && (A_KNOWN == 0 || PARTY == 0)
                {  // debug: the premise, every lane's weight masks equal lane 0's
                    uint64_t diff = 0;
                    for (int j = 1; j < factor; j++)
                        for (uint64_t i = 0; i < w_size; i++) diff += w[j * w_size + i] != w[i];
                    if (diff)
                        std::cout << "CONV_LANES_CHECK layer " << n << ": " << diff << " of " << w_size * (factor - 1)
                                  << " weight masks differ from lane 0" << std::endl;
                }
#endif
                conv_parms.push_back(conv);
                conv_layers.push_back({x, w, y, y_size, c_index, p.y_size_per_batch});
                c_index += y_size;
                continue;
#else
#if CHEETAH_CONV_PACKED == 1
                Iface::generateConvTriplesPacked(keys,
#else
                Iface::generateConvTriplesCheetahWrapper(keys,
#endif
                        x, w, y,
                        conv,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB,
                        factor
                );
#endif
            } else if constexpr (std::is_same_v<LayerParams, FullyConnectedParameter>) {
                Iface::generateFCTriplesCheetah(keys,
                        x, w, y,
                        p.batchSize, p.in_feat, p.out_feat,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB,
                        factor
                );
            } else if constexpr (std::is_same_v<LayerParams, BatchNorm2DParameter>) {
#if CHEETAH_BN_BATCHED == 1
                bn_layers.push_back({x, w, y, p.batchSize, (size_t) p.ch, (size_t) p.h, (size_t) p.w});
                deferred_bn.push_back({x, w, y, y_size, c_index});
                c_index += y_size;
                continue;
#else
                Iface::generateBNTriplesCheetah(keys,
                        x, w, y,
                        p.batchSize, p.ch, p.h, p.w,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB,
                        factor
                );
#endif
            } else {
                std::cerr << "Unsupported Param type\n";
            }
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
            if constexpr (std::is_same_v<LayerParams, ConvolutionParameter> ||
                          std::is_same_v<LayerParams, FullyConnectedParameter>)
                mwk_fix_p1_share<LayerParams>(keys, y, y_size, p.y_size_per_batch);
#endif
            for (uint64_t i = 0; i < y_size; i++) {
                alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
                for (int j = 0; j < factor; j++)
                    temp[j] = y[j * y_size + i];
                orthogonalize_arithmetic(temp, c + c_index + i, 1);
            }

            delete[] x;
            delete[] w;
            delete[] y;
            c_index += y_size;
        }
#if CHEETAH_CONV_LANES_ACTIVE
        if (!conv_layers.empty()) {
            // one output buffer, the layers at consecutive offsets (generateConvTriplesPackedBatch's layout)
            uint64_t total = 0;
            for (auto& d : conv_layers) total += d.y_size * factor;
            std::vector<UINT_TYPE> y_all(total);
            std::vector<UINT_TYPE*> xs, ws;
            for (auto& d : conv_layers) xs.push_back(d.x), ws.push_back(d.w);
            Iface::generateConvTriplesPackedBatch(keys, conv_parms, A_KNOWN == 0 || PARTY == 1 ? xs.data() : nullptr,
                                                  A_KNOWN == 0 || PARTY == 0 ? ws.data() : nullptr, y_all.data(),
                                                  CHEETAH_PARTY, CHEETAH_THREADS,
                                                  A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB, nullptr, side_ios);
            uint64_t off = 0;
            for (auto& d : conv_layers) {
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
                // P1's prescribed shares, layer by layer in the order its masks were recorded (lane-major: lane j's
                // outputs at j * y_size, as mwk_fix_p1_share takes them)
                mwk_fix_p1_share<LayerParams>(keys, y_all.data() + off, d.y_size, d.per_batch,
                                              side_ios ? side_ios[0] : nullptr);
#endif
                for (uint64_t i = 0; i < d.y_size; i++) {
                    alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
                    for (int j = 0; j < factor; j++)
                        temp[j] = y_all[off + j * d.y_size + i];
                    orthogonalize_arithmetic(temp, c + d.c_index + i, 1);
                }
                off += d.y_size * factor;
                delete[] d.x;
                delete[] d.w;
                delete[] d.y;
            }
        }
#endif
#if CHEETAH_BN_BATCHED == 1
        if (!bn_layers.empty()) {
            Iface::generateBNTriplesBatched(keys, bn_layers, CHEETAH_PARTY, CHEETAH_THREADS,
                                            A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB, factor);
            for (auto& d : deferred_bn) {
                for (uint64_t i = 0; i < d.y_size; i++) {
                    alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
                    for (int j = 0; j < factor; j++)
                        temp[j] = d.y[j * d.y_size + i];
                    orthogonalize_arithmetic(temp, c + d.c_index + i, 1);
                }
                delete[] d.x;
                delete[] d.w;
                delete[] d.y;
            }
        }
#endif
    }
    #if CHEETAH_DISCONNECT == 1
#if CHEETAH_CONV_SIDE_ACTIVE
    if (!tl_conv_side)  // the side thread's generator ends while the others still use the regular channels
#endif
    keys.disconnect();
    #endif
}

template <typename type>
void generateRandomMultiplicationDummyTriples(type a[],
                                              type b[],
                                              uint64_t num_muls,
                                              std::string ip,
                                              int port)
{
    std::cout << "RANDOM_MULTIPLICATION\n";

    if (num_muls == 0) return;

    port += CHEETAH_PORT_OFFSET;

    //reinterpret SIMD bitstream as uint8 bitstream
    uint8_t* uint_a = (uint8_t*) a;
    uint8_t* uint_b = (uint8_t*) b;

    Iface::generateRandomMultiplicationsCheetah(uint_a, uint_b, num_muls, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, CHEETAH_IO_OFFSET);
    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
    #endif
}

void CheetahDisconnect(std::string ip, int port) {
    port += CHEETAH_PORT_OFFSET;
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).disconnect();
}

// CHEETAH_RELEASE_OT: the OT packs' memory back after the preprocessing (the online phase extends no OTs)
void CheetahReleaseOT(std::string ip, int port) {
    port += CHEETAH_PORT_OFFSET;
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, CHEETAH_IO_OFFSET).release_ot();
}


#else

#define generateArithmeticTriples generateFakeArithmeticTriples
#define generateBooleanTriples generateFakeBooleanTriples
#define generateArithmeticAB2Triples generateFakeArithmeticTriples
#define generateBooleanAB2Triples generateFakeBooleanTriples
#define generateConvTriples generateFakeLayerTriples
#define generateFCTriples generateFakeLayerTriples
#define generateBatchNorm2DTriples generateFakeLayerTriples
#define generateBooleanAdditionTriples generateFakeBooleanAdditionTriples
#define generateMultiplexerTriples generateFakeMultiplexerTriples
#define generateCOTTriples generateCOTDummyTriples
#define generateRandomMultiplications generateFakeRandomMultiplications

template <typename type>
void generateFakeArithmeticTriples(type a[],
                                   type b[],
                                   type c[],
                                   int bitlength,
                                   uint64_t num_triples,
                                   std::string ip,
                                   int port)
{
}

template <typename type>
void generateFakeBooleanTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

    template <typename type>
void generateFakeAB2ArithmeticTriples(type a[],
                                   type b[],
                                   type c[],
                                   int bitlength,
                                   uint64_t num_triples,
                                   std::string ip,
                                   int port)
{
}

template <typename type>
void generateFakeAB2BooleanTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

    template <typename type>
void generateFakeBooleanAdditionTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

template <typename type>
void generateFakeMultiplexerTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

template <typename type>
void generateFakeCOTTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

template <typename type, typename LayerParams>
void generateFakeLayerTriples(type** a,
                             type** b,
                             type c[],
                             int bitlength,
                             std::vector<LayerParams> params,
                             std::string ip,
                             int port)
{
}

template <typename type>
void generateFakeRandomMultiplications(type a[],
                                       type b[],
                                       uint64_t num_muls,
                                       std::string ip,
                                       int port)
{
}

#endif
