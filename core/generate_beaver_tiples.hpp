#pragma once
#include "include/pch.h"
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

    Iface::generateArithTriplesCheetah(uint_a, uint_b, uint_c, bitlength, num_triples, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, Utils::PROTO::AB, PROCESS_NUM);

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
            ot, PROCESS_NUM);

#if CHEETAH_DISCONNECT == 1
    if (disconnect)
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
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
            CHEETAH_THREADS, PROCESS_NUM);

#if CHEETAH_DISCONNECT == 1
    if (disconnect)
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
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

    Iface::generateBool3TupleCheetah(l_beaver_3_tuples, num_beaver_3_tuples , ip, port, CHEETAH_PARTY, CHEETAH_THREADS, PROCESS_NUM);
#if CHEETAH_WAN_OPT == 1
    if (num_beaver_3_tuples > 0 && num_beaver_4_tuples > 0) {
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).get_ios(CHEETAH_THREADS)[0]->sync();
    }
#endif
    Iface::generateBool4TupleCheetah(l_beaver_4_tuples, num_beaver_4_tuples , ip, port, CHEETAH_PARTY, CHEETAH_THREADS, PROCESS_NUM);
    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
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

    Iface::generateArithTriplesCheetah(uint_a, uint_b, uint_c, bitlength, num_triples, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, Utils::PROTO::AB2, PROCESS_NUM);

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
            num_triples, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, ot, PROCESS_NUM);

#if CHEETAH_DISCONNECT == 1
    if (disconnect)
        Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
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
    auto& keys = Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port + CHEETAH_PORT_OFFSET, CHEETAH_THREADS, PROCESS_NUM);
#endif
    //reinterpret SIMD bitstream as uint8 bitstream
#if PARTY == 0
    auto av = reinterpret_cast<type (*)[num_bits_per_input]> (a);
#else
    auto bv = reinterpret_cast<type (*)[num_bits_per_input]> (b);
#endif
    auto cv = reinterpret_cast<type (*)[num_bits_per_input]> (c);
    num_triples = num_triples / (num_bits_per_input * DATTYPE);
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
                Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
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
                ip, port, CHEETAH_THREADS, PROCESS_NUM);
#else
            Iface::generateCOT(CHEETAH_PARTY, nullptr, uint_a, uint_c, num_triples,
                ip, port, CHEETAH_THREADS, PROCESS_NUM);
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
        ip, port, CHEETAH_THREADS, PROCESS_NUM);
#else
    Iface::generateCOT(CHEETAH_PARTY, nullptr, uint_a, uint_c, num_triples,
        ip, port, CHEETAH_THREADS, PROCESS_NUM);
#endif

    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
#if PARTY == 0
    DELETEARR(uint_a);
#endif
    DELETEARR(uint_c);

    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
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
                    ip, port, PROCESS_NUM, CHEETAH_THREADS);

            return;
        }

    UINT_TYPE* uint_a = NEW(UINT_TYPE[num_triples]); //stores arithmetic share
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor));

    UINT_TYPE* uint_c = NEW(UINT_TYPE[num_triples]);

    Iface::do_multiplex(num_triples, uint_a, uint_b, uint_c, CHEETAH_PARTY,
            ip, port, PROCESS_NUM, CHEETAH_THREADS);

    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
    DELETEARR(uint_a);
    DELETEARR(uint_c);

    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
    #endif

}





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


    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
    auto& keys = Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM);
    const int factor = DATTYPE/BITLENGTH;

    if(factor == 1) { // No need to unvectorize
#if CHEETAH_CONV_TYPE == 1
        std::vector<Utils::ConvParm> parms(params.size());
        size_t total_batches = 0;
#endif
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

#if CHEETAH_CONV_TYPE == 0
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
                Iface::generateBNTriplesCheetah(keys,
                        A_KNOWN == 0 || PARTY == 1 ? uint_x[n] : nullptr,
                        A_KNOWN == 0 || PARTY == 0 ? uint_w[n] : nullptr,
                        uint_y + y_index_counter,
                        p.batchSize, p.ch, p.h, p.w,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 0 ? Utils::PROTO::AB : Utils::PROTO::AB2,
                        factor
                );
            } else {
                std::cerr << "Unsupported Param type\n";
            }
            // Layer(uint_w[i],uint_x[i],uint_y + y_index_counter, p); // calculate layer operation
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
            // Force P1's triple share [lxly]_2 to the r1 it picked in aby2_pre (uint_y == (UINT_TYPE*)c here).
            if constexpr (std::is_same_v<LayerParams, ConvolutionParameter> ||
                          std::is_same_v<LayerParams, FullyConnectedParameter>)
            {
                const uint64_t N = (uint64_t)p.y_size_per_batch;  // per-batch output size (oc*oh*ow)
                uint64_t ysz = N * p.batchSize;
                auto* mwk_io = keys.get_ios(CHEETAH_THREADS)[0];
                // SCATTER: CONV runs ONE GEMM per batch element (convolutional_layer.h), so its recorded index
                // resets per batch (0..N-1); c[] is batch-major, so add the batch offset (k/N)*N. FC runs a SINGLE
                // GEMM with m=batch, so its recorded index is already global (offset stride = ysz => offset 0).
                // P0 and P1 share an identical g_mwk_p1_indices order, so the delta send/recv stays aligned.
                constexpr bool is_conv = std::is_same_v<LayerParams, ConvolutionParameter>;
                const uint64_t bstride = is_conv ? N : ysz;
#if PARTY == 1
                for (uint64_t k = 0; k < ysz; k++) {
                    uint64_t raw = g_mwk_p1_indices[g_mwk_p1_masks_consume + k];
                    uint64_t idx = (raw == G_MWK_LINEAR_SENTINEL) ? k : ((k / bstride) * bstride + raw);
                    DATATYPE r1 = g_mwk_p1_masks[g_mwk_p1_masks_consume + k];
                    DATATYPE delta = OP_SUB(r1, c[y_index_counter + idx]);
                    mwk_io->send_data(&delta, sizeof(DATATYPE));
                    c[y_index_counter + idx] = r1;
                }
                mwk_io->flush();
#else
                for (uint64_t k = 0; k < ysz; k++) {
                    uint64_t raw = g_mwk_p1_indices[g_mwk_p1_masks_consume + k];
                    uint64_t idx = (raw == G_MWK_LINEAR_SENTINEL) ? k : ((k / bstride) * bstride + raw);
                    DATATYPE delta;
                    mwk_io->recv_data(&delta, sizeof(DATATYPE));
                    c[y_index_counter + idx] = OP_SUB(c[y_index_counter + idx], delta);
                }
#endif
                g_mwk_p1_masks_consume += ysz;
            }
#endif
            y_index_counter += p.y_size_per_batch * p.batchSize;
#if CHEETAH_WAN_OPT == 1
            if (n + 1 < params.size()) {
                keys.get_ios(CHEETAH_THREADS)[0]->sync();
            }
#endif
        }
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
            for (uint64_t i = 0; i < w_size; i++) {
                alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&a[n][i], temp, 1);
                for (int j = 0; j < factor; ++j) {
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

                Iface::generateConvTriplesCheetahWrapper(keys,
                        x, w, y,
                        conv,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB,
                        factor
                );
            } else if constexpr (std::is_same_v<LayerParams, FullyConnectedParameter>) {
                Iface::generateFCTriplesCheetah(keys,
                        x, w, y,
                        p.batchSize, p.in_feat, p.out_feat,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB,
                        factor
                );
            } else if constexpr (std::is_same_v<LayerParams, BatchNorm2DParameter>) {
                Iface::generateBNTriplesCheetah(keys,
                        x, w, y,
                        p.batchSize, p.ch, p.h, p.w,
                        CHEETAH_PARTY, CHEETAH_THREADS,
                        A_KNOWN == 1 ? Utils::PROTO::AB2 : Utils::PROTO::AB,
                        factor
                );
            } else {
                std::cerr << "Unsupported Param type\n";
            }
            // Conv2D(w,x,y, p) // calculate layer operation
            for (uint64_t i = 0; i < y_size; i++) {
                alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
                for (int j = 0; j < factor; j++)
                    temp[j] = y[j * y_size + i];
                orthogonalize_arithmetic(temp, c + c_index + i, 1);
            }

#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
            // Force P1's triple share [lxly]_2 to the r1 it picked in aby2_pre (so l_P1 = TRUNC(-r1) and the
            // share are consistent for the reveal). P0 receives delta = r1 - c1 and sets its share to cross-r1.
            if constexpr (std::is_same_v<LayerParams, ConvolutionParameter> ||
                          std::is_same_v<LayerParams, FullyConnectedParameter>)
            {
                const uint64_t N = (uint64_t)p.y_size_per_batch;  // per-batch output size (oc*oh*ow)
                auto* mwk_io = keys.get_ios(CHEETAH_THREADS)[0];
                // SCATTER (see factor==1 path above): CONV resets its index per batch (0..N-1) into batch-major c[]
                // -> add batch offset (k/N)*N; FC index is already global (stride = y_size => offset 0).
                constexpr bool is_conv = std::is_same_v<LayerParams, ConvolutionParameter>;
                const uint64_t bstride = is_conv ? N : y_size;
#if PARTY == 1
                for (uint64_t k = 0; k < y_size; k++) {
                    uint64_t raw = g_mwk_p1_indices[g_mwk_p1_masks_consume + k];
                    uint64_t idx = (raw == G_MWK_LINEAR_SENTINEL) ? k : ((k / bstride) * bstride + raw);
                    DATATYPE r1 = g_mwk_p1_masks[g_mwk_p1_masks_consume + k];
                    DATATYPE delta = OP_SUB(r1, c[c_index + idx]);
                    mwk_io->send_data(&delta, sizeof(DATATYPE));
                    c[c_index + idx] = r1;
                }
                mwk_io->flush();
#else
                for (uint64_t k = 0; k < y_size; k++) {
                    uint64_t raw = g_mwk_p1_indices[g_mwk_p1_masks_consume + k];
                    uint64_t idx = (raw == G_MWK_LINEAR_SENTINEL) ? k : ((k / bstride) * bstride + raw);
                    DATATYPE delta;
                    mwk_io->recv_data(&delta, sizeof(DATATYPE));
                    c[c_index + idx] = OP_SUB(c[c_index + idx], delta);
                }
#endif
                g_mwk_p1_masks_consume += y_size;
            }
#endif

            delete[] x;
            delete[] w;
            delete[] y;
            c_index += y_size;
        }
    }
    #if CHEETAH_DISCONNECT == 1
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

    Iface::generateRandomMultiplicationsCheetah(uint_a, uint_b, num_muls, ip, port, CHEETAH_PARTY, CHEETAH_THREADS, PROCESS_NUM);
    #if CHEETAH_DISCONNECT == 1
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
    #endif
}

void CheetahDisconnect(std::string ip, int port) {
    port += CHEETAH_PORT_OFFSET;
    Iface::Keys<IO::NetIO>::instance(CHEETAH_PARTY, ip, port, CHEETAH_THREADS, PROCESS_NUM).disconnect();
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
