#pragma once

#include "../../MP-SPDZ/lib/Machine.hpp"
#include "../../MP-SPDZ/lib/Shares/CInteger.hpp"
#include "../../MP-SPDZ/lib/Shares/Integer.hpp"
#include "../../core/utils/print.hpp"
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../protocols/Protocols.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>

#if FUNCTION_IDENTIFIER == 500
#define FUNCTION MP_MACHINE_TUTORIAL
#elif FUNCTION_IDENTIFIER == 501
#define FUNCTION MP_MACHINE_TMP
#elif FUNCTION_IDENTIFIER == 502
#define FUNCTION MP_MACHINE_ADD
#elif FUNCTION_IDENTIFIER == 503
#define FUNCTION MP_MACHINE_MUL
#elif FUNCTION_IDENTIFIER == 504
#define FUNCTION MP_MACHINE_MUL_FIX
#elif FUNCTION_IDENTIFIER == 505
#define FUNCTION MP_MACHINE_INT_TESTS
#elif FUNCTION_IDENTIFIER == 506
#define FUNCTION MP_MACHINE_MUL_BENCH
#elif FUNCTION_IDENTIFIER == 507 || FUNCTION_IDENTIFIER == 508 || FUNCTION_IDENTIFIER == 509
#define FUNCTION MP_MACHINE_COMP_BENCH
#elif FUNCTION_IDENTIFIER == 510
#define FUNCTION MP_MACHINE_DIV_BENCH
#elif FUNCTION_IDENTIFIER == 511
#define FUNCTION MP_MACHINE_SHARE_BENCH
#elif FUNCTION_IDENTIFIER == 512
#define FUNCTION MP_MACHINE_REVEAL_BENCH
#elif FUNCTION_IDENTIFIER == 513 || FUNCTION_IDENTIFIER == 514 || FUNCTION_IDENTIFIER == 515
#define FUNCTION MP_MACHINE_MAX_BENCH
#elif FUNCTION_IDENTIFIER == 516 || FUNCTION_IDENTIFIER == 517 || FUNCTION_IDENTIFIER == 518
#define FUNCTION MP_MACHINE_MIN_BENCH
#elif FUNCTION_IDENTIFIER == 519
#define FUNCTION MP_MACHINE_AVG_BENCH
#elif FUNCTION_IDENTIFIER == 520
#define FUNCTION MP_MACHINE_INTERSECTION_BENCH
#elif FUNCTION_IDENTIFIER == 521 || FUNCTION_IDENTIFIER == 522 || FUNCTION_IDENTIFIER == 523
#define FUNCTION MP_MACHINE_AUCTION_BENCH
#elif FUNCTION_IDENTIFIER == 524
#define FUNCTION MP_MACHINE_BIT_AND_BENCH
#elif FUNCTION_IDENTIFIER == 525
#define FUNCTION MP_MACHINE_AES_BENCH
#elif FUNCTION_IDENTIFIER == 526 || FUNCTION_IDENTIFIER == 527 || FUNCTION_IDENTIFIER == 528
#define FUNCTION MP_MACHINE_REG_BENCH
#elif FUNCTION_IDENTIFIER == 529 || FUNCTION_IDENTIFIER == 530 || FUNCTION_IDENTIFIER == 531
#define FUNCTION MP_MACHINE_LENET_BENCH
#elif FUNCTION_IDENTIFIER == 532 || FUNCTION_IDENTIFIER == 533 || FUNCTION_IDENTIFIER == 534
#define FUNCTION MP_MACHINE_VGG_BENCH
#endif

// Boilerplate
#define RESULTTYPE DATATYPE
void generateElements() {}

#if FUNCTION_IDENTIFIER == 500
template <typename Share>
void MP_MACHINE_TUTORIAL(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("tutorial.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 501
template <typename Share>
void MP_MACHINE_TMP(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    Share::communicate();
    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("custom.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 502
template <typename Share>
void MP_MACHINE_ADD(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("add.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 503
template <typename Share>
void MP_MACHINE_MUL(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("mul.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 504
template <typename Share>
void MP_MACHINE_MUL_FIX(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("mul_fix.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 505
template <typename Share>
void MP_MACHINE_INT_TESTS(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    Share::communicate();
#if BITLENGTH == 64
    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("int_test.sch");
    m.run();
#else
    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("int_test_32.sch");
    m.run();
#endif
}

#elif FUNCTION_IDENTIFIER == 506
template <typename Share>
void MP_MACHINE_MUL_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    Share::communicate();
    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("Int_Multiplication.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 507 || FUNCTION_IDENTIFIER == 508 || FUNCTION_IDENTIFIER == 509
template <typename Share>
void MP_MACHINE_COMP_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("Int_Compare.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 510
template <typename Share>
void MP_MACHINE_DIV_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("Int_Division.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 511
template <typename Share>
void MP_MACHINE_SHARE_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("Input.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 512
template <typename Share>
void MP_MACHINE_REVEAL_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("Reveal.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 513 || FUNCTION_IDENTIFIER == 514 || FUNCTION_IDENTIFIER == 515
template <typename Share>
void MP_MACHINE_MAX_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("SecureMax.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 516 || FUNCTION_IDENTIFIER == 517 || FUNCTION_IDENTIFIER == 518
template <typename Share>
void MP_MACHINE_MIN_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("SecureMin.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 519
template <typename Share>
void MP_MACHINE_AVG_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("SecureMean.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 520
template <typename Share>
void MP_MACHINE_INTERSECTION_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("PrivateSetIntersection.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 521 || FUNCTION_IDENTIFIER == 522 || FUNCTION_IDENTIFIER == 523
template <typename Share>
void MP_MACHINE_AUCTION_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("SecureAuction.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 524
template <typename Share>
void MP_MACHINE_BIT_AND_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    Share::communicate();
    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("BIT_AND.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 525
template <typename Share>
void MP_MACHINE_AES_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    Share::communicate();
    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("AES.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 526 || FUNCTION_IDENTIFIER == 527 || FUNCTION_IDENTIFIER == 528
template <typename Share>
void MP_MACHINE_REG_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("LogReg.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 529 || FUNCTION_IDENTIFIER == 530 || FUNCTION_IDENTIFIER == 531
template <typename Share>
void MP_MACHINE_LENET_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("LeNet.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 532 || FUNCTION_IDENTIFIER == 533 || FUNCTION_IDENTIFIER == 534
template <typename Share>
void MP_MACHINE_VGG_BENCH(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("VGG.sch");
    m.run();
}

#endif
