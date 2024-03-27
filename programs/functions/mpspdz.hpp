#pragma once

#include <cstdint>
#include <cstring>
#include <iostream>
#include <cmath>
#include "../../protocols/Protocols.h"
#include "../../protocols/XOR_Share.hpp"
#include "../../protocols/Additive_Share.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../utils/print.hpp"

#include "../../MP-SPDZ/lib/Machine.hpp"
#include "../../MP-SPDZ/lib/Shares/CInteger.hpp"

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
#endif

//Boilerplate
#define RESULTTYPE DATATYPE
void generateElements()
{}

#if FUNCTION_IDENTIFIER == 500
template<typename Share>
void MP_MACHINE_TUTORIAL(DATATYPE* res)
{
    using S = XOR_Share<DATATYPE, Share>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<Share, sint, sbitset_t, S> m("tutorial.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 501
template<typename Share>
void MP_MACHINE_TMP(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    Share::communicate();
    IR::Machine<cint, Share, A, sbitset_t, S> m("tmp.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 502
template<typename Share>
void MP_MACHINE_ADD(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<cint, Share, A, sbitset_t, S> m("add.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 503
template<typename Share>
void MP_MACHINE_MUL(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<cint, Share, A, sbitset_t, S> m("mul.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 504
template<typename Share>
void MP_MACHINE_MUL_FIX(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    IR::Machine<cint, Share, A, sbitset_t, S> m("mul_fix.sch");
    m.run();
}

#elif FUNCTION_IDENTIFIER == 505
template<typename Share>
void MP_MACHINE_INT_TESTS(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;

    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<64, S>;

    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    Share::communicate();
#if BITLENGTH == 64
    IR::Machine<cint, Share, A, sbitset_t, S> m("int_test.sch");
    m.run();
#else
    IR::Machine<cint, Share, A, sbitset_t, S> m("int_test_32.sch");
    m.run();
#endif
}

#endif

