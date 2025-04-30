#pragma once

#include "../MP-SPDZ/lib/Machine.hpp"
#include "../MP-SPDZ/lib/Shares/CInteger.hpp"
#include "../MP-SPDZ/lib/Shares/Integer.hpp"
#include "../datatypes/k_bitset.hpp"
#include "../datatypes/k_sint.hpp"
#include "../protocols/Additive_Share.hpp"
#include "../protocols/Protocols.h"
#include "../protocols/XOR_Share.hpp"
#include "../utils/print.hpp"
#include <cstdint>

#if FUNCTION_IDENTIFIER == 0  // replace with actual number
#define FUNCTION example      // replace with function name
#endif

// Boilerplate
#define RESULTTYPE DATATYPE
void generateElements() {}

#if FUNCTION_IDENTIFIER == 0

template <typename Share>
void example(DATATYPE* res)
{
    using cint = IR::CInteger<INT_TYPE, UINT_TYPE>;
    using int_t = IR::Integer<int64_t, uint64_t>;

    using S = XOR_Share<DATATYPE, Share>;

    using A = Additive_Share<DATATYPE, Share>;

    IR::Machine<int_t, cint, Share, A, sbitset_t, S> m("<schedule-file>");  // replace with actual path
    m.run();
}

#endif
