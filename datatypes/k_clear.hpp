#pragma once
#include "../protocols/Protocols.h"
template <typename Datatype>
class clear_t
{
  private:
    Datatype values[BITLENGTH];

  public:
    // temporary constructor
    clear_t() {}

    clear_t(UINT_TYPE value)
    {
        for (int i = 0; i < BITLENGTH; i++)
            values[i] = PROMOTE(value);
    }

    clear_t(UINT_TYPE value[DATTYPE]) { init(value); }

    void init(UINT_TYPE value[DATTYPE])
    {
        DATATYPE temp_d[BITLENGTH];
        orthogonalize_arithmetic(value, temp_d);
        for (int i = 0; i < BITLENGTH; i++)
            values[i] = temp_d[i];
    }

    void reveal_to_all(UINT_TYPE result[DATTYPE]) const { unorthogonalize_arithmetic(values, result); }
};
