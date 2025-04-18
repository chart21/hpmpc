#pragma once
#include "../../../datatypes/k_bitset.hpp"
#include "../../../protocols/Protocols.h"

template <int k, typename Share>
class BooleanAdder_MSB_Carry
{
    using Bitset = sbitset_t<k, Share>;

  private:
    int r;
    Bitset& x;
    Bitset& y;
    Share carry_last;
    Share carry_this;
    Share msb;

  public:
    // constructor

    BooleanAdder_MSB_Carry() { r = k; }

    BooleanAdder_MSB_Carry(Bitset& x0, Bitset& x1) : x(x0), y(x1) { r = k; }

#if TRUNC_DELAYED == 1
    Share get_msb() { return msb; }

    void update_msb() { msb = x[0] ^ y[0] ^ carry_last; }
#endif
    void set_values(Bitset& x0, Bitset& x1)
    {
        x = x0;
        y = x1;
    }

    int get_rounds() { return r; }

    Share get_carry() { return carry_this; }

    int get_total_rounds() { return k; }

    bool is_done() { return r == -1; }

    template <int m = k, typename std::enable_if<(m > 2), int>::type = 0>
    void step()
    {
        r -= 1;

        switch (r)
        {
            case k - 1:  // special case for lsbs
                carry_last = x[k - 1] & y[k - 1];
                break;
            case k - 2:
                carry_last.complete_and();  // get carry from lsb
                prepare_carry();
                break;
            case 0:
                complete_carry();
#if TRUNC_DELAYED == 1
                update_msb();
#endif
                prepare_carry();
                break;
            case -1:
                complete_carry();  // get final carry
                break;
            default:
                complete_carry();  // get carry from previous round
                prepare_carry();   // prepare carry for next round
                break;
        }
    }

    template <int m = k, typename std::enable_if<(m == 2), int>::type = 0>
    void step()
    {
        r -= 1;

        switch (r)
        {
            case k - 1:  // special case for lsbs
                carry_last = x[k - 1] & y[k - 1];
                break;
            case k - 2:
                carry_last.complete_and();  // get carry from lsb
#if TRUNC_DELAYED == 1
                update_msb();
#endif
                prepare_carry();
                break;
            case -1:
                complete_carry();  // get final carry
                break;
            default:
                complete_carry();  // get carry from previous round
                prepare_carry();   // prepare carry for next round
                break;
        }
    }

    template <int m = k, typename std::enable_if<(m == 1), int>::type = 0>
    void step()
    {
        r -= 1;

        switch (r)
        {
            case k - 1:  // special case for lsbs
                carry_last = x[k - 1] & y[k - 1];
                break;
            case k - 2:
                carry_last.complete_and();  // get carry from lsb
                carry_this = carry_last;
#if TRUNC_DELAYED == 1
                msb = x[0] ^ y[0];
#endif
                break;
        }
    }

    void prepare_carry() { carry_this = (carry_last ^ x[r]) & (carry_last ^ y[r]); }

    void complete_carry()
    {
        carry_this.complete_and();
        carry_this = carry_this ^ carry_last;
        carry_last = carry_this;
    }
};
