#pragma once
#include "../../../datatypes/k_bitset.hpp"
#include "../../../protocols/Protocols.h"
//TODO: Could save around 1% of total 2PC boolean preprocessing by using A_KNOWN optimization for the first layer of gates
template <int k, typename Share>
class BooleanAdder
{
    using Bitset = sbitset_t<k, Share>;

  private:
    int r;
    Bitset& x;
    Bitset& y;
    Bitset& z;
    Share carry_last;
    Share carry_this;

  public:
    // constructor

    BooleanAdder() { r = k; }

    BooleanAdder(Bitset& x0, Bitset& x1, Bitset& y0) : x(x0), y(x1), z(y0) { r = k; }

    void set_values(Bitset& x0, Bitset& x1, Bitset& y0)
    {
        x = x0;
        y = x1;
        z = y0;
    }

    int get_rounds() { return r; }

    int get_total_rounds() { return k; }

    bool is_done() { return r == 0; }

    void step()
    {
        r -= 1;
        switch (r)
        {
            case k - 1:  // special case for lsbs
                z[k - 1] = x[k - 1] ^ y[k - 1];
#if A_KNOWN_TO_EVALUATORS_OPT == 1 
                    carry_last = x[k - 1].mult_a_known_to_evaluators(y[k -1]);
                    carry_last.prepare_remask();
#else
                carry_last = x[k - 1] & y[k - 1];
#endif
                break;
            case k - 2:
#if A_KNOWN_TO_EVALUATORS_OPT == 1
                carry_last.complete_remask();
#else
                carry_last.complete_and();  // get carry from lsb
#endif
                update_z();
                prepare_carry();
                break;
            case 0:
                complete_carry();
                update_z();  // no need to prepare another carry
                break;
            default:
                complete_carry();  // get carry from previous round
                update_z();        // update bit
                prepare_carry();   // prepare carry for next round
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

    void update_z() { z[r] = x[r] ^ y[r] ^ carry_last; }
};
