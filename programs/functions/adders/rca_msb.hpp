#pragma once
#include "../../../datatypes/k_bitset.hpp"
#include "../../../protocols/Protocols.h"

template <int k, typename Share>
class BooleanAdder_MSB
{
    using Bitset = sbitset_t<k, Share>;

  private:
    int r;
    Bitset& x;
    Bitset& y;
    Share& z;
    Share carry_last;
    Share carry_this;

  public:
    // constructor

    BooleanAdder_MSB() { r = k; }

    BooleanAdder_MSB(Bitset& x0, Bitset& x1, Share& y0) : x(x0), y(x1), z(y0) { r = k; }

    void set_values(Bitset& x0, Bitset& x1, Share& y0)
    {
        x = x0;
        y = x1;
        z = y0;
    }

    // CUT_FRACTIONAL_BITS_OPT, protocol-independent form. Under TRUNC_DELAYED == 0 the value feeding
    // this adder has been truncated by FRACTIONAL bits, so its top FRACTIONAL slices (0 .. F-1, slice 0
    // being the numeric MSB) are sign extension and carry no information. The ripple runs from the LSB
    // (slice k-1) upwards, so the cut is simply "stop at slice F and read the sum bit there" - the last
    // F rounds, and their AND gates and communication, never happen. Nothing protocol-specific is
    // involved: the vacant slices are just never read.
    //
    // g_cut_frac_active is set by RELU only; max/min comparisons share this class and run full width,
    // so the bound has to be a runtime query rather than a compile-time constant.
    static constexpr bool cut_supported = (CUT_FRAC_ELIGIBLE_GENERIC && k == BITLENGTH);
    static int cut_lo() { return (cut_supported && g_cut_frac_active) ? FRACTIONAL : 0; }

    int get_rounds() { return r; }

    int get_total_rounds() { return k - cut_lo(); }

    bool is_done() { return r == cut_lo(); }

    void step()
    {
        r -= 1;
        const int lo = cut_lo();
        if (lo != 0 && r == lo)  // cut: finish early on the boundary slice
        {
            complete_carry();
            z = x[lo] ^ y[lo] ^ carry_last;
            return;
        }
        switch (r)
        {
            case k - 1:  // special case for lsbs
#if A_KNOWN_TO_EVALUATORS_OPT == 1
                carry_last = x[k - 1].mult_a_known_to_evaluators(y[k -1]);
                carry_last.prepare_remask();
#else
#if A_KNOWN == 1
                carry_last = x[k - 1].prepare_and_a_known(y[k - 1]);
#else
                carry_last = x[k - 1] & y[k - 1];
#endif
#endif
                break;
            case k - 2:
#if A_KNOWN_TO_EVALUATORS_OPT == 1
                carry_last.complete_remask();
#else
                carry_last.complete_and();  // get carry from lsb
#endif
                prepare_carry();
                break;
            case 0:
                complete_carry();
                update_z();  // final value, no need to prepare another carry
                break;
            default:
                complete_carry();  // get carry from previous round
                prepare_carry();   // prepare carry for next round
                break;
        }
    }

#if A_KNOWN_TO_EVALUATORS_OPT == 0
    void prepare_carry() { carry_this = (carry_last ^ x[r]) & (carry_last ^ y[r]); }
#else
    void prepare_carry() { 
        if(current_phase == PHASE_LIVE)
            carry_this = (carry_last ^ x[r]) & (carry_last ^ y[r]);
        else
            carry_this = carry_last & (!y[r]); 
    }
#endif

    void complete_carry()
    {
        carry_this.complete_and();
#if A_KNOWN_TO_EVALUATORS_OPT == 0
        carry_this = carry_this ^ carry_last;
        carry_last = carry_this;
#else
     if(current_phase == PHASE_LIVE)
     {
        carry_this = carry_this ^ carry_last;
        carry_last = carry_this;
     }
#endif
    }

    void update_z() { z = x[0] ^ y[0] ^ carry_last; }
};
