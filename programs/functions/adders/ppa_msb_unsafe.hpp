#pragma once
#include "../../../datatypes/k_bitset.hpp"
#include "../../../protocols/Protocols.h"

template <int k, typename Share>
class PPA_MSB_Unsafe
{
    using Bitset = sbitset_t<k, Share>;

  private:
    Bitset& a;
    Bitset& b;
    Share& msb;
    int level;
    int startPos;
    int step_length;
    // CUT_FRACTIONAL_BITS_OPT: cst[i] tracks whether prefix wire i currently holds the IDENTITY of the
    // prefix operator, (g, p) = (0, 1). The vacant slices start out identity, and a wire stays identity
    // only while every slice it aggregates is - so this is tracked exactly rather than derived from the
    // level, which keeps the skipping valid at every tree depth.
    bool cst[k];
    // prepare_step mutates cst[], so completion cannot re-derive which combines it skipped - it has to
    // be recorded when the decision is made.
    bool skipped[k];

  public:
    static constexpr bool cut_supported = (CUT_FRAC_ELIGIBLE_GENERIC && k == BITLENGTH);
    // vacant = provably sign extension. Slice 0 is the numeric MSB; slice FRACTIONAL is the boundary
    // whose RAW pair feeds the output tap, so it is NOT substituted.
    static bool vacant(int i)
    {
        return cut_supported && g_cut_frac_active && i >= 1 && i < FRACTIONAL;
    }
    static bool cut_on() { return cut_supported && g_cut_frac_active; }
    // constructor

    void prepare_step()
    {
        startPos = 1 << level;
        step_length = 1 << (level + 1);
        bool first = true;
        for (int i = startPos; i < k; i += step_length)
        {
            int lowWire = k - i;
            int curWire = std::max(lowWire - startPos, 1);

            if (curWire != lowWire)
            {
                skipped[curWire] = false;
                if (cut_on() && cst[lowWire])
                {
                    // low wire is the identity (g=0, p=1): G stays, P stays. Nothing to compute and
                    // nothing to send - cst[curWire] is unchanged.
                    skipped[curWire] = true;
                    first = false;
                    continue;
                }
                if (cut_on() && cst[curWire])
                {
                    // this wire is the identity, so the combine degenerates to a copy of the low wire
                    b[curWire] = b[lowWire];
                    if (!first)
                        a[curWire] = a[lowWire];
                    cst[curWire] = false;
                    skipped[curWire] = true;
                    first = false;
                    continue;
                }
                // G1 = G1 ^ P_1 & G0
                b[curWire] = (a[curWire] & b[lowWire]) ^ b[curWire];

                if (!first)
                {

                    // P_1 = P_1 & P_0
                    a[curWire] = a[lowWire] & a[curWire];
                }

                first = false;
            }
        }
    }

    void complete_Step()
    {
        bool first = true;
        for (int i = startPos; i < k; i += step_length)
        {
            int lowWire = k - i;
            int curWire = std::max(lowWire - startPos, 1);

            if (curWire != lowWire)
            {
                if (cut_on() && skipped[curWire])
                {
                    first = false;
                    continue;  // prepared nothing this round, so there is nothing to complete
                }
                // G1 = G1 ^ P_1 & G0
                b[curWire].complete_and();

                if (!first)
                {

                    // P_1 = P_1 & P_0
                    a[curWire].complete_and();
                }

                first = false;
            }
        }
        level++;
    }

    void step()
    {
        const int log2k = std::ceil(std::log2(k));
        switch (level)
        {
            case -2:
                for (int i = 0; i < k; ++i)
                {
                    cst[i] = false;
                    skipped[i] = false;
                }
                // Output tap: with slices 1..F-1 substituted the tree aggregates from the boundary, so
                // the sum bit to read is the boundary slice's, not slice 0's.
                if (cut_on())
                    msb = a[FRACTIONAL] ^ b[FRACTIONAL];
                else
                {
                    a[0] = a[0] ^ b[0];
                    msb = a[0];
                }
                for (int i = 1; i < k; ++i)
                {
                    if (vacant(i))
                    {
                        a[i] = Share(SET_ALL_ONE());   // p := 1
                        b[i] = Share(SET_ALL_ZERO());  // g := 0
                        cst[i] = true;
                        continue;  // no AND, so no communication for this slice
                    }
                    Share tmp = a[i] ^ b[i];
                    tmp = a[i] ^ b[i];
                    /* G[i - 1] = a[i - 1] & b[i - 1]; */
#if A_KNOWN_TO_EVALUATORS_OPT == 1
                    b[i] = a[i].mult_a_known_to_evaluators(b[i]);
                    b[i].prepare_remask();
#else
#if A_KNOWN == 1
                    b[i] = a[i].prepare_and_a_known(b[i]);
#else
                    b[i] = a[i] & b[i];
#endif

#endif
                    a[i] = tmp;
                }
                level++;
                break;
            case -1:
                for (int i = 1; i < k; ++i)
                {
                    if (vacant(i))
                        continue;  // substituted above, nothing was prepared
                    // G[i - 1].complete_and();
#if A_KNOWN_TO_EVALUATORS_OPT == 1
                    b[i].complete_remask();
#else
                    b[i].complete_and();  // possibly wrong and above is correct
#endif
                }
                level++;
                prepare_step();
                break;
            default:
                complete_Step();
                prepare_step();
                break;
            case log2k - 1:
                complete_Step();
                msb = msb ^ b[1];
                level = -3;
                break;
        }
    }

    PPA_MSB_Unsafe(Bitset& x0, Bitset& x1, Share& y0) : a(x0), b(x1), msb(y0) { level = -2; }

    int get_rounds() { return level; }

    int get_total_rounds() { return std::ceil(std::log2(k)) + 1; }

    bool is_done() { return level == -3; }
};
