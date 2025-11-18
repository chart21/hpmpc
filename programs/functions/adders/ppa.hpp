#pragma once
#include "../../../datatypes/k_bitset.hpp"
#include "../../../protocols/Protocols.h"

template <int k, typename Share>
class PPA
{
    using Bitset = sbitset_t<k, Share>;

  private:
    Bitset& a;
    Bitset& b;
    Bitset& sum;
    Bitset G;
    Bitset P;
    int level;
    int startPos;
    int step_length;

  public:
    // constructor

    void prepare_step()
    {
        startPos = 1 << level;
        step_length = 1 << (level + 1);
        bool first = true;
        for (int i = k - startPos; i >= 0; i -= step_length)
        {
            int lowWire = i + 1;
            int endPos = std::max(i - startPos + 1, 0);
            for (int curWire = i; curWire >= endPos; --curWire)
            {

                if (curWire < k - 1)
                {
                    G[curWire] = (P[curWire] & G[lowWire]) ^ G[curWire];
                }

                if (!first)
                {
                    P[curWire] = P[lowWire] & P[curWire];
                }
            }
            first = false;
        }
    }

    void complete_Step()
    {
        bool first = true;
        for (int i = k - startPos; i >= 0; i -= step_length)
        {
            int lowWire = i + 1;
            int endPos = std::max(i - startPos + 1, 0);
            for (int curWire = i; curWire >= endPos; --curWire)
            {

                if (curWire < k - 1)
                {
                    G[curWire].complete_and();
                }

                if (!first)
                {
                    P[curWire].complete_and();
                }
            }
            first = false;
        }
        level++;
    }

    void step()
    {
        const int log2k = std::ceil(std::log2(k));
        switch (level)
        {
            case -2:
                for (int i = 1; i < k; ++i)
                {
                    P[i] = a[i] ^ b[i];
#if A_KNOWN_TO_EVALUATORS_OPT_SIM == 1 
                    G[i] = a[i].mult_a_known_to_evaluators(b[i]);
                    G[i].prepare_remask();
#else
                    G[i] = a[i] & b[i];
#endif

                }
                P[0] = a[0] ^ b[0];
                level++;
                break;
            case -1:
                for (int i = 1; i < k; ++i)
#if A_KNOWN_TO_EVALUATORS_OPT_SIM == 1
                    G[i].complete_remask(); 
#else
                    G[i].complete_and();  
#endif
                level++;
                prepare_step();
                break;
            default:
                complete_Step();
                prepare_step();
                break;
            case log2k - 1:
                complete_Step();
                sum[BITLENGTH - 1] = P[k - 1];
                for (int i = 0; i < k - 1; ++i)
                {
                    sum[i] = (a[i] ^ b[i]) ^ G[i + 1];
                }

                level = -3;
                break;
        }
    }

    PPA(Bitset& x0, Bitset& x1, Bitset& y0) : a(x0), b(x1), sum(y0) { level = -2; }

    int get_rounds() { return level; }

    int get_total_rounds() { return std::ceil(std::log2(k)) + 1; }

    bool is_done() { return level == -3; }
};
