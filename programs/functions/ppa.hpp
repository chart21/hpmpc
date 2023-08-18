#pragma once
#include "../../protocols/Protocols.h"
#include "../../datatypes/k_bitset.hpp"
#include <cstring>
#include <iostream>

template<typename Share>
class BooleanAdder {
    using Bitset = sbitset_t<Share>;
private:
    Bitset &a;
    Bitset &b;
    Bitset &sum;
    Bitset G;
    Bitset P;
    int level;
    int startPos;
    int step_length;
   
public:
//constructor

void prepare_step() {
    startPos = 1 << level;
    step_length = 1 << (level + 1);
    bool first = true;
    for (int i = BITLENGTH - startPos; i >= 0; i -= step_length) {
        int lowWire = i + 1;
        int endPos = std::max(i - startPos + 1, 0);
        for (int curWire = i; curWire >= endPos; --curWire) {

            if (curWire < BITLENGTH - 1) {
                G[curWire] = (P[curWire] & G[lowWire]) ^ G[curWire];
            }

            if (!first) {
                P[curWire] = P[lowWire] & P[curWire];
            }
        }
        first = false;
    }
}

void complete_Step() {
    bool first = true;
    for (int i = BITLENGTH - startPos; i >= 0; i -= step_length) {
        int lowWire = i + 1;
        int endPos = std::max(i - startPos + 1, 0);
        for (int curWire = i; curWire >= endPos; --curWire) {

            if (curWire < BITLENGTH - 1) {
                G[curWire].complete_and();
            }

            if (!first) {
                P[curWire].complete_and();
            }
        }
        first = false;
    }
    level++;
}

void step() {
    switch(level) {
        case -2:
           for (int i = 1; i < BITLENGTH; ++i) {
                P[i] = a[i] ^ b[i];
                G[i] = a[i] & b[i];
            }
            P[0] = a[0] ^ b[0];
            level++;
            break;
        case -1:
           for (int i = 1; i < BITLENGTH; ++i) 
                G[i].complete_and();
            level++;
            prepare_step();
            break;
        default:
            complete_Step();
            prepare_step();
            break;
        case LOG2_BITLENGTH-1:
            complete_Step();
            sum[BITLENGTH-1] = P[BITLENGTH-1];
            for (int i = 0; i < BITLENGTH - 1; ++i) {
                sum[i] = (a[i] ^ b[i]) ^ G[i + 1];
            }

            level = -3;
            break;
    }
}



BooleanAdder(Bitset &x0, Bitset &x1, Bitset &y0) : a(x0), b(x1), sum(y0) 
    {
        level = -2;
    }




int get_rounds() {
    return level;
}

int get_total_rounds() {
    return LOG2_BITLENGTH + 1;
}

bool is_done() {
    return level == -3;
}

};
