#pragma once
#include "../../protocols/Protocols.h"
#include "../../datatypes/k_bitset.hpp"
#include <cstring>
#include <iostream>

template<typename Share>
class PPA_MSB {
    using Bitset = sbitset_t<Share>;
private:
    Bitset &a;
    Bitset &b;
    Share &msb;
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
      for (int i = startPos; i < BITLENGTH; i += step_length){
            int lowWire = BITLENGTH - i;
            int curWire = std::max(lowWire - startPos, 1);

            if (curWire != lowWire) {
                // G1 = G1 ^ P_1 & G0
                G[curWire] = (P[curWire] & G[lowWire]) ^ G[curWire];

                if (!first) {

                    // P_1 = P_1 & P_0
                    P[curWire] = P[lowWire] & P[curWire];
                }

                first = false;
            }

        }
    }

void complete_Step() {
    bool first = true;
      for (int i = startPos; i < BITLENGTH; i += step_length){
            int lowWire = BITLENGTH - i;
            int curWire = std::max(lowWire - startPos, 1);

            if (curWire != lowWire) {
                // G1 = G1 ^ P_1 & G0
                G[curWire].complete_and();

                if (!first) {

                    // P_1 = P_1 & P_0
                    P[curWire].complete_and();
                }

                first = false;
            }

        }
    level++;
}

void step() {
    switch(level) {
        case -2:
            P[0] = a[0] ^ b[0];
           for (int i = 1; i < BITLENGTH; ++i) {
                P[i] = a[i] ^ b[i];
                /* G[i - 1] = a[i - 1] & b[i - 1]; */
                G[i] = a[i] & b[i]; // possibly wrong and above is correct
            }
            level++;
            break;
        case -1:
           for (int i = 1; i < BITLENGTH; ++i) 
               // G[i - 1].complete_and();
                G[i].complete_and(); // possibly wrong and above is correct

            level++;
            prepare_step();
            break;
        default:
            complete_Step();
            prepare_step();
        break;
    case LOG2_BITLENGTH-1:
            complete_Step();
            msb = a[0] ^ b[0] ^ G[1];
            level = -3;
            break;
    }
}



PPA_MSB(Bitset &x0, Bitset &x1, Share &y0) : a(x0), b(x1), msb(y0) 
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
