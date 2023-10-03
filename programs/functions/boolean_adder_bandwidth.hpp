#pragma once
#include "../../protocols/Protocols.h"
#include "../../datatypes/k_bitset.hpp"
#include <cstring>
#include <iostream>

template<typename Share>
class BooleanAdder {
    using Bitset = sbitset_t<Share>;
private:
    int r;
    Bitset &x;
    Bitset &y;
    Bitset &z;
    Share carry_last;
    Share carry_this;
   
public:
//constructor

BooleanAdder()
    {
        r = BITLENGTH;
    }

BooleanAdder(Bitset &x0, Bitset &x1, Bitset &y0) : x(x0), y(x1), z(y0) 
    {
        r = BITLENGTH;
    }

void set_values(Bitset &x0, Bitset &x1, Bitset &y0) 
    {
        x = x0;
        y = x1;
        z = y0;
    }

int get_rounds() {
    return r;
}

int get_total_rounds() {
    return BITLENGTH;
}

bool is_done() {
    return r == 0;
}

void step() 
{
r-=1;
switch(r)
{
    case BITLENGTH-1: //special case for lsbs
        z[BITLENGTH-1] = x[BITLENGTH-1] ^ y[BITLENGTH-1];
        carry_last = x[BITLENGTH-1] & y[BITLENGTH-1];
        break;
case BITLENGTH-2:
        carry_last.complete_and(); // get carry from lsb
        update_z(); 
        prepare_carry();
      break;
    case 0:
      complete_carry();
      update_z(); // no need to prepare another carry
      break;
    default:
      complete_carry(); // get carry from previous round
      update_z(); // update bit 
      prepare_carry(); // prepare carry for next round
        break;
}
}

void prepare_carry()
{
    carry_this = (carry_last ^ x[r]) & (carry_last ^ y[r]);
}

void complete_carry()
{
    carry_this.complete_and();
    carry_this = carry_this ^ carry_last;
    carry_last = carry_this;

}

void update_z()
{
    z[r] = x[r] ^ y[r] ^ carry_last;

}


};
