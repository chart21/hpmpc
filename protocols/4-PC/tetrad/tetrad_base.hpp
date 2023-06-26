#pragma once
#include "../../generic_share.hpp"
#if PARTY == 3
#define SHARE Dealer_Share
#else
#define SHARE Tetrad_Share
#endif
struct Tetrad_Share{
    DATATYPE mv;
    DATATYPE l0;
    DATATYPE l1;
    DATATYPE storage; // used for storing results needed later
Tetrad_Share(){}
Tetrad_Share(DATATYPE value, DATATYPE rando1, DATATYPE rando2)
{
    mv = value;
    l0 = rando1;
    l1 = rando2;
}
};

struct Dealer_Share{
    DATATYPE l1;
    DATATYPE l2;
    DATATYPE l3;
Dealer_Share(){}
Dealer_Share(DATATYPE r0, DATATYPE r1, DATATYPE r2)
{
    this->l1 = r0;
    this->l2 = r1;
    this->l3 = r2;
}
};
