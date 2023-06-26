#pragma once
#include "../../generic_share.hpp"
#if PARTY == 3
#define SHARE Dealer_Share
#else
#define SHARE OEC_MAL_Share
#endif
struct OEC_MAL_Share{
    DATATYPE v;
    DATATYPE r;
#if PROTOCOL != 11
    //no value store needed
#else
    #if PARTY == 1 || PARTY == 2
    DATATYPE m; // used for saving messages for verification
    #endif
#endif
OEC_MAL_Share(){}
OEC_MAL_Share(DATATYPE value, DATATYPE rando)
{
    v = value;
    r = rando;
}
};

struct Dealer_Share{
    DATATYPE r0;
    DATATYPE r1;
Dealer_Share(){}
Dealer_Share(DATATYPE r0, DATATYPE r1)
{
    this->r0 = r0;
    this->r1 = r1;
}
};
