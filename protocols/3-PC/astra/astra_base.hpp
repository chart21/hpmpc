 
#pragma once
#include "../../generic_share.hpp"
#if PARTY == 0
    #define SHARE DATATYPE
#else
    #define SHARE Evaluator_Share
#endif
#define Coordinator_Share DATATYPE
struct Evaluator_Share{
    DATATYPE mv;
    DATATYPE lv;
Evaluator_Share(){ }
Evaluator_Share(DATATYPE s1, DATATYPE s2)
{
    mv = s1;
    lv = s2;
}
};

