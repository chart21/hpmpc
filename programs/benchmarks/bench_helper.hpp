#pragma once
#include "../../datatypes/XOR_Share.hpp"
// if placed after a function, gurantees that all parties have finished computation and communication
template <typename Share>
void dummy_reveal()
{
    using S = XOR_Share<DATATYPE, Share>;
    S dummy;
    dummy.prepare_reveal_to_all();
    Share::communicate();
    dummy.complete_reveal_to_all();
}
