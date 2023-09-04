#pragma once
#include "oec-mal_base.hpp"
class OEC_MAL3_POST
{
bool optimized_sharing;
public:
OEC_MAL3_POST(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

Dealer_Share public_val(DATATYPE a)
{
    return Dealer_Share(SET_ALL_ZERO(),SET_ALL_ZERO());
}

Dealer_Share Not(Dealer_Share a)
{
   return a;
}


template <typename func_add>
Dealer_Share Add(Dealer_Share a, Dealer_Share b, func_add ADD)
{
    return a;
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Dealer_Share a, Dealer_Share b, Dealer_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
}

template <typename func_add, typename func_sub>
void complete_mult(Dealer_Share &c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(Dealer_Share a)
{
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Dealer_Share a, func_add ADD, func_sub SUB)
{

DATATYPE result = SUB(receive_from_live(P_0),retrieve_output_share());
store_compare_view(P_123, retrieve_output_share());
store_compare_view(P_0123, result);
return result;
}


Dealer_Share* alloc_Share(int l)
{
    return new Dealer_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(Dealer_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
}

template <typename func_add, typename func_sub>
void complete_receive_from(Dealer_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
}




void send()
{
    send_live();
}

void receive()
{
    receive_live();
}

void communicate()
{
    communicate_live();
}

};
