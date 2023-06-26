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

// Receive sharing of ~XOR(a,b) locally
Dealer_Share Xor(Dealer_Share a, Dealer_Share b)
{
   return a; 
}



//prepare AND -> send real value a&b to other P
void prepare_and(Dealer_Share a, Dealer_Share b, Dealer_Share &c)
{
}

void complete_and(Dealer_Share &c)
{
}


void prepare_reveal_to_all(Dealer_Share a)
{
}    



DATATYPE complete_Reveal(Dealer_Share a)
{
DATATYPE result = receive_from_live(P0);
store_compare_view(P0123, result);
return result;
}


Dealer_Share* alloc_Share(int l)
{
    return new Dealer_Share[l];
}



void prepare_receive_from(Dealer_Share a[], int id, int l)
{
}

void complete_receive_from(Dealer_Share a[], int id, int l)
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
