#pragma once
#include "../generic_share.hpp"
#define SHARE DATATYPE

class TTP
{
bool input_srngs;
    public:
TTP(bool use_srngs) {input_srngs = use_srngs;}


XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return NOT(a);
}

template <typename func_add>
DATATYPE Add(DATATYPE a, DATATYPE b, func_add ADD)
{
   return ADD(a, b);
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(XOR_Share a, XOR_Share b, XOR_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c = MULT(a,b);
}

template <typename func_add, typename func_sub>
void complete_mult(XOR_Share &c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(DATATYPE a)
{
    if(PARTY == 2){
    for(int t = 0; t < num_players-1; t++) // for other protocols, sending buffers may be different for each player
        send_to_live(t, a);
    }   
}


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
DATATYPE result = a;
if(PARTY != 2)
{
    result = receive_from_live(P2);
}

return result;
}



template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
#if PARTY != 2
if(id == PSELF)
{
    for(int s = 0; s < l; s++)
    {
        send_to_live(P2, get_input_live());
    }
}
#endif
}


template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{

#if PARTY == 2
if(id == P2)
{
    for(int s = 0; s < l; s++)
    {
        a[s] = get_input_live();
    }
}
else
{
for (int i = 0; i < l; i++) {
    a[i] = receive_from_live(id);
}
}
#endif
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}



void finalize()
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

