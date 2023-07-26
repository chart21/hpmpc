#pragma once
#include "../generic_share.hpp"
#define INIT_SHARE DATATYPE

class TTP_init
{
bool input_srngs;
public:
TTP_init(bool use_srngs) {input_srngs = use_srngs;}




XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return a;
}

template <typename func_add>
DATATYPE Add(DATATYPE a, DATATYPE b, func_add ADD)
{
   return a;
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(DATATYPE &a, DATATYPE &b, DATATYPE &c, func_add ADD, func_sub SUB, func_mul MUL)
{
}

template <typename func_add, typename func_sub>
void complete_mult(DATATYPE c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(DATATYPE a)
{
    if(PARTY == 2)
    {
        for(int t = 0; t < num_players-1; t++) 
        {
            send_to_(t);
        }//add to send buffer
    
    }
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
if(PARTY != 2)
{
    receive_from_(P2);
}
return a;
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF && PARTY != 2)
{
    for(int i = 0; i < l; i++) 
        send_to_(P2);
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
    if(id == PSELF)
{
    return;
}
if(PARTY == 2)
{
        for (int i = 0; i < l; i++) {
receive_from_(id);
        }
    }

}

void send()
{
send_();
}
void receive()
{
    receive_();
}
void communicate()
{
communicate_();
}

void finalize(std::string* ips)
{
    finalize_(ips);
}


};
