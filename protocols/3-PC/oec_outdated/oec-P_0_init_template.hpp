#pragma once
#include "../sharemind/sharemind_base.hpp"

class OEC0_init
{
bool optimized_sharing;
public:
OEC0_init(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

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
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c, func_add ADD, func_sub SUB, func_mul MUL)
{
#if PRE == 1
    pre_send_to_(P_2);
    pre_send_to_(P_2);
#else
    send_to_(P_2);
    send_to_(P_2);
#endif

//return u[player_id] * v[player_id];
}

template <typename func_add, typename func_sub>
void complete_mult(DATATYPE &c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(DATATYPE a)
{
    for(int t = 0; t < 2; t++) 
    {
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
    pre_send_to_(t);
#else
    send_to_(t);
#endif
    }//add to send buffer
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
receive_from_(P_2);
return a;
#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share_();
#endif
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


void share_unoptimized(DATATYPE a[], int id, int l)
{

if(id == P_0)
{
    for(int i = 0; i < l; i++)
    {
        
#if PRE == 1
        pre_send_to_(P_1);
        pre_send_to_(P_2);
#else
        send_to_(P_1);
        send_to_(P_2);
#endif
    }

}
}

template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(optimized_sharing == false)
    share_unoptimized(a,id,l);
return;
}

template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
    return;
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

void finalize(std::string* ips, receiver_args* ra, sender_args* sa)
{
    finalize_(ips, ra, sa);
}


};

