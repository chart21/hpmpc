#pragma once
#include "../../generic_share.hpp"
class OEC_MAL3_init
{
bool optimized_sharing;
public:
OEC_MAL3_init(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}


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
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c, func_add ADD, func_sub SUB, func_mul MULT)
{
#if PROTOCOL == 12 || PROTOCOL == 8
#if PRE == 1
    pre_send_to_(P_2);
#else
    send_to_(P_2);
#endif
#else
store_compare_view_init(P_2);
#endif
#if PROTOCOL == 10 || PROTOCOL == 12 || PROTOCOL == 8
#if PRE == 1
    pre_send_to_(P_0);
#else
    send_to_(P_0);
#endif
#elif PROTOCOL == 11
store_compare_view_init(P_0);
#endif
}

template <typename func_add, typename func_sub>
void complete_mult(DATATYPE &c, func_add ADD, func_sub SUB)
{
}


void prepare_reveal_to_all(DATATYPE)
{
    #if PROTOCOL == 8
    for(int t = 0; t < 3; t++) 
    {
        #if PRE == 1 
    pre_send_to_(t);
    #else
    send_to_(t);
    #endif

    }
 #else
#if PRE == 1
    pre_send_to_(P_0);
#else
    send_to_(P_0);
#endif
#endif
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
receive_from_(P_0);
#if PROTOCOL == 8
store_compare_view_init(P_1);
#else
store_compare_view_init(P_123);
store_compare_view_init(P_0123);
#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share_();
store_output_share_();
#endif

return a;
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}

template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
/* return; */
/* old: */

if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
        #if PRE == 1
        pre_send_to_(P_0);
        pre_send_to_(P_1);
        pre_send_to_(P_2);
        #else
        send_to_(P_0);
        send_to_(P_1);
        send_to_(P_2);
        #endif
    }
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
}





void send()
{
send_();
}

// P_0 only has 1 receive round
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
