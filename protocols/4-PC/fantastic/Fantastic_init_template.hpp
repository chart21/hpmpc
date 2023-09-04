#pragma once
#include "../../generic_share.hpp"
class Fantastic_Four_init
{
bool optimized_sharing;
public:
Fantastic_Four_init(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}


XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return a;
}


// Receive sharing of ~XOR(a,b) locally
template <typename Func_add>
DATATYPE Add(DATATYPE a, DATATYPE b, Func_add ADD)
{
   return a;
}


template <typename Func_add, typename Func_sub, typename Func_mul>
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c, Func_add ADD, Func_sub SUB, Func_mul MUL)
{
    #if PARTY == 0

send_to_(P_2);
send_to_(P_1);
/* store_compare_view_init(P_3); */

#elif PARTY == 1

send_to_(P_3);
send_to_(P_0);
/* store_compare_view_init(P_0); */

#elif PARTY == 2

// c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
send_to_(P_0);
/* store_compare_view_init(P_1); */
/* store_compare_view_init(P_1); */


#elif PARTY == 3

//c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
send_to_(P_1);
/* store_compare_view_init(P_2); */
/* store_compare_view_init(P_0); */

#endif

}

template <typename Func_add, typename Func_sub>
void complete_mult(DATATYPE &c, Func_add ADD, Func_sub SUB)
{
#if PARTY == 0

//c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + r013
receive_from_(P_1);
store_compare_view_init(P_3);
store_compare_view_init(P_3);



//c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
receive_from_(P_2);
store_compare_view_init(P_1);


#elif PARTY == 1

// c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
receive_from_(P_3);
store_compare_view_init(P_0);
store_compare_view_init(P_2);

//c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
receive_from_(P_0);
store_compare_view_init(P_2);

// receive second term from P_0


#elif PARTY == 2


//c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
receive_from_(P_0);
store_compare_view_init(P_1);
store_compare_view_init(P_1);
store_compare_view_init(P_3);
//receive rest from P_0, verify with P_3



#elif PARTY == 3


//c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 - r123) + r013
receive_from_(P_1);
store_compare_view_init(P_0);
store_compare_view_init(P_0);
store_compare_view_init(P_2);
#endif
}


void prepare_reveal_to_all(DATATYPE a)
{
    send_to_(PNEXT);
}    


template <typename Func_add, typename Func_sub>
DATATYPE complete_Reveal(DATATYPE a, Func_add ADD, Func_sub SUB)
{
receive_from_(PPREV);
store_compare_view_init(P_0123);
return a;
}


DATATYPE* alloc_Share(int l)
{
    return new DATATYPE[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
{
#if PARTY == 0
    for(int i = 0; i < l; i++)
    {
        send_to_(P_1);
        send_to_(P_2);
    }
#elif PARTY == 1
    for(int i = 0; i < l; i++)
    {
        send_to_(P_0);
        send_to_(P_2);
    }
#elif PARTY == 2
    for(int i = 0; i < l; i++)
    {
        send_to_(P_0);
        send_to_(P_1);
    }
#else // PARTY == 3
    for(int i = 0; i < l; i++)
    {
        send_to_(P_0);
        send_to_(P_1);
    }
#endif
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id != PSELF)
{
#if PARTY == 0
    if(id == P_1)
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_1);
    store_compare_view_init(P_2);
    }
    }
    else if(id == P_2)
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_2);
    store_compare_view_init(P_1);
    }
    }
    else // id == P_3
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_3);
    store_compare_view_init(P_1);
    }
    }
#elif PARTY == 1
    if(id == P_0)
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_0);
    store_compare_view_init(P_2);
    }
    }
    else if(id == P_2)
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_2);
    store_compare_view_init(P_0);
    }
    }
    else // id == P_3
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_3);
    store_compare_view_init(P_0);
    }
    }
#elif PARTY == 2
    if(id == P_0)
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_0);
    store_compare_view_init(P_1);
    }
    }
    else if(id == P_1)
    {
    for(int i = 0; i < l; i++)
    {
    receive_from_(P_1);
    store_compare_view_init(P_0);
    }
    } 
#endif
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

void finalize(std::string* ips, receiver_args* ra, sender_args* sa)
{
    finalize_(ips, ra, sa);
}


};
