#pragma once
#include "oec-mal_base.hpp"
#define PRE_SHARE Dealer_Share
class OEC_MAL3
{
bool optimized_sharing;
public:
OEC_MAL3(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

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
    return Dealer_Share(ADD(a.r0,b.r0),ADD(a.r1,b.r1));
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Dealer_Share a, Dealer_Share b, Dealer_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c.r0 = getRandomVal(P_123); // r123_1
c.r1 = ADD(getRandomVal(P_023),getRandomVal(P_013)); // x1 

/* DATATYPE r124 = getRandomVal(P_013); // used for verification */
/* DATATYPE r234 = getRandomVal(P_123); // Probably sufficient to only generate with P_2(-> P_3 in paper) -> no because of verification */

/* DATATYPE o1 = ADD( x1y1, getRandomVal(P_013)); */
DATATYPE o1 = ADD(c.r1, ADD(MULT(a.r1,b.r1), getRandomVal(P_013)));

#if PROTOCOL == 11
/* DATATYPE o4 = ADD(SUB(SUB(x1y1, MULT(a.r0,b.r1)) ,MULT(a.r1,b.r0)),getRandomVal(P_123_2)); // r123_2 */
/* DATATYPE o4 = ADD(SUB(MULT(a.r1, SUB(b.r0,b.r1)) ,MULT(b.r1,a.r0)),getRandomVal(P_123_2)); // r123_2 */
DATATYPE o4 = ADD(SUB(MULT(a.r1, SUB(b.r1,b.r0)) ,MULT(b.r1,a.r0)),getRandomVal(P_123_2)); // r123_2
#else
DATATYPE o4 = ADD(SUB(MULT(a.r1, SUB(b.r1,b.r0)) ,MULT(b.r1,a.r0)),SUB(getRandomVal(P_123_2),c.r0)); // r123_2
#endif

/* DATATYPE o4 = ADD( SUB( MULT(a.r0,b.r1) ,MULT(a.r1,b.r0)),getRandomVal(P_123)); */
/* o4 = XOR(o4,o1); //computationally easier to let P_3 do it here instead of P_0 later */
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_2, o1);
#else
send_to_live(P_2, o1);
#endif
#else
store_compare_view(P_2, o1);
#endif


#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_0, o4);
#else
send_to_live(P_0, o4);
#endif
#elif PROTOCOL == 11
store_compare_view(P_0,o4);
#endif
}

template <typename func_add, typename func_sub>
void complete_mult(Dealer_Share &c, func_add ADD, func_sub SUB)
{
}


void prepare_reveal_to_all(Dealer_Share a)
{
#if PRE == 1
    pre_send_to_live(P_0, a.r0);
#else
    send_to_live(P_0, a.r0);
#endif
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Dealer_Share a, func_add ADD, func_sub SUB)
{
#if PRE == 0
DATATYPE result = SUB(receive_from_live(P_0),a.r0);
store_compare_view(P_123, a.r1);
store_compare_view(P_0123, result);
return result;
#else
#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share(a.r0);
store_output_share(a.r1);
#endif
return a.r0;
#endif


}


Dealer_Share* alloc_Share(int l)
{
    return new Dealer_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(Dealer_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    DATATYPE v = get_input_live();
    DATATYPE x_1 = getRandomVal(P_013);
    DATATYPE x_2 = getRandomVal(P_023);
    DATATYPE u = getRandomVal(P_123);

    a[i].r0 = u;
    a[i].r1 = ADD(x_1,x_2);
    DATATYPE complete_masked = ADD(v, ADD(a[i].r1, a[i].r0));
    #if PRE == 1
    pre_send_to_live(P_0, complete_masked);
    pre_send_to_live(P_1, complete_masked);
    pre_send_to_live(P_2, complete_masked);
    #else
    send_to_live(P_0, complete_masked);
    send_to_live(P_1, complete_masked);
    send_to_live(P_2, complete_masked);
    #endif
    } 
}
else if(id == P_0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r0 = SET_ALL_ZERO();
    a[i].r1 = ADD(getRandomVal(P_013),getRandomVal(P_023));
    }
}
else if(id == P_1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r0 = getRandomVal(P_123);
    a[i].r1 = getRandomVal(P_013);
    }
}
else if(id == P_2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r0 = getRandomVal(P_123);
    a[i].r1 = getRandomVal(P_023);
    }
}
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
#if PRE == 0
    communicate_live();
#endif
}

};
