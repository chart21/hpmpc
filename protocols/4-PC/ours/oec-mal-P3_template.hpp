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

#if FUNCTION_IDENTIFIER < 5
// Receive sharing of ~XOR(a,b) locally
Dealer_Share Xor(Dealer_Share a, Dealer_Share b)
{
   return Dealer_Share(XOR(a.r0,b.r0),XOR(a.r1,b.r1));
}



//prepare AND -> send real value a&b to other P
/* void prepare_and(Dealer_Share a, Dealer_Share b, Dealer_Share &c) */
/* { */
/* /1* c.r0 = getRandomVal(P0); *1/ */
/* /1* c.r1 = getRandomVal(P1); *1/ */
/* /1* c.r2 = getRandomVal(P2); *1/ */
/* DATATYPE r0 = getRandomVal(P013); */
/* DATATYPE r1 = getRandomVal(P023); */
/* DATATYPE r2 = getRandomVal(P123); */

/* /1* DATATYPE r124 = getRandomVal(P013); // used for verification *1/ */
/* /1* DATATYPE r234 = getRandomVal(P123); // Probably sufficient to only generate with P2(-> P3 in paper) -> no because of verification *1/ */
/* DATATYPE o1 = XOR( AND(a.r0,b.r0), getRandomVal(P013)); */
/* DATATYPE o4 = XOR( XOR( AND(a.r0,b.r1) ,AND(a.r2,b.r0)),getRandomVal(P123)); */
/* /1* o4 = XOR(o4,o1); //computationally easier to let P3 do it here instead of P0 later *1/ */
/* #if PROTOCOL == 12 */
/* #if PRE == 1 */
/* pre_send_to_live(P2, o1); */
/* #else */
/* send_to_live(P2, o1); */
/* #endif */
/* #else */
/* store_compare_view(P2, o1); */
/* #endif */


/* #if PROTOCOL == 10 || PROTOCOL == 12 */
/* #if PRE == 1 */
/* pre_send_to_live(P0, o4); */
/* #else */
/* send_to_live(P0, o4); */
/* #endif */
/* #elif PROTOCOL == 11 */
/* store_compare_view(P0,o4); */
/* #endif */

/* c.r0 = XOR(r0,r1); */
/* c.r1 = XOR(r0,r2); */
/* c.r2 = XOR(r1,r2); */

/* } */
void prepare_and(Dealer_Share a, Dealer_Share b, Dealer_Share &c)
{
c.r0 = getRandomVal(P123); // r123_1
c.r1 = XOR(getRandomVal(P023),getRandomVal(P013)); // x1 

/* DATATYPE r124 = getRandomVal(P013); // used for verification */
/* DATATYPE r234 = getRandomVal(P123); // Probably sufficient to only generate with P2(-> P3 in paper) -> no because of verification */

DATATYPE o1 = XOR(c.r1,XOR( AND(a.r1,b.r1), getRandomVal(P013)));
#if PROTOCOL == 11
DATATYPE o4 = XOR(XOR(AND(a.r1, XOR(b.r0,b.r1)) ,AND(a.r1,b.r0)),getRandomVal(P123_2)); // ay1 + bx1 + x1y1 + r234_2 should remain
#else
DATATYPE o4 = XOR(XOR(AND(a.r1, XOR(b.r0,b.r1)) ,AND(a.r1,b.r0)),XOR(c.r0,getRandomVal(P123_2))); // ay1 + bx1 + x1y1 + r234_2 should remain
#endif
/* DATATYPE o4 = ADD( SUB( MULT(a.r0,b.r1) ,MULT(a.r1,b.r0)),getRandomVal(P123)); */
/* o4 = XOR(o4,o1); //computationally easier to let P3 do it here instead of P0 later */
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P2, o1);
#else
send_to_live(P2, o1);
#endif
#else
store_compare_view(P2, o1);
#endif


#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P0, o4);
#else
send_to_live(P0, o4);
#endif
#elif PROTOCOL == 11
store_compare_view(P0,o4);
#endif

}

void complete_and(Dealer_Share &c)
{
}
#endif
#if FUNCTION_IDENTIFIER > 4
void prepare_mult(Dealer_Share a, Dealer_Share b, Dealer_Share &c)
{
c.r0 = getRandomVal(P123); // r123_1
c.r1 = ADD(getRandomVal(P023),getRandomVal(P013)); // x1 

/* DATATYPE r124 = getRandomVal(P013); // used for verification */
/* DATATYPE r234 = getRandomVal(P123); // Probably sufficient to only generate with P2(-> P3 in paper) -> no because of verification */

/* DATATYPE o1 = ADD( x1y1, getRandomVal(P013)); */
DATATYPE o1 = ADD(c.r1, ADD(MULT(a.r1,b.r1), getRandomVal(P013)));

#if PROTOCOL == 11
/* DATATYPE o4 = ADD(SUB(SUB(x1y1, MULT(a.r0,b.r1)) ,MULT(a.r1,b.r0)),getRandomVal(P123_2)); // r123_2 */
DATATYPE o4 = ADD(SUB(MULT(a.r1, SUB(b.r0,b.r1)) ,MULT(b.r1,a.r0)),getRandomVal(P123_2)); // r123_2
#else
DATATYPE o4 = ADD(SUB(MULT(a.r1, SUB(b.r0,b.r1)) ,MULT(b.r1,a.r0)),SUB(getRandomVal(P123_2),c.r0)); // r123_2
#endif

/* DATATYPE o4 = ADD( SUB( MULT(a.r0,b.r1) ,MULT(a.r1,b.r0)),getRandomVal(P123)); */
/* o4 = XOR(o4,o1); //computationally easier to let P3 do it here instead of P0 later */
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P2, o1);
#else
send_to_live(P2, o1);
#endif
#else
store_compare_view(P2, o1);
#endif


#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P0, o4);
#else
send_to_live(P0, o4);
#endif
#elif PROTOCOL == 11
store_compare_view(P0,o4);
#endif
}

void complete_mult(Dealer_Share &c)
{
}

#endif

void prepare_reveal_to_all(Dealer_Share a)
{
#if PRE == 1
    pre_send_to_live(P0, a.r0);
    pre_send_to_live(P1, a.r1);
    pre_send_to_live(P2, a.r1);
#else
    send_to_live(P0, a.r0);
    send_to_live(P1, a.r1);
    send_to_live(P2, a.r1);
#endif
}    



DATATYPE complete_Reveal(Dealer_Share a)
{
#if PRE == 0
DATATYPE result = XOR(a.r0, receive_from_live(P0));
store_compare_view(P0123, result);
return result;
#else
return a.r0;
#endif
}


Dealer_Share* alloc_Share(int l)
{
    return new Dealer_Share[l];
}



void prepare_receive_from(Dealer_Share a[], int id, int l)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    DATATYPE v = get_input_live();
    DATATYPE r013 = getRandomVal(P013);
    DATATYPE r023 = getRandomVal(P023);
    DATATYPE r123 = getRandomVal(P123);

    a[i].r0 = XOR(r013,r023);
    a[i].r1 = XOR(r013,r123);
    a[i].r1 = XOR(r023,r123);
    send_to_live(P0, XOR(a[i].r1,v));
    send_to_live(P1, XOR(a[i].r0,v));
    send_to_live(P2, XOR(a[i].r0,v));
    } 
}
else if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r1 = getRandomVal(P013);
    a[i].r1 = getRandomVal(P023);
    a[i].r0 = XOR(a[i].r1,a[i].r1);

    }
}
else if(id == P1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r0 = getRandomVal(P013);
    a[i].r1 = getRandomVal(P123);
    a[i].r1 = XOR(a[i].r0,a[i].r1);
    }
}
else if(id == P2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r0 = getRandomVal(P023);
    a[i].r1 = getRandomVal(P123);
    a[i].r1 = XOR(a[i].r0,a[i].r1);
    }
}
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
#if PRE == 0
    communicate_live();
#endif
}

};
