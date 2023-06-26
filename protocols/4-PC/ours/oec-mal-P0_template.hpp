#pragma once
#include "oec-mal_base.hpp"
#define PRE_SHARE OEC_MAL_Share
class OEC_MAL0
{
bool optimized_sharing;
public:
OEC_MAL0(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

OEC_MAL_Share public_val(DATATYPE a)
{
    return OEC_MAL_Share(a,SET_ALL_ZERO());
}

OEC_MAL_Share Not(OEC_MAL_Share a)
{
   return OEC_MAL_Share(NOT(a.v),a.r);
}

// Receive sharing of ~XOR(a,b) locally
OEC_MAL_Share Xor(OEC_MAL_Share a, OEC_MAL_Share b)
{
    a.v = XOR(a.v,b.v);
    a.r = XOR(a.r,b.r);
   return a;
}

#if FUNCTION_IDENTIFIER < 5
/* void prepare_and(OEC_MAL_Share a, OEC_MAL_Share b, OEC_MAL_Share &c) */
/* { */
/* c.r = XOR(getRandomVal(P013),getRandomVal(P023)); // calculate c_1 */
/* /1* DATATYPE r124 = getRandomVal(P013); *1/ */
/* /1* DATATYPE o1 = XOR( x1y1, r124); *1/ */
/* DATATYPE o1 = XOR(AND(a.r, b.r), getRandomVal(P013)); */

/* #if PROTOCOL == 11 */
/* c.v = XOR(c.r, XOR( AND(a.v,b.r), AND(b.v,a.r))); */
/* #else */
/* c.v = XOR( AND(a.v,b.r), AND(b.v,a.r)); */
/* #endif */

/* /1* DATATYPE m3_flat = AND(a.v,b.v); *1/ */

/* /1* c.m = XOR(x1y1, XOR( XOR(AND(a.v,b.v), AND( XOR(a.v, a.r), XOR(b.v, b.r))), c.r)); *1/ */
/* #if PROTOCOL == 12 */
/* store_compare_view(P2,o1); */
/* #else */
/*     #if PRE == 1 */
/*         pre_send_to_live(P2, o1); */
/*     #else */
/*         send_to_live(P2, o1); */
/*     #endif */
/* #endif */


/* } */

/* void complete_and(OEC_MAL_Share &c) */
/* { */
/* #if PROTOCOL == 10 || PROTOCOL == 12 */
/* #if PRE == 1 */
/* DATATYPE o_4 = pre_receive_from_live(P3); */
/* #else */
/* DATATYPE o_4 = receive_from_live(P3); */
/* #endif */
/* #elif PROTOCOL == 11 */
/* DATATYPE m_2XORm_3 = receive_from_live(P2); */
/* store_compare_view(P1, m_2XORm_3); // Verify if P_2 sent correct message m_2 XOR m_3 */
/* store_compare_view(P3, XOR(m_2XORm_3,c.v)); // x2y2 + x3y3 + r234 should remain */
/* c.v = XOR(c.r,receive_from_live(P2)); // receive ab + c_1 + c_3 from P2 (P3 in paper), need to convert to ab + c_3, maybe conversion not needed in boolean domain? */
/* store_compare_view(P1, c.v); // Verify if P_2 sent correct message of ab */
/* #endif */

/* #if PROTOCOL == 10 || PROTOCOL == 12 */
/* /1* DATATYPE m3_prime = receive_from_live(P2); *1/ */
/* c.v = XOR(c.v, o_4); */

/* /1* c.m = XOR(c.m, o_4); *1/ */
/* store_compare_view(P012,XOR(c.v, c.r)); */
/* c.v = XOR(c.v, receive_from_live(P2)); */
/* store_compare_view(P1, c.v); // P_2 has ab+c1, P1 has ab+c3 -> P1 needs to convert */
/* #endif */
/* } */
void prepare_and(OEC_MAL_Share a, OEC_MAL_Share b, OEC_MAL_Share &c)
{
c.r = XOR(getRandomVal(P013),getRandomVal(P023)); // calculate c_1
/* DATATYPE r124 = getRandomVal(P013); */
/* DATATYPE o1 = XOR( x1y1, r124); */
DATATYPE o1 = XOR(c.r, XOR(AND(a.r, b.r), getRandomVal(P013)));

#if PROTOCOL == 11
c.v = XOR(XOR( AND(a.v,b.r), AND(b.v,a.r)),c.r);
#else
c.v = XOR( AND(a.v,b.r), AND(b.v,a.r));
#endif

/* DATATYPE m3_flat = AND(a.v,b.v); */

/* c.m = XOR(x1y1, XOR( XOR(AND(a.v,b.v), AND( XOR(a.v, a.r), XOR(b.v, b.r))), c.r)); */
#if PROTOCOL == 12
store_compare_view(P2,o1);
#else
    #if PRE == 1
        pre_send_to_live(P2, o1);
    #else
        send_to_live(P2, o1);
    #endif
#endif

}

void complete_and(OEC_MAL_Share &c)
{
#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
DATATYPE o_4 = pre_receive_from_live(P3);
#else
DATATYPE o_4 = receive_from_live(P3);
#endif
#elif PROTOCOL == 11
DATATYPE m_2XORm_3 = receive_from_live(P2);
store_compare_view(P1, m_2XORm_3); // Verify if P_2 sent correct message m_2 XOR m_3
store_compare_view(P3, XOR(m_2XORm_3,c.v)); // x1 y1 - x1 y3 - x 3 y1 - r234 should remain
c.v = receive_from_live(P2); // receive ab + c1 + r_234_1 from P2 (P3 in paper), need to convert to ab+ r234_1 (maybe not? and only for verify?)
store_compare_view(P1, c.v); // Verify if P_2 sent correct message of ab
c.v = XOR(c.v,c.r); // convert to ab + r234_1 (maybe not needed)
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
/* DATATYPE m3_prime = receive_from_live(P2); */
c.v = XOR(c.v, o_4);

/* c.m = XOR(c.m, o_4); */
DATATYPE m3_prime = receive_from_live(P2);
c.v = XOR(m3_prime,c.v);
store_compare_view(P012,XOR(c.v, c.r)); // compare ab + r_234_1 + c_1 with P2,P3
store_compare_view(P1, m3_prime); // compare m_3 prime with P2
#endif
}

#endif

#if FUNCTION_IDENTIFIER > 4
void prepare_mult(OEC_MAL_Share a, OEC_MAL_Share b, OEC_MAL_Share &c)
{
c.r = ADD(getRandomVal(P013),getRandomVal(P023)); // calculate c_1
/* DATATYPE r124 = getRandomVal(P013); */
/* DATATYPE o1 = XOR( x1y1, r124); */
DATATYPE o1 = ADD(c.r, ADD(MULT(a.r, b.r), getRandomVal(P013)));

#if PROTOCOL == 11
c.v = SUB(ADD( MULT(a.v,b.r), MULT(b.v,a.r)),c.r);
#else
c.v = ADD( MULT(a.v,b.r), MULT(b.v,a.r));
#endif

/* DATATYPE m3_flat = AND(a.v,b.v); */

/* c.m = XOR(x1y1, XOR( XOR(AND(a.v,b.v), AND( XOR(a.v, a.r), XOR(b.v, b.r))), c.r)); */
#if PROTOCOL == 12
store_compare_view(P2,o1);
#else
    #if PRE == 1
        pre_send_to_live(P2, o1);
    #else
        send_to_live(P2, o1);
    #endif
#endif


}

void complete_mult(OEC_MAL_Share &c)
{
#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
DATATYPE o_4 = pre_receive_from_live(P3);
#else
DATATYPE o_4 = receive_from_live(P3);
#endif
#elif PROTOCOL == 11
DATATYPE m_2XORm_3 = receive_from_live(P2);
store_compare_view(P1, m_2XORm_3); // Verify if P_2 sent correct message m_2 XOR m_3
store_compare_view(P3, SUB(m_2XORm_3,c.v)); // x1 y1 - x1 y3 - x 3 y1 - r234 should remain
c.v = receive_from_live(P2); // receive ab + c1 + r_234_1 from P2 (P3 in paper), need to convert to ab+ r234_1 (maybe not? and only for verify?)
store_compare_view(P1, c.v); // Verify if P_2 sent correct message of ab
c.v = SUB(c.v,c.r); // convert to ab + r234_1 (maybe not needed)
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
/* DATATYPE m3_prime = receive_from_live(P2); */
c.v = ADD(c.v, o_4);

/* c.m = XOR(c.m, o_4); */
DATATYPE m3_prime = receive_from_live(P2);
c.v = SUB(m3_prime,c.v);
store_compare_view(P012,ADD(c.v, c.r)); // compare ab + r_234_1 + c_1 with P2,P3
store_compare_view(P1, m3_prime); // compare m_3 prime with P2
#endif
}

#endif


void prepare_reveal_to_all(OEC_MAL_Share a)
{
#if PRE == 0
    send_to_live(P3, a.v);
#endif
}    



DATATYPE complete_Reveal(OEC_MAL_Share a)
{
#if PRE == 1
DATATYPE result = XOR(a.v, pre_receive_from_live(P3));
send_to_live(P3, result);
#else
DATATYPE result = XOR(a.v, receive_from_live(P3));
#endif
store_compare_view(P0123, result); 
// Problem, P3 sends all the values -> send in circle
return result;
}


OEC_MAL_Share* alloc_Share(int l)
{
    return new OEC_MAL_Share[l];
}



void prepare_receive_from(OEC_MAL_Share a[], int id, int l)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    a[i].v = get_input_live();
    DATATYPE x_2 = getRandomVal(P013);
    DATATYPE x_3 = getRandomVal(P023);
    a[i].r = XOR(x_2, x_3);
    
    send_to_live(P1,XOR(a[i].v, a[i].r));
    send_to_live(P2,XOR( a[i].v, a[i].r));
    a[i].v = XOR(a[i].v,x_3);
    } 
}
else if(id == P1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r = getRandomVal(P013);
    }
}
else if(id == P2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r = getRandomVal(P023);
    }
}
else if(id == P3)
{
    for(int i = 0; i < l; i++)
    {
    DATATYPE r013 = getRandomVal(P013);
    DATATYPE r023 = getRandomVal(P023);
    a[i].r = XOR(r013, r023);
    }
}
}
void complete_receive_from(OEC_MAL_Share a[], int id, int l)
{
if(id != PSELF)
{
    for(int i = 0; i < l; i++)
    {
    a[i].v = receive_from_live(id);
    }

        if(id != P1)
            for(int i = 0; i < l; i++)
                store_compare_view(P1,a[i].v);
        if(id != P2)
            for(int i = 0; i < l; i++)
                store_compare_view(P2,XOR(a[i].v,a[i].r));

    
}
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
/* #if PRE == 0 */
    communicate_live();
/* #endif */
}

};
