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

template <typename func_add>
OEC_MAL_Share Add(OEC_MAL_Share a, OEC_MAL_Share b, func_add ADD)
{
    return OEC_MAL_Share(ADD(a.v,b.v),ADD(a.r,b.r));
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(OEC_MAL_Share a, OEC_MAL_Share b, OEC_MAL_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c.r = ADD(getRandomVal(P_013),getRandomVal(P_023)); // calculate c_1
/* DATATYPE r124 = getRandomVal(P_013); */
/* DATATYPE o1 = XOR( x1y1, r124); */
DATATYPE o1 = ADD(c.r, ADD(MULT(a.r, b.r), getRandomVal(P_013)));

#if PROTOCOL == 11
c.v = SUB(ADD( MULT(a.v,b.r), MULT(b.v,a.r)),c.r);
#else
c.v = ADD( MULT(a.v,b.r), MULT(b.v,a.r));
#endif

/* DATATYPE m3_flat = AND(a.v,b.v); */

/* c.m = XOR(x1y1, XOR( XOR(AND(a.v,b.v), AND( XOR(a.v, a.r), XOR(b.v, b.r))), c.r)); */
#if PROTOCOL == 12
store_compare_view(P_2,o1);
#else
    #if PRE == 1
        pre_send_to_live(P_2, o1);
    #else
        send_to_live(P_2, o1);
    #endif
#endif


}

template <typename func_add, typename func_sub>
void complete_mult(OEC_MAL_Share &c, func_add ADD, func_sub SUB)
{
#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
DATATYPE o_4 = pre_receive_from_live(P_3);
#else
DATATYPE o_4 = receive_from_live(P_3);
#endif
#elif PROTOCOL == 11
DATATYPE m_2XORm_3 = receive_from_live(P_2);
store_compare_view(P_1, m_2XORm_3); // Verify if P_2 sent correct message m_2 XOR m_3
store_compare_view(P_3, SUB(m_2XORm_3,c.v)); // x1 y1 - x1 y3 - x 3 y1 - r234 should remain
c.v = receive_from_live(P_2); // receive ab + c1 + r_234_1 from P_2 (P_3 in paper), need to convert to ab+ r234_1 (maybe not? and only for verify?)
store_compare_view(P_1, c.v); // Verify if P_2 sent correct message of ab
c.v = SUB(c.v,c.r); // convert to ab + r234_1 (maybe not needed)
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
/* DATATYPE m3_prime = receive_from_live(P_2); */
c.v = ADD(c.v, o_4);

/* c.m = XOR(c.m, o_4); */
DATATYPE m3_prime = receive_from_live(P_2);
c.v = SUB(m3_prime,c.v);
store_compare_view(P_012,ADD(c.v, c.r)); // compare ab + r_234_1 + c_1 with P_2,P_3
store_compare_view(P_1, m3_prime); // compare m_3 prime with P_2
#endif
}


void prepare_reveal_to_all(OEC_MAL_Share a)
{
send_to_live(P_1, a.r);
send_to_live(P_2, a.r);

send_to_live(P_3, a.v);
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(OEC_MAL_Share a, func_add ADD, func_sub SUB)
{
#if PRE == 1
DATATYPE result = SUB(a.v, pre_receive_from_live(P_3));
/* send_to_live(P_3, result); */
#else
DATATYPE result = SUB(a.v, receive_from_live(P_3));
#endif
store_compare_view(P_0123, result); 
// Problem, P_3 sends all the values -> send in circle


return result;
}


OEC_MAL_Share* alloc_Share(int l)
{
    return new OEC_MAL_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(OEC_MAL_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    a[i].v = get_input_live();
    DATATYPE x_1 = getRandomVal(P_013);
    DATATYPE x_2 = getRandomVal(P_023);
    a[i].r = ADD(x_1, x_2);
    
    send_to_live(P_1,ADD(a[i].v, a[i].r));
    send_to_live(P_2,ADD( a[i].v, a[i].r));
    } 
}
else if(id == P_1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r = getRandomVal(P_013); // x_0
    }
}
else if(id == P_2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].r = getRandomVal(P_023); // x_0
    }
}
else if(id == P_3)
{
    for(int i = 0; i < l; i++)
    {
    DATATYPE x_1 = getRandomVal(P_013);
    DATATYPE x_2 = getRandomVal(P_023);
    a[i].r = ADD(x_1, x_2);
    }
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(OEC_MAL_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id != PSELF)
{

    for(int i = 0; i < l; i++)
    {
            #if PRE == 1
        if(id == P_3)
            a[i].v = pre_receive_from_live(id);
        else
            a[i].v = receive_from_live(id);
    #else
    a[i].v = receive_from_live(id);
    #endif
    }


    if(id != P_1)
        for(int i = 0; i < l; i++)
            store_compare_view(P_1,a[i].v);
    if(id != P_2)
        for(int i = 0; i < l; i++)
            store_compare_view(P_2,a[i].v);


    for(int i = 0; i < l; i++)
    {
    a[i].v = SUB(a[i].v,a[i].r); // convert locally to a + u
    }



    
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
