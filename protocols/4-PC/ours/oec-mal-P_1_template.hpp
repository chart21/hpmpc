#pragma once
#include "oec-mal_base.hpp"
class OEC_MAL1
{
bool optimized_sharing;
public:
OEC_MAL1(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

OEC_MAL_Share public_val(DATATYPE a)
{
    return OEC_MAL_Share(a,SET_ALL_ZERO());
}

OEC_MAL_Share Not(OEC_MAL_Share a)
{
   a.v = NOT(a.v);
   return a;
}

template <typename func_add>
OEC_MAL_Share Add(OEC_MAL_Share a, OEC_MAL_Share b, func_add ADD)
{
    return OEC_MAL_Share(ADD(a.v,b.v),ADD(a.r,b.r));
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(OEC_MAL_Share a, OEC_MAL_Share b , OEC_MAL_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
/* DATATYPE cr = XOR(getRandomVal(P_013),getRandomVal(P_123)); */
/* c.r = SUB(getRandomVal(P_013),getRandomVal(P_123)); */
c.r = getRandomVal(P_013);
DATATYPE r124 = getRandomVal(P_013);
/* DATATYPE r234 = getRandomVal(P_123); //used for veryfying m3' sent by P_3 -> probably not needed -> for verification needed */
c.v = ADD( ADD(MULT(a.v,b.r), MULT(b.v,a.r))  , r124);  
/* DATATYPE m_2 = XOR(c.v, c.r); */
send_to_live(P_2,c.v);

/* DATATYPE m3_prime = XOR( XOR(r234,cr) , AND( XOR(a.v,a.r) ,XOR(b.v,b.r))); //computationally wise more efficient to verify ab instead of m_3 prime */

/* store_compare_view(P_0,m3_prime); */
/* c.m = ADD(c.v,getRandomVal(P_123)); */
DATATYPE a1b1 = MULT(a.v,b.v);
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P_0,ADD(a1b1,getRandomVal(P_123_2))); // compare a1b1 + r123_2 with P_0
#endif
/* c.v = XOR( AND(      XOR(a.v,a.r) , XOR(b.v,b.r) ) , c.v); */
#if PROTOCOL == 11
c.m = ADD(c.v,getRandomVal(P_123_2)); // m_2 + r234_2 store to compareview later
#endif

c.v = SUB( a1b1,c.v);

}

template <typename func_add, typename func_sub>
void complete_mult(OEC_MAL_Share &c, func_add ADD, func_sub SUB)
{
DATATYPE m_3 = receive_from_live(P_2);
c.v = SUB(c.v, m_3);

/* c.m = XOR(c.m,m_3); */
/* DATATYPE cm = XOR(c.m,m_3); */

#if PROTOCOL == 11
store_compare_view(P_0,ADD(c.m,m_3)); // compare m_2 + m_3 + r234_2
store_compare_view(P_0,ADD(c.v,getRandomVal(P_123))); //compare ab + c1 + r234_1
#else
store_compare_view(P_012,ADD(c.v,getRandomVal(P_123))); //compare ab + c1 + r234_1
#endif
}


void prepare_reveal_to_all(OEC_MAL_Share a)
{
return;
}    

template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(OEC_MAL_Share a, func_add ADD, func_sub SUB)
{
DATATYPE r = receive_from_live(P_0);
DATATYPE result = SUB(a.v, r);
store_compare_view(P_123, r);
store_compare_view(P_0123, result);
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
    
    DATATYPE x_0 = getRandomVal(P_013);
    DATATYPE u = getRandomVal(P_123);
    a[i].r = x_0; //  = x_1, x_2 = 0
    a[i].v = ADD(get_input_live(),x_0);
    send_to_live(P_0,ADD(a[i].v,u));
    send_to_live(P_2,ADD(a[i].v,u));
}
}
else if(id == P_0)
{
for(int i = 0; i < l; i++)
{
    a[i].r = getRandomVal(P_013);
    a[i].v = SET_ALL_ZERO();
    // u = 0
}
}
else if(id == P_2)
{
for(int i = 0; i < l; i++)
{
    a[i].r = SET_ALL_ZERO();
    a[i].v = getRandomVal(P_123); //u
    
  
}
}
else if(id == P_3)
{
for(int i = 0; i < l; i++)
{
    a[i].r = getRandomVal(P_013); //x1
    a[i].v = getRandomVal(P_123); //u

    
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
        DATATYPE val;
        if(id == P_3)
            val = pre_receive_from_live(P_3);
        else
            val = receive_from_live(id);
    #else
    DATATYPE val = receive_from_live(id);
    #endif

    if(id != P_0)
            store_compare_view(P_0,val);
    if(id != P_2)
            store_compare_view(P_2,val);
    a[i].v = SUB(val,a[i].v); // convert locally to a + x_0
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
    communicate_live();
}

};
