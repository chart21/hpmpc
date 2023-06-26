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

// Receive sharing of ~XOR(a,b) locally
OEC_MAL_Share Xor(OEC_MAL_Share a, OEC_MAL_Share b)
{
   return OEC_MAL_Share(XOR(a.v,b.v),XOR(a.r,b.r));
}

#if FUNCTION_IDENTIFIER < 5
/* void prepare_and(OEC_MAL_Share a, OEC_MAL_Share b , OEC_MAL_Share &c) */
/* { */
/* /1* DATATYPE cr = XOR(getRandomVal(P013),getRandomVal(P123)); *1/ */
/* c.r = XOR(getRandomVal(P013),getRandomVal(P123)); */
/* DATATYPE r124 = getRandomVal(P013); */
/* /1* DATATYPE r234 = getRandomVal(P123); //used for veryfying m3' sent by P3 -> probably not needed -> for verification needed *1/ */
/* c.v = XOR( c.r,  XOR( XOR(AND(a.v,b.r), AND(b.v,a.r))  , r124)); */  
/* /1* DATATYPE m_2 = XOR(c.v, c.r); *1/ */
/* send_to_live(P2,c.v); */

/* /1* DATATYPE m3_prime = XOR( XOR(r234,cr) , AND( XOR(a.v,a.r) ,XOR(b.v,b.r))); //computationally wise more efficient to verify ab instead of m_3 prime *1/ */

/* /1* store_compare_view(P0,m3_prime); *1/ */
/* c.m = XOR(c.v,getRandomVal(P123)); */
/* /1* c.v = XOR( AND(      XOR(a.v,a.r) , XOR(b.v,b.r) ) , c.v); *1/ */
/* c.v = XOR(c.v, AND(a.v,b.v)); */

/* } */

/* void complete_and(OEC_MAL_Share &c) */
/* { */
/* DATATYPE m_3 = receive_from_live(P2); */
/* c.v = XOR(c.v, m_3); */

/* /1* c.m = XOR(c.m,m_3); *1/ */
/* /1* DATATYPE cm = XOR(c.m,m_3); *1/ */

/* #if PROTOCOL == 10 || PROTOCOL == 12 */
/* store_compare_view(P012,XOR(c.m,m_3)); */
/* #elif PROTOCOL == 11 */
/* store_compare_view(P0,XOR(c.m,m_3)); */
/* #endif */
/* store_compare_view(P0,XOR(c.v,c.r)); */
/* } */

void prepare_and(OEC_MAL_Share a, OEC_MAL_Share b , OEC_MAL_Share &c)
{
c.r = getRandomVal(P013);
DATATYPE r124 = getRandomVal(P013);
/* DATATYPE r234 = getRandomVal(P123); //used for veryfying m3' sent by P3 -> probably not needed -> for verification needed */
c.v = XOR( XOR(AND(a.v,b.r), AND(b.v,a.r))  , r124);  
/* DATATYPE m_2 = XOR(c.v, c.r); */
send_to_live(P2,c.v);

/* DATATYPE m3_prime = XOR( XOR(r234,cr) , AND( XOR(a.v,a.r) ,XOR(b.v,b.r))); //computationally wise more efficient to verify ab instead of m_3 prime */

/* store_compare_view(P0,m3_prime); */
/* c.m = ADD(c.v,getRandomVal(P123)); */
DATATYPE a1b1 = AND(a.v,b.v);
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P0,XOR(a1b1,getRandomVal(P123_2))); // compare a1b1 + r123_2 with P0
#endif
/* c.v = XOR( AND(      XOR(a.v,a.r) , XOR(b.v,b.r) ) , c.v); */
#if PROTOCOL == 11
c.m = XOR(c.v,getRandomVal(P123_2)); // m_2 + r234_2 store to compareview later
#endif

c.v = XOR( a1b1,c.v);

}

void complete_and(OEC_MAL_Share &c)
{
DATATYPE m_3 = receive_from_live(P2);
c.v = XOR(c.v, m_3);


#if PROTOCOL == 11
store_compare_view(P0,XOR(c.m,m_3)); // compare m_2 + m_3 + r234_2
store_compare_view(P0,XOR(c.v,getRandomVal(P123))); //compare ab + c1 + r234_1
#endif
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P012,XOR(c.v,getRandomVal(P123))); //compare ab + c1 + r234_1
#endif
}
#endif

#if FUNCTION_IDENTIFIER > 4
void prepare_mult(OEC_MAL_Share a, OEC_MAL_Share b , OEC_MAL_Share &c)
{
/* DATATYPE cr = XOR(getRandomVal(P013),getRandomVal(P123)); */
/* c.r = SUB(getRandomVal(P013),getRandomVal(P123)); */
c.r = getRandomVal(P013);
DATATYPE r124 = getRandomVal(P013);
/* DATATYPE r234 = getRandomVal(P123); //used for veryfying m3' sent by P3 -> probably not needed -> for verification needed */
c.v = ADD( ADD(MULT(a.v,b.r), MULT(b.v,a.r))  , r124);  
/* DATATYPE m_2 = XOR(c.v, c.r); */
send_to_live(P2,c.v);

/* DATATYPE m3_prime = XOR( XOR(r234,cr) , AND( XOR(a.v,a.r) ,XOR(b.v,b.r))); //computationally wise more efficient to verify ab instead of m_3 prime */

/* store_compare_view(P0,m3_prime); */
/* c.m = ADD(c.v,getRandomVal(P123)); */
DATATYPE a1b1 = MULT(a.v,b.v);
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P0,ADD(a1b1,getRandomVal(P123_2))); // compare a1b1 + r123_2 with P0
#endif
/* c.v = XOR( AND(      XOR(a.v,a.r) , XOR(b.v,b.r) ) , c.v); */
#if PROTOCOL == 11
c.m = ADD(c.v,getRandomVal(P123_2)); // m_2 + r234_2 store to compareview later
#endif

c.v = SUB( a1b1,c.v);

}

void complete_mult(OEC_MAL_Share &c)
{
DATATYPE m_3 = receive_from_live(P2);
c.v = SUB(c.v, m_3);

/* c.m = XOR(c.m,m_3); */
/* DATATYPE cm = XOR(c.m,m_3); */

#if PROTOCOL == 11
store_compare_view(P0,ADD(c.m,m_3)); // compare m_2 + m_3 + r234_2
store_compare_view(P0,ADD(c.v,getRandomVal(P123))); //compare ab + c1 + r234_1
#else
store_compare_view(P012,ADD(c.v,getRandomVal(P123))); //compare ab + c1 + r234_1
#endif
}
#endif


void prepare_reveal_to_all(OEC_MAL_Share a)
{
return;
}    


DATATYPE complete_Reveal(OEC_MAL_Share a)
{
#if PRE == 1
DATATYPE result = XOR(a.v, pre_receive_from_live(P3));
#else
DATATYPE result = XOR(a.v, receive_from_live(P3));
#endif
store_compare_view(P0123, result);
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
    DATATYPE x_1 = getRandomVal(P013);
    DATATYPE x_3 = getRandomVal(P123);
    DATATYPE x_2 = XOR(x_1,x_3);
    a[i].r = x_2;
    send_to_live(P0,XOR(a[i].v,x_3));
    send_to_live(P2,XOR(a[i].v,x_1));
    a[i].v = XOR(a[i].v,x_1);
}
}
else if(id == P0)
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
    a[i].r = getRandomVal(P123);
    
  
}
}
else if(id == P3)
{
for(int i = 0; i < l; i++)
{
    DATATYPE r013 = getRandomVal(P013);
    DATATYPE r123 = getRandomVal(P123);

    a[i].r = XOR(r013,r123);
    
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
        if(id != P0)
            for(int i = 0; i < l; i++)
                store_compare_view(P0,XOR(a[i].v,a[i].r));
        if(id != P2)
            for(int i = 0; i < l; i++)
                store_compare_view(P2,a[i].v);

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
