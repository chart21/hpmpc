#pragma once
#include "oec-mal_base.hpp"
class OEC_MAL2
{
bool optimized_sharing;
public:
OEC_MAL2(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

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
    //1* get Random val
    c.r = getRandomVal(P_023);
   /* c.r = ADD(getRandomVal(P_023), getRandomVal(P_123)); */
   /* DATATYPE r234 = getRandomVal(P_123); */
   /* DATATYPE r234_1 = */
   /*     getRandomVal(P_123); // Probably sufficient to only generate with P_3 -> */
   DATATYPE r234_2 =
       getRandomVal(P_123_2); // Probably sufficient to only generate with P_3 ->
                           // Probably not because of verification
/* c.r = getRandomVal(P_3); */
#if PROTOCOL == 12
#if PRE == 1
   DATATYPE o1 = pre_receive_from_live(P_3);
#else
   DATATYPE o1 = receive_from_live(P_3);
#endif
   store_compare_view(P_0, o1);
#else
   DATATYPE o1 = receive_from_live(P_0);
   store_compare_view(P_3, o1);
#endif
   c.v = ADD(SUB(MULT(a.v, b.r),o1), MULT(b.v, a.r));
   send_to_live(P_1, c.v);


   /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */
   DATATYPE a1b1 = MULT(a.v, b.v);

#if PROTOCOL == 10 || PROTOCOL == 12
   /* DATATYPE m3_prime = XOR(XOR(r234, c.r), c.v); */
   /* DATATYPE m3_prime = XOR(XOR(r234, c.r), AND(XOR(a.v, a.r), XOR(b.v, b.r))); */
   /* send_to_live(P_0, ADD(r234 ,ADD(c.v,c.r))); */
   send_to_live(P_0, ADD(a1b1,r234_2));
#endif
#if PROTOCOL == 11
c.m = ADD(c.v, r234_2); // store m_2 + m_3 + r_234_2 to send P_0 later
#endif
   c.v = SUB(a1b1, c.v);
   /* c.m = ADD(c.m, r234); */

}

template <typename func_add, typename func_sub>
void complete_mult(OEC_MAL_Share &c, func_add ADD, func_sub SUB)
{

DATATYPE m2 = receive_from_live(P_1);
c.v = SUB(c.v, m2);
/* c.m = XOR(c.m, m2); */
/* DATATYPE cm = XOR(c.m, m2); */
#if PROTOCOL == 11
send_to_live(P_0, ADD(c.m, m2)); // let P_0 verify m_2 XOR m_3, obtain m_2 + m_3 + r_234_2
send_to_live(P_0, ADD(c.v,getRandomVal(P_123))); // let P_0 obtain ab + c1 + r234_1
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P_012, ADD(getRandomVal(P_123), c.v)); // compare ab + c1 + r234_1
#endif
/* store_compare_view(P_0, c.v); */
}


void prepare_reveal_to_all(OEC_MAL_Share a)
{
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
    
    DATATYPE x_0 = getRandomVal(P_023);
    DATATYPE u = getRandomVal(P_123);
    a[i].r = x_0; //  = x_2, x_1 = 0
    a[i].v = ADD(get_input_live(),x_0);
    send_to_live(P_0,ADD(a[i].v,u));
    send_to_live(P_1,ADD(a[i].v,u));
}
}
else if(id == P_0)
{
for(int i = 0; i < l; i++)
{
    a[i].r = getRandomVal(P_023);
    a[i].v = SET_ALL_ZERO();
    // u = 0
}
}
else if(id == P_1)
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
    a[i].r = getRandomVal(P_023); //x2
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
    if(id != P_1)
            store_compare_view(P_1,val);
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
