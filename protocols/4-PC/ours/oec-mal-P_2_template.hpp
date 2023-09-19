#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL2_Share
{
    private:
    Datatype v;
    Datatype r;
#if PROTOCOL == 11
    Datatype m;
#endif

public:
    
OEC_MAL2_Share() {}
OEC_MAL2_Share(Datatype v, Datatype r) : v(v), r(r) {}
OEC_MAL2_Share(Datatype v) : v(v) {}

OEC_MAL2_Share public_val(Datatype a)
{
    return OEC_MAL2_Share(a,SET_ALL_ZERO());
}

OEC_MAL2_Share Not() const
{
   return OEC_MAL2_Share(NOT(v),r);
}

template <typename func_add>
OEC_MAL2_Share Add(OEC_MAL2_Share b, func_add ADD) const
{
   return OEC_MAL2_Share(ADD(v,b.v),ADD(r,b.r));
}


template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_mult(OEC_MAL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    //1* get Random val
    OEC_MAL2_Share c;
    c.r = getRandomVal(P_023);
   /* c.r = ADD(getRandomVal(P_023), getRandomVal(P_123)); */
   /* Datatype r234 = getRandomVal(P_123); */
   /* Datatype r234_1 = */
   /*     getRandomVal(P_123); // Probably sufficient to only generate with P_3 -> */
   Datatype r234_2 =
       getRandomVal(P_123_2); // Probably sufficient to only generate with P_3 ->
                           // Probably not because of verification
/* c.r = getRandomVal(P_3); */
#if PROTOCOL == 12
#if PRE == 1
   Datatype o1 = pre_receive_from_live(P_3);
#else
   Datatype o1 = receive_from_live(P_3);
#endif
   store_compare_view(P_0, o1);
#else
   Datatype o1 = receive_from_live(P_0);
   store_compare_view(P_3, o1);
#endif
   c.v = ADD(SUB(MULT(v, b.r),o1), MULT(b.v, r));
   send_to_live(P_1, c.v);


   /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */
   Datatype a1b1 = MULT(v, b.v);

#if PROTOCOL == 10 || PROTOCOL == 12
   /* Datatype m3_prime = XOR(XOR(r234, c.r), c.v); */
   /* Datatype m3_prime = XOR(XOR(r234, c.r), AND(XOR(a.v, a.r), XOR(b.v, b.r))); */
   /* send_to_live(P_0, ADD(r234 ,ADD(c.v,c.r))); */
   send_to_live(P_0, ADD(a1b1,r234_2));
#endif
#if PROTOCOL == 11
c.m = ADD(c.v, r234_2); // store m_2 + m_3 + r_234_2 to send P_0 later
#endif
   c.v = SUB(a1b1, c.v);
   /* c.m = ADD(c.m, r234); */
return c;
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
Datatype m2 = receive_from_live(P_1);
v = SUB(v, m2);
/* c.m = XOR(c.m, m2); */
/* Datatype cm = XOR(c.m, m2); */
#if PROTOCOL == 11
send_to_live(P_0, ADD(m, m2)); // let P_0 verify m_2 XOR m_3, obtain m_2 + m_3 + r_234_2
send_to_live(P_0, ADD(v,getRandomVal(P_123))); // let P_0 obtain ab + c1 + r234_1
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P_012, ADD(getRandomVal(P_123), v)); // compare ab + c1 + r234_1
#endif
/* store_compare_view(P_0, c.v); */
}


void prepare_reveal_to_all()
{
}

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
Datatype r0 = receive_from_live(P_0);
Datatype result = SUB(v, r0);
store_compare_view(P_123, r0);
store_compare_view(P_0123, result);
return result;
}


template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    Datatype x_0 = getRandomVal(P_023);
    Datatype u = getRandomVal(P_123);
    r = x_0; //  = x_2, x_1 = 0
    v = ADD(get_input_live(),x_0);
    send_to_live(P_0,ADD(v,u));
    send_to_live(P_1,ADD(v,u));
}
else if constexpr(id == P_0)
{
    r = getRandomVal(P_023);
    v = SET_ALL_ZERO();
    // u = 0
}
else if constexpr(id == P_1)
{
    r = SET_ALL_ZERO();
    v = getRandomVal(P_123); //u
}
else if constexpr(id == P_3)
{
    r = getRandomVal(P_023); //x2
    v = getRandomVal(P_123); //u
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id != PSELF)
{
    
    #if PRE == 1
        Datatype val;
        if constexpr(id == P_3)
            val = pre_receive_from_live(P_3);
        else
            val = receive_from_live(id);
    #else
    Datatype val = receive_from_live(id);
    #endif
    if constexpr(id != P_0)
            store_compare_view(P_0,val);
    if constexpr(id != P_1)
            store_compare_view(P_1,val);
    v = SUB(val,v); // convert locally to a + x_0
    }



}



static void send()
{
    send_live();
}

static void receive()
{
    receive_live();
}

static void communicate()
{
    communicate_live();
}

};
