#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE OEC_MAL0_Share
template <typename Datatype>
class OEC_MAL0_Share
{
private:
    Datatype v;
    Datatype r;
public:
    
OEC_MAL0_Share() {}
OEC_MAL0_Share(Datatype v, Datatype r) : v(v), r(r) {}
OEC_MAL0_Share(Datatype v) : v(v) {}


    

OEC_MAL0_Share public_val(Datatype a)
{
    return OEC_MAL0_Share(a,SET_ALL_ZERO());
}

OEC_MAL0_Share Not() const
{
   return OEC_MAL0_Share(NOT(v),r);
}

template <typename func_add>
OEC_MAL0_Share Add(OEC_MAL0_Share b, func_add ADD) const
{
   return OEC_MAL0_Share(ADD(v,b.v),ADD(r,b.r));
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL0_Share prepare_mult(OEC_MAL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL0_Share c;
c.r = ADD(getRandomVal(P_013),getRandomVal(P_023)); // calculate c_1
/* Datatype r124 = getRandomVal(P_013); */
/* Datatype o1 = XOR( x1y1, r124); */
Datatype o1 = ADD(c.r, ADD(MULT(r, b.r), getRandomVal(P_013)));

#if PROTOCOL == 11
c.v = SUB(ADD( MULT(v,b.r), MULT(b.v,r)),c.r);
#else
c.v = ADD( MULT(v,b.r), MULT(b.v,r));
#endif

/* Datatype m3_flat = AND(a.v,b.v); */

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

return c;
}

template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL0_Share prepare_dot(const OEC_MAL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL0_Share c;
c.r = MULT(r, b.r); 
c.v = ADD( MULT(v,b.r), MULT(b.v,r));
return c;
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
Datatype cr = ADD(getRandomVal(P_013),getRandomVal(P_023)); // calculate c_1
/* Datatype r124 = getRandomVal(P_013); */
/* Datatype o1 = XOR( x1y1, r124); */
Datatype o1 = ADD(cr,ADD( r, getRandomVal(P_013)));

#if PROTOCOL == 11
v = SUB(v,cr);
#endif
r = cr;
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
void complete_mult(func_add ADD, func_sub SUB)
{
#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
Datatype o_4 = pre_receive_from_live(P_3);
#else
Datatype o_4 = receive_from_live(P_3);
#endif
#elif PROTOCOL == 11
Datatype m_2XORm_3 = receive_from_live(P_2);
store_compare_view(P_1, m_2XORm_3); // Verify if P_2 sent correct message m_2 XOR m_3
store_compare_view(P_3, SUB(m_2XORm_3,v)); // x1 y1 - x1 y3 - x 3 y1 - r234 should remain
v = receive_from_live(P_2); // receive ab + c1 + r_234_1 from P_2 (P_3 in paper), need to convert to ab+ r234_1 (maybe not? and only for verify?)
store_compare_view(P_1, v); // Verify if P_2 sent correct message of ab
v = SUB(v,r); // convert to ab + r234_1 (maybe not needed)
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
/* Datatype m3_prime = receive_from_live(P_2); */
v = ADD(v, o_4);

/* c.m = XOR(c.m, o_4); */
Datatype m3_prime = receive_from_live(P_2);
v = SUB(m3_prime,v);
store_compare_view(P_012,ADD(v, r)); // compare ab + r_234_1 + c_1 with P_2,P_3
store_compare_view(P_1, m3_prime); // compare m_3 prime with P_2
#endif
}


void prepare_reveal_to_all()
{
send_to_live(P_1, r);
send_to_live(P_2, r);

send_to_live(P_3, v);
}    

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
#if PRE == 1
Datatype result = SUB(v, pre_receive_from_live(P_3));
/* send_to_live(P_3, result); */
#else
Datatype result = SUB(v, receive_from_live(P_3));
#endif
store_compare_view(P_0123, result); 
// Problem, P_3 sends all the values -> send in circle


return result;
}




template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    v = get_input_live();
    Datatype x_1 = getRandomVal(P_013);
    Datatype x_2 = getRandomVal(P_023);
    r = ADD(x_1, x_2);
    
    send_to_live(P_1,ADD(v, r));
    send_to_live(P_2,ADD( v, r));
     
}
else if constexpr(id == P_1)
{
    r = getRandomVal(P_013); // x_0
}
else if constexpr(id == P_2)
{
    r = getRandomVal(P_023); // x_0
}
else if constexpr(id == P_3)
{
    Datatype x_1 = getRandomVal(P_013);
    Datatype x_2 = getRandomVal(P_023);
    r = ADD(x_1, x_2);
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id != PSELF)
{

    #if PRE == 1
    if(id == P_3)
        v = pre_receive_from_live(id);
    else
        v = receive_from_live(id);
    #else
    v = receive_from_live(id);
    #endif


    if constexpr(id != P_1)
            store_compare_view(P_1,v);
    if constexpr(id != P_2)
            store_compare_view(P_2,v);


    v = SUB(v,r); // convert locally to a + u
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
/* #if PRE == 0 */
    communicate_live();
/* #endif */
}

};
