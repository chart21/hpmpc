#pragma once
#include "../sharemind/sharemind_base.hpp"
/* #include "../../../datatypes/k_bitset.hpp" */
/* #include "../../../datatypes/k_sint.hpp" */
//#include "oecl_base.hpp"
template <typename Datatype>
class OECL2_Share
{
Datatype p1;
bool optimized_sharing;
public:
OECL2_Share() {}
OECL2_Share(Datatype p1) : p1(p1) {}

OECL2_Share public_val(Datatype a)
{
    return a;
}

OECL2_Share Not() const
{
    return OECL2_Share(NOT(p1));
}

template <typename func_add>
OECL2_Share Add(OECL2_Share b, func_add ADD) const
{
    return ADD(p1,b.p1);
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult(OECL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype ap1 = getRandomVal(P0); // P2 mask for P1
OECL2_Share c;
#if PRE == 1
c.p1 = ADD(pre_receive_from_live(P0), MULT(p1,b.1)); // P0_message + (a+rr) (b+rl)
#else
c.p1 = ADD(receive_from_live(P0), MULT(p1,b.p1)); // P0_message + (a+rr) (b+rl)
#endif

send_to_live(P1, ADD(ap1,c.p1)); 
return c;
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
p1 = SUB(p1, receive_from_live(P1)); 
}


void prepare_reveal_to_all()
{
send_to_live(P0, p1);
}

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
return SUB(p1, pre_receive_from_live(P0));
#else
return SUB(p1, receive_from_live(P0));
#endif
}


    template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P2)
{
    p1 = get_input_live();     
    /* p1 = getRandomVal(0); *1/ */
    send_to_live(P1, ADD(getRandomVal(P0),p1));
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P0)
{
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
        p1 = pre_receive_from_live(P0);
#else
        p1 = receive_from_live(P0);
#endif
}
else if constexpr(id == P1)
{
p1 = receive_from_live(P1);
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


//higher level functions


static void A2B_S1(Datatype in[], Datatype out[])
{
    //convert share a + x1 to boolean
    unorthogonalize_arithmetic(in, (UINT_TYPE*) out);
    orthogonalize_boolean((UINT_TYPE*) out, out);
    for(int i = 0; i < BITLENGTH; i++)
    {
        send_to_live(P1, XOR(out[i],getRandomVal(P0))); // send all bits a + x_1 XOR r_0,2 to P1
    }
}

static void A2B_S2(Datatype out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i] = SET_ALL_ZERO();
    }
}

static void prepare_A2B(Datatype in[], Datatype out[])
{
    A2B_S1(in, out);
    A2B_S2(out);
}

static void complete_A2B(Datatype out[])
{
}

};

