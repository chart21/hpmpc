#pragma once
#include "../sharemind/sharemind_base.hpp"
/* #include "../../../datatypes/k_bitset.hpp" */
/* #include "../../../datatypes/k_sint.hpp" */
//#include "oecl_base.hpp"
template <typename Datatype>
class OECL2_Share
{
Datatype p1;
Datatype p2;
bool optimized_sharing;
public:
OECL2_Share() {}
OECL2_Share(Datatype p1) : p1(p1) {}
OECL2_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}

OECL2_Share public_val(Datatype a)
{
    return a;
}

OECL2_Share Not() const
{
    return OECL2_Share(NOT(p1),p2);
}

template <typename func_add>
OECL2_Share Add(OECL2_Share b, func_add ADD) const
{
    return OECL2_Share(ADD(p1,b.p1),ADD(p2,b.p2));
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult(OECL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OECL2_Share c;
c.p2 = getRandomVal(P0); // P2 mask for P1
#if PRE == 1
c.p1 = ADD(pre_receive_from_live(P0), MULT(p1,b.1)); // P0_message + (a+rr) (b+rl)
#else
c.p1 = ADD(receive_from_live(P0), MULT(p1,b.p1)); // P0_message + (a+rr) (b+rl)
#endif

send_to_live(P1, ADD(c.p1,c.p2)); 
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
    p2 = getRandomVal(P0);
    /* p1 = getRandomVal(0); *1/ */
    send_to_live(P1, ADD(p1,p2));
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P0)
{
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
        p2 = pre_receive_from_live(P0);
#else
        p2 = receive_from_live(P0);
#endif
        p1 = SUB(SET_ALL_ZERO(), p2); // set share to - - (a + r0,1)
}
else if constexpr(id == P1)
{
p1 = receive_from_live(P1);
p2 = SET_ALL_ZERO();
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


static void prepare_A2B_S1(OECL2_Share in[], OECL2_Share out[])
{
    //convert share a + x1 to boolean
    Datatype temp[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp[i] = FUNC_ADD64(in[i].p1,in[i].p2); // set share to a + x_0
    }
    unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp);
    orthogonalize_boolean((UINT_TYPE*) temp, temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp[i];
        out[i].p2 = SET_ALL_ZERO();
    }
}

static void prepare_A2B_S2(OECL2_Share in[], OECL2_Share out[])
{
}

static void complete_A2B_S1(OECL2_Share out[])
{
}

static void complete_A2B_S2(OECL2_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
#if PRE == 1
        out[i].p1 = pre_receive_from_live(P0);
#else
        out[i].p1 = receive_from_live(P0);
#endif
        out[i].p2 = out[i].p1; // set both shares to -x0 xor r0,1
    }
        /* out[0].p2 = FUNC_NOT(out[0].p2);// change sign bit -> -x0 xor r0,1 to x0 xor r0,1 */
}

};


