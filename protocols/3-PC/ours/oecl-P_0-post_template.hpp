#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL0_Share 
{
private:
    Datatype p1;
    Datatype p2;
    public:
    //static constexpr int VALS_PER_SHARE = 2;

    OECL0_Share() {}
    OECL0_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}
    OECL0_Share(Datatype p1) : p1(p1) {}


    

OECL0_Share public_val(Datatype a)
{
    return OECL0_Share();
}

OECL0_Share Not() const
{
    return OECL0_Share();
}

template <typename func_add>
OECL0_Share Add(OECL0_Share b, func_add ADD) const
{
    return OECL0_Share();
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_dot_add(OECL0_Share a, OECL0_Share b , OECL0_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
}

template <typename func_add, typename func_sub, typename func_mul>
OECL0_Share prepare_dot(const OECL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OECL0_Share();
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
}
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}

    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}


template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_mult(OECL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OECL0_Share();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}

void prepare_reveal_to_all()
{
}    



template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
return SUB(receive_from_live(P_2),retrieve_output_share());
}

template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
#if OPT_SHARE == 1 && SHARE_PREP == 0
if constexpr(id == P_0)
{
    p1 = get_input_live();
    p2 = getRandomVal(P_1);
    send_to_live(P_2, XOR(p1,p2));
}
#endif
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
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
#if PRE == 0
    communicate_live();
#endif
}

static void prepare_A2B_S1(OECL0_Share in[], OECL0_Share out[])
{
}


static void prepare_A2B_S2(OECL0_Share in[], OECL0_Share out[])
{
}

static void complete_A2B_S1(OECL0_Share out[])
{

}
static void complete_A2B_S2(OECL0_Share out[])
{

}

void prepare_bit_injection_S1(OECL0_Share out[])
{
}

void prepare_bit_injection_S2(OECL0_Share out[])
{
}

static void complete_bit_injection_S1(OECL0_Share out[])
{
    
}

static void complete_bit_injection_S2(OECL0_Share out[])
{


}



};


