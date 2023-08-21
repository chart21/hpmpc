#pragma once
#include "oecl_base.hpp"
template <typename Datatype>
class OECL1_Share
{
Datatype p1;
Datatype p2;
public:
OECL1_Share() {}
OECL1_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}
OECL1_Share(Datatype p1) : p1(p1) {}

OECL1_Share public_val(Datatype a)
{
    return OECL1_Share(a,SET_ALL_ZERO());
}

OECL1_Share Not() const
{
   return OECL1_Share(NOT(p1),p2);
}

template <typename func_add>
OECL1_Share Add(OECL1_Share b, func_add ADD) const
{
   return OECL1_Share(ADD(p1,b.p1),ADD(p2,b.p2));
}

    template <typename func_add, typename func_sub, typename func_mul>
void prepare_dot(OECL1_Share a, OECL1_Share b , OECL1_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c.p1 = ADD(c.p1, ADD(MULT(a.p1,b.p2), MULT(b.p1,a.p2)));
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(OECL1_Share &c, func_add ADD, func_sub SUB)
{
    c.p1 = ADD(getRandomVal(P0), c.p1);
    c.p2 = getRandomVal(P2);
    send_to_live(P2,SUB(c.p1,c.p2));
}



template <typename func_add, typename func_sub, typename func_mul>
    OECL1_Share prepare_mult(OECL1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OECL1_Share c;
c.p1 = ADD(getRandomVal(P0), ADD(MULT(p1,b.p2), MULT(b.p1,p2))); //remove P1_mask, then (a+ra)rl + (b+rb)rr 
c.p2 = getRandomVal(P0); //generate P1_2 mask
send_to_live(P2,SUB(c.p1,c.p2)); 
return c;
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
p1 = SUB(receive_from_live(P2),p1);
}

void prepare_reveal_to_all()
{
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




template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P0)
{
        p2 = getRandomVal(P0);
        p1 = SUB(SET_ALL_ZERO(), p2); // set p1 to - r0,1
}
else if constexpr(id == P1)
{
    p1 = get_input_live();
    p2 = getRandomVal(P0);
    send_to_live(P2,ADD(p1,p2));
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    
#if OPT_SHARE == 0
if constexpr(id == P0)
{
    #if PRE == 1 && SHARE_PREP == 1
        p1 = pre_receive_from_live(P0);
    #else
        p1 = receive_from_live(P0);
    #endif
}
#endif 
if constexpr(id == P2)
{
p1 = receive_from_live(P2);
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


static void prepare_A2B_S1(OECL1_Share in[], OECL1_Share out[])
{
    Datatype temp_p1[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp_p1[i] = FUNC_ADD64(in[i].p1,in[i].p2) ; // set first share to a+x_0
    }
    unorthogonalize_arithmetic(temp_p1, (UINT_TYPE*) temp_p1);
    orthogonalize_boolean((UINT_TYPE*) temp_p1, temp_p1);
    
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp_p1[i];
        out[i].p2 = SET_ALL_ZERO(); // set other share to 0
    }

}

static void complete_A2B_S1(OECL1_Share out[])
{
}

static void prepare_A2B_S2(const OECL1_Share in[], OECL1_Share out[])
{

    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = getRandomVal(P0); 
        out[i].p2 = out[i].p1; // set both shares to r0,1
    } 
}

static void complete_A2B_S2(OECL1_Share out[])
{
}

void prepare_bit_injection_S1(OECL1_Share out[])
{
    DATATYPE temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = FUNC_XOR(p1,p2);
    unorthogonalize_boolean(temp,(UINT_TYPE*)temp);
    orthogonalize_arithmetic((UINT_TYPE*) temp,  temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp[i];// set share to b xor x_0
        out[i].p2 = SET_ALL_ZERO(); // set other share to 0
    }
}

void prepare_bit_injection_S2(OECL1_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p2 = getRandomVal(P0); // set second share to r0,1
        out[i].p1 = FUNC_SUB64(SET_ALL_ZERO(), out[i].p2) ; // set first share -r0,1
    }
}

static void complete_bit_injection_S1(OECL1_Share out[])
{
    
}

static void complete_bit_injection_S2(OECL1_Share out[])
{


}


};
