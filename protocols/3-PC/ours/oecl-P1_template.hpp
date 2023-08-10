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
if constexpr(id == P0)
{

#if OPT_SHARE == 1
        p1 = SET_ALL_ZERO(); 
#else
    #if PRE == 1 && SHARE_PREP == 1
        p1 = pre_receive_from_live(P0);
    #else
        p1 = receive_from_live(P0);
    #endif
#endif 
}
else if constexpr(id == P2)
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


static void complete_A2B_S1(Datatype out[])
{
    auto out_pointer = (Datatype(*)[2]) out;
    for(int i = 0; i < BITLENGTH; i++)
    {
        out_pointer[i][0] = receive_from_live(P2); // receive a + x_1 xor r0,2 from P2
        out_pointer[i][1] = SET_ALL_ZERO(); // set other share to 0
    }
}

static void prepare_A2B_S2(Datatype in[], Datatype out[])
{
    //convert share a + x1 to boolean
    Datatype temp[2][BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[0][j] = OP_SUB(SET_ALL_ZERO(), ((Datatype(*)[2]) in)[j][1]); // set both shares to -x1
            temp[1][j] = temp[0][j];
        }
    unorthogonalize_arithmetic(temp[0], (UINT_TYPE*) temp[0]);
    orthogonalize_boolean((UINT_TYPE*) temp[0], temp[0]);
    unorthogonalize_arithmetic(temp[1], (UINT_TYPE*) temp[1]);
    orthogonalize_boolean((UINT_TYPE*) temp[1], temp[1]);

    auto out_pointer = (Datatype(*)[BITLENGTH]) out;
    for(int i = 0; i < BITLENGTH; i++)
        for(int j = 0; j < 2; j++)
            out_pointer[i][j] = temp[j][i];
    
}



static void prepare_A2B(Datatype in[], Datatype out[])
{
    prepare_A2B_S2(in, out);
}

static void complete_A2B(Datatype out[])
{
    complete_A2B_S1(out);
}

};
