#pragma once
#include "../generic_share.hpp"
#include "inttypes.h"
template <typename Datatype>
class TTP_Share
{
private:
    Datatype p1;
public:
TTP_Share() {}
TTP_Share(Datatype a) {p1 = a;}

void public_val(Datatype a)
{
    p1 = a;
}

TTP_Share Not() const
{
   return TTP_Share(NOT(p1));
}

template <typename func_add>
TTP_Share Add(TTP_Share b, func_add ADD) const
{
    return TTP_Share(ADD(p1, b.p1));
}


template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_mult(TTP_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_Share(MULT(p1,b.p1));
}
template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}


void prepare_reveal_to_all()
{
#if PARTY == 2
    for(int t = 0; t < num_players-1; t++) 
        send_to_live(t, p1);
#endif
}


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
#if PARTY != 2
    Datatype result = receive_from_live(P2);
#else
    Datatype result = p1;
#endif
return result;
}



template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
#if PARTY != 2
if constexpr(id == PSELF)
{
            Datatype tmp = get_input_live();
        send_to_live(P2, tmp);
        
}
#endif
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
#if PARTY == 2
if constexpr(id == P2)
{
        p1 = get_input_live();
}
else
{
    p1 = receive_from_live(id);
}
#endif
}





static void finalize()
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
    communicate_live();
}

static void prepare_A2B_S1(TTP_Share in[], TTP_Share out[])
{
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            /* temp[j] = in[j].p1; */
            temp[j] = getRandomVal(PNEXT);
            in[j].p1 -= temp[j];
        }
    unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp);
    orthogonalize_boolean((UINT_TYPE*) temp, temp);

    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp[i];
    } 
}


static void prepare_A2B_S2(TTP_Share in[], TTP_Share out[])
{
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            /* temp[j] = SET_ALL_ZERO(); */
            temp[j] = in[j].p1;
        }
    unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp);
    orthogonalize_boolean((UINT_TYPE*) temp, temp);

    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp[i];
    } 
}

static void complete_A2B_S1(TTP_Share out[])
{

}
static void complete_A2B_S2(TTP_Share out[])
{

}

};

