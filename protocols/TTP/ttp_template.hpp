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

TTP_Share public_val(Datatype a)
{
    return TTP_Share(a);
}

Datatype get_p1()
{
    return p1;
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
TTP_Share prepare_dot(const TTP_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_Share(MULT(p1,b.p1));
}

template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{

}
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
DATATYPE dummy = getRandomVal(0);
p1 = ADD(TRUNC(SUB(p1,dummy)), TRUNC(dummy));
}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
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
#if PARTY == 2 && PROTOCOL != 13
    for(int t = 0; t < num_players-1; t++) 
        send_to_live(t, p1);
#endif
}


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
#if PARTY != 2 && PROTOCOL != 13
    Datatype result = receive_from_live(P_2);
#else
    Datatype result = p1;
#endif
return result;
}



template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
#if PARTY != 2 && PROTOCOL != 13
if constexpr(id == PSELF)
{
            Datatype tmp = get_input_live();
        send_to_live(P_2, tmp);
        
}
#endif
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
#if PARTY == 2 || PROTOCOL == 13
if constexpr(id == P_2)
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
#if PROTOCOL != 13
    send_live();
#endif
}

static void receive()
{
#if PROTOCOL != 13
    receive_live();
#endif
}

static void communicate()
{
#if PROTOCOL != 13
    communicate_live();
#endif
}

static void prepare_A2B_S1(int k, TTP_Share in[], TTP_Share out[])
{
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            /* temp[j] = in[j].p1; */
            temp[j] = in[j].p1;
            /* in[j].p1 = FUNC_SUB32(in[j].p1,temp[j]); */
        }
    /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
    /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp, temp2);
    orthogonalize_boolean(temp2, temp);

    for(int i = 0; i < k; i++)
    {
        out[i].p1 = temp[i];
    } 
}


static void prepare_A2B_S2(int k, TTP_Share in[], TTP_Share out[])
{
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            /* temp[j] = SET_ALL_ZERO(); */
            temp[j] = in[j].p1;
        }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp, temp2);
    orthogonalize_boolean(temp2, temp);

    for(int i = 0; i < k; i++)
    {
        out[i].p1 = temp[i];
    } 
}

static void complete_A2B_S1(int k, TTP_Share out[])
{

}
static void complete_A2B_S2(int k, TTP_Share out[])
{

}

void prepare_bit_injection_S1(TTP_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = SET_ALL_ZERO(); // set first summand to zero
    }
}

void prepare_bit_injection_S2(TTP_Share out[])
{
    DATATYPE temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = p1;
    unorthogonalize_boolean(temp,(UINT_TYPE*)temp);
    orthogonalize_arithmetic((UINT_TYPE*) temp,  temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp[i]; // set second summand to the msb
    }
}

static void complete_bit_injection_S1(TTP_Share out[])
{
    
}

static void complete_bit_injection_S2(TTP_Share out[])
{


}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_dot3(TTP_Share b, TTP_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_Share(MULT(MULT(p1,b.p1),c.p1));
}


template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_mult3(TTP_Share b, TTP_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_Share(MULT(MULT(p1,b.p1),c.p1));
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_dot4(TTP_Share b, TTP_Share c, TTP_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_Share(MULT(MULT(MULT(p1,b.p1),c.p1),d.p1));
}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_mult4(TTP_Share b, TTP_Share c, TTP_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_Share(MULT(MULT(MULT(p1,b.p1),c.p1),d.p1));
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){}




};

