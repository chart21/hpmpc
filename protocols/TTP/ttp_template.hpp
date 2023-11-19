#pragma once
#include "../generic_share.hpp"
#include "inttypes.h"
template <typename Datatype>
class TTP_Share
{
private:
    Datatype p1;
#if SIMULATE_MPC_FUNCTIONS == 1
    Datatype p2;
#endif
public:
TTP_Share() {}
TTP_Share(Datatype a) {p1 = a;}
#if SIMULATE_MPC_FUNCTIONS == 1
TTP_Share(Datatype a, Datatype b) {p1 = a; p2 = b;}
#endif

static TTP_Share public_val(Datatype a)
{
#if SIMULATE_MPC_FUNCTIONS == 1
    return TTP_Share(a, SET_ALL_ZERO());
#else
    return TTP_Share(a);
#endif
}

Datatype get_p1()
{
    return p1;
}

TTP_Share Not() const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    return TTP_Share(NOT(p1), p2);
#else
   return TTP_Share(NOT(p1));
#endif
}

    template <typename func_mul, typename func_trunc>
TTP_Share mult_public_fixed(const Datatype b, func_mul MULT, func_trunc TRUNC) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    return TTP_Share(TRUNC(MULT(p1, b)), TRUNC(MULT(p2, b)));
#else
   return TTP_Share(TRUNC(MULT(p1, b)));
#endif
}


template <typename func_add>
TTP_Share Add(TTP_Share b, func_add ADD) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    return TTP_Share(ADD(p1, b.p1), ADD(p2, b.p2));
#else
    return TTP_Share(ADD(p1, b.p1));
#endif
}

    template <typename func_add, typename func_sub, typename func_mul>
TTP_Share prepare_dot(const TTP_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto result = MULT(SUB(p1,p2),SUB(b.p1,b.p2)); // (a + x - x)(b + y - y) = ab
    return TTP_Share(result, SET_ALL_ZERO());
#else
return TTP_Share(MULT(p1,b.p1));
#endif
}

template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto randomVal = getRandomVal(0);
    p1 = ADD(p1, randomVal);
    p2 = randomVal;
#endif
}
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
#if SIMULATE_MPC_FUNCTIONS == 1
/* Datatype dummy = getRandomVal(0); */
    auto randomVal = getRandomVal(0);
    p1 = TRUNC(ADD(p1, randomVal));
    p2 = TRUNC(randomVal);
/* std::cout << "dummy: " << dummy << std::endl; */
/* std::cout << "p1 (before): " << p1 << std::endl; */
/* p1 = ADD(TRUNC(SUB(p1,dummy)), TRUNC(dummy)); */
p1 = ADD(p1,PROMOTE(1)); // to avoid negative values
/* std::cout << "p1 (after): " << p1 << std::endl; */
#else
/* std::cout << "p1 (before): " << p1 << std::endl; */
p1 = TRUNC(p1);
/* Datatype dummy = getRandomVal(0); */
/* p1 = ADD(TRUNC(SUB(p1,dummy)), TRUNC(dummy)); */
/* p1 = ADD(p1,PROMOTE(1)); // to avoid negative values */
/* std::cout << "p1 (after): " << p1 << std::endl; */
#endif
}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}


template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_mult(TTP_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto result = MULT(SUB(p1,p2),SUB(b.p1,b.p2));
    auto randomVal = getRandomVal(0);
    return TTP_Share(ADD(result, randomVal), randomVal);
#else
return TTP_Share(MULT(p1,b.p1));
#endif
}
template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){

}


void prepare_reveal_to_all() const
{
#if PARTY == 2 && PROTOCOL != 13

    for(int t = 0; t < num_players-1; t++) 
#if SIMULATE_MPC_FUNCTIONS == 1
        send_to_live(t, p1);
#else
        send_to_live(t, p1);
#endif
#endif
}


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
#if PARTY != 2 && PROTOCOL != 13
    #if SIMULATE_MPC_FUNCTIONS == 1
    Datatype result = SUB(receive_from_live(P_2), p2);
    #else
    Datatype result = receive_from_live(P_2);
    #endif
#else
#if SIMULATE_MPC_FUNCTIONS == 1
    Datatype result = SUB(p1, p2);
#else
    Datatype result = p1;
#endif
#endif
return result;
}


template <int id,typename func_add, typename func_sub>
void prepare_receive_from(Datatype value, func_add ADD, func_sub SUB)
{
#if PARTY != 2 && PROTOCOL != 13
if constexpr(id == PSELF)
{
        send_to_live(P_2, value);
        
}
#else
    p1 = value;
#endif
}


template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    prepare_receive_from<id>(get_input_live(), ADD, SUB); //TODO: change such that input is only fetched if Party is PSELF
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
#if PARTY == 2 && PROTOCOL != 13
    p1 = receive_from_live(id);
    #if SIMULATE_MPC_FUNCTIONS == 1
    p2 = getRandomVal(0);
    p1 = ADD(p1, p2);
    #endif
#else
    #if SIMULATE_MPC_FUNCTIONS == 1
    p2 = getRandomVal(0);
    p1 = ADD(p1, p2);
    #endif
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
            /* #if SIMULATE_MPC_FUNCTIONS == 1 */
            /*     temp[j] = getRandomVal(0); */
            /*     in[j].p1 = OP_SUB(in[j].p1,temp[j]); */
            /* #else */
            #if SIMULATE_MPC_FUNCTIONS == 1
                temp[j] = OP_SUB( SET_ALL_ZERO(), in[j].p2);
            #else
                temp[j] = SET_ALL_ZERO();
            #endif
            /* #endif */
        }
    /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
    /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp, temp2);
    orthogonalize_boolean(temp2, temp);

    for(int i = 0; i < k; i++)
    {
#if SIMULATE_MPC_FUNCTIONS == 1
        out[i].p2 = getRandomVal(0);
        out[i].p1 = FUNC_XOR(out[i].p2,temp[i]);
#else 
        out[i].p1 = temp[i];
#endif
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
#if SIMULATE_MPC_FUNCTIONS == 1
        out[i].p2 = getRandomVal(0);
        out[i].p1 = FUNC_XOR(out[i].p2,temp[i]);
#else
        out[i].p1 = temp[i];
#endif
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
    Datatype temp[BITLENGTH]{0};
#if SIMULATE_MPC_FUNCTIONS == 1
    temp[BITLENGTH - 1] = FUNC_XOR(p1,p2);
#else
    temp[BITLENGTH - 1] = p1;
#endif
    unorthogonalize_boolean(temp,(UINT_TYPE*)temp);
    orthogonalize_arithmetic((UINT_TYPE*) temp,  temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp[i]; // set second summand to the msb
    }
}

static void complete_bit_injection_S1(TTP_Share out[])
{
#if SIMULATE_MPC_FUNCTIONS == 1
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p2 = getRandomVal(0);
        out[i].p1 = OP_ADD(out[i].p2,out[i].p1);
    }
#endif
    
}

static void complete_bit_injection_S2(TTP_Share out[])
{
#if SIMULATE_MPC_FUNCTIONS == 1
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p2 = getRandomVal(0);
        out[i].p1 = OP_ADD(out[i].p2,out[i].p1);
    }
#endif


}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_dot3(TTP_Share b, TTP_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto result = MULT( MULT(SUB(p1,p2),SUB(b.p1,b.p2)),SUB(c.p1,c.p2));
    return TTP_Share(result,SET_ALL_ZERO());
#else
return TTP_Share(MULT(MULT(p1,b.p1),c.p1));
#endif
}


template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_mult3(TTP_Share b, TTP_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto result = MULT( MULT(SUB(p1,p2),SUB(b.p1,b.p2)),SUB(c.p1,c.p2));
    return TTP_Share(result,SET_ALL_ZERO());
#else
return TTP_Share(MULT(MULT(p1,b.p1),c.p1));
#endif
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
#if SIMULATE_MPC_FUNCTIONS == 1
    p2 = getRandomVal(0);
    p1 = ADD(p1,p2);
#endif

}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_dot4(TTP_Share b, TTP_Share c, TTP_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto result = MULT( MULT(MULT(SUB(p1,p2),SUB(b.p1,b.p2)),SUB(c.p1,c.p2)),SUB(d.p1,d.p2));
    return TTP_Share(result,SET_ALL_ZERO());
#else
return TTP_Share(MULT(MULT(MULT(p1,b.p1),c.p1),d.p1));
#endif
}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_Share prepare_mult4(TTP_Share b, TTP_Share c, TTP_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto result = MULT( MULT(MULT(SUB(p1,p2),SUB(b.p1,b.p2)),SUB(c.p1,c.p2)),SUB(d.p1,d.p2));
    return TTP_Share(result,SET_ALL_ZERO());
#else
return TTP_Share(MULT(MULT(MULT(p1,b.p1),c.p1),d.p1));
#endif
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
#if SIMULATE_MPC_FUNCTIONS == 1
    p2 = getRandomVal(0);
    p1 = ADD(p1,p2);
#endif
}


TTP_Share relu() const
{
    return TTP_Share(relu_epi(p1));
}    
};

