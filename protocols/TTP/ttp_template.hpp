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
    /* return OP_ADD(p1,p2); */
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

template <typename func_mul>
TTP_Share mult_public(const Datatype b, func_mul MULT) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    return TTP_Share(MULT(p1,b),MULT(p2,b));
#else
    return TTP_Share(MULT(p1,b));
#endif
}


    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
TTP_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    /* return TTP_Share(TRUNC(MULT(p1, b)), TRUNC(MULT(p2, b))); */
#if TRUNC_THEN_MULT == 1
    Datatype p1_v = MULT(TRUNC(p1), b);
    Datatype p2_v = MULT(TRUNC(p2), b);
    /* auto val = MULT(TRUNC(SUB(p1,p2)), b); */
#else
    /* auto val = TRUNC(MULT(SUB(p1,p2), b)); */
    Datatype p1_v = TRUNC(MULT(p1, b));
    Datatype p2_v = TRUNC(MULT(p2, b));
#endif
    /* return TTP_Share(ADD(val, randomVal), randomVal); */
    auto randomVal = getRandomVal(0);
    return TTP_Share(ADD(p1_v, randomVal), ADD(p2_v, randomVal));
#else
   return TTP_Share(TRUNC(MULT(p1, b)));
#endif
}
    
    template <typename func_add, typename func_sub>
void complete_public_mult_fixed(func_add ADD, func_sub SUB)
{
}

template <typename func_mul>
TTP_Share mult_public(Datatype b, func_mul MULT)
{
#if SIMULATE_MPC_FUNCTIONS == 1
    return TTP_Share(MULT(p1,b),MULT(p2,b));
#else
    return TTP_Share(MULT(p1,b));
#endif
}


template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, TTP_Share& r_mk2, TTP_Share& r_msb, TTP_Share& c, TTP_Share& c_prime) {
    Datatype c_dat_prime = trunc(p1);
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH-FRACTIONAL-1)) - 1;
    Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue
    // Apply the mask using bitwise AND
    c_dat_prime = AND(c_dat_prime, mask); //mod 2^k-m-1
    Datatype c_dat = OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(p1);
#if SIMULATE_MPC_FUNCTIONS == 1
    Datatype tmp = getRandomVal(0);
    c = TTP_Share(ADD(c_dat,tmp),tmp);
    tmp = getRandomVal(0);
    c_prime = TTP_Share(ADD(c_dat_prime,tmp),tmp);
#else
    c = TTP_Share(c_dat);
    c_prime = TTP_Share(c_dat_prime);
#endif
    /* c = TTP_Share(p1); */
    /* c_prime = TTP_Share(trunc(p1)); */
}
    /* Datatype r = getRandomVal(0); */
    /* Datatype c = ADD(SUB(p1,p2),r); //open c = x + r */
    /* Datatype c_prime = trunc(c); */
    /* UINT_TYPE maskValue = (1 << (BITLENGTH-FRACTIONAL-1)) - 1; */ 
    /* Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue */
    /* // Apply the mask using bitwise AND */
    /* c_prime = AND(c_prime, mask); //mod 2^k-m-1 */
    /* c_prime = c_prime % (1 << (BITLENGTH-FRACTIONAL-1)); */
    /* Datatype b = XOR(r >> (BITLENGTH-1), c >> (BITLENGTH-1)); // xor msbs */
    /* c_prime = ADD(SUB(c_prime, (r << (1)) >> (FRACTIONAL+1)), b << (BITLENGTH-FRACTIONAL-1)); */ 
    /* Datatype f_mask = getRandomVal(0); */
    /* c_prime = ADD(c_prime,PROMOTE(1)); // avoid 0 underflow */
    /* return TTP_Share(ADD(c_prime,f_mask),f_mask); */
    

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, TTP_Share& r_mk2, TTP_Share& r_msb, TTP_Share& c, TTP_Share& c_prime) {
#if SIMULATE_MPC_FUNCTIONS == 1
    Datatype rmk2 = OP_SHIFT_LOG_RIGHT<FRACTIONAL+1>( OP_SHIFT_LEFT<1>(p2)); 
    Datatype rmsb = OP_SHIFT_LOG_RIGHT<BITLENGTH -1>(p2);
    Datatype tmp = getRandomVal(0);
    r_mk2 = TTP_Share(ADD(rmk2,tmp),tmp);
    tmp = getRandomVal(0);
    r_msb = TTP_Share(ADD(rmsb,tmp),tmp);
#else
    r_mk2 = TTP_Share(PROMOTE(0));
    r_msb = TTP_Share(PROMOTE(0));
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
    /* p1 = TRUNC(ADD(p1, randomVal)); */
    /* p2 = TRUNC(randomVal); */
    p1 = SUB(TRUNC(ADD(p1, randomVal)), TRUNC(randomVal));
    p2 = getRandomVal(0);
    p1 = ADD(p1, p2);
/* std::cout << "dummy: " << dummy << std::endl; */
/* std::cout << "p1 (before): " << p1 << std::endl; */
/* p1 = ADD(TRUNC(SUB(p1,dummy)), TRUNC(dummy)); */
/* p1 = ADD(p1,PROMOTE(1)); // to avoid negative values */
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
if constexpr(id == PSELF || PROTOCOL == 13)
    prepare_receive_from<id>(get_input_live(), ADD, SUB); //Careful: Simulator is always fetching inputs
else
    prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
#if PARTY == 2 && PROTOCOL != 13
if constexpr(id != PSELF)
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

static void prepare_A2B_S1(int m, int k, TTP_Share in[], TTP_Share out[])
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

    for(int i = m; i < k; i++)
    {
#if SIMULATE_MPC_FUNCTIONS == 1
        out[i-m].p2 = getRandomVal(0);
        out[i-m].p1 = FUNC_XOR(out[i-m].p2,temp[i]);
#else 
        out[i-m].p1 = temp[i];
#endif
    } 
}


static void prepare_A2B_S2(int m, int k, TTP_Share in[], TTP_Share out[])
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

    for(int i = m; i < k; i++)
    {
#if SIMULATE_MPC_FUNCTIONS == 1
        out[i-m].p2 = getRandomVal(0);
        out[i-m].p1 = FUNC_XOR(out[i-m].p2,temp[i]);
#else
        out[i-m].p1 = temp[i];
#endif
    } 
}

static void complete_A2B_S1( int k, TTP_Share out[])
{

}
static void complete_A2B_S2( int k, TTP_Share out[])
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
    /* unorthogonalize_boolean(temp,(UINT_TYPE*)temp); */
    /* orthogonalize_arithmetic((UINT_TYPE*) temp,  temp); */
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(temp, temp2);
    orthogonalize_arithmetic(temp2, temp);
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

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
TTP_Share prepare_trunc_2k(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc) const{
#if SIMULATE_MPC_FUNCTIONS == 1
    Datatype r = getRandomVal(0);
    Datatype c = ADD(SUB(p1,p2),r); //open c = x + r
    Datatype c_prime = trunc(c);
    UINT_TYPE maskValue = (1 << (BITLENGTH-FRACTIONAL-1)) - 1; 
    Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue
    // Apply the mask using bitwise AND
    c_prime = AND(c_prime, mask); //mod 2^k-m-1
    c_prime = c_prime % (1 << (BITLENGTH-FRACTIONAL-1));
    Datatype b = XOR(r >> (BITLENGTH-1), c >> (BITLENGTH-1)); // xor msbs
    c_prime = ADD(SUB(c_prime, (r << (1)) >> (FRACTIONAL+1)), b << (BITLENGTH-FRACTIONAL-1)); 
    Datatype f_mask = getRandomVal(0);
    c_prime = ADD(c_prime,PROMOTE(1)); // avoid 0 underflow
    return TTP_Share(ADD(c_prime,f_mask),f_mask);
    /* return TTP_Share(ADD(trunc(SUB(p1,p2)),f_mask),f_mask); */
#else
    return TTP_Share(trunc(p1));
#endif
}

template <typename func_add, typename func_sub>
void complete_trunc_2k(func_add ADD, func_sub SUB){
}

TTP_Share relu() const
{
#if SIMULATE_MPC_FUNCTIONS == 1
    auto result = FUNC_TRUNC(relu_epi(OP_SUB(p1,p2)));
    auto randVal = getRandomVal(0);
    return TTP_Share(OP_ADD(result,randVal),randVal);
#else
    return TTP_Share(FUNC_TRUNC(relu_epi(p1)));
#endif
}    
};

