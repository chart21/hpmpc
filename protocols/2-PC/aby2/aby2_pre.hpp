#pragma once
#include "../../beaver_triples.hpp"
#include <cstdint>
template <typename Datatype>
class ABY2_PRE_Share{
using BT = triple<Datatype>;
private:
Datatype l;
public:
ABY2_PRE_Share()  {}
ABY2_PRE_Share(Datatype l) { this->l = l; }

template <typename func_mul>
ABY2_PRE_Share mult_public(const Datatype b, func_mul MULT) const
{
    return ABY2_PRE_Share(MULT(l,b));
}

template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
        l = getRandomVal(PSELF);
    else
        l = SET_ALL_ZERO();
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
}

template <typename func_add>
ABY2_PRE_Share Add( ABY2_PRE_Share b, func_add ADD) const
{
    return ABY2_PRE_Share(ADD(l,b.l));
}

void prepare_reveal_to_all() const
{
    pre_send_to_live(PNEXT, l);
    triple_type[triple_type_index++] = 2;
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
    return SET_ALL_ZERO();
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
BT t;
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
{
    t = retrieveBooleanTriple<Datatype>();
    triple_type[triple_type_index++] = 0;
}
else
{
    t = retrieveArithmeticTriple<Datatype>();
    triple_type[triple_type_index++] = 1;
}
//open
auto lta = ADD(l,t.a);
auto ltb = ADD(b.l,t.b);
pre_send_to_live(PNEXT, lta);
pre_send_to_live(PNEXT, ltb);
auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
{
store_output_share_bool(t.a);
store_output_share_bool(b.l);
store_output_share_bool(lxly);
}
else
{
store_output_share_arithmetic(t.a);
store_output_share_arithmetic(b.l);
store_output_share_arithmetic(lxly);
}
return ABY2_PRE_Share(getRandomVal(PSELF)); //new mask
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
BT t;
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
{
    t = retrieveBooleanTriple<Datatype>();
    triple_type[triple_type_index++] = 0;
}
else
{
    t = retrieveArithmeticTriple<Datatype>();
    triple_type[triple_type_index++] = 1;
}
//open
auto lta = ADD(l,t.a);
auto ltb = ADD(b.l,t.b);
pre_send_to_live(PNEXT, lta);
pre_send_to_live(PNEXT, ltb);
auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
{
store_output_share_bool(t.a);
store_output_share_bool(b.l);
store_output_share_bool(lxly);
}
else
{
store_output_share_arithmetic(t.a);
store_output_share_arithmetic(b.l);
store_output_share_arithmetic(lxly);
}
return ABY2_PRE_Share();
}

/* template <typename func_add, typename func_sub, typename func_mul, typename func_trunc> */
/*     ABY2_PRE_Share prepare_dot_with_trunc(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT, func_trunc TRUNC) const */
/* { */
/*     return prepare_mult(b, ADD, SUB, MULT); */
/* } */

template <typename func_add, typename func_sub>

void mask_and_send_dot(func_add ADD, func_sub SUB)
{
    l = getRandomVal(PSELF);
}
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
    l = getRandomVal(PSELF);
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_PRE_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC, int fractional_bits = FRACTIONAL) const
{
return ABY2_PRE_Share(getRandomVal(PSELF));
}
    
    template <typename func_add, typename func_sub>
void complete_public_mult_fixed(func_add ADD, func_sub SUB)
{
}

    
    template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
}
    
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}

static void prepare_A2B_S1(int m, int k, ABY2_PRE_Share in[], ABY2_PRE_Share out[])
{
#if PARTY == 0
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
    }
#endif
}

static void prepare_A2B_S2(int m, int k, ABY2_PRE_Share in[], ABY2_PRE_Share out[])
{
#if PARTY == 1
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
    }
#endif
}

void prepare_bit2a(ABY2_PRE_Share out[])
{
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    Datatype lb[BITLENGTH]{0};
    lb[BITLENGTH - 1] = l;
    unorthogonalize_boolean(lb, temp2);
    orthogonalize_arithmetic(temp2, lb);
    for(int i = 0; i < BITLENGTH; i++)
    {
        auto t = retrieveArithmeticTriple<Datatype>();
        triple_type[triple_type_index++] = 3;
#if PARTY == 0
        auto bl = SET_ALL_ZERO();
        auto al = lb[i];
#else 
        auto bl = lb[i];
        auto al = SET_ALL_ZERO();
#endif
        auto lta = OP_ADD(al, t.a);
        auto ltb = OP_ADD(bl, t.b); //optimization?
        pre_send_to_live(PNEXT, lta); 
        pre_send_to_live(PNEXT, ltb);
        auto lxly = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, t.a)), t.c);
        store_output_share_arithmetic(t.a);
        store_output_share_arithmetic(bl);
        store_output_share_arithmetic(lxly);
        out[i].l = getRandomVal(PSELF); 
    }
}

void complete_bit2a()
{
}


static void complete_A2B_S1(int k, ABY2_PRE_Share out[])
{
}

static void complete_A2B_S2(int k, ABY2_PRE_Share out[])
{
}


static ABY2_PRE_Share public_val(Datatype a)
{
    return ABY2_PRE_Share(SET_ALL_ZERO());
}

ABY2_PRE_Share Not() const
{
    return ABY2_PRE_Share(l);
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
}


static void complete_preprocessing(uint64_t arithmetic_triple_num, uint64_t boolean_triple_num, uint64_t num_output_shares)
{
Datatype* lxly_a = new Datatype[arithmetic_triple_num];
Datatype* lxly_b = new Datatype[boolean_triple_num];
uint64_t arithmetic_triple_counter = 0;
uint64_t boolean_triple_counter = 0;
const auto num_triples = arithmetic_triple_num + boolean_triple_num + num_output_shares;
for(uint64_t i = 0; i < num_triples; i++)
{
if(triple_type[i] == 0)
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share_bool();
    auto bl = retrieve_output_share_bool();
    auto prev_val = retrieve_output_share_bool();
    lxly_b[boolean_triple_counter++] = FUNC_XOR(FUNC_XOR(FUNC_AND(lta, bl), FUNC_AND(ltb, ta)), prev_val);
}
else if(triple_type[i] == 1)
{
    auto lta = pre_receive_from_live(PNEXT);
   auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share_arithmetic();
    auto bl = retrieve_output_share_arithmetic();
    auto prev_val = retrieve_output_share_arithmetic();
    lxly_a[arithmetic_triple_counter++] = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)), prev_val);
}
else if(triple_type[i] == 3)
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share_arithmetic();
    auto bl = retrieve_output_share_arithmetic();
    auto prev_val = retrieve_output_share_arithmetic();
    lxly_a[arithmetic_triple_counter++] = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)), prev_val);
}
else
{
    auto l = pre_receive_from_live(PNEXT);
    store_output_share(l);
}
}
delete[] preprocessed_outputs_bool;
preprocessed_outputs_bool = lxly_b;
preprocessed_outputs_bool_index = 0;
preprocessed_outputs_bool_input_index = 0;

delete[] preprocessed_outputs_arithmetic;
preprocessed_outputs_arithmetic = lxly_a;
preprocessed_outputs_arithmetic_index = 0;
preprocessed_outputs_arithmetic_input_index = 0;
deinit_beaver();
init_srngs();
/* for(int i = 0; i < num_players*player_multiplier; i++) */
/* { */
/*     num_generated[i] = 0; // reset the random number generator to receive identical lx,ly */
/* } */
delete[] triple_type;

}

};
