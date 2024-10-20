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
    
    template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
}

ABY2_PRE_Share public_val(Datatype a)
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
    lxly_b[arithmetic_triple_counter++] = FUNC_XOR(FUNC_XOR(FUNC_AND(lta, bl), FUNC_AND(ltb, ta)), prev_val);
}
else if(triple_type[i] == 1)
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share_arithmetic();
    auto bl = retrieve_output_share_arithmetic();
    auto prev_val = retrieve_output_share_arithmetic();
    lxly_a[boolean_triple_counter++] = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)), prev_val);
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
for(int i = 0; i < num_players*player_multiplier; i++)
{
    num_generated[i] = 0; // reset the random number generator to receive identical lx,ly
}
delete[] triple_type;

}

};
