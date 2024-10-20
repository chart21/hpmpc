#pragma once
#include "../../beaver_triples.hpp"
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
    /* pre_send_to_live(PNEXT, l); */
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
if constexpr(std::is_same_v<func_add, FUNC_XOR>)
    t = retrieveBooleanTriple<Datatype>();
else
    t = retrieveArithmeticTriple<Datatype>();
//open
auto lta = ADD(l,t.a);
auto ltb = ADD(b.l,t.b);
send_to_live(PNEXT, lta);
send_to_live(PNEXT, ltb);
auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
store_output_share(t.a);
store_output_share(b.l);
store_output_share(lxly);
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

template <typename func_add, typename func_sub, typename func_mul>
static void complete_preprocessing(uint64_t n, func_add ADD, func_sub SUB, func_mul MULT)
{
Datatype* lxly = new Datatype[n];
for(int i = 0; i < n; i++)
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share();
    auto tb = retrieve_output_share();
    lxly[i] = ADD(SUB(MULT(lta, tb), MULT(ltb, ta)), retrieve_output_share());
}
delete[] preprocessed_outputs;
preprocessed_outputs = lxly;
preprocessed_outputs_index = 0;
preprocessed_outputs_input_index = 0;
deinit_beaver();
}

};
