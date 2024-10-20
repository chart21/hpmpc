#pragma once
#include "../../beaver_triples.hpp"
#include <cstdint>
#include <functional>
template <typename Datatype>
class ABY2_init{
private:
    public:
ABY2_init()  {}



template <typename func_mul>
ABY2_init mult_public(const Datatype b, func_mul MULT) const
{
    return ABY2_init();
}

// P_i shares mx - lxi, P_j sets lxj to 0
template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
    {
        send_to_(PNEXT);
    }
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id != PSELF)
        receive_from_(id);
}

template <typename func_add>
ABY2_init Add( ABY2_init b, func_add ADD) const
{
    return ABY2_init();
}

    

void prepare_reveal_to_all() const
{
    pre_send_to_(PNEXT);
    store_output_share_();
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
    return SET_ALL_ZERO();
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_mult(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
ABY2_init c;
if constexpr(std::is_same_v<func_add, FUNC_XOR>)
{
    num_boolean_triples++;
    store_output_share_bool_();
    store_output_share_bool_();
    store_output_share_bool_();
}
else
{
    num_arithmetic_triples++;
    store_output_share_arithmetic_();
    store_output_share_arithmetic_();
    store_output_share_arithmetic_();
}

pre_send_to_(PNEXT);
pre_send_to_(PNEXT);
send_to_(PNEXT);
return c;
}
    
    template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    receive_from_(PPREV);
}



ABY2_init public_val(Datatype a)
{
    return ABY2_init();
}

ABY2_init Not() const
{
    return ABY2_init();
}

static void send()
{
send_();
}
static void receive()
{
    receive_();
}
static void communicate()
{
communicate_();
}

static void finalize(std::string* ips)
{
    finalize_(ips);
}

static void finalize(std::string* ips, receiver_args* ra, sender_args* sa)
{
    finalize_(ips, ra, sa);
}

static void complete_preprocessing(uint64_t arithmetic_triple_num, uint64_t boolean_triple_num, uint64_t num_output_shares)
{
for(uint64_t i = 0; i < arithmetic_triple_num + boolean_triple_num; i++)
{
    pre_receive_from_(PNEXT);
    pre_receive_from_(PNEXT);
}
for(uint64_t i = 0; i < num_output_shares; i++)
{
    pre_receive_from_(PNEXT);
}
triple_type = new uint8_t[arithmetic_triple_num + boolean_triple_num];
}



};

