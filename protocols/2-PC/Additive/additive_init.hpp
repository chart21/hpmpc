#pragma once
#include "../../beaver_triples.hpp"
#include <functional>
template <typename Datatype>
class Add_init
{
  private:
  public:
    Add_init() {}

    template <typename func_mul>
    Add_init mult_public(const Datatype b, func_mul MULT) const
    {
        return Add_init();
    }

    // P_i shares mx - lxi, P_j sets lxj to 0
    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            send_to_(PNEXT);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
            receive_from_(id);
    }

    template <typename func_add>
    Add_init Add(Add_init b, func_add ADD) const
    {
        return Add_init();
    }

    void prepare_reveal_to_all() const { send_to_(PNEXT); }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        receive_from_(PNEXT);
        return SET_ALL_ZERO();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Add_init prepare_mult(Add_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        if constexpr (std::is_same_v<func_add(), FUNC_XOR>)
            num_boolean_triples++;
        else
            num_arithmetic_triples++;
        Add_init c;
        send_to_(PNEXT);
        send_to_(PNEXT);
        store_output_share_();
        store_output_share_();
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void complete_mult(func_add ADD, func_sub SUB, func_mul MULT)
    {
        receive_from_(PPREV);
        receive_from_(PPREV);
    }

    Add_init public_val(Datatype a) { return Add_init(); }

    Add_init Not() const { return Add_init(); }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

#if SKIP_PRE == 1
    template <typename func_add, typename func_sub, typename func_mul>
    static void generate_zero_triples(uint64_t triple_num, func_add ADD, func_sub SUB, func_mul MULT)
    {
        init_beaver();
        if constexpr (std::is_same_v<func_add(), FUNC_XOR>)
        {
            for (uint64_t i = 0; i < triple_num; i++)
            {
                boolean_triple_a[i] = SET_ALL_ZERO();
                boolean_triple_b[i] = SET_ALL_ZERO();
                boolean_triple_c[i] = SET_ALL_ZERO();
            }
        }
        else
        {
            for (uint64_t i = 0; i < triple_num; i++)
            {
                arithmetic_triple_a[i] = SET_ALL_ZERO();
                arithmetic_triple_b[i] = SET_ALL_ZERO();
                arithmetic_triple_c[i] = SET_ALL_ZERO();
            }
        }
    }
#endif
};
