#pragma once
#include "../../generic_share.hpp"
#include <functional>
template <typename Datatype>
class ABY2_init
{
  private:
  public:
    ABY2_init() {}

    template <typename func_mul>
    ABY2_init mult_public(const Datatype b, func_mul MULT) const
    {
        return ABY2_init();
    }

    // Fake triple:
    Datatype getFakeBeaverShare() const
    {
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share_();
#endif
        return SET_ALL_ZERO();
    }

    void getFakeRandomVal() const
    {
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share_();
#endif
    }

    // P_i shares mx - lxi, P_j sets lxj to 0
    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            getFakeRandomVal();
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
    ABY2_init Add(ABY2_init b, func_add ADD) const
    {
        return ABY2_init();
    }

    void prepare_reveal_to_all() const
    {
#if PRE == 0
        send_to_(PNEXT);
#else
        pre_send_to_(PNEXT);
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
/* #if PRE == 1 && HAS_POST_PROTOCOL == 1 */
/* store_output_share_(); */
/* #endif */
#if PRE == 0
        receive_from_(PNEXT);
#else
        pre_receive_from_(PNEXT);
#endif
        return SET_ALL_ZERO();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_mult(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_init c;
        getFakeRandomVal();
        getFakeBeaverShare();
        send_to_(PNEXT);
        return c;
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        receive_from_(PPREV);
    }

    ABY2_init public_val(Datatype a) { return ABY2_init(); }

    ABY2_init Not() const { return ABY2_init(); }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }
};
