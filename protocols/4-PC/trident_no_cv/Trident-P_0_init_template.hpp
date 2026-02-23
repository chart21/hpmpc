#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE Trident0_init
template <typename Datatype>
class Trident0_init
{
  public:
    Trident0_init() {}

    Trident0_init(Datatype a, Datatype b, Datatype c)
    {
    }

    Trident0_init public_val(Datatype a) { return Trident0_init(); }

    template <typename func_mul>
    Trident0_init mult_public(const Datatype b, func_mul MULT) const
    {
        return Trident0_init();
    }

    Trident0_init Not() const { return Trident0_init(); }

    template <typename func_add>
    Trident0_init Add(Trident0_init b, func_add ADD) const
    {
        return Trident0_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Trident0_init prepare_mult(Trident0_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        #if PRE == 1
        pre_send_to_(P_1);
        pre_send_to_(P_2);
        pre_send_to_(P_3);
        #else
        send_to_(P_1);
        send_to_(P_2);
        send_to_(P_3);
        #endif
        return Trident0_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_mult2(Trident123_Share a, Trident123_Share b, func_add ADD, func_sub SUB, func_mul MULT)
    {
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    void prepare_reveal_to_all() const { 
        #if PRE == 1
        pre_send_to_(P_1);
        pre_send_to_(P_2);
        pre_send_to_(P_3);
        #else
        send_to_(P_1);
        send_to_(P_2);
        send_to_(P_3);
        #endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        receive_from_(P_1);
        receive_from_(P_2);
        #if PRE == 1
        store_output_share_();
        store_output_share_();
        store_output_share_();
        #endif
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            Datatype mv = ADD(ADD(ADD(l1, l2), l3), val);
            #if PRE == 1
            store_output_share_();
            store_output_share_();
            store_output_share_();
            // pre_send_to_live(P_1, mv);
            // pre_send_to_live(P_2, mv);
            // pre_send_to_live(P_3, mv);
            #endif
            send_to_(P_1);
            send_to_(P_2);
            send_to_(P_3);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
    }
    
    template <int id, typename func_add, typename func_sub>
    void complete_receive_from_2(func_add ADD, func_sub SUB)
    {
    }

    static void send() { send_(); }

    static void receive() { receive_(); }

    static void communicate()
    {
        communicate_();
    }
};
