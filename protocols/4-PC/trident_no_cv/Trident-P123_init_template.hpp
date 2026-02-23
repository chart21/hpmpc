#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Trident123_init
{
  private:
  public:
    Trident123_init() {}

    Trident123_init(Datatype a, Datatype b, Datatype c)
    {
    }

    Trident123_init public_val(Datatype a) { return Trident123_init(); }

    Trident123_init Not() const { return Trident123_init(); }

    template <typename func_add>
    Trident123_init Add(Trident123_init b, func_add ADD) const
    {
        return Trident123_init();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    Trident123_init prepare_mult(Trident123_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        send_to_(PPREV_EX_0);
        return Trident123_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_mult2(Trident123_init a, Trident123_init b, func_add ADD, func_sub SUB, func_mul MULT) 
    {
        receive_from_(PNEXT_EX_0);
        send_to_(PPREV_EX_0);
        send_to_(PNEXT_EX_0);
    }
    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        receive_from_(PPREV_EX_0);
        receive_from_(PNEXT_EX_0);
    }


    void prepare_reveal_to_all() const
    {
        #if PARTY == 1 || PARTY == 2
        send_to_(P_0);
        #endif
        send_to_(PNEXT_EX_0);
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
        receive_from_(PPREV_EX_0);
        return SET_ALL_ZERO();
    }

    template <typename func_mul>
    Trident123_init mult_public(const Datatype b, func_mul MULT) const
    {
        return Trident123_init();
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            #if P_1 != PSELF
            send_to_(P_0);
            #endif
            #if P_2 != PSELF
            send_to_(P_1);
            #endif
            #if P_3 != PSELF
            send_to_(P_2);
            #endif
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {
            mv = receive_from_(id);
            if(constexpr (id != PPREV_EX_0))
            {
                send_to_(PPREV_EX_0);
            }
            if(constexpr (id != PNEXT_EX_0))
            {
                send_to_(PNEXT_EX_0);
            }
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from_2(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PPREV_EX_0)
        {
            receive_from_(PPREV_EX_0);
        }
        if constexpr (id != PNEXT_EX_0)
        {
            receive_from_(PNEXT_EX_0);
        }
    }

    static void send() { send_(); }

    static void receive() { receive_(); }

    static void communicate()
    {
        communicate_();
    }
};
