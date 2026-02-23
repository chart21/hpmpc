#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE TRIDENT0_POST_Share
template <typename Datatype>
class TRIDENT0_POST_Share
{
  private:
    Datatype l1;
    Datatype l2;
    Datatype l3;
  public:
    TRIDENT0_POST_Share() {}

    TRIDENT0_POST_Share(Datatype a, Datatype b, Datatype c)
    {
        l1 = a;
        l2 = b;
        l3 = c;
    }

    TRIDENT0_POST_Share public_val(Datatype a) { return TRIDENT0_POST_Share(); }

    template <typename func_mul>
    TRIDENT0_POST_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return TRIDENT0_POST_Share();
    }

    TRIDENT0_POST_Share Not() const { return TRIDENT0_POST_Share(l1, l2, l3); }

    template <typename func_add>
    TRIDENT0_POST_Share Add(TRIDENT0_POST_Share b, func_add ADD) const
    {
        return TRIDENT0_POST_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    TRIDENT0_POST_Share prepare_mult(TRIDENT0_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return TRIDENT0_POST_Share();
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
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        l1 = retrieve_output_share();
        l2 = retrieve_output_share();
        l3 = retrieve_output_share();
        Datatype mv = ADD(ADD(ADD(l1, l2), l3), receive_from_live(P_1));
        Datatype mv2 = ADD(ADD(ADD(l1, l2), l3), receive_from_live(P_2));
        if(! dat_equal(mv, mv2))
        {
            printf("P%i: Compareviews failed! \n", PARTY);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            l1 = retrieve_output_share();
            l2 = retrieve_output_share();
            l3 = retrieve_output_share();
            Datatype mv = ADD(ADD(ADD(l1, l2), l3), val);
            send_to_live(P_1, mv);
            send_to_live(P_2, mv);
            send_to_live(P_3, mv);
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

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate()
    {
        communicate_live();
    }
};
