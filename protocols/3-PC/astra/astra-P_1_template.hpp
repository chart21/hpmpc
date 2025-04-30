#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class ASTRA1_Share
{
    Datatype mv;
    Datatype lv;

  public:
    ASTRA1_Share() {}
    ASTRA1_Share(Datatype a, Datatype b)
    {
        mv = a;
        lv = b;
    }

    ASTRA1_Share public_val(Datatype a) { return ASTRA1_Share(a, SET_ALL_ZERO()); }

    ASTRA1_Share Not() const { return ASTRA1_Share(NOT(mv), lv); }

    template <typename func_add>
    ASTRA1_Share Add(ASTRA1_Share b, func_add ADD) const
    {
        return ASTRA1_Share(ADD(mv, b.mv), lv);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ASTRA1_Share prepare_mult(ASTRA1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ASTRA1_Share c;
        Datatype yz1 = getRandomVal(P_0);  // yz1
        Datatype yxy1 = getRandomVal(P_0);
        c.mv = SUB(ADD(yz1, yxy1), ADD(MULT(mv, b.lv), MULT(b.mv, lv)));
        c.lv = yz1;
        send_to_live(P_2, c.mv);
        return c;
    }
    template <typename func_add, typename func_sub, typename func_mul>
    ASTRA1_Share prepare_dot(const ASTRA1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ASTRA1_Share c;
        c.mv = ADD(MULT(mv, b.lv), MULT(b.mv, lv));
        return c;
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        Datatype yz1 = getRandomVal(P_0);  // yz1
        Datatype yxy1 = getRandomVal(P_0);
        mv = SUB(ADD(yz1, yxy1), mv);
        lv = yz1;
        send_to_live(P_2, mv);
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        // a.p2 already set in last round
        mv = ADD(mv, receive_from_live(P_2));
    }

    void prepare_reveal_to_all() const {}

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        return SUB(mv, receive_from_live(P_0));
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_0)
        {
            lv = getRandomVal(P_0);
        }
        else if constexpr (id == P_1)  // -> lv = lv2, lv1=0
        {
            lv = getRandomVal(P_0);
            mv = ADD(val, lv);
            send_to_live(P_2, mv);
        }
    }

    template <typename func_mul>
    ASTRA1_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return ASTRA1_Share(MULT(mv, b), MULT(lv, b));
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_0)
        {
#if OPT_SHARE == 1
            mv = SET_ALL_ZERO();  // check options
#else
            mv = receive_from_live(P_0);
#endif
        }
        else if constexpr (id == P_2)
        {
            mv = receive_from_live(P_2);
            lv = SET_ALL_ZERO();
        }
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }
};
