#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class ASTRA2_Share
{
    Datatype mv;
    Datatype lv;

  public:
    ASTRA2_Share() {}
    ASTRA2_Share(Datatype a, Datatype b)
    {
        mv = a;
        lv = b;
    }

    ASTRA2_Share public_val(DATATYPE a) { return ASTRA2_Share(a, SET_ALL_ZERO()); }

    ASTRA2_Share Not() const { return ASTRA2_Share(NOT(mv), lv); }

    template <typename func_add>
    ASTRA2_Share Add(ASTRA2_Share b, func_add ADD) const
    {
        return ASTRA2_Share(ADD(mv, b.mv), lv);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ASTRA2_Share prepare_mult(ASTRA2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ASTRA2_Share c;
        DATATYPE yz2 = getRandomVal(P_0);  // yz1

#if PRE == 0
        DATATYPE yxy2 = receive_from_live(P_0);
#else
        DATATYPE yxy2 = pre_receive_from_live(P_0);
#endif

        c.mv = ADD(ADD(SUB(MULT(mv, b.mv), ADD(MULT(mv, b.lv), MULT(b.mv, lv))), yz2), yxy2);
        send_to_live(P_1, c.mv);
        c.lv = yz2;
        return c;
    }
    template <typename func_add, typename func_sub, typename func_mul>
    ASTRA2_Share prepare_dot(const ASTRA2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ASTRA2_Share c;
        c.mv = SUB(MULT(mv, b.mv), ADD(MULT(mv, b.lv), MULT(b.mv, lv)));
        return c;
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        DATATYPE yz2 = getRandomVal(P_0);  // yz1
#if PRE == 0
        DATATYPE yxy2 = receive_from_live(P_0);
#else
        DATATYPE yxy2 = pre_receive_from_live(P_0);
#endif
        mv = ADD(mv, ADD(yz2, yxy2));
        send_to_live(P_1, mv);
        lv = yz2;
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        mv = ADD(mv, receive_from_live(P_1));
    }

    void prepare_reveal_to_all() const { send_to_live(P_0, mv); }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 0
        return SUB(mv, receive_from_live(P_0));
#else
        return SUB(mv, pre_receive_from_live(P_0));
#endif
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_1)
        {
#if OPT_SHARE == 0
            lv = getRandomVal(P_0);
#endif
        }
        else if constexpr (id == P_2)  // -> lv = lv1, lv2=0
        {
            lv = getRandomVal(P_0);
            mv = ADD(val, lv);
            send_to_live(P_1, mv);
        }
    }

    template <typename func_mul>
    ASTRA2_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return ASTRA2_Share(MULT(mv, b), MULT(lv, b));
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_0)
        {
#if OPT_SHARE == 1
            mv = SET_ALL_ZERO();  // Check options
#if PRE == 0
            lv = receive_from_live(P_0);
#else
            lv = pre_receive_from_live(P_0);
#endif
#else
            mv = receive_from_live(P_0);
#endif
        }
        else if constexpr (id == P_1)
        {
            mv = receive_from_live(P_1);
            /* lv = SET_ALL_ZERO(); */
        }
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }
};
