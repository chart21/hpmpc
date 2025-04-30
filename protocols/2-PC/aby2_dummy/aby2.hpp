#pragma once
#include "../../generic_share.hpp"
#include <functional>
template <typename Datatype>
class ABY2_Share
{
  private:
#if PRE == 0
    Datatype m;
#endif
    Datatype l;

  public:
    ABY2_Share() {}
    ABY2_Share(Datatype l) { this->l = l; }
#if PRE == 0
    ABY2_Share(Datatype x, Datatype l)
    {
        this->m = x;
        this->l = l;
    }
#endif

    template <typename func_mul>
    ABY2_Share mult_public(const Datatype b, func_mul MULT) const
    {
#if PRE == 0
        return ABY2_Share(MULT(m, b), MULT(l, b));
#else
        return ABY2_Share(MULT(l, b));
#endif
    }

    // Fake triple:
    Datatype getFakeBeaverShare(Datatype al, Datatype bl) const
    {
        Datatype share = SET_ALL_ZERO();
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share(share);
#endif
        return share;
    }

    Datatype getFakeRandomVal(int id) const
    {
        Datatype randVal = SET_ALL_ZERO();
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share(randVal);
#endif
        return randVal;
    }

    // P_i shares mx - lxi, P_j sets lxj to 0
    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            l = getFakeRandomVal(id);
#if PRE == 0
            m = ADD(val, l);
            send_to_live(PNEXT, m);
#endif
        }
        else
            l = SET_ALL_ZERO();
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
#if PRE == 0
        if constexpr (id != PSELF)
            m = receive_from_live(id);
#endif
    }

    template <typename func_add>
    ABY2_Share Add(ABY2_Share b, func_add ADD) const
    {
#if PRE == 0
        return ABY2_Share(ADD(m, b.m), ADD(l, b.l));
#else
        return ABY2_Share(ADD(l, b.l));
#endif
    }

    void prepare_reveal_to_all() const
    {
#if PRE == 0
        send_to_live(PNEXT, l);
#else
        pre_send_to_live(PNEXT, l);
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 0
        return SUB(m, ADD(l, receive_from_live(PNEXT)));
#else
        return SET_ALL_ZERO();
#endif
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_Share prepare_mult(ABY2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_Share c;
        Datatype lalb = getFakeBeaverShare(l, b.l);  // Fake triple
        c.l = getFakeRandomVal(PSELF);               // get new mask
#if PRE == 0
        Datatype msg = ADD(ADD(ADD(MULT(m, b.l), MULT(l, b.m)), lalb), c.l);
        send_to_live(PNEXT, msg);
        c.m = SUB(MULT(m, b.m), msg);
#endif
        return c;
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PRE == 0
        Datatype msg = receive_from_live(PPREV);
        m = SUB(m, msg);
#endif
    }

    ABY2_Share public_val(Datatype a)
    {
#if PRE == 0
        return ABY2_Share(a, SET_ALL_ZERO());
#else
        return ABY2_Share(SET_ALL_ZERO());
#endif
    }

    ABY2_Share Not() const
    {
#if PRE == 0
        return ABY2_Share(NOT(m), l);
#else
        return ABY2_Share(l);
#endif
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate()
    {
#if PRE == 0
        communicate_live();
#endif
    }
};
