#pragma once
#include "../../generic_share.hpp"
#include <functional>
template <typename Datatype>
class ABY2_POST_Share
{
  private:
    Datatype m;
    Datatype l;

  public:
    ABY2_POST_Share() {}
    ABY2_POST_Share(Datatype l) { this->l = l; }
    ABY2_POST_Share(Datatype x, Datatype l)
    {
        this->m = x;
        this->l = l;
    }

    template <typename func_mul>
    ABY2_POST_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return ABY2_POST_Share(MULT(m, b), MULT(l, b));
    }

    // Fake triple:
    Datatype getFakeBeaverShare(Datatype al, Datatype bl) const { return retrieve_output_share(); }

    Datatype getFakeRandomVal(int id) const { return retrieve_output_share(); }

    // P_i shares mx - lxi, P_j sets lxj to 0
    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            l = getFakeRandomVal(id);
            m = ADD(val, l);
            send_to_live(PNEXT, m);
        }
        else
            l = SET_ALL_ZERO();
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
            m = receive_from_live(id);
    }

    template <typename func_add>
    ABY2_POST_Share Add(ABY2_POST_Share b, func_add ADD) const
    {
        return ABY2_POST_Share(ADD(m, b.m), ADD(l, b.l));
    }

    void prepare_reveal_to_all() const {}

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        return SUB(m, ADD(pre_receive_from_live(PNEXT), l));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_POST_Share prepare_mult(ABY2_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_POST_Share c;
        Datatype lalb = getFakeBeaverShare(l, b.l);  // Fake triple
        c.l = getFakeRandomVal(PSELF);               // get new mask
        Datatype msg = ADD(ADD(ADD(MULT(m, b.l), MULT(l, b.m)), lalb), c.l);
        send_to_live(PNEXT, msg);
        c.m = SUB(MULT(m, b.m), msg);
        return c;
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Datatype msg = receive_from_live(PPREV);
        m = SUB(m, msg);
    }

    ABY2_POST_Share public_val(Datatype a) { return ABY2_POST_Share(a, SET_ALL_ZERO()); }

    ABY2_POST_Share Not() const { return ABY2_POST_Share(NOT(m), l); }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }
};
