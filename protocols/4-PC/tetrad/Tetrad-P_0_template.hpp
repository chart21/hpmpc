#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE Tetrad0_Share
template <typename Datatype>
class Tetrad0_Share
{
  private:
    Datatype mv;
    Datatype l0;
    Datatype l1;
    Datatype storage;  // used for storing results needed later
  public:
    Tetrad0_Share() {}

    Tetrad0_Share(Datatype a, Datatype b, Datatype c)
    {
        mv = a;
        l0 = b;
        l1 = c;
    }

    Tetrad0_Share public_val(Datatype a) { return Tetrad0_Share(a, SET_ALL_ZERO(), SET_ALL_ZERO()); }

    template <typename func_mul>
    Tetrad0_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return Tetrad0_Share(MULT(mv, b), MULT(l0, b), MULT(l1, b));
    }

    Tetrad0_Share Not() const { return Tetrad0_Share(NOT(mv), l0, l1); }

    template <typename func_add>
    Tetrad0_Share Add(Tetrad0_Share b, func_add ADD) const
    {
        return Tetrad0_Share(ADD(mv, b.mv), ADD(l0, b.l0), ADD(l1, b.l1));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Tetrad0_Share prepare_mult(Tetrad0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        // l0 -> l1
        // l1 -> l2
        Tetrad0_Share c;
        Datatype y3ab = ADD(ADD(MULT(l0, b.l1), MULT(l1, b.l0)), MULT(l0, l0));
        Datatype u1 = getRandomVal(P_013);
        Datatype u2 = getRandomVal(P_023);
        Datatype r = SUB(y3ab, ADD(u1, u2));
        Tetrad0_Share q;

        // q:
        c.mv = SET_ALL_ZERO();
        c.l0 = getRandomVal(P_013);                // -> lamda1, not held by P2
        c.l1 = SUB(SET_ALL_ZERO(), ADD(r, c.l0));  // lambda2 -> modified
        store_compare_view(P_2, c.l1);             // verify if P_2 gets correct value from P_3

        Datatype v = ADD(u1, u2);

        v = SUB(v, ADD(MULT(ADD(l0, l1), b.mv), MULT(ADD(b.l0, b.l1), mv)));
        c.mv = v;  // Trick, can be set to zero later on
        return c;
    }
    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PRE == 0
        Datatype w = receive_from_live(P_3);
#else
        Datatype w = pre_receive_from_live(P_3);
#endif

        mv = ADD(mv, w);
        store_compare_view(P_012, mv);
        mv = SET_ALL_ZERO();  // restore actual value of c.mv

        Tetrad0_Share p;
        p.l0 = SET_ALL_ZERO();  // lambda1
        p.l1 = SET_ALL_ZERO();  // lambda2
        p.mv = receive_from_live(P_2);
        store_compare_view(P_1, p.mv);

        // o = p + q
        mv = ADD(mv, p.mv);
        l0 = ADD(l0, p.l0);
        l1 = ADD(l1, p.l1);

        /* std::cout << "mv: " << mv << std::endl; */
        /* std::cout << "l1: " << l0 << std::endl; */
        /* std::cout << "l2: " << l1 << std::endl; */
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Tetrad0_Share prepare_dot(const Tetrad0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Tetrad0_Share c;
        c.mv = ADD(ADD(MULT(l0, b.l1), MULT(l1, b.l0)), MULT(l0, l0));  // y3ab
        c.l0 = SUB(SET_ALL_ZERO(), ADD(MULT(ADD(l0, l1), b.mv), MULT(ADD(b.l0, b.l1), mv)));
        return c;
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        Datatype u1 = getRandomVal(P_013);
        Datatype u2 = getRandomVal(P_023);
        Datatype u1u2 = ADD(u1, u2);
        Datatype r = SUB(mv, u1u2);
        Tetrad0_Share q;

        Datatype v = SUB(u1u2, l0);

        // q:
        mv = SET_ALL_ZERO();
        l1 = getRandomVal(P_013);              // lambda2
        l0 = SUB(SET_ALL_ZERO(), ADD(r, l1));  // lambda1
        store_compare_view(P_2, l0);           // verify if P_2 gets correct value from P_3

        mv = v;  // Trick, can be set to zero later on
    }

    void prepare_reveal_to_all() const { send_to_live(P_3, mv); }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 0
        Datatype l3 = receive_from_live(P_3);
#else
        Datatype l3 = pre_receive_from_live(P_3);
#endif
        Datatype result = SUB(mv, l3);
        result = SUB(result, l0);
        result = SUB(result, l1);
        store_compare_view(P_1, l3);  // verify own value

        store_compare_view(P_1, l1);  // verify others, l2 for P1
        store_compare_view(P_2, l0);  // l1 for P2
        return result;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        // l0 -> lamda1, l1 -> lambda2
        if constexpr (id == PSELF)
        {
            mv = val;
            l0 = getRandomVal(P_013);  // l1
            l1 = getRandomVal(P_023);  // l2
            Datatype l3 = SET_ALL_ZERO();
            mv = ADD(ADD(mv, l3), ADD(l0, l1));
            send_to_live(P_1, mv);
            send_to_live(P_2, mv);
        }
        else if constexpr (id == P_1)
        {
            l0 = getRandomVal(P_013);
            l1 = SET_ALL_ZERO();
        }
        else if constexpr (id == P_2)
        {
            l0 = SET_ALL_ZERO();
            l1 = getRandomVal(P_023);
        }
        else if constexpr (id == P_3)
        {
            l0 = getRandomVal(P_013);
            l1 = getRandomVal(P_023);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {
            mv = receive_from_live(id);

            if constexpr (id != P_1)
                store_compare_view(P_1, mv);
            if constexpr (id != P_2)
                store_compare_view(P_2, mv);
        }
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate()
    {
        /* #if PRE == 0 */
        communicate_live();
        /* #endif */
    }
};
