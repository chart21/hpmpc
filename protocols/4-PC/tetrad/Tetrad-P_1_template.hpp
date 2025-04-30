#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Tetrad1_Share
{
  private:
    Datatype mv;
    Datatype l0;
    Datatype l1;
    Datatype storage;  // used for storing results needed later
  public:
    Tetrad1_Share() {}

    Tetrad1_Share(Datatype a, Datatype b, Datatype c)
    {
        mv = a;
        l0 = b;
        l1 = c;
    }

    Tetrad1_Share public_val(Datatype a) { return Tetrad1_Share(a, SET_ALL_ZERO(), SET_ALL_ZERO()); }

    Tetrad1_Share Not() const { return Tetrad1_Share(NOT(mv), l0, l1); }

    template <typename func_add>
    Tetrad1_Share Add(Tetrad1_Share b, func_add ADD) const
    {
        return Tetrad1_Share(ADD(mv, b.mv), ADD(l0, b.l0), ADD(l1, b.l1));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Tetrad1_Share prepare_mult(Tetrad1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        // l0 -> lambda1, l1 -> lambda3
        Tetrad1_Share c;
        Datatype y1ab = ADD(ADD(MULT(l0, b.l1), MULT(l1, b.l0)), MULT(l1, l1));
        Datatype u1 = getRandomVal(P_013);

        Datatype s = getRandomVal(P_123);

        // q:
        c.mv = SET_ALL_ZERO();
        c.l0 = getRandomVal(P_013);  // lambda1
        c.l1 = SET_ALL_ZERO();       // lambda3

        Datatype y1 = SUB(ADD(y1ab, u1), ADD(MULT(l0, b.mv), MULT(mv, b.l0)));
        Datatype y3 = SUB(SET_ALL_ZERO(), ADD(MULT(l1, b.mv), MULT(mv, b.l1)));
        send_to_live(P_2, y1);
        Datatype z_r = ADD(ADD(y1, y3), MULT(mv, b.mv));

        // Trick to store values neede later
        c.storage = c.l0;
        c.l0 = y1;
        c.l1 = s;
        c.mv = z_r;
        return c;
    }
    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Datatype y2 = receive_from_live(P_2);
        Datatype v = ADD(ADD(l0, l1), y2);  // y1 + y2 + s for verification
        store_compare_view(P_012, v);
        mv = ADD(mv, y2);

        // p:
        /* Datatype pl1 = SET_ALL_ZERO(); // known by all */
        /* Datatype pl2 = SET_ALL_ZERO(); // known by all */
        Datatype pl3 = getRandomVal(P_123);  // hide from P_0
        Datatype pmv = ADD(mv, pl3);         //
        store_compare_view(P_0, pmv);

        // o = p + q
        mv = pmv;
        l0 = storage;  // lambda1
        l1 = pl3;      // lambda3
        /* std::cout << "mv: " << mv << std::endl; */
        /* std::cout << "l1: " << l0 << std::endl; */
        /* std::cout << "l3: " << l1 << std::endl; */
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Tetrad1_Share prepare_dot(const Tetrad1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Tetrad1_Share c;
        Datatype y1ab = ADD(ADD(MULT(l0, b.l1), MULT(l1, b.l0)), MULT(l1, l1));
        Datatype y3 = SUB(y1ab, ADD(MULT(l0, b.mv), MULT(mv, b.l0)));
        c.l0 = SUB(SET_ALL_ZERO(), ADD(MULT(l1, b.mv), MULT(mv, b.l1)));  // y1
        c.l1 = ADD(ADD(c.l0, y3), MULT(mv, b.mv));                        // z_r
        return c;
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        Datatype y1 = l0;
        Datatype z_r = l1;
        Tetrad1_Share c;
        Datatype u1 = getRandomVal(P_013);

        // q:
        Datatype cmv = SET_ALL_ZERO();
        Datatype cl0 = getRandomVal(P_013);  // lambda1
        Datatype cl1 = SET_ALL_ZERO();       // lambda3

        Datatype s = getRandomVal(P_123);

        y1 = ADD(u1, y1);
        send_to_live(P_2, y1);
        z_r = ADD(u1, z_r);
        // Trick to store values neede later
        storage = cl0;
        l0 = y1;
        l1 = s;
        mv = z_r;
    }

    void prepare_reveal_to_all() const {}

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {

        // receive lambda2 from P_3
        store_compare_view(P_0, l1);  // help P_0 verifying
        store_compare_view(P_3, mv);  // help P_3 verifying
#if PRE == 0
        Datatype lambda2 = receive_from_live(P_3);
#else
        Datatype lambda2 = pre_receive_from_live(P_3);
#endif
        store_compare_view(P_0, lambda2);  // get help from P_0 to veriy
        Datatype result = SUB(mv, lambda2);
        result = SUB(result, l0);
        result = SUB(result, l1);
        return result;
    }

    template <typename func_mul>
    Tetrad1_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return Tetrad1_Share(MULT(mv, b), MULT(l0, b), MULT(l1, b));
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            mv = val;
            l0 = getRandomVal(P_013);  // l1
            l1 = getRandomVal(P_123);  // l3
            Datatype l2 = SET_ALL_ZERO();
            mv = ADD(ADD(mv, l2), ADD(l0, l1));
            send_to_live(P_0, mv);
            send_to_live(P_2, mv);
        }
        else if constexpr (id == P_0)
        {
            l0 = getRandomVal(P_013);
            l1 = SET_ALL_ZERO();
        }
        else if constexpr (id == P_2)
        {
            l0 = SET_ALL_ZERO();
            l1 = getRandomVal(P_123);
        }
        else if constexpr (id == P_3)
        {
            l0 = getRandomVal(P_013);
            l1 = getRandomVal(P_123);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {
            mv = receive_from_live(id);

            if constexpr (id != P_0)
                store_compare_view(P_0, mv);
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
