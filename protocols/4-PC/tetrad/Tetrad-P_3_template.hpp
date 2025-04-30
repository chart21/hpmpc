#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Tetrad3_Share
{
  private:
    Datatype l1;
    Datatype l2;
    Datatype l3;
    Datatype storage;  // used for storing results needed later
  public:
    Tetrad3_Share() {}

    Tetrad3_Share(Datatype a, Datatype b, Datatype c)
    {
        l1 = a;
        l2 = b;
        l3 = c;
    }

    Tetrad3_Share public_val(Datatype a) { return Tetrad3_Share(SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()); }

    Tetrad3_Share Not() const { return Tetrad3_Share(l1, l2, l3); }

    template <typename func_add>
    Tetrad3_Share Add(Tetrad3_Share b, func_add ADD) const
    {
        return Tetrad3_Share(ADD(l1, b.l1), ADD(l2, b.l2), ADD(l3, b.l3));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Tetrad3_Share prepare_mult(Tetrad3_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Tetrad3_Share c;
        Datatype y1ab = ADD(ADD(MULT(l1, b.l3), MULT(l3, b.l1)), MULT(l3, l3));
        Datatype y2ab = ADD(ADD(MULT(l2, b.l3), MULT(l3, b.l2)), MULT(l2, l2));
        Datatype y3ab = ADD(ADD(MULT(l1, b.l2), MULT(l2, b.l1)), MULT(l1, l1));
        Datatype u1 = getRandomVal(P_013);
        Datatype u2 = getRandomVal(P_023);
        Datatype r = SUB(y3ab, ADD(u1, u2));
        Tetrad3_Share q;

        Datatype s = getRandomVal(P_123);
        Datatype w = ADD(s, ADD(y1ab, y2ab));
#if PRE == 1
        pre_send_to_live(P_0, w);
#else
        send_to_live(P_0, w);
#endif
        // q:
        c.l1 = getRandomVal(P_013);                // modified since we use P2 instad of P1
        c.l2 = SUB(SET_ALL_ZERO(), ADD(r, c.l1));  // lambda1 -> lamda2, modified
        c.l3 = SET_ALL_ZERO();                     // lambda3
#if PRE == 1
        pre_send_to_live(P_2, c.l2);
#else
        send_to_live(P_2, c.l2);
#endif
        return c;
    }
    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Tetrad3_Share p;
        p.l1 = SET_ALL_ZERO();       // lambda1
        p.l2 = SET_ALL_ZERO();       // lambda2
        p.l3 = getRandomVal(P_123);  // lambda3

        // o = p + q
        l1 = ADD(l1, p.l1);
        l2 = ADD(l2, p.l2);
        l3 = ADD(l3, p.l3);
        /* std::cout << "l1: " << l1 << std::endl; */
        /* std::cout << "l2: " << l2 << std::endl; */
        /* std::cout << "l3: " << l3 << std::endl; */
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Tetrad3_Share prepare_dot(const Tetrad3_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Tetrad3_Share c;
        Datatype y1ab = ADD(ADD(MULT(l1, b.l3), MULT(l3, b.l1)), MULT(l3, l3));
        Datatype y2ab = ADD(ADD(MULT(l2, b.l3), MULT(l3, b.l2)), MULT(l2, l2));
        Datatype y3ab = ADD(ADD(MULT(l1, b.l2), MULT(l2, b.l1)), MULT(l1, l1));
        c.l1 = y1ab;
        c.l2 = y2ab;
        c.l3 = y3ab;
        return c;
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        Datatype u1 = getRandomVal(P_013);
        Datatype u2 = getRandomVal(P_023);
        Datatype r = SUB(l3, ADD(u1, u2));
        Tetrad3_Share q;

        Datatype s = getRandomVal(P_123);
        Datatype w = ADD(s, ADD(l1, l2));
#if PRE == 1
        pre_send_to_live(P_0, w);
#else
        send_to_live(P_0, w);
#endif
        // q:
        l2 = getRandomVal(P_013);              // lambda2
        l1 = SUB(SET_ALL_ZERO(), ADD(r, l2));  // lambda1
        l3 = SET_ALL_ZERO();                   // lambda3
#if PRE == 1
        pre_send_to_live(P_2, l2);
#else
        send_to_live(P_2, l2);
#endif
    }

    void prepare_reveal_to_all() const
    {
#if PRE == 1
        pre_send_to_live(P_2, l1);
        pre_send_to_live(P_1, l2);
        pre_send_to_live(P_0, l3);
#else
        send_to_live(P_2, l1);
        send_to_live(P_1, l2);
        send_to_live(P_0, l3);
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 0
        // receive lambda3 from P_3
        Datatype mv = receive_from_live(P_0);
        store_compare_view(P_1, mv);  // verify own value
        Datatype result = SUB(mv, l3);
        result = SUB(result, l1);
        result = SUB(result, l2);
#elif PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share(l1);
        store_output_share(l2);
        store_output_share(l3);
        Datatype result = SET_ALL_ZERO();
#endif
        return result;
    }

    template <typename func_mul>
    Tetrad3_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return Tetrad3_Share(MULT(l1, b), MULT(l2, b), MULT(l3, b));
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            Datatype mv = val;
            l1 = getRandomVal(P_013);  // l1
            l2 = getRandomVal(P_023);
            l3 = getRandomVal(P_123);
            mv = ADD(ADD(mv, l3), ADD(l1, l2));
#if PRE == 1
            pre_send_to_live(P_0, mv);
            pre_send_to_live(P_1, mv);
            pre_send_to_live(P_2, mv);
#else
            send_to_live(P_0, mv);
            send_to_live(P_1, mv);
            send_to_live(P_2, mv);
#endif
        }
        else if constexpr (id == P_1)
        {
            l1 = getRandomVal(P_013);
            l2 = SET_ALL_ZERO();
            l3 = getRandomVal(P_123);
        }
        else if constexpr (id == P_2)
        {
            l1 = SET_ALL_ZERO();
            l2 = getRandomVal(P_023);
            l3 = getRandomVal(P_123);
        }
        else if constexpr (id == P_0)
        {
            l1 = getRandomVal(P_013);
            l2 = getRandomVal(P_023);
            l3 = SET_ALL_ZERO();
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
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
