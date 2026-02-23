#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE OEC_MAL0_NO_CV_Share
template <typename Datatype>
class OEC_MAL0_NO_CV_Share
{
  private:
    Datatype v;
    Datatype r;

  public:
    OEC_MAL0_NO_CV_Share() {}
    OEC_MAL0_NO_CV_Share(Datatype v, Datatype r) : v(v), r(r) {}
    OEC_MAL0_NO_CV_Share(Datatype v) : v(v) {}

    static OEC_MAL0_NO_CV_Share public_val(Datatype a) { return OEC_MAL0_NO_CV_Share(a, SET_ALL_ZERO()); }

    OEC_MAL0_NO_CV_Share Not() const { return OEC_MAL0_NO_CV_Share(NOT(v), r); }



    template <typename func_add>
    OEC_MAL0_NO_CV_Share Add(OEC_MAL0_NO_CV_Share b, func_add ADD) const
    {
        return OEC_MAL0_NO_CV_Share(ADD(v, b.v), ADD(r, b.r));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL0_NO_CV_Share prepare_mult(OEC_MAL0_NO_CV_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL0_NO_CV_Share c;
        c.r = ADD(getRandomVal(P_013), getRandomVal(P_023));  // calculate c_1
        Datatype o1 = ADD(c.r, ADD(MULT(r, b.r), getRandomVal(P_013)));
        send_to_live(P_2, o1);
        c.v = ADD(MULT(v, b.r), MULT(b.v, r));
        return c;
    }


    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        Datatype o_4 = pre_receive_from_live(P_3);
#else
        Datatype o_4 = receive_from_live(P_3);
#endif

        /* Datatype m3_prime = receive_from_live(P_2); */
        v = ADD(v, o_4);

        /* c.m = XOR(c.m, o_4); */
        Datatype m3_prime = receive_from_live(P_2);
        check_eqs(m3_prime, receive_from_live(P_1));
        v = SUB(m3_prime, v);
    }
    
    template <typename func_add, typename func_sub>
    void complete_mult2(func_add ADD, func_sub SUB)
    {
        Datatype mv1 = receive_from_live(P_1);
        Datatype mv2 = receive_from_live(P_2);
        Datatype local = ADD(v, r);
        check_eqs(mv1, local);
        check_eqs(mv2, local);
    }

    void prepare_reveal_to_all() const
    {
        send_to_live(P_1, r);
        send_to_live(P_2, r);
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 1
        Datatype result = SUB(v, pre_receive_from_live(P_3));
#else
        Datatype result = SUB(v, receive_from_live(P_3));
#endif
        Datatype result2 = SUB(v, receive_from_live(P_2));
        check_eqs(result, result2);
        return result;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            v = val;
            Datatype x_1 = getRandomVal(P_013);
            Datatype x_2 = getRandomVal(P_023);
            r = ADD(x_1, x_2);

            send_to_live(P_1, ADD(v, r));
            send_to_live(P_2, ADD(v, r));
        }
        else if constexpr (id == P_1)
        {
            r = getRandomVal(P_013);  // x_0
        }
        else if constexpr (id == P_2)
        {
            r = getRandomVal(P_023);  // x_0
        }
        else if constexpr (id == P_3)
        {
            Datatype x_1 = getRandomVal(P_013);
            Datatype x_2 = getRandomVal(P_023);
            r = ADD(x_1, x_2);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {

#if PRE == 1
            if (id == P_3)
                v = pre_receive_from_live(id);
            else
                v = receive_from_live(id);
#else
            v = receive_from_live(id);
#endif

            if constexpr (id != P_1)
                check_eqs(v, receive_from_live(P_1));
            if constexpr (id != P_2)
                check_eqs(v, receive_from_live(P_2));

            v = SUB(v, r);  // convert locally to a + u
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
