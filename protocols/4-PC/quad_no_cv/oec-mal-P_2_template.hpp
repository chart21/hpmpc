#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL2_NO_CV_Share
{
  private:
    Datatype v;
    Datatype r;
    Datatype m;

  public:
    OEC_MAL2_NO_CV_Share() {}
    OEC_MAL2_NO_CV_Share(Datatype v, Datatype r) : v(v), r(r) {}
    OEC_MAL2_NO_CV_Share(Datatype v, Datatype r, Datatype m) : v(v), r(r), m(m) {}

    static OEC_MAL2_NO_CV_Share public_val(Datatype a)
    {
        return OEC_MAL2_NO_CV_Share(a, SET_ALL_ZERO(), SET_ALL_ZERO());
    }

    OEC_MAL2_NO_CV_Share Not() const
    {
        return OEC_MAL2_NO_CV_Share(NOT(v), r, m);
    }

    template <typename func_add>
    OEC_MAL2_NO_CV_Share Add(OEC_MAL2_NO_CV_Share b, func_add ADD) const
    {
        return OEC_MAL2_NO_CV_Share(ADD(v, b.v), ADD(r, b.r), ADD(m, b.m));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_NO_CV_Share prepare_mult(OEC_MAL2_NO_CV_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL2_NO_CV_Share c;
        c.r = getRandomVal(P_023);
        Datatype r234_2 = getRandomVal(P_123);  // Probably sufficient to only generate with P_3 ->
                                                // Probably not because of verification
#if PRE == 1
        Datatype o1 = pre_receive_from_live(P_3);
#else
        Datatype o1 = receive_from_live(P_3);
#endif
        check_eqs(o1, receive_from_live(P_0));
        c.v = ADD(SUB(MULT(v, b.r), o1), MULT(b.v, r));
        send_to_live(P_1, c.v);
        Datatype a1b1 = MULT(v, b.v);
        send_to_live(P_0, ADD(a1b1, r234_2));
        c.v = SUB(a1b1, c.v);
        return c;
    }


    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Datatype m2 = receive_from_live(P_1);
        v = SUB(v, m2);
        m = getRandomVal(P_123);
        send_to_live(P_0, ADD(v, m));
    }

    template <typename func_add, typename func_sub>
    void complete_mult2(func_add ADD, func_sub SUB)
    {
        // Datatype mv1 = receive_from_live(P_0);
        // Datatype local = ADD(v, m);
        // check_eqs(mv1, local);
    }

    void prepare_reveal_to_all() const {
        send_to_live(P_0, m);
        send_to_live(P_3, v);
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        Datatype r0 = receive_from_live(P_0);
        Datatype result = SUB(v, r0);
        #if PRE == 1
        Datatype r0p = pre_receive_from_live(P_3);
        #else
        Datatype r0p = receive_from_live(P_3);
        #endif
        check_eqs(r0, r0p);
        return result;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            Datatype x_0 = getRandomVal(P_023);
            Datatype u = getRandomVal(P_123);
            r = x_0;  //  = x_2, x_1 = 0
            v = ADD(val, x_0);
            send_to_live(P_0, ADD(v, u));
            send_to_live(P_1, ADD(v, u));
            m = u;
        }
        else if constexpr (id == P_0)
        {
            r = getRandomVal(P_023);
            v = SET_ALL_ZERO();
            m = SET_ALL_ZERO();
        }
        else if constexpr (id == P_1)
        {
            r = SET_ALL_ZERO();
            v = getRandomVal(P_123);  // u
            m = v;
        }
        else if constexpr (id == P_3)
        {
            r = getRandomVal(P_023);  // x2
            v = getRandomVal(P_123);  // u
            m = v;
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {

#if PRE == 1
            Datatype val;
            if constexpr (id == P_3)
                val = pre_receive_from_live(P_3);
            else
                val = receive_from_live(id);
#else
            Datatype val = receive_from_live(id);
#endif
            if constexpr (id != P_0)
                send_to_live(P_0, val);
            if constexpr (id != P_1)
                send_to_live(P_1, val);
            v = SUB(val, v);  // convert locally to a + x_0
        }
    }
    
    template <int id, typename func_add, typename func_sub>
    void complete_receive_from2(func_add ADD, func_sub SUB)
    {
    }
    
    template <typename func_mul>
    OEC_MAL2_NO_CV_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return OEC_MAL2_NO_CV_Share(MULT(v, b), MULT(r, b), MULT(m, b));
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }

};