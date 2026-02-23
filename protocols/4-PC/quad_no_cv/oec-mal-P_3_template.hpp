#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE OEC_MAL3_Share
template <typename Datatype>
class OEC_MAL3_Share
{
  private:
    Datatype r0;
    Datatype r1;

  public:
    OEC_MAL3_Share() {}
    OEC_MAL3_Share(Datatype r0, Datatype r1) : r0(r0), r1(r1) {}
    OEC_MAL3_Share(Datatype r0) : r0(r0) {}

    static OEC_MAL3_Share public_val(Datatype a) { return OEC_MAL3_Share(SET_ALL_ZERO(), SET_ALL_ZERO()); }

    OEC_MAL3_Share Not() const { return *this; }

    template <typename func_add>
    OEC_MAL3_Share Add(OEC_MAL3_Share b, func_add ADD) const
    {
        return OEC_MAL3_Share(ADD(r0, b.r0), ADD(r1, b.r1));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_mult(OEC_MAL3_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL3_Share c;
        c.r1 = ADD(getRandomVal(P_023), getRandomVal(P_013));  // x1
        Datatype o4 = ADD(SUB(MULT(r1, SUB(b.r1, b.r0)), MULT(b.r1, r0)), getRandomVal(P_123));  // r123_2
        c.r0 = o4;
#if PRE == 1
        pre_send_to_live(P_2, o1);
#else
        send_to_live(P_2, o1);
#endif
        return c;
    }


    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Datatype rc0 = getRandomVal(P_123);  // w
        Datatype o4 = SUB(r0, rc0);
#if PRE == 1
        pre_send_to_live(P_0, o4);
#else
        send_to_live(P_0, o4);
#endif
        r0 = rc0;
    }

    void prepare_reveal_to_all() const
    {
#if PRE == 1
        pre_send_to_live(P_0, r0);
        pre_send_to_live(P_1, r1);
        pre_send_to_live(P_2, r1);
#else
        send_to_live(P_0, r0);
        send_to_live(P_1, r1);
        send_to_live(P_2, r1);
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 0
        Datatype result = SUB(receive_from_live(P_1), r1);
        check_eqs(result, SUB(receive_from_live(P_2), r1));
        return result;
#else
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share(r1);
#endif
        return r0;
#endif
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            Datatype v = val;
            Datatype x_1 = getRandomVal(P_013);
            Datatype x_2 = getRandomVal(P_023);
            Datatype u = getRandomVal(P_123);

            r0 = u;
            r1 = ADD(x_1, x_2);
            Datatype complete_masked = ADD(v, ADD(r1, r0));
#if PRE == 1
            pre_send_to_live(P_0, complete_masked);
            pre_send_to_live(P_1, complete_masked);
            pre_send_to_live(P_2, complete_masked);
#else
            send_to_live(P_0, complete_masked);
            send_to_live(P_1, complete_masked);
            send_to_live(P_2, complete_masked);
#endif
        }
        else if constexpr (id == P_0)
        {
            r0 = SET_ALL_ZERO();
            r1 = ADD(getRandomVal(P_013), getRandomVal(P_023));
        }
        else if constexpr (id == P_1)
        {
            r0 = getRandomVal(P_123);
            r1 = getRandomVal(P_013);
        }
        else if constexpr (id == P_2)
        {
            r0 = getRandomVal(P_123);
            r1 = getRandomVal(P_023);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
    }
    
    template <typename func_add, typename func_sub>
    void complete_mult2(func_add ADD, func_sub SUB)
    {
    }
    
    template <int id, typename func_add, typename func_sub>
    void complete_receive_from2(func_add ADD, func_sub SUB)
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