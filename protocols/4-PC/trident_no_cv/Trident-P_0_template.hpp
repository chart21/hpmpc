#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE Trident0_Share
template <typename Datatype>
class Trident0_Share
{
  private:
    Datatype l1;
    Datatype l2;
    Datatype l3;
  public:
    Trident0_Share() {}

    Trident0_Share(Datatype a, Datatype b, Datatype c)
    {
        l1 = a;
        l2 = b;
        l3 = c;
    }

    Trident0_Share public_val(Datatype a) { return Trident0_Share(SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO()); }

    template <typename func_mul>
    Trident0_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return Trident0_Share(MULT(l1, b), MULT(l2, b), MULT(l3, b));
    }

    Trident0_Share Not() const { return Trident0_Share(l1, l2, l3); }

    template <typename func_add>
    Trident0_Share Add(Trident0_Share b, func_add ADD) const
    {
        return Trident0_Share(ADD(l1, b.l1), ADD(l2, b.l2), ADD(l3, b.l3));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Trident0_Share prepare_mult(Trident0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        const Datatype k1 = getRandomVal(P_013);
        const Datatype k2 = getRandomVal(P_012);
        const Datatype k3 = getRandomVal(P_023);
        const Datatype za = SUB(k2,k1); //P0,1
        const Datatype zb = SUB(k3,k2); //P0,2
        const Datatype ztau = SUB(k1,k3); //P0,3
        Trident0_Share c;
        Datatype yx2 = ADD(ADD(MULT(l2, b.l2), MULT(l2, b.l3)), MULT(l3, b.l2));  
        Datatype yx3 = ADD(ADD(MULT(l3, b.l3), MULT(l3, b.l1)), MULT(l1, b.l3));
        Datatype yx1 = ADD(ADD(MULT(l1, b.l1), MULT(l1, b.l2)), MULT(l2, b.l1));
        #if PRE == 1
        pre_send_to_live(P_1, yx3);
        pre_send_to_live(P_2, yx2);
        pre_send_to_live(P_3, yx1);
        #else
        send_to_live(P_1, yx3);
        send_to_live(P_2, yx2);
        send_to_live(P_3, yx1);
        #endif
        c.l1 = getRandomVal(P_023);
        c.l2 = getRandomVal(P_013);
        c.l3 = getRandomVal(P_012);
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_mult2(Trident123_Share a, Trident123_Share b, func_add ADD, func_sub SUB, func_mul MULT)
    {
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    void prepare_reveal_to_all() const { 
        #if PRE == 1
        pre_send_to_live(P_1, l1);
        pre_send_to_live(P_2, l2);
        pre_send_to_live(P_3, l3);
        #else
        send_to_live(P_1, l1);
        send_to_live(P_2, l2);
        send_to_live(P_3, l3);
        #endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        #if PRE == 0
        Datatype mv = ADD(ADD(ADD(l1, l2), l3), receive_from_live(P_1));
        Datatype mv2 = ADD(ADD(ADD(l1, l2), l3), receive_from_live(P_2));
        if(! dat_equal(mv, mv2))
        {
            printf("P%i: Compareviews failed! \n", PARTY);
        }
        #else
        store_output_share(l1);
        store_output_share(l2);
        store_output_share(l3);
        #endif
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_1)
            l1 = SET_ALL_ZERO();
        else
            l1 = getRandomVal(P_023);
        if constexpr (id == P_2)
            l2 = SET_ALL_ZERO();
        else
            l2 = getRandomVal(P_013);
        if constexpr (id == P_3)
            l3 = SET_ALL_ZERO();
        else
            l3 = getRandomVal(P_012);
        if constexpr (id == PSELF)
        {
            Datatype mv = ADD(ADD(ADD(l1, l2), l3), val);
            #if PRE == 1
            store_output_share(l1);
            store_output_share(l2);
            store_output_share(l3);
            // pre_send_to_live(P_1, mv);
            // pre_send_to_live(P_2, mv);
            // pre_send_to_live(P_3, mv);
            #else
            send_to_live(P_1, mv);
            send_to_live(P_2, mv);
            send_to_live(P_3, mv);
            #endif
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
    }
    
    template <int id, typename func_add, typename func_sub>
    void complete_receive_from_2(func_add ADD, func_sub SUB)
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
