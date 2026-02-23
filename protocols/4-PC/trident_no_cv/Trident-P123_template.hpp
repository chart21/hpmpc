#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Trident123_Share
{
  private:
    Datatype mv;
    Datatype li; //l_{v+1}
    Datatype lj; //l_{v-1}
  public:
    Trident123_Share() {}

    Trident123_Share(Datatype a, Datatype b, Datatype c)
    {
        mv = a;
        li = b;
        lj = c;
    }

    Trident123_Share public_val(Datatype a) { return Trident123_Share(a, SET_ALL_ZERO(), SET_ALL_ZERO()); }

    Trident123_Share Not() const { return Trident123_Share(NOT(mv), li, lj); }

    template <typename func_add>
    Trident123_Share Add(Trident123_Share b, func_add ADD) const
    {
        return Trident123_Share(ADD(mv, b.mv), ADD(li, b.li), ADD(lj, b.lj));
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    Trident123_Share prepare_mult(Trident123_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Datatype yxi = ADD(ADD(MULT(li, b.li), MULT(li, b.lj)), MULT(lj, b.li));
        send_to_live(PPREV, yxi);
        Trident123_Share c;
        #if PARTY == 1
        c.li = getRandomVal(P_013);
        c.lj = getRandomVal(P_012);
        #elif PARTY == 2
        c.li = getRandomVal(P_012);
        c.lj = getRandomVal(P_023);
        #elif PARTY == 3
        c.li = getRandomVal(P_013);
        c.lj = getRandomVal(P_023);
        #endif
        c.mv = yxi;
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_mult2(Trident123_Share a, Trident123_Share b, func_add ADD, func_sub SUB, func_mul MULT) 
    {
        Datatype yxj = receive_from_live(PNEXT_EX_0);
        Datatype mzi = SUB(ADD(c.mv,c.li), ADD(MULT(b.li, a.mv), MULT(a.li, b.mv)));
        Datatype mzj = SUB(ADD(c.lj, yxj), ADD(MULT(b.lj, a.mv), MULT(a.lj, b.mv)));
        send_to_live(PNEXT_EX_0, mzi);
        send_to_live(PPREV_EX_0, mzj);
        mv = ADD(mzi, mzj);
    }
    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Datatype mz = receive_from_live(PNEXT_EX_0);
        Datatype mzp = receive_from_live(PPREV_EX_0);
        if(! dat_equal(mz, mzp))
        {
            printf("P%i: Compareviews failed! \n", PARTY);
        }
        c.mv = ADD(c.mv, mz);
    }


    void prepare_reveal_to_all() const
    {
        #if PARTY == 1 || PARTY == 2
        send_to_live(P_0, mv);
        #endif
        send_to_live(PNEXT_EX_0, li);
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 1
        Datatype lv = pre_receive_from_live(P_0);
#else
        Datatype lv = receive_from_live(P_0);
#endif
        Datatype lvp = receive_from_live(PPREV_EX_0);
        if(! dat_equal(lv, lvp))
        {
            printf("P%i: Compareviews failed! \n", PARTY);
        }
        Datatype result = SUB(mv, ADD(ADD(li, lj), lv));
        return result;
    }

    template <typename func_mul>
    Trident123_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return Trident123_Share(MULT(mv, b), MULT(li, b), MULT(lj, b));
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr ((id == P_1 && PSELF == P_3) || (id == P_2 && PSELF == P_1) || (id == P_3 && PSELF == P_2))
            li = SET_ALL_ZERO();
        else
        {
            #if PARTY == 1
            li = getRandomVal(P_012);
            #elif PARTY == 2
            li = getRandomVal(P_023);
            #elif PARTY == 3
            li = getRandomVal(P_013);
            #endif

        }
        if constexpr ((id == P_1 && PSELF == P_2) || (id == P_2 && PSELF == P_3) || (id == P_3 && PSELF == P_1))
            lj = SET_ALL_ZERO();
        else
        {
            #if PARTY == 1
            lj = getRandomVal(P_013);
            #elif PARTY == 2
            lj = getRandomVal(P_012);
            #elif PARTY == 3
            lj = getRandomVal(P_023);
            #endif

        }
        if constexpr (id == PSELF)
        {
            mv = ADD(ADD(mv, li), lj);
            #if P_1 != PSELF
            send_to_live(P_0, mv);
            #endif
            #if P_2 != PSELF
            send_to_live(P_1, mv);
            #endif
            #if P_3 != PSELF
            send_to_live(P_2, mv);
            #endif
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {
            mv = receive_from_live(id);
            if(constexpr (id != PPREV_EX_0))
            {
                send_to_live(PPREV_EX_0, mv);
            }
            if(constexpr (id != PNEXT_EX_0))
            {
                send_to_live(PNEXT_EX_0, mv);
            }
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from_2(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PPREV_EX_0)
        {
            Datatype mvp = receive_from_live(PPREV_EX_0);
            if(! dat_equal(mv, mvp))
            {
                printf("P%i: Compareviews failed! \n", PARTY);
            }
        }
        if constexpr (id != PNEXT_EX_0)
        {
            Datatype mvp = receive_from_live(PNEXT_EX_0);
            if(! dat_equal(mv, mvp))
            {
                printf("P%i: Compareviews failed! \n", PARTY);
            }
        }
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate()
    {
        communicate_live();
    }
};
