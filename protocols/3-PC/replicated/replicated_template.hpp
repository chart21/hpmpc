#pragma once
#include "../../generic_share.hpp"
#include <functional>
template <typename Datatype>
class Replicated_Share
{
  private:
    Datatype x;
    Datatype a;

  public:
    Replicated_Share() {}
    Replicated_Share(Datatype x) { this->x = x; }
    Replicated_Share(Datatype x, Datatype a)
    {
        this->x = x;
        this->a = a;
    }

    template <typename func_add, typename func_sub>
    Replicated_Share share_SRNG(Datatype a, func_add ADD, func_sub SUB)
    {
        Replicated_Share s[3];
        s[PPREV].x = getRandomVal(PPREV);
        s[PNEXT].x = getRandomVal(PNEXT);
        s[2].x = SUB(SET_ALL_ZERO(), ADD(s[PPREV].x, s[PNEXT].x));  // r_{i-1} - r_{i+1}

        s[PPREV].a = SUB(s[PNEXT].x, a);  // x_{i+1} - a
        s[PNEXT].a = SUB(s[2].x, a);      // x_i - a
        s[2].a = SUB(s[PPREV].x, a);      // x_{i-1} - a

        send_to_live(PPREV, s[PPREV].a);
        send_to_live(PNEXT, s[PNEXT].a);

        return s[2];
    }

    template <typename func_mul>
    Replicated_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return Replicated_Share(MULT(x, b), MULT(a, b));
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            *this = share_SRNG(val, ADD, SUB);
        }
        else
            x = getRandomVal(id);
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
            a = receive_from_live(id);
    }

    template <typename func_add>
    Replicated_Share Add(Replicated_Share b, func_add ADD) const
    {
        return Replicated_Share(x, ADD(a, b.a));
    }

    Replicated_Share public_val(Datatype a) { return Replicated_Share(SET_ALL_ZERO(), a); }

    Replicated_Share Not() const { return Replicated_Share(x, NOT(a)); }

    /* void reshare(Datatype a, Datatype u[]) */
    /* { */
    /* u[PPREV] = getRandomVal(PPREV); */
    /* u[PNEXT] = getRandomVal(PNEXT); */
    /* u[2] = SUB(u[PPREV],u[PNEXT]); */
    /* u[2] = SUB(u[2],a); */
    /* } */

    // division by 3 by multiplication with inverse
    template <typename func_mul>
    Datatype div3(Datatype a, func_mul MULT) const
    {
// Divide by 3
#if BITLENGTH == 16
        const uint16_t INVERSE_3 = 0xAAAB;
#elif BITLENGTH == 32
        const uint32_t INVERSE_3 = 0xAAAAAAAB;
#elif BITLENGTH == 64
        const uint64_t INVERSE_3 = 0xAAAAAAAAAAAAAAAB;
#endif
        return MULT(a, PROMOTE(INVERSE_3));
    }

    // overload for boolean circuits
    Datatype div3(Datatype a, std::bit_and<Datatype>) const { return a; }

    template <typename func_add, typename func_sub, typename func_mul>
    Replicated_Share prepare_mult(Replicated_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Replicated_Share c;
        Datatype corr = SUB(getRandomVal(PPREV), getRandomVal(PNEXT));
        Datatype r = ADD(SUB(MULT(a, b.a), MULT(x, b.x)), corr);  // a_i b_i - x_1 y_1 + corr
        r = div3(r, MULT);
        c.a = r;  // used to access value in complete mult
        send_to_live(PNEXT, r);
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Replicated_Share prepare_dot(const Replicated_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Replicated_Share c;
        c.a = SUB(MULT(x, b.x), MULT(a, b.a));
        return c;
    }

    template <typename func_sub, typename func_mul>
    void mask_and_send_dot(func_sub SUB, func_mul MULT)
    {
        a = SUB(a, getRandomVal(PPREV));
        a = SUB(a, getRandomVal(PNEXT));
        a = div3(a, MULT);
        send_to_live(PNEXT, a);
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        auto r_prev = receive_from_live(PPREV);
        x = SUB(r_prev, a);                                    // z_i = r_{i-1} - r_i
        a = SUB(SET_ALL_ZERO(), ADD(ADD(r_prev, r_prev), a));  // c_i = -2 r_{i-1} - r_i */
    }

    void prepare_reveal_to_all() const { send_to_live(PNEXT, x); }

    /* void prepare_reveal_to(Datatype a, int id) */
    /* { */
    /*     if(PSELF != id) */
    /*     { */
    /*         send_to_live(id, a); */
    /* } */
    /* } */

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        Datatype result;
        result = SUB(SET_ALL_ZERO(), SUB(a, receive_from_live(PPREV)));
        return result;
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }
};
