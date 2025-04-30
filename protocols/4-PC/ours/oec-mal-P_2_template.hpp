#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL2_Share
{
  private:
    Datatype v;
    Datatype r;
#if PROTOCOL == 11 || FRACTIONAL > 0 || MULTI_INPUT == 1
    Datatype m;
#endif

  public:
    OEC_MAL2_Share() {}
    OEC_MAL2_Share(Datatype v, Datatype r) : v(v), r(r) {}
#if MULTI_INPUT == 1
    OEC_MAL2_Share(Datatype v, Datatype r, Datatype m) : v(v), r(r), m(m) {}
#endif
    OEC_MAL2_Share(Datatype v) : v(v) {}

    static OEC_MAL2_Share public_val(Datatype a)
    {
#if MULTI_INPUT == 1
        return OEC_MAL2_Share(a, SET_ALL_ZERO(), SET_ALL_ZERO());
#else
        return OEC_MAL2_Share(a, SET_ALL_ZERO());
#endif
    }

    OEC_MAL2_Share Not() const
    {
#if MULTI_INPUT == 1
        return OEC_MAL2_Share(NOT(v), r, m);
#else
        return OEC_MAL2_Share(NOT(v), r);
#endif
    }

    template <typename func_add>
    OEC_MAL2_Share Add(OEC_MAL2_Share b, func_add ADD) const
    {
#if MULTI_INPUT == 1
        return OEC_MAL2_Share(ADD(v, b.v), ADD(r, b.r), ADD(m, b.m));
#else
        return OEC_MAL2_Share(ADD(v, b.v), ADD(r, b.r));
#endif
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_mult(OEC_MAL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        // 1* get Random val
        OEC_MAL2_Share c;
        c.r = getRandomVal(P_023);
        /* c.r = ADD(getRandomVal(P_023), getRandomVal(P_123)); */
        /* Datatype r234 = getRandomVal(P_123); */
        /* Datatype r234_1 = */
        /*     getRandomVal(P_123); // Probably sufficient to only generate with P_3 -> */
        Datatype r234_2 = getRandomVal(P_123);  // Probably sufficient to only generate with P_3 ->
                                                // Probably not because of verification
/* c.r = getRandomVal(P_3); */
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        Datatype o1 = pre_receive_from_live(P_3);
#else
        Datatype o1 = receive_from_live(P_3);
#endif
        store_compare_view(P_0, o1);
#else
        Datatype o1 = receive_from_live(P_0);
        store_compare_view(P_3, o1);
#endif
        c.v = ADD(SUB(MULT(v, b.r), o1), MULT(b.v, r));
        send_to_live(P_1, c.v);

        /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */
        Datatype a1b1 = MULT(v, b.v);

#if PROTOCOL == 10 || PROTOCOL == 12
        /* Datatype m3_prime = XOR(XOR(r234, c.r), c.v); */
        /* Datatype m3_prime = XOR(XOR(r234, c.r), AND(XOR(a.v, a.r), XOR(b.v, b.r))); */
        /* send_to_live(P_0, ADD(r234 ,ADD(c.v,c.r))); */
        send_to_live(P_0, ADD(a1b1, r234_2));
#endif
#if PROTOCOL == 11
        c.m = ADD(c.v, r234_2);  // store m_2 + m_3 + r_234_2 to send P_0 later
#endif
        c.v = SUB(a1b1, c.v);
        /* c.m = ADD(c.m, r234); */
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_dot(const OEC_MAL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL2_Share c;
        c.r = ADD(MULT(v, b.r), MULT(b.v, r));  // a0 y_2 + b_0 x_2
        c.v = MULT(v, b.v);                     // a0b0
        return c;
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_dot(const OEC_MAL2_Share b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL2_Share c;
        if (i == 0)
        {
            c.r = MULT(r, b.v);
        }
        else if (i == 1)
        {
            c.r = MULT(v, b.r);
        }
        else if (i == 2)
        {
            c.v = MULT(v, b.v);
        }
        return c;
    }

    template <typename func_add, typename func_sub>
    void join_dots(OEC_MAL2_Share c[], func_add ADD, func_sub SUB)
    {
        v = ADD(v, c[2].v);
        r = ADD(r, ADD(c[0].r, c[1].r));
    }

    static constexpr int getNumDotProducts() { return 3; }
#endif

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        Datatype cr = getRandomVal(P_023);
        /* c.r = ADD(getRandomVal(P_023), getRandomVal(P_123)); */
        /* Datatype r234 = getRandomVal(P_123); */
        /* Datatype r234_1 = */
        /*     getRandomVal(P_123); // Probably sufficient to only generate with P_3 -> */
        Datatype r234_2 = getRandomVal(P_123);  // Probably sufficient to only generate with P_3 ->
                                                // Probably not because of verification
/* c.r = getRandomVal(P_3); */
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        Datatype o1 = pre_receive_from_live(P_3);
#else
        Datatype o1 = receive_from_live(P_3);
#endif
        store_compare_view(P_0, o1);
#else
        Datatype o1 = receive_from_live(P_0);
        store_compare_view(P_3, o1);
#endif
        r = SUB(r, o1);
        send_to_live(P_1, r);  // if one party xored with v (m20) here for multi input and gates it should cancel out

        /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */

#if PROTOCOL == 10 || PROTOCOL == 12
        send_to_live(P_0, ADD(v, r234_2));
#endif
#if PROTOCOL == 11
        m = ADD(r, r234_2);  // store m_2 + m_3 + r_234_2 to send P_0 later
#endif
        v = SUB(v, r);  // For multi input AND gates, we should not add those two -> possible hack: both parties Add v
                        // to r -> v cancels out
                        /* c.m = ADD(c.m, r234); */
        r = cr;
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Datatype m2 = receive_from_live(P_1);
        v = SUB(v, m2);
/* c.m = XOR(c.m, m2); */
/* Datatype cm = XOR(c.m, m2); */
#if PROTOCOL == 11
        send_to_live(P_0, ADD(m, m2));  // let P_0 verify m_2 XOR m_3, obtain m_2 + m_3 + r_234_2
#if MULTI_INPUT == 1
        m = getRandomVal(P_123);
        send_to_live(P_0, ADD(v, m));  // send c_0 + w to P_0
#else
        send_to_live(P_0, ADD(v, getRandomVal(P_123)));          // let P_0 obtain ab + c0 + w
#endif
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
#if MULTI_INPUT == 1
        m = getRandomVal(P_123);
        store_compare_view(P_012, ADD(v, m));  // send c_0 + w to P_0
#else
        store_compare_view(P_012, ADD(getRandomVal(P_123), v));  // compare ab + c1 + r234_1
#endif
#endif
        /* store_compare_view(P_0, c.v); */
    }

    void prepare_reveal_to_all() const {}

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        Datatype r0 = receive_from_live(P_0);
        Datatype result = SUB(v, r0);
        store_compare_view(P_123, r0);
        store_compare_view(P_0123, result);
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
#if MULTI_INPUT == 1
            m = u;
#endif
        }
        else if constexpr (id == P_0)
        {
            r = getRandomVal(P_023);
            v = SET_ALL_ZERO();
            // u = 0
#if MULTI_INPUT == 1
            m = SET_ALL_ZERO();
#endif
        }
        else if constexpr (id == P_1)
        {
            r = SET_ALL_ZERO();
            v = getRandomVal(P_123);  // u
#if MULTI_INPUT == 1
            m = v;
#endif
        }
        else if constexpr (id == P_3)
        {
            r = getRandomVal(P_023);  // x2
            v = getRandomVal(P_123);  // u
#if MULTI_INPUT == 1
            m = v;
#endif
        }
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
            prepare_receive_from<id>(get_input_live(), ADD, SUB);
        else
            prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
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
                store_compare_view(P_0, val);
            if constexpr (id != P_1)
                store_compare_view(P_1, val);
            v = SUB(val, v);  // convert locally to a + x_0
        }
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul>
    OEC_MAL2_Share mult_public(const Datatype b, func_mul MULT) const
    {
#if MULTI_INPUT == 1
        return OEC_MAL2_Share(MULT(v, b), MULT(r, b), MULT(m, b));
#else
        return OEC_MAL2_Share(MULT(v, b), MULT(r, b));
#endif
    }
    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL2_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        auto result = v;
        for (int i = 2; i <= b; i *= 2)
            result = OP_TRUNC2(result);

        OEC_MAL2_Share c(result);
#if MULTI_INPUT == 1
        c.m = getRandomVal(P_123);
        send_to_live(P_0, ADD(c.v, c.m));
#else
        send_to_live(P_0, ADD(c.v, getRandomVal(P_123)));  // compare v*b + r123 with P_0
#endif
        return c;
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL2_Share prepare_mult_public_fixed(const Datatype b,
                                             func_mul MULT,
                                             func_add ADD,
                                             func_sub SUB,
                                             func_trunc TRUNC,
                                             int fractional_bits = FRACTIONAL) const
    {
        OEC_MAL2_Share c;
        /* #if TRUNC_THEN_MULT == 1 */
        /*     c.v = MULT(TRUNC(v,fractional_bits),b); */
        /* #else */
        c.v = TRUNC(MULT(v, b), fractional_bits);

/* #endif */
#if MULTI_INPUT == 1
        c.m = getRandomVal(P_123);
        send_to_live(P_0, ADD(c.v, c.m));
#else
        send_to_live(P_0, ADD(c.v, getRandomVal(P_123)));  // compare v*b + r123 with P_0
#endif
        return c;
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL2_Share prepare_trunc_share(func_mul MULT,
                                       func_add ADD,
                                       func_sub SUB,
                                       func_trunc TRUNC,
                                       int fractional_bits = FRACTIONAL) const
    {
        OEC_MAL2_Share c;
        c.v = TRUNC(v, fractional_bits);
#if MULTI_INPUT == 1
        c.m = getRandomVal(P_123);
        send_to_live(P_0, ADD(c.v, c.m));
#else
        send_to_live(P_0, ADD(c.v, getRandomVal(P_123)));  // compare v*b + r123 with P_0
#endif
        return c;
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        r = pre_receive_from_live(P_3);
#else
        r = receive_from_live(P_3);
#endif
        store_compare_view(P_0, r);
#else
#if PRE == 1
        r = pre_receive_from_live(P_0);
#else
        r = receive_from_live(P_0);
#endif
        store_compare_view(P_3, r);
#endif
        /* v = ADD(v,val); */
    }

#if FRACTIONAL > 0

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        r = SUB(r, getRandomVal(P_023));  // a_0 y_2 + b_0 x_2 - r_0,2,3
        send_to_live(P_1, r);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        r = ADD(r, receive_from_live(P_1));  // v^1,2 = m^1 + m^2
        v = TRUNC(SUB(v, r));                // [a_0 b_0 - v^1,2]^t

#if PROTOCOL == 11
        send_to_live(P_0, ADD(r, getRandomVal(P_123)));  // send m1 + m2 + r123 to P_0
#else
        store_compare_view(P_012, ADD(r, getRandomVal(P_123)));  // v^1,2 + r_1,2,3
#endif

#if MULTI_INPUT == 1
        m = getRandomVal(P_123);       // w
        send_to_live(P_0, ADD(v, m));  // c_0 + w
#else
        send_to_live(P_0, ADD(v, getRandomVal(P_123)));          // c_0 + w
#endif

#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        r = pre_receive_from_live(P_3);  // z_2 = m0
#else
        r = receive_from_live(P_3);  // z_2 = m0
#endif
        store_compare_view(P_0, r);  // compare view of m0
#else
#if PRE == 1
        r = pre_receive_from_live(P_0);                          // z_2 = m0
#else
        r = receive_from_live(P_0);  // z_2 = m0
#endif
        store_compare_view(P_3, r);                              // compare view of m0
#endif
    }

#endif

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OEC_MAL2_Share& r_mk2,
                                 OEC_MAL2_Share& r_msb,
                                 OEC_MAL2_Share& c,
                                 OEC_MAL2_Share& c_prime,
                                 int fractional_bits = FRACTIONAL) const
    {
        Datatype c_dat_prime = trunc(v, fractional_bits);
        UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH - fractional_bits - 1)) - 1;
        Datatype mask = PROMOTE(maskValue);    // Set all elements to maskValue
        c_dat_prime = AND(c_dat_prime, mask);  // mod 2^k-m-1
        Datatype c_dat = OP_SHIFT_LOG_RIGHT<BITLENGTH - 1>(v);
        c = OEC_MAL2_Share(c_dat, SET_ALL_ZERO());
        c_prime = OEC_MAL2_Share(c_dat_prime, SET_ALL_ZERO());
#if MULTI_INPUT == 1
        r_mk2.m = SET_ALL_ZERO();
        r_msb.m = SET_ALL_ZERO();
        c.m = getRandomVal(P_123);
        c_prime.m = getRandomVal(P_123);
        send_to_live(P_0, ADD(c.v, c.m));
        send_to_live(P_0, ADD(c_prime.v, c_prime.m));
#else
        send_to_live(P_0, ADD(c_dat, getRandomVal(P_123)));
        send_to_live(P_0, ADD(c_dat_prime, getRandomVal(P_123)));
#endif
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OEC_MAL2_Share& r_mk2,
                                  OEC_MAL2_Share& r_msb,
                                  OEC_MAL2_Share& c,
                                  OEC_MAL2_Share& c_prime) const
    {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        r_mk2.r = pre_receive_from_live(P_3);
        r_msb.r = pre_receive_from_live(P_3);
#else
        r_mk2.r = receive_from_live(P_3);
        r_msb.r = receive_from_live(P_3);
#endif
        store_compare_view(P_0, r_mk2.r);
        store_compare_view(P_0, r_msb.r);
#else
#if PRE == 0
        r_mk2.r = receive_from_live(P_0);
        r_msb.r = receive_from_live(P_0);
#else
        r_mk2.r = pre_receive_from_live(P_0);
        r_msb.r = pre_receive_from_live(P_0);
#endif
        store_compare_view(P_3, r_mk2.r);
        store_compare_view(P_3, r_msb.r);
#endif
        r_mk2.v = SET_ALL_ZERO();
        r_msb.v = SET_ALL_ZERO();
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OEC_MAL2_Share prepare_trunc_exact_xmod2t(func_add ADD,
                                              func_sub SUB,
                                              func_xor XOR,
                                              func_and AND,
                                              int fractional_bits = FRACTIONAL) const
    {
        Datatype mx = v;
        // Step 1, Compute [x/2t] -> delt with public mult fixed
        // Step 2, Compute [x mod t]
        UINT_TYPE maskValue = (UINT_TYPE(1) << (fractional_bits)) - 1;
        Datatype mask = PROMOTE(maskValue);  // Set all elements to maskValue
        // Apply the mask using bitwise AND
        Datatype mxmodt = AND(mx, mask);  // mod 2^t
        // Step3, Compute [x]^B -> delt with prepareA2B
        return OEC_MAL2_Share(mxmodt, SET_ALL_ZERO());
    }

    void get_random_B2A()
    {
        v = SET_ALL_ZERO();
        r = getRandomVal(P_023);
#if MULTI_INPUT == 1
        m = SET_ALL_ZERO();
#endif
    }

    static void prepare_B2A(OEC_MAL2_Share z[], OEC_MAL2_Share random_mask[], OEC_MAL2_Share out[])
    {
        // 1. Reveal z to P_1 and P_2
        for (int i = 0; i < BITLENGTH; i++)
        {
            send_to_live(P_1, z[i].r);  // reveal z to P_2
        }
        // 2. Get random mask from P_0,3
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].v = SET_ALL_ZERO();
#if MULTI_INPUT == 1
            out[i].m = SET_ALL_ZERO();
#endif
        }
    }

    static void complete_B2A(OEC_MAL2_Share z_bool[], OEC_MAL2_Share out[])
    {
        Datatype z[BITLENGTH];
        // reconstruct z
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
            out[i].r = pre_receive_from_live(P_3);
#else
            out[i].r = receive_from_live(P_3);
#endif
            store_compare_view(P_0, out[i].r);
#else
#if PRE == 1
            out[i].r = pre_receive_from_live(P_0);
#else
            out[i].r = receive_from_live(P_0);
#endif
            store_compare_view(P_3, out[i].r);
#endif

            Datatype mask = FUNC_XOR(z_bool[i].r, receive_from_live(P_1));
            store_compare_view(P_012, mask);
            z[i] = FUNC_XOR(z_bool[i].v, mask);
        }

        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(z, temp2);
        orthogonalize_arithmetic(temp2, z);
        for (int i = 0; i < BITLENGTH; i++)
        {
            z_bool[i].r = SET_ALL_ZERO();
            z_bool[i].v = z[i];
#if MULTI_INPUT == 1
            z_bool[i].m = getRandomVal(P_123);
            send_to_live(P_0, OP_ADD(z_bool[i].v, z_bool[i].m));
#else
            send_to_live(P_0, OP_ADD(z_bool[i].v, getRandomVal(P_123)));
#endif
        }
    }
    static void complete_B2A2(OEC_MAL2_Share z_bool[], OEC_MAL2_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i] = z_bool[i].Add(out[i], OP_SUB);
        }
    }

    static void prepare_A2B_S1(int m, int k, OEC_MAL2_Share in[], OEC_MAL2_Share out[])
    {
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = in[j].v;  // a0
        }
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_arithmetic(temp, temp2);
        orthogonalize_boolean(temp2, temp);
        /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
        /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */
        for (int j = m; j < k; j++)
        {
            out[j - m].v = temp[j];
            out[j - m].r = SET_ALL_ZERO();  // set share to 0
#if MULTI_INPUT == 1
            out[j - m].m = getRandomVal(P_123);
            send_to_live(P_0, FUNC_XOR(out[j - m].v, out[j - m].m));
#else
            send_to_live(P_0, FUNC_XOR(out[j - m].v, getRandomVal(P_123)));
#endif
        }
    }

    static void prepare_A2B_S2(int m, int k, OEC_MAL2_Share in[], OEC_MAL2_Share out[])
    {
        for (int i = m; i < k; i++)
        {
            out[i - m].v = SET_ALL_ZERO();
#if MULTI_INPUT == 1
            out[i - m].m = SET_ALL_ZERO();
#endif
        }
    }

    static void complete_A2B_S1(int k, OEC_MAL2_Share out[]) {}

    static void complete_A2B_S2(int k, OEC_MAL2_Share out[])
    {
        for (int i = 0; i < k; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 0
            Datatype m0 = receive_from_live(P_0);
#else
            Datatype m0 = pre_receive_from_live(P_0);
#endif
            store_compare_view(P_3, m0);
#else
#if PRE == 0
            Datatype m0 = receive_from_live(P_3);
#else
            Datatype m0 = pre_receive_from_live(P_3);
#endif
            store_compare_view(P_0, m0);
#endif

            out[i].r = m0;
        }
    }

    void prepare_bit2a(OEC_MAL2_Share out[])
    {
        Datatype b0[BITLENGTH]{0};
        b0[BITLENGTH - 1] = v;  // convert b0 to an arithemtic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(b0, temp2);
        orthogonalize_arithmetic(temp2, b0);
        Datatype b0v[BITLENGTH]{0};
        b0v[BITLENGTH - 1] = FUNC_XOR(v, m);  // convert b0v to an arithemtic value
        unorthogonalize_boolean(b0v, temp2);
        orthogonalize_arithmetic(temp2, b0v);
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 1
            Datatype m00 = pre_receive_from_live(P_3);
            store_compare_view(P_0, m00);
#else
            Datatype m00 = receive_from_live(P_0);
            store_compare_view(P_3, m00);
#endif
#else
#if PRE == 1
            Datatype m00 = pre_receive_from_live(P_3);
#else
            Datatype m00 = receive_from_live(P_3);
#endif
            store_compare_view(P_0, m00);
#endif

            Datatype tmp = OP_SUB(m00, OP_MULT(OP_ADD(b0[i], b0[i]), m00));
            Datatype out_r = getRandomVal(P_023);
            Datatype m21 = OP_ADD(out_r, tmp);
            send_to_live(P_1, m21);  // m21

            Datatype r123 = getRandomVal(P_123);

            tmp = OP_SUB(r123, OP_MULT(OP_ADD(b0v[i], b0v[i]), r123));
            out[i].m = getRandomVal(P_123);
            send_to_live(P_0, OP_ADD(out[i].m, tmp));  // m20
            out[i].r = out_r;
            out[i].v = OP_ADD(b0[i], m21);
        }
    }

    void complete_bit2a()
    {
        Datatype m1 = receive_from_live(P_1);
        v = OP_ADD(v, m1);
        store_compare_view(P_012, OP_ADD(v, m));
    }

    void prepare_opt_bit_injection(OEC_MAL2_Share a[], OEC_MAL2_Share out[])
    {
        Datatype b0[BITLENGTH]{0};
        b0[BITLENGTH - 1] = v;  // convert b0 to an arithemtic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(b0, temp2);
        orthogonalize_arithmetic(temp2, b0);
        Datatype b0v[BITLENGTH]{0};
        b0v[BITLENGTH - 1] = FUNC_XOR(v, m);  // convert b0v to an arithemtic value
        unorthogonalize_boolean(b0v, temp2);
        orthogonalize_arithmetic(temp2, b0v);
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 1
            Datatype m00 = pre_receive_from_live(P_0);
            Datatype m01 = pre_receive_from_live(P_0);
#else
            Datatype m00 = receive_from_live(P_0);
            Datatype m01 = receive_from_live(P_0);
#endif
            store_compare_view(P_3, m00);
            store_compare_view(P_3, m01);
#else
#if PRE == 1
            Datatype m00 = pre_receive_from_live(P_3);
            Datatype m01 = pre_receive_from_live(P_3);
#else
            Datatype m00 = receive_from_live(P_3);
            Datatype m01 = receive_from_live(P_3);
#endif
            store_compare_view(P_0, m00);
            store_compare_view(P_0, m01);
#endif

            Datatype tmp = OP_SUB(OP_ADD(b0[i], b0[i]), PROMOTE(1));
            tmp = OP_MULT(tmp, OP_SUB(m01, OP_MULT(a[i].v, m00)));
            tmp = OP_SUB(tmp, OP_MULT(b0[i], a[i].r));
            Datatype out_r = getRandomVal(P_023);
            Datatype m21 = OP_ADD(out_r, tmp);
            send_to_live(P_1, m21);  // m21

            Datatype r123 = getRandomVal(P_123);
            Datatype r123_2 = getRandomVal(P_123);
            Datatype a0u = OP_ADD(a[i].v, a[i].m);  // set share to a_0 + u

            tmp = OP_SUB(OP_ADD(b0v[i], b0v[i]), PROMOTE(1));
            tmp = OP_MULT(tmp, OP_SUB(r123_2, OP_MULT(a0u, r123)));
            tmp = OP_SUB(tmp, OP_MULT(b0v[i], a[i].m));
            out[i].m = getRandomVal(P_123);
            send_to_live(P_0, OP_ADD(out[i].m, tmp));  // m20
            out[i].r = out_r;
            out[i].v = OP_ADD(OP_MULT(a[i].v, b0[i]), m21);
        }
    }

    void complete_opt_bit_injection()
    {
        Datatype m1 = receive_from_live(P_1);
        v = OP_ADD(v, m1);
        store_compare_view(P_012, OP_ADD(v, m));
    }

    void prepare_bit_injection_S1(OEC_MAL2_Share out[])
    {
        Datatype temp[BITLENGTH]{0};
        temp[BITLENGTH - 1] = v;
        /* unorthogonalize_boolean(temp, (UINT_TYPE*) temp); */
        /* orthogonalize_arithmetic((UINT_TYPE*) temp, temp); */
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(temp, temp2);
        orthogonalize_arithmetic(temp2, temp);
        for (int j = 0; j < BITLENGTH; j++)
        {
            out[j].r = SET_ALL_ZERO();  // set share to 0
            out[j].v = temp[j];
#if MULTI_INPUT == 1
            out[j].m = getRandomVal(P_123);
            send_to_live(P_0, OP_ADD(out[j].v, out[j].m));
#else
            send_to_live(P_0, OP_ADD(out[j].v, getRandomVal(P_123)));
#endif
        }
    }

    void prepare_bit_injection_S2(OEC_MAL2_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].v = SET_ALL_ZERO();
#if MULTI_INPUT == 1
            out[i].m = SET_ALL_ZERO();
#endif
        }
    }

    static void complete_bit_injection_S1(OEC_MAL2_Share out[]) {}

    static void complete_bit_injection_S2(OEC_MAL2_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 0
            Datatype m0 = receive_from_live(P_0);
#else
            Datatype m0 = pre_receive_from_live(P_0);
#endif
            store_compare_view(P_3, m0);
#else
#if PRE == 0
            Datatype m0 = receive_from_live(P_3);
#else
            Datatype m0 = pre_receive_from_live(P_3);
#endif
            store_compare_view(P_0, m0);
#endif
            out[i].r = OP_SUB(SET_ALL_ZERO(), m0);
        }
    }

#if MULTI_INPUT == 1

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_dot3(const OEC_MAL2_Share b,
                                const OEC_MAL2_Share c,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        Datatype mxy = pre_receive_from_live(P_3);
        Datatype mxz = pre_receive_from_live(P_3);
        Datatype myz = pre_receive_from_live(P_3);
#else
        Datatype mxy = receive_from_live(P_3);
        Datatype mxz = receive_from_live(P_3);
        Datatype myz = receive_from_live(P_3);
#endif
        store_compare_view(P_0, mxy);
        store_compare_view(P_0, mxz);
        store_compare_view(P_0, myz);
#else
        Datatype mxy = receive_from_live(P_0);
        Datatype mxz = receive_from_live(P_0);
        Datatype myz = receive_from_live(P_0);
        store_compare_view(P_3, mxy);
        store_compare_view(P_3, mxz);
        store_compare_view(P_3, myz);
#endif
        Datatype sxy = ADD(mxy, ADD(MULT(r, b.m), MULT(m, b.r)));
        Datatype sxz = ADD(mxz, ADD(MULT(r, c.m), MULT(m, c.r)));
        Datatype syz = ADD(myz, ADD(MULT(b.r, c.m), MULT(b.m, c.r)));
        /* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), MULT(MULT(m,b.m),c.r)); */
        /* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, */
        /*                     MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), */
        /*         ADD( MULT(MULT(m,b.m),c.r), ADD(MULT(MULT(m,c.m),b.r),MULT(MULT(m,b.m),c.r)))); */
        /* Datatype sxyz = */
        /* ADD(mxyz, */
        /* ADD( */
        /*     ADD(ADD(MULT(mxy,c.m),MULT(mxz,b.m)),MULT(myz,m)), */
        /*     ADD(ADD(MULT(MULT(m,b.m),c.r),MULT(MULT(m,c.m),b.r)),MULT(MULT(b.m,c.m),r)) */
        /*     )); */
        Datatype sxyz = ADD(ADD(MULT(m, (ADD(myz, MULT(b.r, c.m)))), MULT(b.m, (ADD(mxz, MULT(c.r, m))))),
                            MULT(c.m, (ADD(mxy, MULT(r, b.m)))));
        Datatype a0 = ADD(v, m);
        Datatype b0 = ADD(b.v, b.m);
        Datatype c0 = ADD(c.v, c.m);
        Datatype rxy = getRandomVal(P_123);
        Datatype rxz = getRandomVal(P_123);
        Datatype ryz = getRandomVal(P_123);
        OEC_MAL2_Share d;
        d.r = SUB(ADD(ADD(MULT(a0, ADD(syz, MULT(b0, SUB(c0, c.r)))), (MULT(b0, SUB(sxz, MULT(c0, r))))),
                      MULT(c0, SUB(sxy, MULT(a0, b.r)))),
                  sxyz);  // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
        d.v = ADD(ADD(MULT(a0, SUB(ryz, MULT(b0, c.m))), (MULT(b0, SUB(rxz, MULT(c0, m))))),
                  MULT(c0, SUB(rxy, MULT(a0, b.m))));  // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
        /* d.r = ADD(d.r, d.v); // hack for mask_and_send_dot */
        d.r = SUB(SET_ALL_ZERO(), d.r);  // hack for mask_and_send_dot

        return d;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_mult3(const OEC_MAL2_Share b,
                                 const OEC_MAL2_Share c,
                                 func_add ADD,
                                 func_sub SUB,
                                 func_mul MULT) const
    {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        Datatype mxy = pre_receive_from_live(P_3);
        Datatype mxz = pre_receive_from_live(P_3);
        Datatype myz = pre_receive_from_live(P_3);
        Datatype mxyz = pre_receive_from_live(P_3);
#else
        Datatype mxy = receive_from_live(P_3);
        Datatype mxz = receive_from_live(P_3);
        Datatype myz = receive_from_live(P_3);
        Datatype mxyz = receive_from_live(P_3);
#endif
        store_compare_view(P_0, mxy);
        store_compare_view(P_0, mxz);
        store_compare_view(P_0, myz);
        store_compare_view(P_0, mxyz);
#else
        Datatype mxy = receive_from_live(P_0);
        Datatype mxz = receive_from_live(P_0);
        Datatype myz = receive_from_live(P_0);
        Datatype mxyz = receive_from_live(P_0);
        store_compare_view(P_3, mxy);
        store_compare_view(P_3, mxz);
        store_compare_view(P_3, myz);
        store_compare_view(P_3, mxyz);
#endif
        Datatype sxy = ADD(mxy, ADD(MULT(r, b.m), MULT(m, b.r)));
        Datatype sxz = ADD(mxz, ADD(MULT(r, c.m), MULT(m, c.r)));
        Datatype syz = ADD(myz, ADD(MULT(b.r, c.m), MULT(b.m, c.r)));
        /* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), MULT(MULT(m,b.m),c.r)); */
        /* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, */
        /*                     MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), */
        /*         ADD( MULT(MULT(m,b.m),c.r), ADD(MULT(MULT(m,c.m),b.r),MULT(MULT(m,b.m),c.r)))); */
        /* Datatype sxyz = */
        /* ADD(mxyz, */
        /* ADD( */
        /*     ADD(ADD(MULT(mxy,c.m),MULT(mxz,b.m)),MULT(myz,m)), */
        /*     ADD(ADD(MULT(MULT(m,b.m),c.r),MULT(MULT(m,c.m),b.r)),MULT(MULT(b.m,c.m),r)) */
        /*     )); */
        Datatype sxyz = ADD(ADD(MULT(m, (ADD(myz, MULT(b.r, c.m)))), MULT(b.m, (ADD(mxz, MULT(c.r, m))))),
                            ADD(MULT(c.m, (ADD(mxy, MULT(r, b.m)))), mxyz));
        Datatype a0 = ADD(v, m);
        Datatype b0 = ADD(b.v, b.m);
        Datatype c0 = ADD(c.v, c.m);
        Datatype rxy = getRandomVal(P_123);
        Datatype rxz = getRandomVal(P_123);
        Datatype ryz = getRandomVal(P_123);
        Datatype rxyz = getRandomVal(P_123);
        OEC_MAL2_Share d;
        d.v = SUB(ADD(ADD(MULT(a0, ADD(syz, MULT(b0, SUB(c0, c.r)))), (MULT(b0, SUB(sxz, MULT(c0, r))))),
                      MULT(c0, SUB(sxy, MULT(a0, b.r)))),
                  sxyz);  // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
        Datatype m20 = SUB(ADD(ADD(MULT(a0, SUB(ryz, MULT(b0, c.m))), (MULT(b0, SUB(rxz, MULT(c0, m))))),
                               MULT(c0, SUB(rxy, MULT(a0, b.m)))),
                           rxyz);  // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
        d.m = getRandomVal(P_123);
        d.r = getRandomVal(P_023);
        d.v = ADD(d.v, d.r);
        send_to_live(P_0, ADD(m20, d.m));  // + s
        send_to_live(P_1, d.v);
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
        Datatype m21 = receive_from_live(P_1);
        v = ADD(v, m21);
        store_compare_view(P_012, ADD(v, m));  // compare d_0 s
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_dot4(const OEC_MAL2_Share b,
                                const OEC_MAL2_Share c,
                                const OEC_MAL2_Share d,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        Datatype mxy = pre_receive_from_live(P_3);
        Datatype mxz = pre_receive_from_live(P_3);
        Datatype mxw = pre_receive_from_live(P_3);
        Datatype myz = pre_receive_from_live(P_3);
        Datatype myw = pre_receive_from_live(P_3);
        Datatype mzw = pre_receive_from_live(P_3);
        Datatype mxyz = pre_receive_from_live(P_3);
        Datatype mxyw = pre_receive_from_live(P_3);
        Datatype mxzw = pre_receive_from_live(P_3);
        Datatype myzw = pre_receive_from_live(P_3);
#else
        Datatype mxy = receive_from_live(P_3);
        Datatype mxz = receive_from_live(P_3);
        Datatype mxw = receive_from_live(P_3);
        Datatype myz = receive_from_live(P_3);
        Datatype myw = receive_from_live(P_3);
        Datatype mzw = receive_from_live(P_3);
        Datatype mxyz = receive_from_live(P_3);
        Datatype mxyw = receive_from_live(P_3);
        Datatype mxzw = receive_from_live(P_3);
        Datatype myzw = receive_from_live(P_3);
#endif
        store_compare_view(P_0, mxy);
        store_compare_view(P_0, mxz);
        store_compare_view(P_0, mxw);
        store_compare_view(P_0, myz);
        store_compare_view(P_0, myw);
        store_compare_view(P_0, mzw);
        store_compare_view(P_0, mxyz);
        store_compare_view(P_0, mxyw);
        store_compare_view(P_0, mxzw);
        store_compare_view(P_0, myzw);
#else
        Datatype mxy = receive_from_live(P_0);
        Datatype mxz = receive_from_live(P_0);
        Datatype mxw = receive_from_live(P_0);
        Datatype myz = receive_from_live(P_0);
        Datatype myw = receive_from_live(P_0);
        Datatype mzw = receive_from_live(P_0);
        Datatype mxyz = receive_from_live(P_0);
        Datatype mxyw = receive_from_live(P_0);
        Datatype mxzw = receive_from_live(P_0);
        Datatype myzw = receive_from_live(P_0);
        store_compare_view(P_3, mxy);
        store_compare_view(P_3, mxz);
        store_compare_view(P_3, mxw);
        store_compare_view(P_3, myz);
        store_compare_view(P_3, myw);
        store_compare_view(P_3, mzw);
        store_compare_view(P_3, mxyz);
        store_compare_view(P_3, mxyw);
        store_compare_view(P_3, mxzw);
        store_compare_view(P_3, myzw);
#endif
        Datatype sxy = ADD(mxy, ADD(MULT(r, b.m), MULT(m, b.r)));
        Datatype sxz = ADD(mxz, ADD(MULT(r, c.m), MULT(m, c.r)));
        Datatype sxw = ADD(mxw, ADD(MULT(r, d.m), MULT(m, d.r)));
        Datatype syz = ADD(myz, ADD(MULT(b.r, c.m), MULT(b.m, c.r)));
        Datatype syw = ADD(myw, ADD(MULT(b.r, d.m), MULT(b.m, d.r)));
        Datatype szw = ADD(mzw, ADD(MULT(c.r, d.m), MULT(c.m, d.r)));
        Datatype sxyz = ADD(ADD(MULT(m, (ADD(myz, MULT(b.r, c.m)))), MULT(b.m, (ADD(mxz, MULT(c.r, m))))),
                            ADD(MULT(c.m, (ADD(mxy, MULT(r, b.m)))), mxyz));
        Datatype sxyw = ADD(ADD(MULT(m, (ADD(myw, MULT(b.r, d.m)))), MULT(b.m, (ADD(mxw, MULT(d.r, m))))),
                            ADD(MULT(d.m, (ADD(mxy, MULT(r, b.m)))), mxyw));
        Datatype sxzw = ADD(ADD(MULT(m, (ADD(mzw, MULT(c.r, d.m)))), MULT(c.m, (ADD(mxw, MULT(d.r, m))))),
                            ADD(MULT(d.m, (ADD(mxz, MULT(r, c.m)))), mxzw));
        Datatype syzw = ADD(ADD(MULT(b.m, (ADD(mzw, MULT(c.r, d.m)))), MULT(c.m, (ADD(myw, MULT(d.r, b.m))))),
                            ADD(MULT(d.m, (ADD(myz, MULT(b.r, c.m)))), myzw));
        Datatype sxyzw = ADD(ADD(MULT(m, ADD(MULT(d.m, ADD(MULT(b.m, ADD(c.m, c.r)), myz)), myzw)),
                                 MULT(b.m, ADD(MULT(m, ADD(mzw, MULT(c.m, d.r))), ADD(MULT(c.m, mxw), mxzw)))),
                             ADD(MULT(c.m, ADD(MULT(m, ADD(myw, MULT(d.m, b.r))), mxyw)),
                                 MULT(d.m, ADD(MULT(b.m, ADD(mxz, MULT(c.m, r))), ADD(MULT(c.m, mxy), mxyz)))));

        /* Datatype sxyz = ADD(ADD(mxyz,MULT(mxy,c.m)), MULT(MULT(m,b.m),c.r)); */
        /* Datatype sxyw = ADD(ADD(mxyw,MULT(mxy,d.m)), MULT(MULT(m,b.m),d.r)); */
        /* Datatype sxzw = ADD(ADD(mxzw,MULT(mxz,d.m)), MULT(MULT(m,c.m),d.r)); */
        /* Datatype syzw = ADD(ADD(myzw,MULT(myz,d.m)), MULT(MULT(m,b.m),d.r)); */
        /* Datatype sxyzw = ADD(  ADD(  ADD(mxyzw,MULT(mxyz,d.m))  , MULT(mxy,c.r)), ADD(MULT(MULT(m,b.m),c.m),d.r)); */

        Datatype a0 = ADD(v, m);
        Datatype b0 = ADD(b.v, b.m);
        Datatype c0 = ADD(c.v, c.m);
        Datatype d0 = ADD(d.v, d.m);
        Datatype rxy = getRandomVal(P_123);
        Datatype rxz = getRandomVal(P_123);
        Datatype rxw = getRandomVal(P_123);
        Datatype ryz = getRandomVal(P_123);
        Datatype ryw = getRandomVal(P_123);
        Datatype rzw = getRandomVal(P_123);
        Datatype rxyz = getRandomVal(P_123);
        Datatype rxyw = getRandomVal(P_123);
        Datatype rxzw = getRandomVal(P_123);
        Datatype ryzw = getRandomVal(P_123);
        OEC_MAL2_Share e;
        e.r =

            ADD(ADD(MULT(a0, SUB(MULT(d0, ADD(MULT(b0, SUB(c0, c.r)), syz)), syzw)),
                    MULT(b0, ADD(MULT(a0, SUB(szw, MULT(c0, d.r))), SUB(MULT(c0, sxw), sxzw)))),
                ADD(ADD(sxyzw, MULT(c0, SUB(MULT(a0, SUB(syw, MULT(d0, b.r))), sxyw))),
                    MULT(d0,
                         ADD(MULT(b0, SUB(sxz, MULT(c0, r))),
                             SUB(MULT(c0, sxy),
                                 sxyz)))));  // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) +
                                             // c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
        e.v =

            ADD(ADD(MULT(a0, SUB(MULT(d0, SUB(ryz, MULT(b0, c.m))), ryzw)),
                    MULT(b0, ADD(MULT(a0, SUB(rzw, MULT(c0, d.m))), SUB(MULT(c0, rxw), rxzw)))),
                ADD(MULT(c0, SUB(MULT(a0, SUB(ryw, MULT(d0, b.m))), rxyw)),
                    MULT(d0, ADD(MULT(b0, SUB(rxz, MULT(c0, m))), SUB(MULT(c0, rxy), rxyz))))

            );  // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1)
                // + c0rxy - rxyz) + rxyzw
        /* e.r = ADD(e.r, e.v); // hack for mask_and_send_dot */
        e.r = SUB(SET_ALL_ZERO(), e.r);  // hack for mask_and_send_dot
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_mult4(const OEC_MAL2_Share b,
                                 const OEC_MAL2_Share c,
                                 const OEC_MAL2_Share d,
                                 func_add ADD,
                                 func_sub SUB,
                                 func_mul MULT) const
    {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        Datatype mxy = pre_receive_from_live(P_3);
        Datatype mxz = pre_receive_from_live(P_3);
        Datatype mxw = pre_receive_from_live(P_3);
        Datatype myz = pre_receive_from_live(P_3);
        Datatype myw = pre_receive_from_live(P_3);
        Datatype mzw = pre_receive_from_live(P_3);
        Datatype mxyz = pre_receive_from_live(P_3);
        Datatype mxyw = pre_receive_from_live(P_3);
        Datatype mxzw = pre_receive_from_live(P_3);
        Datatype myzw = pre_receive_from_live(P_3);
        Datatype mxyzw = pre_receive_from_live(P_3);
#else
        Datatype mxy = receive_from_live(P_3);
        Datatype mxz = receive_from_live(P_3);
        Datatype mxw = receive_from_live(P_3);
        Datatype myz = receive_from_live(P_3);
        Datatype myw = receive_from_live(P_3);
        Datatype mzw = receive_from_live(P_3);
        Datatype mxyz = receive_from_live(P_3);
        Datatype mxyw = receive_from_live(P_3);
        Datatype mxzw = receive_from_live(P_3);
        Datatype myzw = receive_from_live(P_3);
        Datatype mxyzw = receive_from_live(P_3);
#endif
        store_compare_view(P_0, mxy);
        store_compare_view(P_0, mxz);
        store_compare_view(P_0, mxw);
        store_compare_view(P_0, myz);
        store_compare_view(P_0, myw);
        store_compare_view(P_0, mzw);
        store_compare_view(P_0, mxyz);
        store_compare_view(P_0, mxyw);
        store_compare_view(P_0, mxzw);
        store_compare_view(P_0, myzw);
        store_compare_view(P_0, mxyzw);
#else
        Datatype mxy = receive_from_live(P_0);
        Datatype mxz = receive_from_live(P_0);
        Datatype mxw = receive_from_live(P_0);
        Datatype myz = receive_from_live(P_0);
        Datatype myw = receive_from_live(P_0);
        Datatype mzw = receive_from_live(P_0);
        Datatype mxyz = receive_from_live(P_0);
        Datatype mxyw = receive_from_live(P_0);
        Datatype mxzw = receive_from_live(P_0);
        Datatype myzw = receive_from_live(P_0);
        Datatype mxyzw = receive_from_live(P_0);
        store_compare_view(P_3, mxy);
        store_compare_view(P_3, mxz);
        store_compare_view(P_3, mxw);
        store_compare_view(P_3, myz);
        store_compare_view(P_3, myw);
        store_compare_view(P_3, mzw);
        store_compare_view(P_3, mxyz);
        store_compare_view(P_3, mxyw);
        store_compare_view(P_3, mxzw);
        store_compare_view(P_3, myzw);
        store_compare_view(P_3, mxyzw);
#endif
        Datatype sxy = ADD(mxy, ADD(MULT(r, b.m), MULT(m, b.r)));
        Datatype sxz = ADD(mxz, ADD(MULT(r, c.m), MULT(m, c.r)));
        Datatype sxw = ADD(mxw, ADD(MULT(r, d.m), MULT(m, d.r)));
        Datatype syz = ADD(myz, ADD(MULT(b.r, c.m), MULT(b.m, c.r)));
        Datatype syw = ADD(myw, ADD(MULT(b.r, d.m), MULT(b.m, d.r)));
        Datatype szw = ADD(mzw, ADD(MULT(c.r, d.m), MULT(c.m, d.r)));
        Datatype sxyz = ADD(ADD(MULT(m, (ADD(myz, MULT(b.r, c.m)))), MULT(b.m, (ADD(mxz, MULT(c.r, m))))),
                            ADD(MULT(c.m, (ADD(mxy, MULT(r, b.m)))), mxyz));
        Datatype sxyw = ADD(ADD(MULT(m, (ADD(myw, MULT(b.r, d.m)))), MULT(b.m, (ADD(mxw, MULT(d.r, m))))),
                            ADD(MULT(d.m, (ADD(mxy, MULT(r, b.m)))), mxyw));
        Datatype sxzw = ADD(ADD(MULT(m, (ADD(mzw, MULT(c.r, d.m)))), MULT(c.m, (ADD(mxw, MULT(d.r, m))))),
                            ADD(MULT(d.m, (ADD(mxz, MULT(r, c.m)))), mxzw));
        Datatype syzw = ADD(ADD(MULT(b.m, (ADD(mzw, MULT(c.r, d.m)))), MULT(c.m, (ADD(myw, MULT(d.r, b.m))))),
                            ADD(MULT(d.m, (ADD(myz, MULT(b.r, c.m)))), myzw));
        Datatype sxyzw = ADD(ADD(MULT(m, ADD(MULT(d.m, ADD(MULT(b.m, ADD(c.m, c.r)), myz)), myzw)),
                                 MULT(b.m, ADD(MULT(m, ADD(mzw, MULT(c.m, d.r))), ADD(MULT(c.m, mxw), mxzw)))),
                             ADD(ADD(mxyzw, MULT(c.m, ADD(MULT(m, ADD(myw, MULT(d.m, b.r))), mxyw))),
                                 MULT(d.m, ADD(MULT(b.m, ADD(mxz, MULT(c.m, r))), ADD(MULT(c.m, mxy), mxyz)))));

        /* Datatype sxyz = ADD(ADD(mxyz,MULT(mxy,c.m)), MULT(MULT(m,b.m),c.r)); */
        /* Datatype sxyw = ADD(ADD(mxyw,MULT(mxy,d.m)), MULT(MULT(m,b.m),d.r)); */
        /* Datatype sxzw = ADD(ADD(mxzw,MULT(mxz,d.m)), MULT(MULT(m,c.m),d.r)); */
        /* Datatype syzw = ADD(ADD(myzw,MULT(myz,d.m)), MULT(MULT(m,b.m),d.r)); */
        /* Datatype sxyzw = ADD(  ADD(  ADD(mxyzw,MULT(mxyz,d.m))  , MULT(mxy,c.r)), ADD(MULT(MULT(m,b.m),c.m),d.r)); */

        Datatype a0 = ADD(v, m);
        Datatype b0 = ADD(b.v, b.m);
        Datatype c0 = ADD(c.v, c.m);
        Datatype d0 = ADD(d.v, d.m);
        Datatype rxy = getRandomVal(P_123);
        Datatype rxz = getRandomVal(P_123);
        Datatype rxw = getRandomVal(P_123);
        Datatype ryz = getRandomVal(P_123);
        Datatype ryw = getRandomVal(P_123);
        Datatype rzw = getRandomVal(P_123);
        Datatype rxyz = getRandomVal(P_123);
        Datatype rxyw = getRandomVal(P_123);
        Datatype rxzw = getRandomVal(P_123);
        Datatype ryzw = getRandomVal(P_123);
        Datatype rxyzw = getRandomVal(P_123);
        OEC_MAL2_Share e;
        e.v =

            ADD(ADD(MULT(a0, SUB(MULT(d0, ADD(MULT(b0, SUB(c0, c.r)), syz)), syzw)),
                    MULT(b0, ADD(MULT(a0, SUB(szw, MULT(c0, d.r))), SUB(MULT(c0, sxw), sxzw)))),
                ADD(ADD(sxyzw, MULT(c0, SUB(MULT(a0, SUB(syw, MULT(d0, b.r))), sxyw))),
                    MULT(d0,
                         ADD(MULT(b0, SUB(sxz, MULT(c0, r))),
                             SUB(MULT(c0, sxy),
                                 sxyz)))));  // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) +
                                             // c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
        Datatype m20 =

            ADD(ADD(MULT(a0, SUB(MULT(d0, SUB(ryz, MULT(b0, c.m))), ryzw)),
                    MULT(b0, ADD(MULT(a0, SUB(rzw, MULT(c0, d.m))), SUB(MULT(c0, rxw), rxzw)))),
                ADD(ADD(rxyzw, MULT(c0, SUB(MULT(a0, SUB(ryw, MULT(d0, b.m))), rxyw))),
                    MULT(d0, ADD(MULT(b0, SUB(rxz, MULT(c0, m))), SUB(MULT(c0, rxy), rxyz))))

            );  // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1)
                // + c0rxy - rxyz) + rxyzw
        e.m = getRandomVal(P_123);
        e.r = getRandomVal(P_023);
        e.v = ADD(e.v, e.r);
        send_to_live(P_0, ADD(m20, e.m));  // + s
        send_to_live(P_1, e.v);
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
        Datatype m21 = receive_from_live(P_1);
        v = ADD(v, m21);
        store_compare_view(P_012, ADD(v, m));  // compare d_0 s
    }

#endif

#if USE_CUDA_GEMM == 2

    static void CONV_2D(const OEC_MAL2_Share* X,
                        const OEC_MAL2_Share* W,
                        OEC_MAL2_Share* Y,
                        int batchSize,
                        int inh,
                        int inw,
                        int din,
                        int dout,
                        int wh,
                        int ww,
                        int padding,
                        int stride,
                        int dilation = 1)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int xSize = inh * inw * din * batchSize;
        const int wSize = wh * ww * din * dout;
        const int outh = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
        const int outw = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
        const int ySize = outh * outw * dout * batchSize;
        batchSize *= factor;

        UINT_TYPE* r = new UINT_TYPE[factor * xSize];
        UINT_TYPE* v = new UINT_TYPE[factor * xSize];
        UINT_TYPE* b_r = new UINT_TYPE[wSize];
        UINT_TYPE* b_v = new UINT_TYPE[wSize];
        UINT_TYPE* v_br = new UINT_TYPE[factor * ySize];
        UINT_TYPE* bv_r = new UINT_TYPE[factor * ySize];
        UINT_TYPE* v_bv = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&X[i].r, temp, 1);
            for (int j = 0; j < factor; j++)
                r[j * xSize + i] = temp[j];
            unorthogonalize_arithmetic(&X[i].v, temp, 1);
            for (int j = 0; j < factor; j++)
                v[j * xSize + i] = temp[j];
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].r, temp, 1);
            b_r[i] = temp[0];
            unorthogonalize_arithmetic(&W[i].v, temp, 1);
            b_v[i] = temp[0];
        }

        conv2d_cutlass(v, b_v, v_bv, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(v, b_r, v_br, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(r, b_v, bv_r, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = v_br[j * ySize + i] + bv_r[j * ySize + i];
            orthogonalize_arithmetic(temp, &Y[i].r, 1);
            for (int j = 0; j < factor; j++)
                temp[j] = v_bv[j * ySize + i];
            orthogonalize_arithmetic(temp, &Y[i].v, 1);
        }

        delete[] r;
        delete[] v;
        delete[] b_r;
        delete[] b_v;
        delete[] v_br;
        delete[] bv_r;
        delete[] v_bv;
    }
#elif USE_CUDA_GEMM == 4

    static void CONV_2D(const OEC_MAL2_Share* X,
                        const OEC_MAL2_Share* W,
                        OEC_MAL2_Share* Y,
                        int batchSize,
                        int inh,
                        int inw,
                        int din,
                        int dout,
                        int wh,
                        int ww,
                        int padding,
                        int stride,
                        int dilation = 1)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int xSize = inh * inw * din * batchSize;
        const int wSize = wh * ww * din * dout;
        const int outh = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
        const int outw = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
        const int ySize = outh * outw * dout * batchSize;
        batchSize *= factor;

        UINT_TYPE* r = new UINT_TYPE[factor * xSize];
        UINT_TYPE* v = new UINT_TYPE[factor * xSize];
        UINT_TYPE* b_r = new UINT_TYPE[wSize];
        UINT_TYPE* b_v = new UINT_TYPE[wSize];
        UINT_TYPE* v_br = new UINT_TYPE[factor * ySize];
        UINT_TYPE* bv_r = new UINT_TYPE[factor * ySize];
        UINT_TYPE* v_bv = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            unorthogonalize_arithmetic(&X[i].r, r + i * factor, 1);
            unorthogonalize_arithmetic(&X[i].v, v + i * factor, 1);
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].r, temp, 1);
            b_r[i] = temp[0];
            unorthogonalize_arithmetic(&W[i].v, temp, 1);
            b_v[i] = temp[0];
        }

        conv2d_cutlass(v, b_v, v_bv, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(v, b_r, v_br, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(r, b_v, bv_r, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = v_br[i * factor + j] + bv_r[i * factor + j];
            orthogonalize_arithmetic(temp, &Y[i].r, 1);
            for (int j = 0; j < factor; j++)
                temp[j] = v_bv[i * factor + j];
            orthogonalize_arithmetic(temp, &Y[i].v, 1);
        }

        delete[] r;
        delete[] v;
        delete[] b_r;
        delete[] b_v;
        delete[] v_br;
        delete[] bv_r;
        delete[] v_bv;
    }
#endif
#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1

    static void GEMM(OEC_MAL2_Share* a, OEC_MAL2_Share* b, OEC_MAL2_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;

        UINT_TYPE* b_r = new UINT_TYPE[b_size * factor];
        UINT_TYPE* b_v = new UINT_TYPE[b_size * factor];
        UINT_TYPE* r;
        UINT_TYPE* v;
        if (a_fixed)
        {
            r = new UINT_TYPE[a_size];
            v = new UINT_TYPE[a_size];
        }
        else
        {
            r = new UINT_TYPE[factor * a_size];
            v = new UINT_TYPE[factor * a_size];
        }
        UINT_TYPE* v_bv = new UINT_TYPE[factor * c_size];
        UINT_TYPE* v_br = new UINT_TYPE[factor * c_size];
        UINT_TYPE* bv_r = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].r, temp, 1);
            if (a_fixed)
            {
                r[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    r[j * a_size + i] = temp[j];

            unorthogonalize_arithmetic(&a[i].v, temp, 1);
            if (a_fixed)
            {
                v[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    v[j * a_size + i] = temp[j];
        }
        for (int i = 0; i < b_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&b[i].r, temp, 1);
            for (int j = 0; j < factor; j++)
                b_r[j * b_size + i] = temp[j];
            unorthogonalize_arithmetic(&b[i].v, temp, 1);
            for (int j = 0; j < factor; j++)
                b_v[j * b_size + i] = temp[j];
        }

        for (int i = 0; i < factor; i++)
        {
            if (a_fixed)
            {
                gemm_cutlass(m, n, k, v, &b_v[i * b_size], &v_bv[i * c_size]);
                gemm_cutlass(m, n, k, v, &b_r[i * b_size], &v_br[i * c_size]);
                gemm_cutlass(m, n, k, r, &b_v[i * b_size], &bv_r[i * c_size]);
            }

            else
            {
                gemm_cutlass(m, n, k, &v[i * a_size], &b_v[i * b_size], &v_bv[i * c_size]);
                gemm_cutlass(m, n, k, &v[i * a_size], &b_r[i * b_size], &v_br[i * c_size]);
                gemm_cutlass(m, n, k, &r[i * a_size], &b_v[i * b_size], &bv_r[i * c_size]);
            }
        }

        for (int j = 0; j < c_size; j++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int i = 0; i < factor; i++)
                temp[i] = v_br[i * c_size + j] + bv_r[i * c_size + j];
            orthogonalize_arithmetic(temp, &c[j].r, 1);
            for (int i = 0; i < factor; i++)
                temp[i] = v_bv[i * c_size + j];
            orthogonalize_arithmetic(temp, &c[j].v, 1);
        }

        delete[] r;
        delete[] v;
        delete[] b_r;
        delete[] b_v;
        delete[] v_bv;
        delete[] v_br;
        delete[] bv_r;
    }
#else

    static void GEMM(OEC_MAL2_Share* a, OEC_MAL2_Share* b, OEC_MAL2_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;
        UINT_TYPE* b_r = new UINT_TYPE[b_size * factor];
        UINT_TYPE* b_v = new UINT_TYPE[b_size * factor];
        UINT_TYPE* r;
        UINT_TYPE* v;
        if (a_fixed)
        {
            r = new UINT_TYPE[a_size];
            v = new UINT_TYPE[a_size];
        }
        else
        {
            r = new UINT_TYPE[factor * a_size];
            v = new UINT_TYPE[factor * a_size];
        }

        UINT_TYPE* v_bv = new UINT_TYPE[factor * c_size];
        UINT_TYPE* v_br = new UINT_TYPE[factor * c_size];
        UINT_TYPE* bv_r = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].r, temp, 1);
            if (a_fixed)
            {
                r[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    r[j * a_size + i] = temp[j];

            unorthogonalize_arithmetic(&a[i].v, temp, 1);
            if (a_fixed)
            {
                v[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    v[j * a_size + i] = temp[j];
        }
        if (a_fixed)
        {
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    unorthogonalize_arithmetic(&b[i * n + j].r, temp, 1);
                    for (int l = 0; l < factor; l++)
                        b_r[i * n * factor + l * n + j] = temp[l];
                    unorthogonalize_arithmetic(&b[i * n + j].v, temp, 1);
                    for (int l = 0; l < factor; l++)
                        b_v[i * n * factor + l * n + j] = temp[l];
                }
            }

            gemm_cutlass(m, n * factor, k, v, b_v, v_bv);
            gemm_cutlass(m, n * factor, k, v, b_r, v_br);
            gemm_cutlass(m, n * factor, k, r, b_v, bv_r);

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    for (int l = 0; l < factor; l++)
                        temp[l] = v_br[i * n * factor + l * n + j] + bv_r[i * n * factor + l * n + j];
                    orthogonalize_arithmetic(temp, &c[i * n + j].r, 1);
                    for (int l = 0; l < factor; l++)
                        temp[l] = v_bv[i * n * factor + l * n + j];
                    orthogonalize_arithmetic(temp, &c[i * n + j].v, 1);
                }
            }
        }
        else
        {
            for (int i = 0; i < b_size; i++)
            {
                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&b[i].r, temp, 1);
                for (int j = 0; j < factor; j++)
                    b_r[j * b_size + i] = temp[j];
                unorthogonalize_arithmetic(&b[i].v, temp, 1);
                for (int j = 0; j < factor; j++)
                    b_v[j * b_size + i] = temp[j];
            }

            for (int i = 0; i < factor; i++)
            {
                gemm_cutlass(m, n, k, &v[i * a_size], &b_v[i * b_size], &v_bv[i * c_size]);
                gemm_cutlass(m, n, k, &v[i * a_size], &b_r[i * b_size], &v_br[i * c_size]);
                gemm_cutlass(m, n, k, &r[i * a_size], &b_v[i * b_size], &bv_r[i * c_size]);
            }

            for (int j = 0; j < c_size; j++)
            {
                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                for (int i = 0; i < factor; i++)
                    temp[i] = v_br[i * c_size + j] + bv_r[i * c_size + j];
                orthogonalize_arithmetic(temp, &c[j].r, 1);
                for (int i = 0; i < factor; i++)
                    temp[i] = v_bv[i * c_size + j];
                orthogonalize_arithmetic(temp, &c[j].v, 1);
            }
        }
        delete[] r;
        delete[] v;
        delete[] b_r;
        delete[] b_v;
        delete[] v_bv;
        delete[] v_br;
        delete[] bv_r;
    }
#endif
#endif

#endif
};
