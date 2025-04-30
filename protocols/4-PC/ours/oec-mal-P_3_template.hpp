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

        /* Datatype r124 = getRandomVal(P_013); // used for verification */
        /* Datatype r234 = getRandomVal(P_123); // Probably sufficient to only generate with P_2(-> P_3 in paper) -> no
         * because of verification */

        /* Datatype o1 = ADD( x1y1, getRandomVal(P_013)); */
        Datatype o1 = ADD(c.r1, ADD(MULT(r1, b.r1), getRandomVal(P_013)));

#if PROTOCOL == 11
        /* Datatype o4 = ADD(SUB(SUB(x1y1, MULT(a.r0,b.r1)) ,MULT(a.r1,b.r0)),getRandomVal(P_123)); // r123_2 */
        /* Datatype o4 = ADD(SUB(MULT(a.r1, SUB(b.r0,b.r1)) ,MULT(b.r1,a.r0)),getRandomVal(P_123)); // r123_2 */
        Datatype o4 = ADD(SUB(MULT(r1, SUB(b.r1, b.r0)), MULT(b.r1, r0)), getRandomVal(P_123));  // r123_2
#else
        Datatype o4 = ADD(SUB(MULT(r1, SUB(b.r1, b.r0)), MULT(b.r1, r0)), getRandomVal(P_123));  // r123_2
#endif
        c.r0 = o4;
/* Datatype o4 = ADD( SUB( MULT(a.r0,b.r1) ,MULT(a.r1,b.r0)),getRandomVal(P_123)); */
/* o4 = XOR(o4,o1); //computationally easier to let P_3 do it here instead of P_0 later */
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_2, o1);
#else
        send_to_live(P_2, o1);
#endif
#else
        store_compare_view(P_2, o1);
#endif

        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_dot(const OEC_MAL3_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL3_Share c;
        c.r1 = MULT(r1, b.r1);                                  // store o_1
        c.r0 = SUB(MULT(r1, SUB(b.r1, b.r0)), MULT(b.r1, r0));  // store o_4
        return c;
    }
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        Datatype r0123 = ADD(getRandomVal(P_013), getRandomVal(P_023));
        r1 = TRUNC(SUB(r0123, r1));  // z_0 = [r_0,1,3 + r_0,2,3 - x_0 y_0]^t
        r0 = SUB(r0, r0123);

#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_2, SUB(r1, getRandomVal(P_013)));  // compare m^0 - z_1
#else
        send_to_live(P_2, SUB(r1, getRandomVal(P_013)));  // compare m^0 - z_1
#endif
#else
        store_compare_view(P_2, SUB(r1, getRandomVal(P_013)));  // compare m^0 - z_1
#endif
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_dot(const OEC_MAL3_Share b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL3_Share c;
        if (i == 0)
        {
            c.r0 = MULT(r1, SUB(b.r1, b.r0));
        }
        else if (i == 1)
        {
            c.r0 = MULT(b.r1, r0);
        }
        else if (i == 2)
        {
            c.r1 = MULT(r1, b.r1);
        }
        return c;
    }

    template <typename func_add, typename func_sub>
    void join_dots(OEC_MAL3_Share c[], func_add ADD, func_sub SUB)
    {
        r0 = ADD(r0, SUB(c[0].r0, c[1].r0));
        r1 = ADD(r1, c[2].r1);
    }

    static constexpr int getNumDotProducts() { return 3; }
#endif

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {

        Datatype rc1 = ADD(getRandomVal(P_023), getRandomVal(P_013));  // x0

        Datatype o1 = ADD(rc1, ADD(r1, getRandomVal(P_013)));
#if PROTOCOL == 11
        Datatype o4 = ADD(r0, getRandomVal(P_123));  // r123_2
#else
        Datatype o4 = ADD(r0, getRandomVal(P_123));             // - w + r123
#endif
        r0 = o4;
        r1 = rc1;

#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_2, o1);
#else
        send_to_live(P_2, o1);
#endif
#else
        store_compare_view(P_2, o1);
#endif
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        Datatype rc0 = getRandomVal(P_123);  // w
#if PROTOCOL == 11
        Datatype o4 = r0;
#else
        Datatype o4 = SUB(r0, rc0);
#endif
#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
        pre_send_to_live(P_0, o4);
#else
        send_to_live(P_0, o4);
#endif
#elif PROTOCOL == 11
        store_compare_view(P_0, o4);
#endif
        r0 = rc0;
    }

    void prepare_reveal_to_all() const
    {
#if PRE == 1
        pre_send_to_live(P_0, r0);
#else
        send_to_live(P_0, r0);
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 0
        Datatype result = SUB(receive_from_live(P_0), r0);
        store_compare_view(P_123, r1);
        store_compare_view(P_0123, result);
        return result;
#else
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share(r0);
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
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate()
    {
#if PRE == 0
        communicate_live();
#endif
    }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul>
    OEC_MAL3_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return OEC_MAL3_Share(MULT(r0, b), MULT(r1, b));
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL3_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        auto result = r1;
        for (int i = 2; i <= b; i *= 2)
            result = OP_TRUNC2(result);
        auto rand_val = getRandomVal(P_013);
        auto val = SUB(result, rand_val);
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_2, val);
#else
        send_to_live(P_2, val);
#endif
#else
        store_compare_view(P_2, val);
#endif

        return OEC_MAL3_Share(getRandomVal(P_123), result);
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL3_Share prepare_mult_public_fixed(const Datatype b,
                                             func_mul MULT,
                                             func_add ADD,
                                             func_sub SUB,
                                             func_trunc TRUNC,
                                             int fractional_bits) const
    {
        /* #if TRUNC_THEN_MULT == 1 */
        /*     auto result = MULT(TRUNC(r1,fractional_bits),b); */
        /* #else */
        auto result = TRUNC(MULT(r1, b), fractional_bits);
        /* #endif */
        auto rand_val = getRandomVal(P_013);
        auto val = SUB(result, rand_val);
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_2, val);
#else
        send_to_live(P_2, val);
#endif
#else
        store_compare_view(P_2, val);
#endif

        return OEC_MAL3_Share(getRandomVal(P_123), result);
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL3_Share prepare_trunc_share(func_mul MULT,
                                       func_add ADD,
                                       func_sub SUB,
                                       func_trunc TRUNC,
                                       int fractional_bits = FRACTIONAL) const
    {
        auto result = SUB(SET_ALL_ZERO(), TRUNC(SUB(SET_ALL_ZERO(), r1), fractional_bits));
        auto rand_val = getRandomVal(P_013);
        auto val = SUB(result, rand_val);
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_2, val);
#else
        send_to_live(P_2, val);
#endif
#else
        store_compare_view(P_2, val);
#endif

        return OEC_MAL3_Share(getRandomVal(P_123), result);
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        r0 = ADD(r0, getRandomVal(P_123));  // mask v^3
#if PROTOCOL == 11
        store_compare_view(P_0, r0);  // v^3 = .. - r_0,1,2 - r_0,2,3 + r_1,2,3 TODO: Recent change: Verify
#else
#if PRE == 1
        pre_send_to_live(P_0, r0);  // m^3 = .. - r_0,1,2 - r_0,2,3 + r_1,2,3 TODO: Recent change: Verify
#else
        send_to_live(P_0, r0);  // m^3 = .. - r_0,1,2 - r_0,2,3 + r_1,2,3 TODO: Recent change: Verify
#endif
#endif
        r0 = getRandomVal(P_123);  // w
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OEC_MAL3_Share& r_mk2,
                                 OEC_MAL3_Share& r_msb,
                                 OEC_MAL3_Share& c,
                                 OEC_MAL3_Share& c_prime,
                                 int fractional_bits = FRACTIONAL) const
    {
        Datatype rmk2 = OP_SHIFT_LOG_RIGHTF(OP_SHIFT_LEFT<1>(r1), fractional_bits + 1);
        Datatype rmsb = OP_SHIFT_LOG_RIGHT<BITLENGTH - 1>(r1);

        r_mk2.r0 = SET_ALL_ZERO();
        r_mk2.r1 = SUB(SET_ALL_ZERO(), rmk2);
        r_msb.r0 = SET_ALL_ZERO();
        r_msb.r1 = SUB(SET_ALL_ZERO(), rmsb);
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_2, SUB(r_mk2.r1, getRandomVal(P_013)));
        pre_send_to_live(P_2, SUB(r_msb.r1, getRandomVal(P_013)));
#else
        send_to_live(P_2, SUB(r_mk2.r1, getRandomVal(P_013)));
        send_to_live(P_2, SUB(r_msb.r1, getRandomVal(P_013)));
#endif
#else
        store_compare_view(P_2, SUB(r_mk2.r1, getRandomVal(P_013)));
        store_compare_view(P_2, SUB(r_msb.r1, getRandomVal(P_013)));
#endif

        c.r0 = getRandomVal(P_123);
        c.r1 = SET_ALL_ZERO();
        c_prime.r0 = getRandomVal(P_123);
        c_prime.r1 = SET_ALL_ZERO();
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OEC_MAL3_Share& r_mk2,
                                  OEC_MAL3_Share& r_msb,
                                  OEC_MAL3_Share& c,
                                  OEC_MAL3_Share& c_prime) const
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OEC_MAL3_Share prepare_trunc_exact_xmod2t(func_add ADD,
                                              func_sub SUB,
                                              func_xor XOR,
                                              func_and AND,
                                              int fractional_bits = FRACTIONAL) const
    {
        Datatype lx = SUB(SET_ALL_ZERO(), r1);
        // Step 1, Compute [x/2t] -> delt with public mult fixed
        // Step 2, Compute [x mod t]
        UINT_TYPE maskValue = (UINT_TYPE(1) << (fractional_bits)) - 1;
        Datatype mask = PROMOTE(maskValue);  // Set all elements to maskValue
        // Apply the mask using bitwise AND
        Datatype lxmodt = AND(lx, mask);  // mod 2^t

        // Step3, Compute [x]^B -> delt with prepareA2B
        return OEC_MAL3_Share(r0, SUB(SET_ALL_ZERO(), lxmodt));
    }

    static void prepare_A2B_S1(int m, int k, OEC_MAL3_Share in[], OEC_MAL3_Share out[])
    {
        for (int i = m; i < k; i++)
        {
            out[i - m].r0 = getRandomVal(P_123);  // r123
            out[i - m].r1 = SET_ALL_ZERO();       // set share to 0
        }
    }

    void get_random_B2A()
    {
        r0 = SET_ALL_ZERO();
        r1 = FUNC_XOR(getRandomVal(P_013), getRandomVal(P_023));
    }

    static void prepare_A2B_S2(int m, int k, OEC_MAL3_Share in[], OEC_MAL3_Share out[])
    {
        // convert share x0 to boolean
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            /* in[j].r1 = SET_ALL_ZERO(); */
            temp[j] = OP_SUB(SET_ALL_ZERO(), in[j].r1);  // set share to -x0
        }
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_arithmetic(temp, temp2);
        orthogonalize_boolean(temp2, temp);
        /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
        /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */

        for (int i = m; i < k; i++)
        {
            out[i - m].r0 = SET_ALL_ZERO();
            out[i - m].r1 = temp[i];
#if PROTOCOL != 12 && PRE == 0
            store_compare_view(P_2, FUNC_XOR(temp[i], getRandomVal(P_013)));  // compare -x0 xor r0,1 with $P_2
#else
#if PRE == 1
            pre_send_to_live(P_2, FUNC_XOR(temp[i], getRandomVal(P_013)));    // -x0 xor r0,1 to P_2
#else
            send_to_live(P_2, FUNC_XOR(temp[i], getRandomVal(P_013)));  // -x0 xor r0,1 to P_2
#endif
#endif
        }
    }

    static void complete_A2B_S1(int k, OEC_MAL3_Share out[]) {}

    static void complete_A2B_S2(int k, OEC_MAL3_Share out[]) {}

    static void prepare_B2A(OEC_MAL3_Share z[], OEC_MAL3_Share random_mask[], OEC_MAL3_Share out[])
    {
        // 2. Share random mask
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = random_mask[j].r1;  // set share to random mask
        }
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(temp, temp2);
        orthogonalize_arithmetic(temp2, temp);
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].r0 = SET_ALL_ZERO();
            out[i].r1 = OP_SUB(SET_ALL_ZERO(), temp[i]);  // set share to - random mask
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
            pre_send_to_live(P_2, OP_SUB(out[i].r1, getRandomVal(P_013)));  // - random mask - r013
#else
            send_to_live(P_2, OP_SUB(out[i].r1, getRandomVal(P_013)));  // - random mask - r013
#endif
#else
            store_compare_view(P_2, OP_SUB(out[i].r1, getRandomVal(P_013)));  // - random mask - r013
#endif
        }
    }

    static void complete_B2A(OEC_MAL3_Share z[], OEC_MAL3_Share out[]) {}

    static void complete_B2A2(OEC_MAL3_Share z[], OEC_MAL3_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            z[i].r0 = getRandomVal(P_123);
            z[i].r1 = SET_ALL_ZERO();
            out[i] = z[i].Add(out[i], OP_SUB);
        }
    }

    void prepare_bit2a(OEC_MAL3_Share out[])
    {
        Datatype y0[BITLENGTH]{0};
        y0[BITLENGTH - 1] = r1;  // convert y0 to an arithemtic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(y0, temp2);
        orthogonalize_arithmetic(temp2, y0);
        Datatype y0v[BITLENGTH]{0};
        y0v[BITLENGTH - 1] = FUNC_XOR(r1, r0);  // convert y_0 xor v to an arithemtic value
        unorthogonalize_boolean(y0v, temp2);
        orthogonalize_arithmetic(temp2, y0v);
        for (int i = 0; i < BITLENGTH; i++)
        {
            Datatype r013 = getRandomVal(P_013);
            Datatype m00 = OP_SUB(y0[i], r013);
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
            pre_send_to_live(P_2, m00);
#else
            send_to_live(P_2, m00);
#endif
#else
            store_compare_view(P_2, m00);
#endif

            Datatype r123 = getRandomVal(P_123);
            Datatype m30 = OP_SUB(y0v[i], r123);

#if PRE == 1
            pre_send_to_live(P_0, m30);
#else
            send_to_live(P_0, m30);
#endif
            out[i].r0 = getRandomVal(P_123);
            out[i].r1 = OP_ADD(getRandomVal(P_013), getRandomVal(P_023));
        }
    }

    void complete_bit2a() {}

    void prepare_opt_bit_injection(OEC_MAL3_Share x[], OEC_MAL3_Share out[])
    {
        Datatype y0[BITLENGTH]{0};
        y0[BITLENGTH - 1] = r1;  // convert y0 to an arithemtic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(y0, temp2);
        orthogonalize_arithmetic(temp2, y0);
        Datatype y0v[BITLENGTH]{0};
        y0v[BITLENGTH - 1] = FUNC_XOR(r1, r0);  // convert y_0 xor v to an arithemtic value
        unorthogonalize_boolean(y0v, temp2);
        orthogonalize_arithmetic(temp2, y0v);
        for (int i = 0; i < BITLENGTH; i++)
        {
            Datatype r013 = getRandomVal(P_013);
            Datatype r013_2 = getRandomVal(P_013);
            Datatype m00 = OP_SUB(y0[i], r013);
            Datatype m01 = OP_SUB(OP_MULT(x[i].r1, y0[i]), r013_2);
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
            pre_send_to_live(P_2, m00);
            pre_send_to_live(P_2, m01);
#else
            send_to_live(P_2, m00);
            send_to_live(P_2, m01);
#endif
#else
            store_compare_view(P_2, m00);
            store_compare_view(P_2, m01);
#endif

            Datatype r123 = getRandomVal(P_123);
            Datatype r123_2 = getRandomVal(P_123);
            Datatype m30 = OP_SUB(y0v[i], r123);
            Datatype m31 = OP_SUB(OP_MULT(OP_ADD(x[i].r0, x[i].r1), y0v[i]), r123_2);

#if PRE == 1
            pre_send_to_live(P_0, m30);
            pre_send_to_live(P_0, m31);
#else
            send_to_live(P_0, m30);
            send_to_live(P_0, m31);
#endif
            out[i].r0 = getRandomVal(P_123);
            out[i].r1 = OP_ADD(getRandomVal(P_013), getRandomVal(P_023));
        }
    }

    void complete_opt_bit_injection() {}

    void prepare_bit_injection_S1(OEC_MAL3_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].r0 = getRandomVal(P_123);  // r123
            out[i].r1 = SET_ALL_ZERO();       // set share to 0
        }
    }

    void prepare_bit_injection_S2(OEC_MAL3_Share out[])
    {
        Datatype temp[BITLENGTH]{0};
        temp[BITLENGTH - 1] = r1;
        /* unorthogonalize_boolean(temp,(UINT_TYPE*)temp); */
        /* orthogonalize_arithmetic((UINT_TYPE*) temp,  temp); */
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(temp, temp2);
        orthogonalize_arithmetic(temp2, temp);
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].r0 = SET_ALL_ZERO();                   // w = 0
            out[i].r1 = OP_SUB(SET_ALL_ZERO(), temp[i]);  // z_0 = - x_0
#if PROTOCOL != 12 && PRE == 0
            store_compare_view(P_2, OP_ADD(temp[i], getRandomVal(P_013)));  // compare -x0 xor r0,1 with $P_2
#else
#if PRE == 1
            pre_send_to_live(P_2, OP_ADD(temp[i], getRandomVal(P_013)));  // -x0 xor r0,1 to P_2
#else
            send_to_live(P_2, OP_ADD(temp[i], getRandomVal(P_013)));    // -x0 xor r0,1 to P_2
#endif
#endif
        }
    }

    static void complete_bit_injection_S1(OEC_MAL3_Share out[]) {}

    static void complete_bit_injection_S2(OEC_MAL3_Share out[]) {}

#if MULTI_INPUT == 1

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_dot3(const OEC_MAL3_Share b,
                                const OEC_MAL3_Share c,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
        Datatype mxy = SUB(MULT(r1, b.r1), getRandomVal(P_013));
        Datatype mxz = SUB(MULT(r1, c.r1), getRandomVal(P_013));
        Datatype myz = SUB(MULT(b.r1, c.r1), getRandomVal(P_013));
        Datatype mxyz = MULT(MULT(r1, b.r1), c.r1);
        mxyz = SUB(SET_ALL_ZERO(), mxyz);  // trick to be compatible with dot2
        Datatype ax = ADD(r0, r1);
        Datatype by = ADD(b.r0, b.r1);
        Datatype cz = ADD(c.r0, c.r1);
        Datatype m3xy = SUB(MULT(ax, by), getRandomVal(P_123));
        Datatype m3xz = SUB(MULT(ax, cz), getRandomVal(P_123));
        Datatype m3yz = SUB(MULT(by, cz), getRandomVal(P_123));
        Datatype m3xyz = MULT(MULT(ax, by), cz);
/* m3xyz = SUB( SET_ALL_ZERO(), m3xyz); // trick to be compatible with dot2 */
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_0, m3xy);
        pre_send_to_live(P_0, m3xz);
        pre_send_to_live(P_0, m3yz);
        pre_send_to_live(P_2, mxy);
        pre_send_to_live(P_2, mxz);
        pre_send_to_live(P_2, myz);
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3yz);
        send_to_live(P_2, mxy);
        send_to_live(P_2, mxz);
        send_to_live(P_2, myz);
#endif
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3yz);
        store_compare_view(P_2, mxy);
        store_compare_view(P_2, mxz);
        store_compare_view(P_2, myz);
#endif
        OEC_MAL3_Share d;
        d.r0 = m3xyz;
        d.r1 = mxyz;
        return d;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_mult3(const OEC_MAL3_Share b,
                                 const OEC_MAL3_Share c,
                                 func_add ADD,
                                 func_sub SUB,
                                 func_mul MULT) const
    {
        Datatype mxy = SUB(MULT(r1, b.r1), getRandomVal(P_013));
        Datatype mxz = SUB(MULT(r1, c.r1), getRandomVal(P_013));
        Datatype myz = SUB(MULT(b.r1, c.r1), getRandomVal(P_013));
        Datatype mxyz = SUB(MULT(MULT(r1, b.r1), c.r1), getRandomVal(P_013));
        Datatype ax = ADD(r0, r1);
        Datatype by = ADD(b.r0, b.r1);
        Datatype cz = ADD(c.r0, c.r1);
        Datatype m3xy = SUB(MULT(ax, by), getRandomVal(P_123));
        Datatype m3xz = SUB(MULT(ax, cz), getRandomVal(P_123));
        Datatype m3yz = SUB(MULT(by, cz), getRandomVal(P_123));
        Datatype m3xyz = SUB(MULT(MULT(ax, by), cz), getRandomVal(P_123));
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_0, m3xy);
        pre_send_to_live(P_0, m3xz);
        pre_send_to_live(P_0, m3yz);
        pre_send_to_live(P_0, m3xyz);
        pre_send_to_live(P_2, mxy);
        pre_send_to_live(P_2, mxz);
        pre_send_to_live(P_2, myz);
        pre_send_to_live(P_2, mxyz);
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3yz);
        send_to_live(P_0, m3xyz);
        send_to_live(P_2, mxy);
        send_to_live(P_2, mxz);
        send_to_live(P_2, myz);
        send_to_live(P_2, mxyz);
#endif
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3yz);
        send_to_live(P_0, m3xyz);
        store_compare_view(P_2, mxy);
        store_compare_view(P_2, mxz);
        store_compare_view(P_2, myz);
        store_compare_view(P_2, mxyz);
#endif
        OEC_MAL3_Share d;
        d.r0 = getRandomVal(P_123);
        d.r1 = ADD(getRandomVal(P_013), getRandomVal(P_023));
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_dot4(const OEC_MAL3_Share b,
                                const OEC_MAL3_Share c,
                                const OEC_MAL3_Share d,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
        Datatype mxy = SUB(MULT(r1, b.r1), getRandomVal(P_013));
        Datatype mxz = SUB(MULT(r1, c.r1), getRandomVal(P_013));
        Datatype mxw = SUB(MULT(r1, d.r1), getRandomVal(P_013));
        Datatype myz = SUB(MULT(b.r1, c.r1), getRandomVal(P_013));
        Datatype myw = SUB(MULT(b.r1, d.r1), getRandomVal(P_013));
        Datatype mzw = SUB(MULT(c.r1, d.r1), getRandomVal(P_013));
        Datatype mxyz = SUB(MULT(MULT(r1, b.r1), c.r1), getRandomVal(P_013));
        Datatype mxyw = SUB(MULT(MULT(r1, b.r1), d.r1), getRandomVal(P_013));
        Datatype mxzw = SUB(MULT(MULT(r1, c.r1), d.r1), getRandomVal(P_013));
        Datatype myzw = SUB(MULT(MULT(b.r1, c.r1), d.r1), getRandomVal(P_013));
        Datatype mxyzw = MULT(MULT(r1, b.r1), MULT(c.r1, d.r1));
        Datatype ax = ADD(r0, r1);
        Datatype by = ADD(b.r0, b.r1);
        Datatype cz = ADD(c.r0, c.r1);
        Datatype dw = ADD(d.r0, d.r1);
        Datatype m3xy = SUB(MULT(ax, by), getRandomVal(P_123));
        Datatype m3xz = SUB(MULT(ax, cz), getRandomVal(P_123));
        Datatype m3xw = SUB(MULT(ax, dw), getRandomVal(P_123));
        Datatype m3yz = SUB(MULT(by, cz), getRandomVal(P_123));
        Datatype m3yw = SUB(MULT(by, dw), getRandomVal(P_123));
        Datatype m3zw = SUB(MULT(cz, dw), getRandomVal(P_123));
        Datatype m3xyz = SUB(MULT(MULT(ax, by), cz), getRandomVal(P_123));
        Datatype m3xyw = SUB(MULT(MULT(ax, by), dw), getRandomVal(P_123));
        Datatype m3xzw = SUB(MULT(MULT(ax, cz), dw), getRandomVal(P_123));
        Datatype m3yzw = SUB(MULT(MULT(by, cz), dw), getRandomVal(P_123));
        Datatype m3xyzw = MULT(MULT(ax, by), MULT(cz, dw));
        m3xyzw = SUB(SET_ALL_ZERO(), m3xyzw);  // trick to be compatible with dot2
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_0, m3xy);
        pre_send_to_live(P_0, m3xz);
        pre_send_to_live(P_0, m3xw);
        pre_send_to_live(P_0, m3yz);
        pre_send_to_live(P_0, m3yw);
        pre_send_to_live(P_0, m3zw);
        pre_send_to_live(P_0, m3xyz);
        pre_send_to_live(P_0, m3xyw);
        pre_send_to_live(P_0, m3xzw);
        pre_send_to_live(P_0, m3yzw);
        pre_send_to_live(P_2, mxy);
        pre_send_to_live(P_2, mxz);
        pre_send_to_live(P_2, mxw);
        pre_send_to_live(P_2, myz);
        pre_send_to_live(P_2, myw);
        pre_send_to_live(P_2, mzw);
        pre_send_to_live(P_2, mxyz);
        pre_send_to_live(P_2, mxyw);
        pre_send_to_live(P_2, mxzw);
        pre_send_to_live(P_2, myzw);
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3xw);
        send_to_live(P_0, m3yz);
        send_to_live(P_0, m3yw);
        send_to_live(P_0, m3zw);
        send_to_live(P_0, m3xyz);
        send_to_live(P_0, m3xyw);
        send_to_live(P_0, m3xzw);
        send_to_live(P_0, m3yzw);
        send_to_live(P_2, mxy);
        send_to_live(P_2, mxz);
        send_to_live(P_2, mxw);
        send_to_live(P_2, myz);
        send_to_live(P_2, myw);
        send_to_live(P_2, mzw);
        send_to_live(P_2, mxyz);
        send_to_live(P_2, mxyw);
        send_to_live(P_2, mxzw);
        send_to_live(P_2, myzw);
#endif
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3xw);
        send_to_live(P_0, m3yz);
        send_to_live(P_0, m3yw);
        send_to_live(P_0, m3zw);
        send_to_live(P_0, m3xyz);
        send_to_live(P_0, m3xyw);
        send_to_live(P_0, m3xzw);
        send_to_live(P_0, m3yzw);
        store_compare_view(P_2, mxy);
        store_compare_view(P_2, mxz);
        store_compare_view(P_2, mxw);
        store_compare_view(P_2, myz);
        store_compare_view(P_2, myw);
        store_compare_view(P_2, mzw);
        store_compare_view(P_2, mxyz);
        store_compare_view(P_2, mxyw);
        store_compare_view(P_2, mxzw);
        store_compare_view(P_2, myzw);
#endif
        OEC_MAL3_Share e;
        e.r0 = m3xyzw;
        e.r1 = mxyzw;
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_mult4(const OEC_MAL3_Share b,
                                 const OEC_MAL3_Share c,
                                 const OEC_MAL3_Share d,
                                 func_add ADD,
                                 func_sub SUB,
                                 func_mul MULT) const
    {
        Datatype mxy = SUB(MULT(r1, b.r1), getRandomVal(P_013));
        Datatype mxz = SUB(MULT(r1, c.r1), getRandomVal(P_013));
        Datatype mxw = SUB(MULT(r1, d.r1), getRandomVal(P_013));
        Datatype myz = SUB(MULT(b.r1, c.r1), getRandomVal(P_013));
        Datatype myw = SUB(MULT(b.r1, d.r1), getRandomVal(P_013));
        Datatype mzw = SUB(MULT(c.r1, d.r1), getRandomVal(P_013));
        Datatype mxyz = SUB(MULT(MULT(r1, b.r1), c.r1), getRandomVal(P_013));
        Datatype mxyw = SUB(MULT(MULT(r1, b.r1), d.r1), getRandomVal(P_013));
        Datatype mxzw = SUB(MULT(MULT(r1, c.r1), d.r1), getRandomVal(P_013));
        Datatype myzw = SUB(MULT(MULT(b.r1, c.r1), d.r1), getRandomVal(P_013));
        Datatype mxyzw = SUB(MULT(MULT(r1, b.r1), MULT(c.r1, d.r1)), getRandomVal(P_013));
        Datatype ax = ADD(r0, r1);
        Datatype by = ADD(b.r0, b.r1);
        Datatype cz = ADD(c.r0, c.r1);
        Datatype dw = ADD(d.r0, d.r1);
        Datatype m3xy = SUB(MULT(ax, by), getRandomVal(P_123));
        Datatype m3xz = SUB(MULT(ax, cz), getRandomVal(P_123));
        Datatype m3xw = SUB(MULT(ax, dw), getRandomVal(P_123));
        Datatype m3yz = SUB(MULT(by, cz), getRandomVal(P_123));
        Datatype m3yw = SUB(MULT(by, dw), getRandomVal(P_123));
        Datatype m3zw = SUB(MULT(cz, dw), getRandomVal(P_123));
        Datatype m3xyz = SUB(MULT(MULT(ax, by), cz), getRandomVal(P_123));
        Datatype m3xyw = SUB(MULT(MULT(ax, by), dw), getRandomVal(P_123));
        Datatype m3xzw = SUB(MULT(MULT(ax, cz), dw), getRandomVal(P_123));
        Datatype m3yzw = SUB(MULT(MULT(by, cz), dw), getRandomVal(P_123));
        Datatype m3xyzw = SUB(MULT(MULT(ax, by), MULT(cz, dw)), getRandomVal(P_123));
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
        pre_send_to_live(P_0, m3xy);
        pre_send_to_live(P_0, m3xz);
        pre_send_to_live(P_0, m3xw);
        pre_send_to_live(P_0, m3yz);
        pre_send_to_live(P_0, m3yw);
        pre_send_to_live(P_0, m3zw);
        pre_send_to_live(P_0, m3xyz);
        pre_send_to_live(P_0, m3xyw);
        pre_send_to_live(P_0, m3xzw);
        pre_send_to_live(P_0, m3yzw);
        pre_send_to_live(P_0, m3xyzw);
        pre_send_to_live(P_2, mxy);
        pre_send_to_live(P_2, mxz);
        pre_send_to_live(P_2, mxw);
        pre_send_to_live(P_2, myz);
        pre_send_to_live(P_2, myw);
        pre_send_to_live(P_2, mzw);
        pre_send_to_live(P_2, mxyz);
        pre_send_to_live(P_2, mxyw);
        pre_send_to_live(P_2, mxzw);
        pre_send_to_live(P_2, myzw);
        pre_send_to_live(P_2, mxyzw);
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3xw);
        send_to_live(P_0, m3yz);
        send_to_live(P_0, m3yw);
        send_to_live(P_0, m3zw);
        send_to_live(P_0, m3xyz);
        send_to_live(P_0, m3xyw);
        send_to_live(P_0, m3xzw);
        send_to_live(P_0, m3yzw);
        send_to_live(P_0, m3xyzw);
        send_to_live(P_2, mxy);
        send_to_live(P_2, mxz);
        send_to_live(P_2, mxw);
        send_to_live(P_2, myz);
        send_to_live(P_2, myw);
        send_to_live(P_2, mzw);
        send_to_live(P_2, mxyz);
        send_to_live(P_2, mxyw);
        send_to_live(P_2, mxzw);
        send_to_live(P_2, myzw);
        send_to_live(P_2, mxyzw);
#endif
#else
        send_to_live(P_0, m3xy);
        send_to_live(P_0, m3xz);
        send_to_live(P_0, m3xw);
        send_to_live(P_0, m3yz);
        send_to_live(P_0, m3yw);
        send_to_live(P_0, m3zw);
        send_to_live(P_0, m3xyz);
        send_to_live(P_0, m3xyw);
        send_to_live(P_0, m3xzw);
        send_to_live(P_0, m3yzw);
        send_to_live(P_0, m3xyzw);
        store_compare_view(P_2, mxy);
        store_compare_view(P_2, mxz);
        store_compare_view(P_2, mxw);
        store_compare_view(P_2, myz);
        store_compare_view(P_2, myw);
        store_compare_view(P_2, mzw);
        store_compare_view(P_2, mxyz);
        store_compare_view(P_2, mxyw);
        store_compare_view(P_2, mxzw);
        store_compare_view(P_2, myzw);
        store_compare_view(P_2, mxyzw);
#endif
        OEC_MAL3_Share e;
        e.r0 = getRandomVal(P_123);
        e.r1 = ADD(getRandomVal(P_013), getRandomVal(P_023));
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
    }
#endif

#if USE_CUDA_GEMM == 2

    static void CONV_2D(const OEC_MAL3_Share* X,
                        const OEC_MAL3_Share* W,
                        OEC_MAL3_Share* Y,
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

        UINT_TYPE* r0 = new UINT_TYPE[factor * xSize];
        UINT_TYPE* r1 = new UINT_TYPE[factor * xSize];
        UINT_TYPE* b_r1 = new UINT_TYPE[wSize];
        UINT_TYPE* br1_br0 = new UINT_TYPE[wSize];
        UINT_TYPE* r_br = new UINT_TYPE[factor * ySize];
        UINT_TYPE* r_br1_br0 = new UINT_TYPE[factor * ySize];
        UINT_TYPE* b_r_r0 = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&X[i].r0, temp, 1);
            for (int j = 0; j < factor; j++)
                r0[j * xSize + i] = temp[j];
            unorthogonalize_arithmetic(&X[i].r1, temp, 1);
            for (int j = 0; j < factor; j++)
                r1[j * xSize + i] = temp[j];
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].r1, temp, 1);
            b_r1[i] = temp[0];
            auto temp2 = OP_SUB(W[i].r1, W[i].r0);
            unorthogonalize_arithmetic(&temp2, temp, 1);
            br1_br0[i] = temp[0];
        }

        conv2d_cutlass(r1, b_r1, r_br, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(r0, b_r1, b_r_r0, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(r1, br1_br0, r_br1_br0, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = r_br1_br0[j * ySize + i] + b_r_r0[j * ySize + i];
            orthogonalize_arithmetic(temp, &Y[i].r0, 1);
            for (int j = 0; j < factor; j++)
                temp[j] = r_br[j * ySize + i];
            orthogonalize_arithmetic(temp, &Y[i].r1, 1);
        }

        delete[] r0;
        delete[] r1;
        delete[] b_r1;
        delete[] br1_br0;
        delete[] r_br;
        delete[] r_br1_br0;
        delete[] b_r_r0;
    }
#elif USE_CUDA_GEMM == 4

    static void CONV_2D(const OEC_MAL3_Share* X,
                        const OEC_MAL3_Share* W,
                        OEC_MAL3_Share* Y,
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

        UINT_TYPE* r0 = new UINT_TYPE[factor * xSize];
        UINT_TYPE* r1 = new UINT_TYPE[factor * xSize];
        UINT_TYPE* b_r1 = new UINT_TYPE[wSize];
        UINT_TYPE* br1_br0 = new UINT_TYPE[wSize];
        UINT_TYPE* r_br = new UINT_TYPE[factor * ySize];
        UINT_TYPE* r_br1_br0 = new UINT_TYPE[factor * ySize];
        UINT_TYPE* b_r_r0 = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&X[i].r0, r0 + i * factor, 1);
            unorthogonalize_arithmetic(&X[i].r1, r1 + i * factor, 1);
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].r1, temp, 1);
            b_r1[i] = temp[0];
            auto temp2 = OP_SUB(W[i].r1, W[i].r0);
            unorthogonalize_arithmetic(&temp2, temp, 1);
            br1_br0[i] = temp[0];
        }

        conv2d_cutlass(r1, b_r1, r_br, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(r0, b_r1, b_r_r0, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(r1, br1_br0, r_br1_br0, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = r_br1_br0[i * factor + j] + b_r_r0[i * factor + j];
            orthogonalize_arithmetic(temp, &Y[i].r0, 1);
            for (int j = 0; j < factor; j++)
                temp[j] = r_br[i * factor + j];
            orthogonalize_arithmetic(temp, &Y[i].r1, 1);
        }

        delete[] r0;
        delete[] r1;
        delete[] b_r1;
        delete[] br1_br0;
        delete[] r_br;
        delete[] r_br1_br0;
        delete[] b_r_r0;
    }

#endif
#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1

    static void GEMM(OEC_MAL3_Share* a, OEC_MAL3_Share* b, OEC_MAL3_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;

        UINT_TYPE* br1 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* br1_br0 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* r0;
        UINT_TYPE* r1;
        if (a_fixed)
        {
            r0 = new UINT_TYPE[a_size];
            r1 = new UINT_TYPE[a_size];
        }
        else
        {
            r0 = new UINT_TYPE[factor * a_size];
            r1 = new UINT_TYPE[factor * a_size];
        }
        UINT_TYPE* r_br = new UINT_TYPE[factor * c_size];
        UINT_TYPE* r_br1_br0 = new UINT_TYPE[factor * c_size];
        UINT_TYPE* b_r_r0 = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].r0, temp, 1);
            if (a_fixed)
            {
                r0[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    r0[j * a_size + i] = temp[j];

            unorthogonalize_arithmetic(&a[i].r1, temp, 1);
            if (a_fixed)
            {
                r1[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    r1[j * a_size + i] = temp[j];
        }
        for (int i = 0; i < b_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&b[i].r1, temp, 1);
            for (int j = 0; j < factor; j++)
                br1[j * b_size + i] = temp[j];
            auto temp2 = OP_SUB(b[i].r1, b[i].r0);
            unorthogonalize_arithmetic(&temp2, temp, 1);
            for (int j = 0; j < factor; j++)
                br1_br0[j * b_size + i] = temp[j];
        }

        for (int i = 0; i < factor; i++)
        {
            if (a_fixed)
            {
                gemm_cutlass(m, n, k, r1, &br1[i * b_size], &r_br[i * c_size]);
                gemm_cutlass(m, n, k, r1, &br1_br0[i * b_size], &r_br1_br0[i * c_size]);
                gemm_cutlass(m, n, k, r0, &br1[i * b_size], &b_r_r0[i * c_size]);
            }

            else
            {
                gemm_cutlass(m, n, k, &r1[i * a_size], &br1[i * b_size], &r_br[i * c_size]);
                gemm_cutlass(m, n, k, &r1[i * a_size], &br1_br0[i * b_size], &r_br1_br0[i * c_size]);
                gemm_cutlass(m, n, k, &r0[i * a_size], &br1[i * b_size], &b_r_r0[i * c_size]);
            }
        }

        for (int j = 0; j < c_size; j++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int i = 0; i < factor; i++)
                temp[i] = r_br1_br0[i * c_size + j] + b_r_r0[i * c_size + j];
            orthogonalize_arithmetic(temp, &c[j].r0, 1);
            for (int i = 0; i < factor; i++)
                temp[i] = r_br[i * c_size + j];
            orthogonalize_arithmetic(temp, &c[j].r1, 1);
        }

        delete[] r0;
        delete[] r1;
        delete[] br1;
        delete[] br1_br0;
        delete[] r_br;
        delete[] r_br1_br0;
        delete[] b_r_r0;
    }
#else

    static void GEMM(OEC_MAL3_Share* a, OEC_MAL3_Share* b, OEC_MAL3_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;
        UINT_TYPE* br1 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* br1_br0 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* r0;
        UINT_TYPE* r1;
        if (a_fixed)
        {
            r0 = new UINT_TYPE[a_size];
            r1 = new UINT_TYPE[a_size];
        }
        else
        {
            r0 = new UINT_TYPE[factor * a_size];
            r1 = new UINT_TYPE[factor * a_size];
        }

        UINT_TYPE* r_br = new UINT_TYPE[factor * c_size];
        UINT_TYPE* r_br1_br0 = new UINT_TYPE[factor * c_size];
        UINT_TYPE* b_r_r0 = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].r0, temp, 1);
            if (a_fixed)
            {
                r0[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    r0[j * a_size + i] = temp[j];

            unorthogonalize_arithmetic(&a[i].r1, temp, 1);
            if (a_fixed)
            {
                r1[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    r1[j * a_size + i] = temp[j];
        }

        if (a_fixed)
        {
            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    unorthogonalize_arithmetic(&b[i * n + j].r1, temp, 1);
                    for (int l = 0; l < factor; l++)
                        br1[i * n * factor + l * n + j] = temp[l];
                    auto temp2 = OP_SUB(b[i * n + j].r1, b[i * n + j].r0);
                    unorthogonalize_arithmetic(&temp2, temp, 1);
                    for (int l = 0; l < factor; l++)
                        br1_br0[i * n * factor + l * n + j] = temp[l];
                }
            }

            gemm_cutlass(m, n * factor, k, r1, br1, r_br);
            gemm_cutlass(m, n * factor, k, r1, br1_br0, r_br1_br0);
            gemm_cutlass(m, n * factor, k, r0, br1, b_r_r0);

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    for (int l = 0; l < factor; l++)
                        temp[l] = r_br1_br0[i * n * factor + l * n + j] + b_r_r0[i * n * factor + l * n + j];
                    orthogonalize_arithmetic(temp, &c[i * n + j].r0, 1);
                    for (int l = 0; l < factor; l++)
                        temp[l] = r_br[i * n * factor + l * n + j];
                    orthogonalize_arithmetic(temp, &c[i * n + j].r1, 1);
                }
            }
        }
        else
        {

            for (int i = 0; i < b_size; i++)
            {
                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&b[i].r1, temp, 1);
                for (int j = 0; j < factor; j++)
                    br1[j * b_size + i] = temp[j];
                auto temp2 = OP_SUB(b[i].r1, b[i].r0);
                unorthogonalize_arithmetic(&temp2, temp, 1);
                for (int j = 0; j < factor; j++)
                    br1_br0[j * b_size + i] = temp[j];
            }

            for (int i = 0; i < factor; i++)
            {
                gemm_cutlass(m, n, k, &r1[i * a_size], &br1[i * b_size], &r_br[i * c_size]);
                gemm_cutlass(m, n, k, &r1[i * a_size], &br1_br0[i * b_size], &r_br1_br0[i * c_size]);
                gemm_cutlass(m, n, k, &r0[i * a_size], &br1[i * b_size], &b_r_r0[i * c_size]);
            }

            for (int j = 0; j < c_size; j++)
            {
                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                for (int i = 0; i < factor; i++)
                    temp[i] = r_br1_br0[i * c_size + j] + b_r_r0[i * c_size + j];
                orthogonalize_arithmetic(temp, &c[j].r0, 1);
                for (int i = 0; i < factor; i++)
                    temp[i] = r_br[i * c_size + j];
                orthogonalize_arithmetic(temp, &c[j].r1, 1);
            }
        }

        delete[] r0;
        delete[] r1;
        delete[] br1;
        delete[] br1_br0;
        delete[] r_br;
        delete[] r_br1_br0;
        delete[] b_r_r0;
    }
#endif
#endif

#endif
};
