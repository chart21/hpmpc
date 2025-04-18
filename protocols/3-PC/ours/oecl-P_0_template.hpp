#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE OECL0_Share
/* #define VALS_PER_SHARE 2 */
#define SHARE OECL0_Share

template <typename Datatype>
class OECL0_Share
{
  private:
    Datatype p1;
    Datatype p2;

  public:
    // static constexpr int VALS_PER_SHARE = 2;

    OECL0_Share() {}
    OECL0_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}
    OECL0_Share(Datatype p1) : p1(p1) {}

    static OECL0_Share public_val(Datatype a) { return OECL0_Share(SET_ALL_ZERO(), SET_ALL_ZERO()); }

    OECL0_Share Not() const { return OECL0_Share(p1, p2); }

    template <typename func_add>
    OECL0_Share Add(OECL0_Share b, func_add ADD) const
    {
        return OECL0_Share(ADD(p1, b.p1), ADD(p2, b.p2));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_dot(const OECL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OECL0_Share c;
        /* #if FRACTIONAL > 0 */
        /* c.p1 = SUB( MULT(p1,b.p1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  )); // -> -e = x2y2 - (x1-x2)(y1-y2) = x1 y2 +
         * x2 y+1 - x1_y1 */
        c.p1 = SUB(MULT(SUB(p1, p2), SUB(b.p1, b.p2)),
                   MULT(p1, b.p1));  // -> e = (x1-x2)(y1-y2) - x2y2 = x1 y1 - x1 y2 - x2 y1
        /* #else */
        /* c.p1 = SUB( MULT(p1,b.p1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  )); // e = x2y2 - (x1-x2)(y1-y2) */

        /* #endif */
        return c;
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_dot(const OECL0_Share b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OECL0_Share c;
        if (i == 0)
            c.p1 = MULT(p1, b.p1);  // -> e = (x1-x2)(y1-y2) - x2y2 = x1 y1 - x1 y2 - x2 y1
        else
            c.p2 = MULT(SUB(p1, p2), SUB(b.p1, b.p2));
        return c;
    }

    template <typename func_add, typename func_sub>
    void join_dots(OECL0_Share c[], func_add ADD, func_sub SUB)
    {
        p1 = ADD(p1, SUB(c[0].p1, c[1].p1));
        p2 = SET_ALL_ZERO();
    }

    static constexpr int getNumDotProducts() { return 2; }
#endif
    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        Datatype r01 = getRandomVal(P_1);
        Datatype z1 = getRandomVal(P_1);
        Datatype z2 = getRandomVal(P_2);
#if PRE == 1
        pre_send_to_live(P_2, SUB(r01, p1));
#else
        send_to_live(P_2, SUB(r01, p1));
#endif
        p1 = z2;
        p2 = z1;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_mult(OECL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Datatype maskP_1 = getRandomVal(P_1);
/* Datatype maskP_1_2 = getRandomVal(P_1); */
/* Datatype maskP_2 = getRandomVal(P_2); */
#if PRE == 1
        pre_send_to_live(P_2, SUB(ADD(MULT(p1, b.p1), maskP_1), MULT(SUB(p1, p2), SUB(b.p1, b.p2))));
#else
        send_to_live(P_2, SUB(ADD(MULT(p1, b.p1), maskP_1), MULT(SUB(p1, p2), SUB(b.p1, b.p2))));
#endif
        // for arithmetic circuits this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 +
        // b.p2)
        return OECL0_Share(getRandomVal(P_2), getRandomVal(P_1));
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    void prepare_reveal_to_all() const
    {
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
        pre_send_to_live(P_1, p1);
        pre_send_to_live(P_2, p2);
#else
        send_to_live(P_1, p1);
        send_to_live(P_2, p2);
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share(p2);
#endif
#if PRE == 1
        return p1;
#else
        return SUB(receive_from_live(P_2), p2);
#endif
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_0)
        {
#if OPT_SHARE == 1
            p2 = getRandomVal(P_1);                  // r0,1
            p1 = SUB(SET_ALL_ZERO(), ADD(val, p2));  // share -(a + r0,1)
#if PRE == 1 && SHARE_PREP == 1
            pre_send_to_live(P_2, p1);  // share -(a + r0,1) to P_2
#else
            send_to_live(P_2, p1);
#endif
#else
            p1 = getRandomVal(P_2);  // P_1 does not need to the share -> thus not srng but 2 -> with updated share
                                     // conversion it needs it
            p2 = getRandomVal(P_1);
            Datatype input = val;
#if PRE == 1
            pre_send_to_live(P_1, ADD(p1, input));
            pre_send_to_live(P_2, ADD(p2, input));
#else
            send_to_live(P_1, ADD(p1, input));
            send_to_live(P_2, ADD(p2, input));
#endif
#endif
        }
        else if constexpr (id == P_1)
        {
            p1 = SET_ALL_ZERO();
            p2 = getRandomVal(P_1);
        }
        else if constexpr (id == P_2)  // id ==2
        {
            p1 = getRandomVal(P_2);
            p2 = SET_ALL_ZERO();
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

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        auto result = ADD(p1, p2);
        for (int i = 2; i <= b; i *= 2)
            result = OP_TRUNC2(result);
        OECL0_Share res;
        res.p2 = getRandomVal(P_1);
        res.p1 = SUB(result, res.p2);
#if PRE == 1
        pre_send_to_live(P_2, res.p1);
#else
        send_to_live(P_2, res.p1);
#endif
        return res;
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_Share prepare_mult_public_fixed(const Datatype b,
                                          func_mul MULT,
                                          func_add ADD,
                                          func_sub SUB,
                                          func_trunc TRUNC,
                                          int fractional_bits = FRACTIONAL) const
    {
        /* #if TRUNC_THEN_MULT == 1 */
        /*         auto result = MULT(TRUNC(ADD(p1,p2),fractional_bits),b); */
        /* #else */
        auto result = TRUNC(MULT(ADD(p1, p2), b), fractional_bits);
        /* #endif */
        OECL0_Share res;
        res.p2 = getRandomVal(P_1);
        res.p1 = SUB(result, res.p2);

#if PRE == 1
        pre_send_to_live(P_2, res.p1);
#else
        send_to_live(P_2, res.p1);
#endif
        return res;
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_Share prepare_trunc_share(func_mul MULT,
                                    func_add ADD,
                                    func_sub SUB,
                                    func_trunc TRUNC,
                                    int fractional_bits = FRACTIONAL) const
    {
        auto result = SUB(SET_ALL_ZERO(), TRUNC(SUB(SET_ALL_ZERO(), ADD(p1, p2)), fractional_bits));
        OECL0_Share res;
        res.p2 = getRandomVal(P_1);
        res.p1 = SUB(result, res.p2);

#if PRE == 1
        pre_send_to_live(P_2, res.p1);
#else
        send_to_live(P_2, res.p1);
#endif
        return res;
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_dot_add(OECL0_Share a, OECL0_Share b, OECL0_Share& c, func_add ADD, func_sub SUB, func_mul MULT)
    {
        c.p1 = ADD(c.p1, SUB(MULT(a.p1, b.p1), MULT(SUB(a.p1, a.p2), SUB(b.p1, b.p2))));
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        Datatype maskP_1 = getRandomVal(P_1);
        Datatype maskP_1_2 = getRandomVal(P_1);
        Datatype maskP_2 = getRandomVal(P_2);

        p1 = SUB(TRUNC(ADD(ADD(p1, maskP_1), maskP_2)), maskP_1_2);  // (e + r0,1 + r0,2)^t - z_1
        p2 = maskP_1_2;                                              // z_1

#if PRE == 1
        pre_send_to_live(P_2, p1);
#else
        send_to_live(P_2, p1);
#endif
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    /* template <typename func_add, typename func_sub, typename func_trunc> */
    /* void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
    /* { */
    /* Datatype maskP_1 = getRandomVal(P_1); */
    /* Datatype maskP_1_2 = getRandomVal(P_1); */
    /* Datatype maskP_2 = getRandomVal(P_2); */

    /* p1 = ADD( TRUNC(ADD(ADD(p1,maskP_1),maskP_2)), maskP_1_2); // (e + r0,1 + r0,2)^t + r0,1_2 */
    /* p2 = SUB(SET_ALL_ZERO(),maskP_1_2); // - r0,1_2 */

    /* #if PRE == 1 */
    /* pre_send_to_live(P_2, p1); */
    /* #else */
    /* send_to_live(P_2, p1); */
    /* #endif */
    /* } */

    template <typename func_mul>
    OECL0_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return OECL0_Share(MULT(p1, b), MULT(p2, b));
    }

    void get_random_B2A()
    {
        p1 = getRandomVal(P_2);
        p2 = getRandomVal(P_1);
    }

    static void prepare_A2B_S1(int m, int k, OECL0_Share in[], OECL0_Share out[])
    {
        for (int i = m; i < k; i++)
        {
            /* out[i].p1 = getRandomVal(P_2); // set share to r0,2 */
            out[i - m].p1 = SET_ALL_ZERO();  // set share to 0
            out[i - m].p2 = SET_ALL_ZERO();  // set other share to 0
        }
    }

    /* static void get_dabit(OECL0_Share bool_out, OECL0_Share arith_out[]) */
    /* { */
    /*     Datatype temp[BITLENGTH]{0}; */
    /*     bool_out.p1 = getRandomVal(P_2); */
    /*     bool_out.p2 = getRandomVal(P_1); */
    /*     temp[BITLENGTH - 1] = FUNC_XOR(bool_out.p1,bool_out.p2); */
    /*     alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE]; */
    /*     unorthogonalize_boolean(temp, temp2); */
    /*     orthogonalize_arithmetic(temp2, temp); */
    /*     for(int i = 0; i < BITLENGTH; i++) */
    /*     { */
    /*         arith_out[i].p2 = getRandomVal(P_1); // set second share to r0,1 */
    /*         arith_out[i].p1 = OP_SUB(SET_ALL_ZERO(), OP_ADD(temp[i], out[i].p2)) ; // set first share to -(x0 + r0,1)
     */
    /*         #if PRE == 1 */
    /*             pre_send_to_live(P_2, arith_out[i].p1); //  - (x0 + r0,1) to P_2 */
    /*         #else */
    /*             send_to_live(P_2, arith_out[i].p1); // - (x0 + r0,1) to P_2 */
    /*         #endif */

    /*     } */
    /* } */

    static void prepare_A2B_S2(int m, int k, OECL0_Share in[], OECL0_Share out[])
    {
        // convert share x0 to boolean
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = OP_SUB(SET_ALL_ZERO(), OP_ADD(in[j].p1, in[j].p2));  // set share to -x0
        }
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_arithmetic(temp, temp2);
        orthogonalize_boolean(temp2, temp);
        /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
        /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */

        for (int i = m; i < k; i++)
        {
            out[i - m].p2 = getRandomVal(P_1);                 // set second share to r0,1
            out[i - m].p1 = FUNC_XOR(temp[i], out[i - m].p2);  // set first share to -x0 xor r0,1
#if PRE == 1
            pre_send_to_live(P_2, out[i - m].p1);  // -x0 xor r0,1 to P_2
#else
            send_to_live(P_2, out[i - m].p1);  // -x0 xor r0,1 to P_2
#endif
        }
        /* out[0].p1 = FUNC_NOT(out[0].p1);// change sign bit -> -x0 xor r0,1 to x0 xor r0,1 */
    }

    static void complete_A2B_S1(int k, OECL0_Share out[]) {}
    static void complete_A2B_S2(int k, OECL0_Share out[]) {}

    static void prepare_B2A(OECL0_Share z[], OECL0_Share random_mask[], OECL0_Share out[])
    {
        // 1. Reveal z to P_1 and P_2
        for (int i = 0; i < BITLENGTH; i++)
        {
            /* send_to_live(P_1, z[i].p1); */
            /* send_to_live(P_2, z[i].p2); */
            z[i].p1 = SET_ALL_ZERO();  // set mask to 0 since it is reveald
            z[i].p2 = SET_ALL_ZERO();  // set mask to 0 since it is reveald
        }
        // 2. Share random mask
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = FUNC_XOR(random_mask[j].p1, random_mask[j].p2);  // set share to r01 xor r02
        }
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(temp, temp2);
        orthogonalize_arithmetic(temp2, temp);
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template prepare_receive_from<P_0>(temp[i], OP_ADD, OP_SUB);
        }
    }

    static void complete_B2A(OECL0_Share z[], OECL0_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
            out[i] = z[i].Add(out[i], OP_SUB);  // calculate z - randmon mask
        }
    }

    void prepare_opt_bit_injection(OECL0_Share x[], OECL0_Share out[])
    {
        Datatype y0[BITLENGTH]{0};
        y0[BITLENGTH - 1] = FUNC_XOR(p1, p2);  // convert b to an arithemtic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(y0, temp2);
        orthogonalize_arithmetic(temp2, y0);
        for (int i = 0; i < BITLENGTH; i++)
        {
            Datatype r01 = getRandomVal(P_1);
            Datatype r01_2 = getRandomVal(P_1);
            Datatype m00 = OP_SUB(y0[i], r01);
            Datatype m01 = OP_SUB(OP_MULT(OP_ADD(x[i].p1, x[i].p2), y0[i]), r01_2);
#if PRE == 1
            pre_send_to_live(P_2, m00);
            pre_send_to_live(P_2, m01);
#else
            send_to_live(P_2, m00);
            send_to_live(P_2, m01);
#endif
            out[i].p1 = getRandomVal(P_2);  // set share to z_2
            out[i].p2 = getRandomVal(P_1);  // set other share to z_1
        }
    }

    void complete_opt_bit_injection() {}

    void prepare_bit2a(OECL0_Share out[])
    {
        Datatype temp[BITLENGTH]{0};
        temp[BITLENGTH - 1] = FUNC_XOR(p1, p2);  // convert y_0 to an arithmetic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(temp, temp2);
        orthogonalize_arithmetic(temp2, temp);

        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            pre_send_to_live(P_2, OP_SUB(temp[i], getRandomVal(P_1)));  // send y_0 - r_0,1 to P_2
#else
            send_to_live(P_2, OP_SUB(temp[i], getRandomVal(P_1)));  // send y_0 - r_0,1 to P_2
#endif
            out[i].p1 = getRandomVal(P_2);  // set share to z_2
            out[i].p2 = getRandomVal(P_1);  // set other share to z_1
        }
    }

    void complete_bit2a() {}

    void prepare_bit_injection_S1(OECL0_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].p1 = SET_ALL_ZERO();  // set share to 0
            out[i].p2 = SET_ALL_ZERO();  // set other share to 0
        }
    }

    void prepare_bit_injection_S2(OECL0_Share out[])
    {
        Datatype temp[BITLENGTH]{0};
        temp[BITLENGTH - 1] = FUNC_XOR(p1, p2);
        /* unorthogonalize_boolean(temp,(UINT_TYPE*)temp); */
        /* orthogonalize_arithmetic((UINT_TYPE*) temp,  temp); */
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(temp, temp2);
        orthogonalize_arithmetic(temp2, temp);
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].p2 = getRandomVal(P_1);                                   // set second share to r0,1
            out[i].p1 = OP_SUB(SET_ALL_ZERO(), OP_ADD(temp[i], out[i].p2));  // set first share to -(x0 + r0,1)
#if PRE == 1
            pre_send_to_live(P_2, out[i].p1);  //  - (x0 + r0,1) to P_2
#else
            send_to_live(P_2, out[i].p1);                           // - (x0 + r0,1) to P_2
#endif
        }
    }

    static void complete_bit_injection_S1(OECL0_Share out[]) {}

    static void complete_bit_injection_S2(OECL0_Share out[]) {}

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_dot3(const OECL0_Share b, const OECL0_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Datatype x0 = ADD(p1, p2);
        Datatype y0 = ADD(b.p1, b.p2);
        Datatype z0 = ADD(c.p1, c.p2);
        Datatype mxy = SUB(MULT(x0, y0), getRandomVal(P_1));
        Datatype mxz = SUB(MULT(x0, z0), getRandomVal(P_1));
        Datatype myz = SUB(MULT(y0, z0), getRandomVal(P_1));
        Datatype mxyz = MULT(MULT(x0, y0), z0);
#if PRE == 1
        pre_send_to_live(P_2, mxy);
        pre_send_to_live(P_2, mxz);
        pre_send_to_live(P_2, myz);
#else
        send_to_live(P_2, mxy);
        send_to_live(P_2, mxz);
        send_to_live(P_2, myz);
#endif
        // for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 +
        // b.p2)
        return OECL0_Share(mxyz, SET_ALL_ZERO());
    }
    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_dot4(OECL0_Share b, OECL0_Share c, OECL0_Share d, func_add ADD, func_sub SUB, func_mul MULT)
        const
    {
        Datatype x0 = ADD(p1, p2);
        Datatype y0 = ADD(b.p1, b.p2);
        Datatype z0 = ADD(c.p1, c.p2);
        Datatype w0 = ADD(d.p1, d.p2);
        Datatype xy = MULT(x0, y0);
        Datatype xz = MULT(x0, z0);
        Datatype xw = MULT(x0, w0);
        Datatype yz = MULT(y0, z0);
        Datatype yw = MULT(y0, w0);
        Datatype zw = MULT(z0, w0);
        Datatype mxy = SUB(xy, getRandomVal(P_1));
        Datatype mxz = SUB(xz, getRandomVal(P_1));
        Datatype mxw = SUB(xw, getRandomVal(P_1));
        Datatype myz = SUB(yz, getRandomVal(P_1));
        Datatype myw = SUB(yw, getRandomVal(P_1));
        Datatype mzw = SUB(zw, getRandomVal(P_1));
        Datatype mxyz = SUB(MULT(xy, z0), getRandomVal(P_1));
        Datatype mxyw = SUB(MULT(xy, w0), getRandomVal(P_1));
        Datatype mxzw = SUB(MULT(xz, w0), getRandomVal(P_1));
        Datatype myzw = SUB(MULT(yz, w0), getRandomVal(P_1));
        Datatype mxyzw = MULT(xy, zw);
        mxyzw = SUB(SET_ALL_ZERO(), mxyzw);  // trick do be comptatible with 2PC dot product
#if PRE == 1
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
/* pre_send_to_live(P_2, mxyzw); */
#else
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
/* send_to_live(P_2, mxyzw); */
#endif
        // for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 +
        // b.p2)
        return OECL0_Share(mxyzw, SET_ALL_ZERO());
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_mult3(OECL0_Share b, OECL0_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Datatype x0 = ADD(p1, p2);
        Datatype y0 = ADD(b.p1, b.p2);
        Datatype z0 = ADD(c.p1, c.p2);
        Datatype mxy = SUB(MULT(x0, y0), getRandomVal(P_1));
        Datatype mxz = SUB(MULT(x0, z0), getRandomVal(P_1));
        Datatype myz = SUB(MULT(y0, z0), getRandomVal(P_1));
        Datatype mxyz = SUB(MULT(MULT(x0, y0), z0), getRandomVal(P_1));
#if PRE == 1
        pre_send_to_live(P_2, mxy);
        pre_send_to_live(P_2, mxz);
        pre_send_to_live(P_2, myz);
        pre_send_to_live(P_2, mxyz);
#else
        send_to_live(P_2, mxy);
        send_to_live(P_2, mxz);
        send_to_live(P_2, myz);
        send_to_live(P_2, mxyz);
#endif
        // for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 +
        // b.p2)
        return OECL0_Share(getRandomVal(P_2), getRandomVal(P_1));
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_mult4(OECL0_Share b, OECL0_Share c, OECL0_Share d, func_add ADD, func_sub SUB, func_mul MULT)
        const
    {
        Datatype x0 = ADD(p1, p2);
        Datatype y0 = ADD(b.p1, b.p2);
        Datatype z0 = ADD(c.p1, c.p2);
        Datatype w0 = ADD(d.p1, d.p2);
        Datatype xy = MULT(x0, y0);
        Datatype xz = MULT(x0, z0);
        Datatype xw = MULT(x0, w0);
        Datatype yz = MULT(y0, z0);
        Datatype yw = MULT(y0, w0);
        Datatype zw = MULT(z0, w0);
        Datatype mxy = SUB(xy, getRandomVal(P_1));
        Datatype mxz = SUB(xz, getRandomVal(P_1));
        Datatype mxw = SUB(xw, getRandomVal(P_1));
        Datatype myz = SUB(yz, getRandomVal(P_1));
        Datatype myw = SUB(yw, getRandomVal(P_1));
        Datatype mzw = SUB(zw, getRandomVal(P_1));
        Datatype mxyz = SUB(MULT(xy, z0), getRandomVal(P_1));
        Datatype mxyw = SUB(MULT(xy, w0), getRandomVal(P_1));
        Datatype mxzw = SUB(MULT(xz, w0), getRandomVal(P_1));
        Datatype myzw = SUB(MULT(yz, w0), getRandomVal(P_1));
        Datatype mxyzw = SUB(MULT(xy, zw), getRandomVal(P_1));
#if PRE == 1
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
        // for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 +
        // b.p2)
        return OECL0_Share(getRandomVal(P_2), getRandomVal(P_1));
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OECL0_Share& r_mk2,
                                 OECL0_Share& r_msb,
                                 OECL0_Share& c,
                                 OECL0_Share& c_prime,
                                 int fractional_bits = FRACTIONAL) const
    {
        /* Datatype rmk2 = (ADD(p1,p2) << 1) >> (FRACTIONAL + 1); */
        /* Datatype rmsb = ADD(p1,p2) >> (BITLENGTH - 1); */
        Datatype rmk2 = OP_SHIFT_LOG_RIGHTF(OP_SHIFT_LEFT<1>(ADD(p1, p2)), fractional_bits + 1);
        Datatype rmsb = OP_SHIFT_LOG_RIGHT<BITLENGTH - 1>(ADD(p1, p2));
        /* Datatype rmk2 = OP_SHIFT_LOG_RIGHTF( OP_SHIFT_LEFT<1>(SUB(SET_ALL_ZERO(),ADD(p1,p2))), fractional_bits+1 );
         */
        /* Datatype rmsb = OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(SUB(SET_ALL_ZERO(),ADD(p1,p2))); */
        /* Datatype rmk2 = SUB(SET_ALL_ZERO(), OP_SHIFT_LOG_RIGHT<FRACTIONAL+1>(
         * OP_SHIFT_LEFT<1>(SUB(SET_ALL_ZERO(),ADD(p1,p2))) )); */
        /* Datatype rmsb = SUB(SET_ALL_ZERO(), OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(SUB(SET_ALL_ZERO(),ADD(p1,p2)))); */
        /* Datatype rmk2 = ADD(p1,p2); */
        /* Datatype rmsb = ADD(p1,p2); */

        /* Datatype rmk2 = (ADD(p1,p2) << 1) >> (FRACTIONAL + 1); */
        /* Datatype rmsb = ADD(p1,p2) >> (BITLENGTH - 1); */
        r_mk2.prepare_receive_from<PSELF>(rmk2, ADD, SUB);
        r_msb.prepare_receive_from<PSELF>(rmsb, ADD, SUB);
        /* r_mk2.template prepare_receive_from<PSELF>(400, ADD, SUB); */
        /* r_msb.template prepare_receive_from<PSELF>(600, ADD, SUB); */
        c.p1 = SET_ALL_ZERO();
        c.p2 = SET_ALL_ZERO();
        c_prime.p1 = SET_ALL_ZERO();
        c_prime.p2 = SET_ALL_ZERO();
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OECL0_Share& r_mk2,
                                  OECL0_Share& r_msb,
                                  OECL0_Share& c,
                                  OECL0_Share& c_prime) const
    {
        r_mk2.template complete_receive_from<PSELF>(ADD, SUB);
        r_msb.template complete_receive_from<PSELF>(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OECL0_Share prepare_trunc_exact_xmod2t(func_add ADD,
                                           func_sub SUB,
                                           func_xor XOR,
                                           func_and AND,
                                           int fractional_bits = FRACTIONAL) const
    {
        Datatype lx = ADD(p1, p2);
        // Step 1, Compute [x/2t] -> delt with public mult fixed
        // Step 2, Compute [x mod t]
        UINT_TYPE maskValue = (UINT_TYPE(1) << (fractional_bits)) - 1;
        Datatype mask = PROMOTE(maskValue);  // Set all elements to maskValue
        // Apply the mask using bitwise AND
        Datatype lxmodt = AND(lx, mask);  // mod 2^t
        // Step3, Compute [x]^B -> delt with prepareA2B
        return OECL0_Share(lxmodt, SET_ALL_ZERO());
    }

    /* #if USE_CUDA_GEMM == 1 */

    /*     template <typename func_add, typename func_sub, typename func_mul> */
    /* static void GEMM(OECL0_Share* a, OECL0_Share* b, OECL0_Share* c, int m, int n, int k, func_add ADD, func_sub SUB,
     * func_mul MULT) */
    /* { */
    /*     const int factor = DATTYPE/BITLENGTH; */
    /*     const int mn = m * n; */
    /*     const int mk = m * k; */
    /*     const int nk = n * k; */

    /*     UINT_TYPE* p1_p2 = NEW(UINT_TYPE[factor * mk]); */
    /*     UINT_TYPE* p1 = NEW(UINT_TYPE[factor * mk]); */
    /*     UINT_TYPE* bp1_bp2 = NEW(UINT_TYPE[factor * nk]); */
    /*     UINT_TYPE* bp1 = NEW(UINT_TYPE[factor * nk]); */
    /*     UINT_TYPE* cp1_1 = NEW(UINT_TYPE[factor * mn]); */
    /*     UINT_TYPE* cp1_2 = NEW(UINT_TYPE[factor * mn]); */

    /*     for (int i = 0; i < m; i++) */
    /*     { */
    /*         for (int j = 0; j < k; j++) */
    /*         { */
    /*             alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
    /*             unorthogonalize_arithmetic(&a[i * k + j].p1, temp, 1); */
    /*             for (int l = 0; l < factor; l++) */
    /*                 p1[l * mk + i * k + j] = temp[l];  // Access p1 like a 1D array */
    /*             auto p1minp2 = SUB(a[i * k + j].p1, a[i * k + j].p2); // p1 - p2 */
    /*             unorthogonalize_arithmetic(&p1minp2, temp, 1); */
    /*             for (int l = 0; l < factor; l++) */
    /*                 p1_p2[l * mk + i * k + j] = temp[l];  // Access p1_p2 like a 1D array */
    /*         } */
    /*     } */

    /*     for (int i = 0; i < k; i++) */
    /*     { */
    /*         for (int j = 0; j < n; j++) */
    /*         { */
    /*             alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
    /*             unorthogonalize_arithmetic(&b[i * n + j].p1, temp, 1); */
    /*             for (int l = 0; l < factor; l++) */
    /*                 bp1[l * nk + i * n + j] = temp[l];  // Access bp1 like a 1D array */
    /*             auto bp1minbp2 = SUB(b[i * n + j].p1, b[i * n + j].p2); // bp1 - bp2 */
    /*             unorthogonalize_arithmetic(&bp1minbp2, temp, 1); */
    /*             for (int l = 0; l < factor; l++) */
    /*                 bp1_bp2[l * nk + i * n + j] = temp[l];  // Access bp1_bp2 like a 1D array */
    /*         } */
    /*     } */

    /*     for (int i = 0; i < factor; i++) */
    /*     { */
    /*         gemm_cutlass(m,n,k,&p1[i * mk], &bp1[i * nk], &cp1_1[i * mn]); */
    /*         gemm_cutlass(m,n,k,&p1_p2[i * mk], &bp1_bp2[i * nk], &cp1_2[i * mn]); */

    /*         /1* test_cuda(); *1/ */
    /*     } */

    /*     for (int j = 0; j < mn; j++) */
    /*     { */
    /*         alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
    /*         for (int i = 0; i < factor; i++) */
    /*             temp[i] = cp1_2[i * mn + j] - cp1_1[i * mn + j]; */
    /*         orthogonalize_arithmetic(temp, &c[j].p1, 1); */
    /*     } */

    /*     /1* for(int i = 0; i < m; i++) *1/ */
    /*     /1* { *1/ */
    /*     /1*     for(int j = 0; j < k; j++) *1/ */
    /*     /1*     { *1/ */
    /*     /1*         auto p1minp2 = SUB(a[i*k+j].p1,a[i*k+j].p2); *1/ */
    /*     /1*         alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; *1/ */
    /*     /1*         unorthogonalize_arithmetic(&p1minp2, temp, 1); *1/ */
    /*     /1*         for(int l = 0; l < factor; l++) *1/ */
    /*     /1*             p1_p2[l*mk + i*k + j] = temp[l]; *1/ */
    /*     /1*         unorthogonalize_arithmetic(&a[i * k + j].p1, temp, 1); *1/ */
    /*     /1*         for (int l = 0; l < factor; l++) *1/ */
    /*     /1*             p1[l * mk + i * k + j] = temp[l];  // Access p1 like a 1D array *1/ */

    /*     /1*     } *1/ */
    /*     /1* } *1/ */

    /*     /1* for(int i = 0; i < k; i++) *1/ */
    /*     /1* { *1/ */
    /*     /1*     for(int j = 0; j < n; j++) *1/ */
    /*     /1*     { *1/ */
    /*     /1*         auto bp1minbp2 = SUB(b[i*n+j].p1, b[i*n+j].p2); *1/ */
    /*     /1*         alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; *1/ */
    /*     /1*         unorthogonalize_arithmetic(&bp1minbp2, temp, 1); *1/ */
    /*     /1*         for(int l = 0; l < factor; l++) *1/ */
    /*     /1*             bp1_bp2[l*kn + i*n + j] = temp[l]; *1/ */
    /*     /1*         unorthogonalize_arithmetic(&b[i * n + j ].p1, temp, 1); *1/ */
    /*     /1*         for (int l = 0; l < factor; l++) *1/ */
    /*     /1*             bp1[l * kn + i * n + j] = temp[l];  // Access bp1 like a 1D array *1/ */
    /*     /1*     } *1/ */
    /*     /1* } *1/ */

    /*     /1* for(int i = 0; i < factor; i++) *1/ */
    /*     /1* { *1/ */
    /*     /1*     gemm_cutlass(m,n,k,p1_p2 + i*mk, bp1_bp2 + i*kn, cp1_1 + i*mn); *1/ */
    /*     /1*     gemm_cutlass(m,n,k,p1 + i*mk, bp1 + i*kn, cp1_2 + i*mn); *1/ */
    /*     /1*     for(int j = 0; j < mn; j++) *1/ */
    /*     /1*     { *1/ */
    /*     /1*         cp1_1[i*mn + j] = cp1_1[i*mn + j] - cp1_2[i*mn + j]; *1/ */
    /*     /1*     } *1/ */
    /*     /1* } *1/ */

    /*     /1* for(int j = 0; j < mn; j++) *1/ */
    /*     /1* { *1/ */
    /*     /1*     alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; *1/ */
    /*     /1*     for(int i = 0; i < factor; i++) *1/ */
    /*     /1*         temp[i] = cp1_1[i*mn + j]; *1/ */
    /*     /1*     orthogonalize_arithmetic(temp, &c[j].p1, 1); *1/ */
    /*     /1* } *1/ */

    /*     delete[] p1_p2; */
    /*     delete[] bp1_bp2; */
    /*     delete[] p1; */
    /*     delete[] bp1; */
    /*     delete[] cp1_1; */
    /*     delete[] cp1_2; */
    /* } */

#if USE_CUDA_GEMM == 2

    static void CONV_2D(const OECL0_Share* X,
                        const OECL0_Share* W,
                        OECL0_Share* Y,
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
        const int factor = DATTYPE / BITLENGTH;  // e.g. 16 if 32-bit values are vectorized in a 512-bit register
        const int xSize = inh * inw * din * batchSize;
        const int wSize = wh * ww * din * dout;
        const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
        const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
        const int ySize = out_h * out_w * dout * batchSize;
        batchSize *= factor;

        UINT_TYPE* x_p1 = new UINT_TYPE[factor * xSize];
        UINT_TYPE* x_p1_x_p2 = new UINT_TYPE[factor * xSize];
        UINT_TYPE* w_p1 = new UINT_TYPE[wSize];  // W is always constant
        UINT_TYPE* w_p1_w_p2 = new UINT_TYPE[wSize];
        UINT_TYPE* y_p1 = new UINT_TYPE[factor * ySize];
        UINT_TYPE* y_p1_2 = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&X[i].p1, temp, 1);
            for (int j = 0; j < factor; j++)
                x_p1[j * xSize + i] = temp[j];
            auto temp2 = OP_SUB(X[i].p1, X[i].p2);
            unorthogonalize_arithmetic(&temp2, temp, 1);
            for (int j = 0; j < factor; j++)
                x_p1_x_p2[j * xSize + i] = temp[j];
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].p1, temp, 1);
            w_p1[i] = temp[0];
            auto temp2 = OP_SUB(W[i].p1, W[i].p2);
            unorthogonalize_arithmetic(&temp2, temp, 1);
            w_p1_w_p2[i] = temp[0];
        }
        conv2d_cutlass(x_p1, w_p1, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(x_p1_x_p2, w_p1_w_p2, y_p1_2, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = y_p1_2[j * ySize + i] - y_p1[j * ySize + i];
            orthogonalize_arithmetic(temp, &Y[i].p1, 1);
        }

        delete[] x_p1;
        delete[] x_p1_x_p2;
        delete[] w_p1;
        delete[] w_p1_w_p2;
        delete[] y_p1;
        delete[] y_p1_2;
    }

#elif USE_CUDA_GEMM == 4

    static void CONV_2D(const OECL0_Share* X,
                        const OECL0_Share* W,
                        OECL0_Share* Y,
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
        const int factor = DATTYPE / BITLENGTH;  // e.g. 16 if 32-bit values are vectorized in a 512-bit register
        const int xSize = inh * inw * din * batchSize;
        const int wSize = wh * ww * din * dout;
        const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
        const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
        const int ySize = out_h * out_w * dout * batchSize;
        batchSize *= factor;

        alignas(sizeof(Datatype)) UINT_TYPE* x_p1 = new UINT_TYPE[factor * xSize];
        alignas(sizeof(Datatype)) UINT_TYPE* x_p1_x_p2 = new UINT_TYPE[factor * xSize];
        alignas(sizeof(Datatype)) UINT_TYPE* w_p1 = new UINT_TYPE[wSize];  // W is always constant
        alignas(sizeof(Datatype)) UINT_TYPE* w_p1_w_p2 = new UINT_TYPE[wSize];
        alignas(sizeof(Datatype)) UINT_TYPE* y_p1 = new UINT_TYPE[factor * ySize];
        alignas(sizeof(Datatype)) UINT_TYPE* y_p1_2 = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            unorthogonalize_arithmetic(&X[i].p1, x_p1 + i * factor, 1);
            auto temp2 = OP_SUB(X[i].p1, X[i].p2);
            unorthogonalize_arithmetic(&temp2, x_p1_x_p2 + i * factor, 1);
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].p1, temp, 1);
            w_p1[i] = temp[0];
            auto temp2 = OP_SUB(W[i].p1, W[i].p2);
            unorthogonalize_arithmetic(&temp2, temp, 1);
            w_p1_w_p2[i] = temp[0];
        }
        // spawn two CPU threads to compute the two convolutions
        conv2d_cutlass(x_p1, w_p1, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
        conv2d_cutlass(x_p1_x_p2, w_p1_w_p2, y_p1_2, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = y_p1_2[i * factor + j] - y_p1[i * factor + j];
            orthogonalize_arithmetic(temp, &Y[i].p1, 1);
        }

        delete[] x_p1;
        delete[] x_p1_x_p2;
        delete[] w_p1;
        delete[] w_p1_w_p2;
        delete[] y_p1;
        delete[] y_p1_2;
    }
#endif
#if USE_CUDA_GEMM > 0

#if USE_CUDA_GEMM == 1

    static void GEMM(OECL0_Share* a, OECL0_Share* b, OECL0_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;
        UINT_TYPE* p1;
        UINT_TYPE* p1_p2;
        if (a_fixed)
        {
            p1 = new UINT_TYPE[a_size];
            p1_p2 = new UINT_TYPE[a_size];
        }
        else
        {
            p1 = new UINT_TYPE[factor * a_size];
            p1_p2 = new UINT_TYPE[factor * a_size];
        }
        UINT_TYPE* bp1 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* bp1_bp2 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* cp1_1 = new UINT_TYPE[factor * c_size];
        UINT_TYPE* cp1_2 = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].p1, temp, 1);
            if (a_fixed)
            {
                p1[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    p1[j * a_size + i] = temp[j];
            auto p1minp2 = OP_SUB(a[i].p1, a[i].p2);
            unorthogonalize_arithmetic(&p1minp2, temp, 1);
            if (a_fixed)
            {
                p1_p2[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    p1_p2[j * a_size + i] = temp[j];
        }

        for (int i = 0; i < b_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&b[i].p1, temp, 1);
            for (int j = 0; j < factor; j++)
                bp1[j * b_size + i] = temp[j];
            auto bp1minbp2 = OP_SUB(b[i].p1, b[i].p2);
            unorthogonalize_arithmetic(&bp1minbp2, temp, 1);
            for (int j = 0; j < factor; j++)
                bp1_bp2[j * b_size + i] = temp[j];
        }

        for (int i = 0; i < factor; i++)
        {
            if (a_fixed)
            {

                gemm_cutlass(m, n, k, p1, &bp1[i * b_size], &cp1_1[i * c_size]);
                gemm_cutlass(m, n, k, p1_p2, &bp1_bp2[i * b_size], &cp1_2[i * c_size]);
            }

            else
            {
                gemm_cutlass(m, n, k, &p1[i * a_size], &bp1[i * b_size], &cp1_1[i * c_size]);
                gemm_cutlass(m, n, k, &p1_p2[i * a_size], &bp1_bp2[i * b_size], &cp1_2[i * c_size]);
            }
        }

        for (int j = 0; j < c_size; j++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int i = 0; i < factor; i++)
                temp[i] = cp1_2[i * c_size + j] - cp1_1[i * c_size + j];
            orthogonalize_arithmetic(temp, &c[j].p1, 1);
        }

        delete[] p1;
        delete[] p1_p2;
        delete[] bp1;
        delete[] bp1_bp2;
        delete[] cp1_1;
        delete[] cp1_2;
    }

#else
    // If matrix A is fixed, then this method converts l kxn matrices B_1,B_2,...B_l to a matrix B of size kx(l*n) and
    // computes A*B.
    static void GEMM(OECL0_Share* a, OECL0_Share* b, OECL0_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;
        UINT_TYPE* p1;
        UINT_TYPE* p1_p2;
        if (a_fixed)
        {
            p1 = new UINT_TYPE[a_size];
            p1_p2 = new UINT_TYPE[a_size];
        }
        else
        {
            p1 = new UINT_TYPE[factor * a_size];
            p1_p2 = new UINT_TYPE[factor * a_size];
        }
        UINT_TYPE* bp1 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* bp1_bp2 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* cp1_1 = new UINT_TYPE[factor * c_size];
        UINT_TYPE* cp1_2 = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].p1, temp, 1);
            if (a_fixed)
            {
                p1[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    p1[j * a_size + i] = temp[j];
            auto p1minp2 = OP_SUB(a[i].p1, a[i].p2);
            unorthogonalize_arithmetic(&p1minp2, temp, 1);
            if (a_fixed)
            {
                p1_p2[i] = temp[0];
            }
            else
                for (int j = 0; j < factor; j++)
                    p1_p2[j * a_size + i] = temp[j];
        }

        if (a_fixed)
        {

            for (int i = 0; i < k; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    unorthogonalize_arithmetic(&b[i * n + j].p1, temp, 1);
                    for (int l = 0; l < factor; l++)
                        bp1[i * n * factor + l * n + j] = temp[l];
                    auto bp1minbp2 = OP_SUB(b[i * n + j].p1, b[i * n + j].p2);
                    unorthogonalize_arithmetic(&bp1minbp2, temp, 1);
                    for (int l = 0; l < factor; l++)
                        bp1_bp2[i * n * factor + l * n + j] = temp[l];
                }
            }

            gemm_cutlass(m, n * factor, k, p1, bp1, cp1_1);
            gemm_cutlass(m, n * factor, k, p1_p2, bp1_bp2, cp1_2);

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    for (int l = 0; l < factor; l++)
                        temp[l] = cp1_2[i * n * factor + j + l * n] - cp1_1[i * n * factor + j + l * n];
                    orthogonalize_arithmetic(temp, &c[j + i * n].p1, 1);
                }
            }
        }
        else
        {
            for (int i = 0; i < b_size; i++)
            {
                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&b[i].p1, temp, 1);
                for (int j = 0; j < factor; j++)
                    bp1[j * b_size + i] = temp[j];
                auto bp1minbp2 = OP_SUB(b[i].p1, b[i].p2);
                unorthogonalize_arithmetic(&bp1minbp2, temp, 1);
                for (int j = 0; j < factor; j++)
                    bp1_bp2[j * b_size + i] = temp[j];
            }

            for (int i = 0; i < factor; i++)
            {
                gemm_cutlass(m, n, k, &p1[i * a_size], &bp1[i * b_size], &cp1_1[i * c_size]);
                gemm_cutlass(m, n, k, &p1_p2[i * a_size], &bp1_bp2[i * b_size], &cp1_2[i * c_size]);
            }

            for (int j = 0; j < c_size; j++)
            {
                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                for (int i = 0; i < factor; i++)
                    temp[i] = cp1_2[i * c_size + j] - cp1_1[i * c_size + j];
                orthogonalize_arithmetic(temp, &c[j].p1, 1);
            }
        }

        delete[] p1;
        delete[] p1_p2;
        delete[] bp1;
        delete[] bp1_bp2;
        delete[] cp1_1;
        delete[] cp1_2;
    }
#endif
#endif
    /* template <typename func_add, typename func_sub, typename func_mul> */
    /* static void GEMM(const OECL0_Share* a, const OECL0_Share* b, OECL0_Share* c, int m, int n, int k, func_add ADD,
     * func_sub SUB, func_mul MULT) */
    /* { */
    /* const int factor = DATTYPE/BITLENGTH; */
    /* UINT_TYPE* p1_p2 = NEW(UINT_TYPE[factor][m*k]); */
    /* UINT_TYPE* p1 = NEW(UINT_TYPE[factor][m*k]); */
    /* UINT_TYPE* bp1_bp2 = NEW(UINT_TYPE[factor][k*n]); */
    /* UINT_TYPE* bp1 = NEW(UINT_TYPE[factor][k*n]); */
    /* UINT_TYPE* cp1_1 = NEW(UINT_TYPE[factor][m*n]); */
    /* UINT_TYPE* cp1_2 = NEW(UINT_TYPE[factor][m*n]); */

    /* for(int i = 0; i < m; i++) */
    /* { */
    /* for(int j = 0; j < k; j++) */
    /* { */
    /* auto p1minp2 = SUB(a[i*k+j].p1,a[i*k+j].p2); */
    /* alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
    /* unorthogonalize_arithmetic(p1minp2, temp,1); */
    /* for(int l = 0; l < factor; l++) */
    /*     p1_p2[l][i*k+j] = temp[l]; */
    /* unorthogonalize_arithmetic(p1, temp,1); */
    /* for(int l = 0; l < factor; l++) */
    /*     p1[l][j] = temp[l]; */
    /* } */
    /* } */

    /* for(int i = 0; i < k; i++) */
    /* { */
    /* for(int j = 0; j < n; j++) */
    /* { */
    /* auto bp1minbp2 = SUB(b[i*n+j].p1,b[i*n+j].p2); */
    /* alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
    /* unorthogonalize_arithmetic(bp1minbp2, temp,1); */
    /* for(int l = 0; l < factor; l++) */
    /*     bp1_bp2[l][i*n+j] = temp[l]; */
    /* unorthogonalize_arithmetic(bp1, temp,1); */
    /* for(int l = 0; l < factor; l++) */
    /*     bp1[l][j] = temp[l]; */
    /* } */
    /* } */

    /* for(int i = 0; i < factor; i++) */
    /* { */
    /* CUDA_GEMM(p1_p2[i], bp1_bp2[i], cp1_1[i], m, n, k); */
    /* CUDA_GEMM(p1[i], bp1[i], cp1_2[i], m, n, k); */
    /* for(int j = 0; j < m*n; j++) */
    /* { */
    /*     cp1_1[i][j] = cp1_1[i][j] - cp1_2[i][j]; */
    /* } */
    /* } */

    /* for(int j = 0; j < m*n; j++) */
    /* { */
    /*     alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
    /*     for(int i = 0; i < factor; i++) */
    /*         temp[i] = cp1_1[i][j]; */
    /*     orthogonalize_arithmetic(temp, c[j].p1,1); */
    /* } */
    /* delete p1_p2; */
    /* delete bp1_bp2; */
    /* delete p1; */
    /* delete bp1; */
    /* delete cp1_1; */
    /* delete cp1_2; */
    /* } */
    /* c.p1 = SUB( MULT( SUB(p1,p2), SUB(b.p1,b.p2)), MULT(p1,b.p1)  ); // -> e = (x1-x2)(y1-y2) - x2y2 = x1 y1 - x1 y2
     * - x2 y1 */
    /* c.p1 = SUB( MULT( SUB(p1,p2), SUB(b.p1,b.p2)), MULT(p1,b.p1)  ); // -> e = (x1-x2)(y1-y2) - x2y2 = x1 y1 - x1 y2
     * - x2 y1 */

#endif
};
