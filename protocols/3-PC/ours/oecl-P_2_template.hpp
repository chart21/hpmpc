#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL2_Share
{
    Datatype p1;
    Datatype p2;

  public:
    OECL2_Share() {}
    OECL2_Share(Datatype p1) : p1(p1) {}
    OECL2_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}

    static OECL2_Share public_val(Datatype a) { return OECL2_Share(a, SET_ALL_ZERO()); }

    OECL2_Share Not() const { return OECL2_Share(NOT(p1), p2); }

    template <typename func_add>
    OECL2_Share Add(OECL2_Share b, func_add ADD) const
    {
        return OECL2_Share(ADD(p1, b.p1), ADD(p2, b.p2));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_dot(const OECL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OECL2_Share c;
        c.p1 = MULT(p1, b.p1);  // ab_2 + e_2, e_2 = x1 y_1
        return c;
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_dot(const OECL2_Share b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OECL2_Share c;
        c.p1 = MULT(p1, b.p1);
        return c;
    }

    template <typename func_add, typename func_sub>
    void join_dots(OECL2_Share c[], func_add ADD, func_sub SUB)
    {
        p1 = ADD(c[0].p1, p1);
    }

    static constexpr int getNumDotProducts() { return 1; }
#endif

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        p1 = ADD(pre_receive_from_live(P_0), p1);
#else
        p1 = ADD(receive_from_live(P_0), p1);
#endif
        p2 = getRandomVal(P_0);
        send_to_live(P_1, ADD(p1, p2));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult(OECL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        /* OECL2_Share c; */
        Datatype cp2 = getRandomVal(P_0);  // P_2 mask for P_1
#if PRE == 1
        Datatype cp1 = ADD(pre_receive_from_live(P_0), MULT(p1, b.p1));  // P_0_message + (a+rr) (b+rl)
#else
        Datatype cp1 = ADD(receive_from_live(P_0), MULT(p1, b.p1));  // P_0_message + (a+rr) (b+rl)
#endif

        send_to_live(P_1, ADD(cp1, cp2));
        return OECL2_Share(cp1, cp2);  // (a+rr) (b+rl)
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        p1 = SUB(p1, receive_from_live(P_1));
    }

    void prepare_reveal_to_all() const { send_to_live(P_0, p1); }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 1 && (OPT_SHARE == 0 || \
                 SHARE_PREP == 1)  // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
        return SUB(p1, pre_receive_from_live(P_0));
#else
        return SUB(p1, receive_from_live(P_0));
#endif
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_2)
        {
            p1 = val;
            p2 = getRandomVal(P_0);
            /* p1 = getRandomVal(0); *1/ */
            send_to_live(P_1, ADD(p1, p2));
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_0)
        {
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
            p2 = pre_receive_from_live(P_0);
#else
            p2 = receive_from_live(P_0);
#endif
            p1 = SUB(SET_ALL_ZERO(), p2);  // set own share to - - (a + r0,1)
        }
        else if constexpr (id == P_1)
        {
            p1 = receive_from_live(P_1);
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

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL2_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        auto result = ADD(p1, p2);
        for (int i = 2; i <= b; i *= 2)
            result = OP_TRUNC2(result);

        OECL2_Share res(result);
        return res;
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL2_Share prepare_mult_public_fixed(const Datatype b,
                                          func_mul MULT,
                                          func_add ADD,
                                          func_sub SUB,
                                          func_trunc TRUNC,
                                          int fractional_bits = FRACTIONAL) const
    {
        OECL2_Share res;
        /* #if TRUNC_THEN_MULT == 1 */
        /*     res.p1 = MULT(TRUNC(ADD(p1,p2),fractional_bits),b); */
        /* #else */
        res.p1 = TRUNC(MULT(ADD(p1, p2), b), fractional_bits);
        /* #endif */
        return res;
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL2_Share prepare_trunc_share(func_mul MULT,
                                    func_add ADD,
                                    func_sub SUB,
                                    func_trunc TRUNC,
                                    int fractional_bits = FRACTIONAL) const
    {
        return OECL2_Share(TRUNC(ADD(p1, p2), fractional_bits));
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        p2 = pre_receive_from_live(P_0);
#else
        p2 = receive_from_live(P_0);
#endif
        p1 = SUB(p1, p2);
    }

    template <typename func_mul>
    OECL2_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return OECL2_Share(MULT(p1, b), MULT(p2, b));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_dot_add(OECL2_Share a, OECL2_Share b, OECL2_Share& c, func_add ADD, func_sub SUB, func_mul MULT)
    {
        c.p1 = ADD(c.p1, MULT(a.p1, b.p1));
    }
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {

        p1 = ADD(p1, getRandomVal(P_0));  // a1b1 + r_0,2
        send_to_live(P_1, p1);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
#if PRE == 1
        p2 = pre_receive_from_live(P_0);  // (e + r0,1 + r0,2)^T - r_0,1
#else
        p2 = receive_from_live(P_0);  // (e + r0,1 + r0,2)^T - r_0,1
#endif
        p1 = SUB(TRUNC(SUB(p1, receive_from_live(P_1))), p2);  // [m2 -m1]^t - m^0
    }

    /* template <typename func_add, typename func_sub, typename func_trunc> */
    /* void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
    /* { */

    /* p1 = ADD(p1, getRandomVal(P_0)); // ab_2 + e_2 + r0,2 */
    /* send_to_live(P_1, p1); // ab_2 + e_2 + r0,2 */
    /* } */

    /* template <typename func_add, typename func_sub, typename func_trunc> */
    /* void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
    /* { */
    /* #if PRE == 1 */
    /* p2 = pre_receive_from_live(P_0); // (e + r0,1 + r0,2)^T + r0,1_2 */
    /* #else */
    /* p2 = receive_from_live(P_0); // (e + r0,1 + r0,2)^T + r0,1_2 */
    /* #endif */
    /* p1 = TRUNC( SUB(p1,receive_from_live(P_1))); // (ab + e + r0,1 + r0,2)^T */
    /* p1 = SUB(p1, p2); // - [ ( (e + r0,1 + r0,2)^T + r0,1_2 ) ] */
    /* } */

    void get_random_B2A()
    {
        p1 = getRandomVal(P_0);
        p2 = p1;
    }

    // higher level functions

    static void prepare_B2A(OECL2_Share z[], OECL2_Share random_mask[], OECL2_Share out[])
    {
        // 1. Reveal z to P_1 and P_2
        for (int i = 0; i < BITLENGTH; i++)
        {
            /* send_to_live(P_1, z[i].p1); */
            send_to_live(P_1, z[i].p2);  // reveal z to P_2
            z[i].p2 = SET_ALL_ZERO();    // set mask to 0 since it is reveald
        }
        // 2. Share random mask
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template prepare_receive_from<P_0>(OP_ADD, OP_SUB);
        }
    }

    static void complete_B2A(OECL2_Share z_bool[], OECL2_Share out[])
    {
        Datatype z[BITLENGTH];
        for (int i = 0; i < BITLENGTH; i++)
            z[i] = FUNC_XOR(z_bool[i].p1, receive_from_live(P_1));
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(z, temp2);
        orthogonalize_arithmetic(temp2, z);

        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
            out[i] = public_val(z[i]).Add(out[i], OP_SUB);  // z - r
        }
    }

    static void prepare_A2B_S1(int m, int k, OECL2_Share in[], OECL2_Share out[])
    {
        // convert share a + x1 to boolean
        Datatype temp[BITLENGTH];
        for (int i = 0; i < BITLENGTH; i++)
        {
            temp[i] = OP_ADD(in[i].p1, in[i].p2);  // set share to a + x_0
        }
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_arithmetic(temp, temp2);
        orthogonalize_boolean(temp2, temp);
        /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
        /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */
        for (int i = m; i < k; i++)
        {
            out[i - m].p1 = temp[i];
            out[i - m].p2 = SET_ALL_ZERO();
        }
    }

    static void prepare_A2B_S2(int m, int k, OECL2_Share in[], OECL2_Share out[]) {}

    static void complete_A2B_S1(int k, OECL2_Share out[]) {}

    static void complete_A2B_S2(int k, OECL2_Share out[])
    {
        for (int i = 0; i < k; i++)
        {
#if PRE == 1
            out[i].p1 = pre_receive_from_live(P_0);
#else
            out[i].p1 = receive_from_live(P_0);
#endif
            out[i].p2 = out[i].p1;  // set both shares to -x0 xor r0,1
        }
        /* out[0].p2 = FUNC_NOT(out[0].p2);// change sign bit -> -x0 xor r0,1 to x0 xor r0,1 */
    }

    void prepare_opt_bit_injection(OECL2_Share a[], OECL2_Share out[])
    {
        Datatype b0[BITLENGTH]{0};
        b0[BITLENGTH - 1] = FUNC_XOR(p1, p2);  // convert b to an arithemtic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(b0, temp2);
        orthogonalize_arithmetic(temp2, b0);
        for (int i = 0; i < BITLENGTH; i++)
        {
            Datatype a0 = OP_ADD(a[i].p1, a[i].p2);
            Datatype z2 = getRandomVal(P_0);
#if PRE == 1
            Datatype m00 = pre_receive_from_live(P_0);
            Datatype m01 = pre_receive_from_live(P_0);
#else
            Datatype m00 = receive_from_live(P_0);
            Datatype m01 = receive_from_live(P_0);
#endif
            Datatype m1 = OP_SUB(OP_ADD(b0[i], b0[i]), PROMOTE(1));
            m1 = OP_MULT(m1, OP_SUB(m01, OP_MULT(a0, m00)));
            m1 = OP_SUB(m1, OP_MULT(b0[i], a[i].p2));
            send_to_live(P_1, OP_ADD(m1, z2));
            out[i].p1 = OP_ADD(m1, OP_MULT(a0, b0[i]));
            out[i].p2 = z2;
        }
    }

    void complete_opt_bit_injection() { p1 = OP_ADD(p1, receive_from_live(P_1)); }

    void prepare_bit2a(OECL2_Share out[])
    {
        Datatype b0[BITLENGTH]{0};
        b0[BITLENGTH - 1] = FUNC_XOR(p1, p2);  // convert b_0 to an arithmetic value
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_boolean(b0, temp2);
        orthogonalize_arithmetic(temp2, b0);

        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            Datatype m0 = pre_receive_from_live(P_0);
#else
            Datatype m0 = receive_from_live(P_0);
#endif
            Datatype m2_prime = OP_SUB(m0, OP_MULT(OP_ADD(b0[i], b0[i]), m0));
            out[i].p1 = OP_ADD(m2_prime, b0[i]);  // set share to m2' + b_0
            out[i].p2 = getRandomVal(P_0);        // set other share to z_1
            send_to_live(P_1, OP_ADD(m2_prime, out[i].p2));
        }
    }

    void complete_bit2a() { p1 = OP_ADD(p1, receive_from_live(P_1)); }

    void prepare_bit_injection_S1(OECL2_Share out[])
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
            out[i].p1 = temp[i];         // set share to b xor x_0
            out[i].p2 = SET_ALL_ZERO();  // set other share to 0
        }
    }

    void prepare_bit_injection_S2(OECL2_Share out[]) {}

    static void complete_bit_injection_S1(OECL2_Share out[]) {}

    static void complete_bit_injection_S2(OECL2_Share out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            out[i].p2 = pre_receive_from_live(P_0);
#else
            out[i].p2 = receive_from_live(P_0);
#endif
            out[i].p1 = OP_SUB(SET_ALL_ZERO(), out[i].p2);  // set first share to x0 + r0,1
        }
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_dot3(const OECL2_Share b, const OECL2_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        Datatype rxy = pre_receive_from_live(P_0);
        Datatype rxz = pre_receive_from_live(P_0);
        Datatype ryz = pre_receive_from_live(P_0);
#else
        Datatype rxy = receive_from_live(P_0);
        Datatype rxz = receive_from_live(P_0);
        Datatype ryz = receive_from_live(P_0);
#endif

        Datatype a0 = ADD(p1, p2);
        Datatype b0 = ADD(b.p1, b.p2);
        Datatype c0 = ADD(c.p1, c.p2);

        OECL2_Share d;
        d.p1 = ADD(ADD(MULT(a0, ADD(MULT(b0, SUB(c0, c.p2)), ryz)), (MULT(b0, SUB(rxz, MULT(c0, p2))))),
                   MULT(c0, SUB(rxy, MULT(a0, b.p2))));  // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
        d.p2 = SET_ALL_ZERO();
        return d;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult3(OECL2_Share b, OECL2_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        Datatype rxy = pre_receive_from_live(P_0);
        Datatype rxz = pre_receive_from_live(P_0);
        Datatype ryz = pre_receive_from_live(P_0);
        Datatype rxyz = pre_receive_from_live(P_0);
#else
        Datatype rxy = receive_from_live(P_0);
        Datatype rxz = receive_from_live(P_0);
        Datatype ryz = receive_from_live(P_0);
        Datatype rxyz = receive_from_live(P_0);
#endif

        Datatype a0 = ADD(p1, p2);
        Datatype b0 = ADD(b.p1, b.p2);
        Datatype c0 = ADD(c.p1, c.p2);

        OECL2_Share d;
        d.p1 = SUB(ADD(ADD(MULT(a0, ADD(MULT(b0, SUB(c0, c.p2)), ryz)), (MULT(b0, SUB(rxz, MULT(c0, p2))))),
                       MULT(c0, SUB(rxy, MULT(a0, b.p2)))),
                   rxyz);  // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
        d.p2 = getRandomVal(P_0);
        send_to_live(P_1, ADD(d.p1, d.p2));
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
        p1 = ADD(p1, receive_from_live(P_1));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_dot4(OECL2_Share b, OECL2_Share c, OECL2_Share d, func_add ADD, func_sub SUB, func_mul MULT)
        const
    {
#if PRE == 1
        Datatype rxy = pre_receive_from_live(P_0);
        Datatype rxz = pre_receive_from_live(P_0);
        Datatype rxw = pre_receive_from_live(P_0);
        Datatype ryz = pre_receive_from_live(P_0);
        Datatype ryw = pre_receive_from_live(P_0);
        Datatype rzw = pre_receive_from_live(P_0);
        Datatype rxyz = pre_receive_from_live(P_0);
        Datatype rxyw = pre_receive_from_live(P_0);
        Datatype rxzw = pre_receive_from_live(P_0);
        Datatype ryzw = pre_receive_from_live(P_0);
#else
        Datatype rxy = receive_from_live(P_0);
        Datatype rxz = receive_from_live(P_0);
        Datatype rxw = receive_from_live(P_0);
        Datatype ryz = receive_from_live(P_0);
        Datatype ryw = receive_from_live(P_0);
        Datatype rzw = receive_from_live(P_0);
        Datatype rxyz = receive_from_live(P_0);
        Datatype rxyw = receive_from_live(P_0);
        Datatype rxzw = receive_from_live(P_0);
        Datatype ryzw = receive_from_live(P_0);
#endif

        Datatype a0 = ADD(p1, p2);
        Datatype b0 = ADD(b.p1, b.p2);
        Datatype c0 = ADD(c.p1, c.p2);
        Datatype d0 = ADD(d.p1, d.p2);

        OECL2_Share e;
        e.p1 =

            ADD(ADD(MULT(a0, SUB(MULT(d0, ADD(MULT(b0, SUB(c0, c.p2)), ryz)), ryzw)),
                    MULT(b0, ADD(MULT(a0, SUB(rzw, MULT(c0, d.p2))), SUB(MULT(c0, rxw), rxzw)))),
                ADD(MULT(c0, SUB(MULT(a0, SUB(ryw, MULT(d0, b.p2))), rxyw)),
                    MULT(d0, ADD(MULT(b0, SUB(rxz, MULT(c0, p2))), SUB(MULT(c0, rxy), rxyz))))

            );  // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) +
                // d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
        e.p2 = SET_ALL_ZERO();
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult4(OECL2_Share b, OECL2_Share c, OECL2_Share d, func_add ADD, func_sub SUB, func_mul MULT)
        const
    {
#if PRE == 1
        Datatype rxy = pre_receive_from_live(P_0);
        Datatype rxz = pre_receive_from_live(P_0);
        Datatype rxw = pre_receive_from_live(P_0);
        Datatype ryz = pre_receive_from_live(P_0);
        Datatype ryw = pre_receive_from_live(P_0);
        Datatype rzw = pre_receive_from_live(P_0);
        Datatype rxyz = pre_receive_from_live(P_0);
        Datatype rxyw = pre_receive_from_live(P_0);
        Datatype rxzw = pre_receive_from_live(P_0);
        Datatype ryzw = pre_receive_from_live(P_0);
        Datatype rxyzw = pre_receive_from_live(P_0);
#else
        Datatype rxy = receive_from_live(P_0);
        Datatype rxz = receive_from_live(P_0);
        Datatype rxw = receive_from_live(P_0);
        Datatype ryz = receive_from_live(P_0);
        Datatype ryw = receive_from_live(P_0);
        Datatype rzw = receive_from_live(P_0);
        Datatype rxyz = receive_from_live(P_0);
        Datatype rxyw = receive_from_live(P_0);
        Datatype rxzw = receive_from_live(P_0);
        Datatype ryzw = receive_from_live(P_0);
        Datatype rxyzw = receive_from_live(P_0);
#endif

        Datatype a0 = ADD(p1, p2);
        Datatype b0 = ADD(b.p1, b.p2);
        Datatype c0 = ADD(c.p1, c.p2);
        Datatype d0 = ADD(d.p1, d.p2);

        OECL2_Share e;
        e.p1 =

            ADD(ADD(MULT(a0, SUB(MULT(d0, ADD(MULT(b0, SUB(c0, c.p2)), ryz)), ryzw)),
                    MULT(b0, ADD(MULT(a0, SUB(rzw, MULT(c0, d.p2))), SUB(MULT(c0, rxw), rxzw)))),
                ADD(ADD(MULT(c0, SUB(MULT(a0, SUB(ryw, MULT(d0, b.p2))), rxyw)), rxyzw),
                    MULT(d0, ADD(MULT(b0, SUB(rxz, MULT(c0, p2))), SUB(MULT(c0, rxy), rxyz))))

            );  // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) +
                // d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
        e.p2 = getRandomVal(P_0);
        send_to_live(P_1, ADD(e.p1, e.p2));

        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
        p1 = ADD(p1, receive_from_live(P_1));
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OECL2_Share& r_mk2,
                                 OECL2_Share& r_msb,
                                 OECL2_Share& c,
                                 OECL2_Share& c_prime,
                                 int fractional_bits = FRACTIONAL) const
    {
        r_mk2.template prepare_receive_from<P_0>(ADD, SUB);
        r_msb.template prepare_receive_from<P_0>(ADD, SUB);

        Datatype c_dat_prime = trunc(ADD(p1, p2), fractional_bits);
        UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH - fractional_bits - 1)) - 1;
        Datatype mask = PROMOTE(maskValue);  // Set all elements to maskValue
        // Apply the mask using bitwise AND
        c_dat_prime = AND(c_dat_prime, mask);  // mod 2^k-m-1
        /* Datatype c_dat = ADD(p1,p2) >> (BITLENGTH - 1); */
        Datatype c_dat = OP_SHIFT_LOG_RIGHT<BITLENGTH - 1>(ADD(p1, p2));
        c = OECL2_Share(c_dat, SET_ALL_ZERO());
        c_prime = OECL2_Share(c_dat_prime, SET_ALL_ZERO());

        /* c_prime.p1 = trunc(ADD(p1,p2)); */
        /* c_prime.p2 = SET_ALL_ZERO(); */
        /* UINT_TYPE maskValue = (1 << (BITLENGTH-FRACTIONAL-1)) - 1; */
        /* Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue */
        /* // Apply the mask using bitwise AND */
        /* c_prime.p1 = AND(c_prime.p1, mask); //mod 2^k-m-1 */

        /* c.p1 = ADD(p1,p2) >> (BITLENGTH - 1); //open c = x + r */
        /* c.p2 = SET_ALL_ZERO(); */
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OECL2_Share& r_mk2,
                                  OECL2_Share& r_msb,
                                  OECL2_Share& c,
                                  OECL2_Share& c_prime) const
    {
        r_mk2.template complete_receive_from<P_0>(ADD, SUB);
        r_msb.template complete_receive_from<P_0>(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OECL2_Share prepare_trunc_exact_xmod2t(func_add ADD,
                                           func_sub SUB,
                                           func_xor XOR,
                                           func_and AND,
                                           int fractional_bits = FRACTIONAL) const
    {
        Datatype mx = ADD(p1, p2);
        // Step 1, Compute [x/2t] -> delt with public mult fixed
        // Step 2, Compute [x mod t]
        UINT_TYPE maskValue = (UINT_TYPE(1) << (fractional_bits)) - 1;
        Datatype mask = PROMOTE(maskValue);  // Set all elements to maskValue
        // Apply the mask using bitwise AND
        Datatype mxmodt = AND(mx, mask);  // mod 2^t
        // Step3, Compute [x]^B -> delt with prepareA2B
        return OECL2_Share(mxmodt, SET_ALL_ZERO());
    }

/* #if USE_CUDA_GEMM == 1 */
/* template <typename func_add, typename func_sub, typename func_mul> */
/* static void GEMM(OECL2_Share* a, OECL2_Share* b, OECL2_Share* c, int m, int n, int k, func_add ADD, func_sub SUB,
 * func_mul MULT) */
/* { */
/*     const int factor = DATTYPE / BITLENGTH; */
/*     const int nk = k * n;  // Total elements in a 2D array of dimensions k x n */
/*     const int mn = m * n;  // Total elements in a 2D array of dimensions m x n */
/*     const int mk = m * k;  // Total elements in a 2D array of dimensions m x k */

/*     UINT_TYPE* p1 = new UINT_TYPE[factor * mk]; */
/*     UINT_TYPE* bp1 = new UINT_TYPE[factor * nk]; */
/*     UINT_TYPE* cp1_1 = new UINT_TYPE[factor * mn]; */

/*     for (int i = 0; i < m; i++) */
/*     { */
/*         for (int j = 0; j < k; j++) */
/*         { */
/*             alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/*             unorthogonalize_arithmetic(&a[i * k + j].p1, temp, 1); */
/*             for (int l = 0; l < factor; l++) */
/*                 p1[l * mk + i * k + j] = temp[l];  // Access p1 like a 1D array */
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
/*         } */
/*     } */

/*     for (int i = 0; i < factor; i++) */
/*     { */
/*         gemm_cutlass(m,n,k,&p1[i * mk], &bp1[i * nk], &cp1_1[i * mn]); */
/*         /1* test_cuda(); *1/ */
/*     } */

/*     for (int j = 0; j < mn; j++) */
/*     { */
/*         alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/*         for (int i = 0; i < factor; i++) */
/*             temp[i] = cp1_1[i * mn + j]; */
/*         orthogonalize_arithmetic(temp, &c[j].p1, 1); */
/*     } */

/*     delete[] p1; */
/*     delete[] bp1; */
/*     delete[] cp1_1; */
/* const int factor = DATTYPE/BITLENGTH; */
/* const int mk = m * k; */
/* const int kn = k * n; */
/* const int mn = m * n; */

/* UINT_TYPE* p1 = NEW(UINT_TYPE[factor * mk]); */
/* UINT_TYPE* bp1 = NEW(UINT_TYPE[factor * kn]); */
/* UINT_TYPE* cp1_1 = NEW(UINT_TYPE[factor * mn]); */

/* for(int i = 0; i < m; i++) */
/* { */
/*     for(int j = 0; j < k; j++) */
/*     { */
/*         alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/*         unorthogonalize_arithmetic(&a[i * k + j].p1, temp, 1); */
/*         for(int l = 0; l < factor; l++) */
/*             p1[l * mk + i * k + j] = temp[l]; */
/*     } */
/* } */

/* for(int i = 0; i < k; i++) */
/* { */
/*     for(int j = 0; j < n; j++) */
/*     { */
/*         alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/*         unorthogonalize_arithmetic(bp1 + i * n + j, temp, 1); */
/*         for(int l = 0; l < factor; l++) */
/*             bp1[l * kn + i * n + j] = temp[l]; */
/*     } */
/* } */

/* for(int i = 0; i < factor; i++) */
/* { */
/*     gemm_cutlass(m,n,k,p1 + i * mk, bp1 + i * kn, cp1_1 + i * mn); */
/* } */

/* for(int j = 0; j < mn; j++) */
/* { */
/*     alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/*     for(int i = 0; i < factor; i++) */
/*         temp[i] = cp1_1[i * mn + j]; */
/*     orthogonalize_arithmetic(temp, &c[j].p1, 1); */
/* } */

/* delete[] p1; */
/* delete[] bp1; */
/* delete[] cp1_1; */

/* const int factor = DATTYPE/BITLENGTH; */
/* UINT_TYPE* p1 = NEW(UINT_TYPE[factor][m*k]); */
/* UINT_TYPE* bp1 = NEW(UINT_TYPE[factor][k*n]); */
/* UINT_TYPE* cp1_1 = NEW(UINT_TYPE[factor][m*n]); */

/* for(int i = 0; i < m; i++) */
/* { */
/* for(int j = 0; j < k; j++) */
/* { */
/* alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/* unorthogonalize_arithmetic(p1, temp,1); */
/* for(int l = 0; l < factor; l++) */
/*     p1[l][j] = temp[l]; */
/* } */
/* } */

/* for(int i = 0; i < k; i++) */
/* { */
/* for(int j = 0; j < n; j++) */
/* { */
/* alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/* unorthogonalize_arithmetic(bp1, temp,1); */
/* for(int l = 0; l < factor; l++) */
/*     bp1[l][j] = temp[l]; */
/* } */
/* } */

/* for(int i = 0; i < factor; i++) */
/* { */
/* CUDA_GEMM(p1[i], bp1[i], cp1_1[i], m, n, k); */
/* } */

/* for(int j = 0; j < m*n; j++) */
/* { */
/*     alignas(sizeof(Datatype)) UINT_TYPE temp[factor]; */
/*     for(int i = 0; i < factor; i++) */
/*         temp[i] = cp1_1[i][j]; */
/*     orthogonalize_arithmetic(temp, c[j].p1,1); */
/* } */
/* delete p1; */
/* delete bp1; */
/* delete cp1_1; */
/* } */

/* #endif */
#if USE_CUDA_GEMM == 2
    static void CONV_2D(const OECL2_Share* X,
                        const OECL2_Share* W,
                        OECL2_Share* Y,
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
        const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
        const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
        const int ySize = out_h * out_w * dout * batchSize;
        batchSize *= factor;

        UINT_TYPE* x_p1 = new UINT_TYPE[factor * xSize];
        UINT_TYPE* w_p1 = new UINT_TYPE[wSize];  // W is always constant
        UINT_TYPE* y_p1 = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&X[i].p1, temp, 1);
            for (int j = 0; j < factor; j++)
                x_p1[j * xSize + i] = temp[j];
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].p1, temp, 1);
            w_p1[i] = temp[0];
        }

        conv2d_cutlass(x_p1, w_p1, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = y_p1[j * ySize + i];
            orthogonalize_arithmetic(temp, &Y[i].p1, 1);
        }

        delete[] x_p1;
        delete[] w_p1;
        delete[] y_p1;
    }

#elif USE_CUDA_GEMM == 4
    static void CONV_2D(const OECL2_Share* X,
                        const OECL2_Share* W,
                        OECL2_Share* Y,
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
        const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
        const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
        const int ySize = out_h * out_w * dout * batchSize;
        batchSize *= factor;

        alignas(sizeof(Datatype)) UINT_TYPE* x_p1 = new UINT_TYPE[factor * xSize];
        alignas(sizeof(Datatype)) UINT_TYPE* w_p1 = new UINT_TYPE[wSize];  // W is always constant
        alignas(sizeof(Datatype)) UINT_TYPE* y_p1 = new UINT_TYPE[factor * ySize];

        for (int i = 0; i < xSize; i++)
        {
            unorthogonalize_arithmetic(&X[i].p1, x_p1 + i * factor, 1);
        }

        for (int i = 0; i < wSize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&W[i].p1, temp, 1);
            w_p1[i] = temp[0];
        }

        conv2d_cutlass(x_p1, w_p1, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);

        for (int i = 0; i < ySize; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = y_p1[i * factor + j];
            orthogonalize_arithmetic(temp, &Y[i].p1, 1);
        }

        delete[] x_p1;
        delete[] w_p1;
        delete[] y_p1;
    }
#endif
#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1

    static void GEMM(OECL2_Share* a, OECL2_Share* b, OECL2_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;
        UINT_TYPE* p1;
        if (a_fixed)
            p1 = new UINT_TYPE[a_size];
        else
            p1 = new UINT_TYPE[factor * a_size];
        UINT_TYPE* bp1 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* cp1_1 = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].p1, temp, 1);
            if (a_fixed)
                p1[i] = temp[0];
            else
                for (int j = 0; j < factor; j++)
                    p1[j * a_size + i] = temp[j];
        }

        for (int i = 0; i < b_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&b[i].p1, temp, 1);
            for (int j = 0; j < factor; j++)
                bp1[j * b_size + i] = temp[j];
        }

        for (int i = 0; i < factor; i++)
        {
            if (a_fixed)
                gemm_cutlass(m, n, k, p1, &bp1[i * b_size], &cp1_1[i * c_size]);
            else
                gemm_cutlass(m, n, k, &p1[i * a_size], &bp1[i * b_size], &cp1_1[i * c_size]);
        }

        for (int j = 0; j < c_size; j++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for (int i = 0; i < factor; i++)
                temp[i] = cp1_1[i * c_size + j];
            orthogonalize_arithmetic(temp, &c[j].p1, 1);
        }

        delete[] p1;
        delete[] bp1;
        delete[] cp1_1;
    }
#else

    static void GEMM(OECL2_Share* a, OECL2_Share* b, OECL2_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        const int factor = DATTYPE / BITLENGTH;
        const int a_size = m * k;
        const int b_size = k * n;
        const int c_size = m * n;
        UINT_TYPE* p1;
        if (a_fixed)
            p1 = new UINT_TYPE[a_size];
        else
            p1 = new UINT_TYPE[factor * a_size];
        UINT_TYPE* bp1 = new UINT_TYPE[factor * b_size];
        UINT_TYPE* cp1_1 = new UINT_TYPE[factor * c_size];

        for (int i = 0; i < a_size; i++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[i].p1, temp, 1);
            if (a_fixed)
                p1[i] = temp[0];
            else
                for (int j = 0; j < factor; j++)
                    p1[j * a_size + i] = temp[j];
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
                }
            }
            gemm_cutlass(m, n * factor, k, p1, bp1, cp1_1);

            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    for (int l = 0; l < factor; l++)
                        temp[l] = cp1_1[i * n * factor + l * n + j];
                    orthogonalize_arithmetic(temp, &c[i * n + j].p1, 1);
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
            }

            for (int i = 0; i < factor; i++)
            {
                gemm_cutlass(m, n, k, &p1[i * a_size], &bp1[i * b_size], &cp1_1[i * c_size]);
            }

            for (int j = 0; j < c_size; j++)
            {
                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                for (int i = 0; i < factor; i++)
                    temp[i] = cp1_1[i * c_size + j];
                orthogonalize_arithmetic(temp, &c[j].p1, 1);
            }
        }

        delete[] p1;
        delete[] bp1;
        delete[] cp1_1;
    }
#endif
#endif

#endif
};
