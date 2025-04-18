#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL2_init
{
  public:
    OEC_MAL2_init() {}

    static OEC_MAL2_init public_val(Datatype a) { return OEC_MAL2_init(); }

    OEC_MAL2_init Not() const { return OEC_MAL2_init(); }

    template <typename func_add>
    OEC_MAL2_init Add(OEC_MAL2_init b, func_add ADD) const
    {
        return OEC_MAL2_init();
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_dot(const OEC_MAL2_init b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL2_init c;
        return c;
    }

    template <typename func_add, typename func_sub>
    void join_dots(OEC_MAL2_init c[], func_add ADD, func_sub SUB)
    {
    }

    static constexpr int getNumDotProducts() { return 1; }

#endif

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_dot(const OEC_MAL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OEC_MAL2_init();
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
        store_compare_view_init(P_0);
#if PRE == 1
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
#endif
#else
        store_compare_view_init(P_3);
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
#endif
        send_to_(P_1);
#if PROTOCOL == 10 || PROTOCOL == 12
        send_to_(P_0);
#endif
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_mult(OEC_MAL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
        store_compare_view_init(P_0);
#if PRE == 1
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
#endif
#else
        store_compare_view_init(P_3);
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
#endif
        send_to_(P_1);
#if PROTOCOL == 10 || PROTOCOL == 12
        send_to_(P_0);
#endif

        // return u[player_id] * v[player_id];
        return OEC_MAL2_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
#if PROTOCOL == 11
        send_to_(P_0);  // let P_0 obtain ab
        send_to_(P_0);  // let P_0 verify m_2 XOR m_3
#endif

#if PROTOCOL == 10 || PROTOCOL == 12 || PROTOCOL == 8
        store_compare_view_init(P_012);
#endif

#if PROTOCOL == 8
        send_to_(P_0);  // let P_0 obtain ab
#endif
    }

    void prepare_reveal_to_all() const {}

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PROTOCOL == 8
#if PRE == 1
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
#endif

        store_compare_view_init(P_0);
#else
        receive_from_(P_0);
        store_compare_view_init(P_123);
        store_compare_view_init(P_0123);
#endif
        Datatype dummy;
        return dummy;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            send_to_(P_0);
            send_to_(P_1);
        }
    }
    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        prepare_receive_from<id>(ADD, SUB);
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {
#if PRE == 1
            if constexpr (id == P_3)
                pre_receive_from_(P_3);
            else
                receive_from_(id);
#else
            receive_from_(id);
#endif
            if constexpr (id != P_0)
                store_compare_view_init(P_0);
            if constexpr (id != P_1)
                store_compare_view_init(P_1);
        }
    }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul>
    OEC_MAL2_init mult_public(const Datatype b, func_mul MULT) const
    {
        return OEC_MAL2_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL2_init prepare_mult_public_fixed(const Datatype b,
                                            func_mul MULT,
                                            func_add ADD,
                                            func_sub SUB,
                                            func_trunc TRUNC,
                                            int fractional_bits = FRACTIONAL) const
    {
        send_to_(P_0);
        return OEC_MAL2_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL2_init prepare_trunc_share(func_mul MULT,
                                      func_add ADD,
                                      func_sub SUB,
                                      func_trunc TRUNC,
                                      int fractional_bits = FRACTIONAL) const
    {
        send_to_(P_0);
        return OEC_MAL2_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL2_init prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        Datatype dummy;
        return prepare_mult_public_fixed(dummy, MULT, ADD, SUB, TRUNC);
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
#endif
        store_compare_view_init(P_0);
#else
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
        store_compare_view_init(P_3);
#endif
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OEC_MAL2_init& r_mk2,
                                 OEC_MAL2_init& r_msb,
                                 OEC_MAL2_init& c,
                                 OEC_MAL2_init& c_prime,
                                 int fractional_bits = FRACTIONAL) const
    {
        send_to_(P_0);
        send_to_(P_0);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OEC_MAL2_init& r_mk2,
                                  OEC_MAL2_init& r_msb,
                                  OEC_MAL2_init& c,
                                  OEC_MAL2_init& c_prime)
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
        receive_from_(P_3);
#endif
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
#else
#if PRE == 0
        receive_from_(P_0);
        receive_from_(P_0);
#else
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
#endif
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
#endif
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OEC_MAL2_init prepare_trunc_exact_xmod2t(func_add ADD,
                                             func_sub SUB,
                                             func_xor XOR,
                                             func_and AND,
                                             int fractional_bits = FRACTIONAL) const
    {
        return OEC_MAL2_init();
    }

    static void prepare_B2A(OEC_MAL2_init z[], OEC_MAL2_init random_mask[], OEC_MAL2_init out[])
    {
        // 1. Reveal z to P_1 and P_2
        for (int i = 0; i < BITLENGTH; i++)
        {
            send_to_(P_1);
        }
    }

    static void complete_B2A(OEC_MAL2_init z_bool[], OEC_MAL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
            pre_receive_from_(P_3);
#else
            receive_from_(P_3);
#endif
            store_compare_view_init(P_0);
#else
#if PRE == 1
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
#endif
            store_compare_view_init(P_3);
#endif

            receive_from_(P_1);
            store_compare_view_init(P_012);

            send_to_(P_0);
        }
    }
    static void complete_B2A2(OEC_MAL2_init z_bool[], OEC_MAL2_init out[]) {}

    void get_random_B2A() {}

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        send_to_(P_1);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        receive_from_(P_1);  // v^1,2 = m^1 + m^2

        send_to_(P_0);

#if PROTOCOL == 11
        send_to_(P_0);
#else
        store_compare_view_init(P_012);
#endif
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_receive_from_(P_3);  // z_2 = m0
#else
        receive_from_(P_3);  // z_2 = m0
#endif
        store_compare_view_init(P_0);  // compare view of m0
#else
#if PRE == 1
        pre_receive_from_(P_0);        // z_2 = m0
#else
        receive_from_(P_0);  // z_2 = m0
#endif
        store_compare_view_init(P_3);  // compare view of m0
#endif
    }

    static void prepare_A2B_S1(int m, int k, OEC_MAL2_init in[], OEC_MAL2_init out[])
    {
        for (int j = m; j < k; j++)
        {
            send_to_(P_0);
        }
    }

    static void prepare_A2B_S2(int m, int k, OEC_MAL2_init in[], OEC_MAL2_init out[]) {}

    static void complete_A2B_S1(int k, OEC_MAL2_init out[]) {}

    static void complete_A2B_S2(int k, OEC_MAL2_init out[])
    {
        for (int i = 0; i < k; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 0
            receive_from_(P_0);
#else
            pre_receive_from_(P_0);
#endif
            store_compare_view_init(P_3);
#else
#if PRE == 0
            receive_from_(P_3);
#else
            pre_receive_from_(P_3);
#endif
            store_compare_view_init(P_0);
#endif
        }
    }

    void prepare_bit2a(OEC_MAL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 1
            pre_receive_from_(P_3);
            store_compare_view_init(P_0);
#else
            receive_from_(P_0);
            store_compare_view_init(P_3);
#endif
#else
#if PRE == 1
            pre_receive_from_(P_3);
#else
            receive_from_(P_3);
#endif
            store_compare_view_init(P_0);
#endif
            send_to_(P_1);
            send_to_(P_0);
        }
    }

    void complete_bit2a()
    {
        receive_from_(P_1);
        store_compare_view_init(P_012);
    }

    void prepare_opt_bit_injection(OEC_MAL2_init a[], OEC_MAL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 1
            pre_receive_from_(P_0);
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
            receive_from_(P_0);
#endif
            store_compare_view_init(P_3);
            store_compare_view_init(P_3);
#else
#if PRE == 1
            pre_receive_from_(P_3);
            pre_receive_from_(P_3);
#else
            receive_from_(P_3);
            receive_from_(P_3);
#endif
            store_compare_view_init(P_0);
            store_compare_view_init(P_0);
#endif
            send_to_(P_1);
            send_to_(P_0);
        }
    }

    void complete_opt_bit_injection()
    {
        receive_from_(P_1);
        store_compare_view_init(P_012);
    }

    void prepare_bit_injection_S1(OEC_MAL2_init out[])
    {
        for (int j = 0; j < BITLENGTH; j++)
        {
            send_to_(P_0);
        }
    }

    void prepare_bit_injection_S2(OEC_MAL2_init out[]) {}

    static void complete_bit_injection_S1(OEC_MAL2_init out[]) {}

    static void complete_bit_injection_S2(OEC_MAL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL != 12 && PRE == 0
#if PRE == 0
            receive_from_(P_0);
#else
            pre_receive_from_(P_0);
#endif
            store_compare_view_init(P_3);
#else
#if PRE == 0
            receive_from_(P_3);
#else
            pre_receive_from_(P_3);
#endif
            store_compare_view_init(P_0);
#endif
        }
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_dot3(OEC_MAL2_init b, OEC_MAL2_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
#endif
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
#else
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
#endif
        OEC_MAL2_init d;
        return d;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_mult3(OEC_MAL2_init b, OEC_MAL2_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
#endif
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
#else
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
#endif
        OEC_MAL2_init d;
        send_to_(P_0);
        send_to_(P_1);
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
        store_compare_view_init(P_012);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_dot4(OEC_MAL2_init b,
                               OEC_MAL2_init c,
                               OEC_MAL2_init d,
                               func_add ADD,
                               func_sub SUB,
                               func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
#endif
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
#else
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
#endif
        OEC_MAL2_init e;
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_mult4(OEC_MAL2_init b,
                                OEC_MAL2_init c,
                                OEC_MAL2_init d,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
        receive_from_(P_3);
#endif
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
#else
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);
#endif
        OEC_MAL2_init e;
        send_to_(P_0);
        send_to_(P_1);
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
        store_compare_view_init(P_012);
    }

#if USE_CUDA_GEMM > 0
    static void GEMM(OEC_MAL2_init* a, OEC_MAL2_init* b, OEC_MAL2_init* c, int m, int n, int k, bool a_fixed) {}
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const OEC_MAL2_init* X,
                        const OEC_MAL2_init* W,
                        OEC_MAL2_init* Y,
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
    }
#endif

#endif
};
