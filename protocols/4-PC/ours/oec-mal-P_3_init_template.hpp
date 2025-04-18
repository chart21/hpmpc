#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL3_init
{
  public:
    OEC_MAL3_init() {}

    static OEC_MAL3_init public_val(Datatype a) { return OEC_MAL3_init(); }

    OEC_MAL3_init Not() const { return OEC_MAL3_init(); }

    template <typename func_add>
    OEC_MAL3_init Add(OEC_MAL3_init b, func_add ADD) const
    {
        return OEC_MAL3_init();
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_dot(const OEC_MAL3_init b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OEC_MAL3_init c;
        return c;
    }

    template <typename func_add, typename func_sub>
    void join_dots(OEC_MAL3_init c[], func_add ADD, func_sub SUB)
    {
    }

    static constexpr int getNumDotProducts() { return 1; }

#endif

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_dot(const OEC_MAL3_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OEC_MAL3_init();
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
#else
        store_compare_view_init(P_2);
#endif

#if PROTOCOL == 8
#if PRE == 1
        pre_send_to_(P_0);
#else
        send_to_(P_0);
#endif
#endif
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_mult(OEC_MAL3_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {

#if PROTOCOL == 8
#if PRE == 1
        pre_send_to_(P_0);
#else
        send_to_(P_0);
#endif
#endif

#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
#else
        store_compare_view_init(P_2);
#endif
        return OEC_MAL3_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
        pre_send_to_(P_0);
#else
        send_to_(P_0);
#endif
#elif PROTOCOL == 11
        store_compare_view_init(P_0);
#endif
    }

    void prepare_reveal_to_all() const
    {
#if PROTOCOL == 8
#if PRE == 1
        pre_send_to_(P_0);
        pre_send_to_(P_1);
        pre_send_to_(P_2);
#else
        send_to_(P_0);
        send_to_(P_1);
        send_to_(P_2);
#endif

#else
#if PRE == 1
        pre_send_to_(P_0);
#else
        send_to_(P_0);
#endif
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        receive_from_(P_0);
#if PROTOCOL == 8
        store_compare_view_init(P_1);
#else
        store_compare_view_init(P_123);
        store_compare_view_init(P_0123);
#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
#if PROTOCOL == 8
        store_output_share_();
#endif
        store_output_share_();
        store_output_share_();
#endif
        Datatype dummy;
        return dummy;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
        /* return; */
        /* old: */

        if constexpr (id == PSELF)
        {
#if PRE == 1
            pre_send_to_(P_0);
            pre_send_to_(P_1);
            pre_send_to_(P_2);
#else
            send_to_(P_0);
            send_to_(P_1);
            send_to_(P_2);
#endif
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
    }

    static void send() { send_(); }

    // P_0 only has 1 receive round
    static void receive() { receive_(); }

    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul>
    OEC_MAL3_init mult_public(const Datatype b, func_mul MULT) const
    {
        return OEC_MAL3_init();
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OEC_MAL3_init& r_mk2,
                                 OEC_MAL3_init& r_msb,
                                 OEC_MAL3_init& c,
                                 OEC_MAL3_init& c_prime,
                                 int fractional_bits = FRACTIONAL) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_2);
        pre_send_to_(P_2);
#else
        send_to_(P_2);
        send_to_(P_2);
#endif
#else
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
#endif
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OEC_MAL3_init& r_mk2,
                                  OEC_MAL3_init& r_msb,
                                  OEC_MAL3_init& c,
                                  OEC_MAL3_init& c_prime)
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OEC_MAL3_init prepare_trunc_exact_xmod2t(func_add ADD,
                                             func_sub SUB,
                                             func_xor XOR,
                                             func_and AND,
                                             int fractional_bits = FRACTIONAL) const
    {
        return OEC_MAL3_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL3_init prepare_mult_public_fixed(const Datatype b,
                                            func_mul MULT,
                                            func_add ADD,
                                            func_sub SUB,
                                            func_trunc TRUNC,
                                            int fractional_bits = FRACTIONAL) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
#else
        store_compare_view_init(P_2);
#endif
        return OEC_MAL3_init();
    }
    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL3_init prepare_trunc_share(func_mul MULT,
                                      func_add ADD,
                                      func_sub SUB,
                                      func_trunc TRUNC,
                                      int fractional_bits = FRACTIONAL) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
#else
        store_compare_view_init(P_2);
#endif
        return OEC_MAL3_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OEC_MAL3_init prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        Datatype dummy;
        return prepare_mult_public_fixed(dummy, MULT, ADD, SUB, TRUNC);
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    static void prepare_B2A(OEC_MAL3_init z[], OEC_MAL3_init random_mask[], OEC_MAL3_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif
#else
            store_compare_view_init(P_2);
#endif
        }
    }

    void get_random_B2A() {}

    static void complete_B2A(OEC_MAL3_init z[], OEC_MAL3_init out[]) {}

    static void complete_B2A2(OEC_MAL3_init z[], OEC_MAL3_init out[]) {}

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
#else
        store_compare_view_init(P_2);
#endif
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
#if PROTOCOL == 11
        store_compare_view_init(P_0);
#else
#if PRE == 1
        pre_send_to_(P_0);
#else
        send_to_(P_0);
#endif
#endif
    }

    static void prepare_A2B_S1(int m, int k, OEC_MAL3_init in[], OEC_MAL3_init out[]) {}

    static void prepare_A2B_S2(int m, int k, OEC_MAL3_init in[], OEC_MAL3_init out[])
    {
        for (int i = m; i < k; i++)
        {
#if PROTOCOL != 12 && PRE == 0
            store_compare_view_init(P_2);
#else
#if PRE == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif
#endif
        }
    }

    static void complete_A2B_S1(int k, OEC_MAL3_init out[]) {}

    static void complete_A2B_S2(int k, OEC_MAL3_init out[]) {}

    void prepare_bit2a(OEC_MAL3_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL == 12 || PRE == 1
#if PRE == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif
#else
            store_compare_view_init(P_2);
#endif

#if PRE == 1
            pre_send_to_(P_0);
#else
            send_to_(P_0);
#endif
        }
    }

    void complete_bit2a() {}

    void prepare_opt_bit_injection(OEC_MAL3_init x[], OEC_MAL3_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
            pre_send_to_(P_2);
            pre_send_to_(P_2);
#else
            send_to_(P_2);
            send_to_(P_2);
#endif
#else
            store_compare_view_init(P_2);
            store_compare_view_init(P_2);
#endif

#if PRE == 1
            pre_send_to_(P_0);
            pre_send_to_(P_0);
#else
            send_to_(P_0);
            send_to_(P_0);
#endif
        }
    }

    void complete_opt_bit_injection() {}

    void prepare_bit_injection_S1(OEC_MAL3_init out[]) {}

    void prepare_bit_injection_S2(OEC_MAL3_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PROTOCOL != 12 && PRE == 0
            store_compare_view_init(P_2);
#else
#if PRE == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif
#endif
        }
    }

    static void complete_bit_injection_S1(OEC_MAL3_init out[]) {}

    static void complete_bit_injection_S2(OEC_MAL3_init out[]) {}
    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_dot3(OEC_MAL3_init b, OEC_MAL3_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
#endif
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
#endif
        OEC_MAL3_init d;
        return d;
    }
    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_mult3(OEC_MAL3_init b, OEC_MAL3_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
#endif
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
#endif
        OEC_MAL3_init d;
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_dot4(OEC_MAL3_init b,
                               OEC_MAL3_init c,
                               OEC_MAL3_init d,
                               func_add ADD,
                               func_sub SUB,
                               func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
#endif
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
#endif
        OEC_MAL3_init e;
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_mult4(OEC_MAL3_init b,
                                OEC_MAL3_init c,
                                OEC_MAL3_init d,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8 || PRE == 1
#if PRE == 1
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_0);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
#endif
#else
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        send_to_(P_0);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
        store_compare_view_init(P_2);
#endif
        OEC_MAL3_init e;
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
    }

#if USE_CUDA_GEMM > 0
    static void GEMM(OEC_MAL3_init* a, OEC_MAL3_init* b, OEC_MAL3_init* c, int m, int n, int k, bool a_fixed) {}
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const OEC_MAL3_init* X,
                        const OEC_MAL3_init* W,
                        OEC_MAL3_init* Y,
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
