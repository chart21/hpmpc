#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL0_init
{
  public:
    OECL0_init() {}

    static OECL0_init public_val(Datatype a) { return OECL0_init(); }

    OECL0_init Not() const { return OECL0_init(); }

    template <typename func_add>
    OECL0_init Add(OECL0_init b, func_add ADD) const
    {
        return OECL0_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_dot(const OECL0_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL0_init();
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_dot(const OECL0_init b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OECL0_init c;
        return c;
    }

    template <typename func_add, typename func_sub>
    void join_dots(OECL0_init c[], func_add ADD, func_sub SUB)
    {
    }

    static constexpr int getNumDotProducts() { return 1; }

#endif

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_mult(OECL0_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
        return OECL0_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    void prepare_reveal_to_all() const
    {
        for (int t = 0; t < 2; t++)
        {
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
            pre_send_to_(t);
#else
            send_to_(t);
#endif

        }  // add to send buffer
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        /* for(int t = 0; t < num_players-1; t++) */
        /*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
        receive_from_(P_2);
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share_();
#endif
        return SET_ALL_ZERO();
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {

        if constexpr (id == P_0)
        {
#if OPT_SHARE == 1
#if PRE == 1 && SHARE_PREP == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif

#else
#if PRE == 1
            pre_send_to_(P_1);
            pre_send_to_(P_2);
#else
            send_to_(P_1);
            send_to_(P_2);
#endif

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
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul>
    OECL0_init mult_public(const Datatype b, func_mul MULT) const
    {
        return OECL0_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_init prepare_mult_public_fixed(const Datatype b,
                                         func_mul MULT,
                                         func_add ADD,
                                         func_sub SUB,
                                         func_trunc TRUNC,
                                         int fractional_bits = FRACTIONAL) const
    {
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
        return OECL0_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_init prepare_trunc_share(func_mul MULT,
                                   func_add ADD,
                                   func_sub SUB,
                                   func_trunc TRUNC,
                                   int fractiona_bits = FRACTIONAL) const
    {
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
        return OECL0_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_init prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        Datatype dummy;
        return prepare_mult_public_fixed(dummy, MULT, ADD, SUB, TRUNC);
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
    }
    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_dot_add(OECL0_init a, OECL0_init b, OECL0_init& c, func_add ADD, func_sub SUB, func_mul MULT)
    {
    }

    static void prepare_A2B_S1(int m, int k, OECL0_init in[], OECL0_init out[]) {}

    static void prepare_A2B_S2(int m, int k, OECL0_init in[], OECL0_init out[])
    {
        for (int i = m; i < k; i++)
        {
#if PRE == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif
        }
    }

    static void prepare_B2A(OECL0_init z[], OECL0_init random_mask[], OECL0_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template prepare_receive_from<P_0>(SET_ALL_ZERO(), OP_ADD, OP_SUB);
        }
    }

    static void complete_B2A(OECL0_init out[], OECL0_init z[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
        }
    }

    static void complete_A2B_S1(int k, OECL0_init out[]) {}
    static void complete_A2B_S2(int k, OECL0_init out[]) {}

    void prepare_opt_bit_injection(OECL0_init x[], OECL0_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            pre_send_to_(P_2);
            pre_send_to_(P_2);
#else
            send_to_(P_2);
            send_to_(P_2);
#endif
        }
    }

    void complete_opt_bit_injection() {}

    void prepare_bit2a(OECL0_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif
        }
    }
    void complete_bit2a() {}

    void prepare_bit_injection_S1(OECL0_init out[]) {}

    void prepare_bit_injection_S2(OECL0_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            pre_send_to_(P_2);
#else
            send_to_(P_2);
#endif
        }
    }
    void get_random_B2A() {}

    static void complete_bit_injection_S1(OECL0_init out[]) {}

    static void complete_bit_injection_S2(OECL0_init out[]) {}
    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_dot3(OECL0_init b, OECL0_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
#else
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
#endif
        OECL0_init d;
        return d;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_mult3(OECL0_init b, OECL0_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
        pre_send_to_(P_2);
#else
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
        send_to_(P_2);
#endif
        OECL0_init d;
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_dot4(OECL0_init b, OECL0_init c, OECL0_init w, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
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
        // for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 +
        // b.p2)
        OECL0_init e;
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_mult4(OECL0_init b, OECL0_init c, OECL0_init w, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
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
        // for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 +
        // b.p2)
        OECL0_init e;
        return e;
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
                                 OECL0_init& r_mk2,
                                 OECL0_init& r_msb,
                                 OECL0_init& c,
                                 OECL0_init& c_prime,
                                 int fractional_bits = FRACTIONAL)
    {
        this->template prepare_receive_from<PSELF>(ADD, SUB);
        this->template prepare_receive_from<PSELF>(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OECL0_init& r_mk2,
                                  OECL0_init& r_msb,
                                  OECL0_init& c,
                                  OECL0_init& c_prime)
    {
        this->template complete_receive_from<PSELF>(ADD, SUB);
        this->template complete_receive_from<PSELF>(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OECL0_init prepare_trunc_exact_xmod2t(func_add ADD,
                                          func_sub SUB,
                                          func_xor XOR,
                                          func_and AND,
                                          int fractional_bits = FRACTIONAL) const
    {
        return OECL0_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    static void GEMM(const OECL0_init* a,
                     const OECL0_init* b,
                     OECL0_init* c,
                     int m,
                     int n,
                     int k,
                     func_add ADD,
                     func_sub SUB,
                     func_mul MULT)
    {
    }

#if USE_CUDA_GEMM > 0
    static void GEMM(OECL0_init* a, OECL0_init* b, OECL0_init* c, int m, int n, int k, bool a_fixed) {}
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const OECL0_init* X,
                        const OECL0_init* W,
                        OECL0_init* Y,
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
