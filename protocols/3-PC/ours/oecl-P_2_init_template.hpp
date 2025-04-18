#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL2_init
{
  public:
    OECL2_init() {}

    static OECL2_init public_val(Datatype a) { return OECL2_init(); }

    OECL2_init Not() const { return OECL2_init(); }

    template <typename func_add>
    OECL2_init Add(OECL2_init b, func_add ADD) const
    {
        return OECL2_init();
    }
    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_dot_add(OECL2_init a, OECL2_init b, OECL2_init& c, func_add ADD, func_sub SUB, func_mul MULT)
    {
    }
    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_dot(const OECL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL2_init();
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_dot(const OECL2_init b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL2_init();
    }

    template <typename func_add, typename func_sub>
    void join_dots(OECL2_init c[], func_add ADD, func_sub SUB)
    {
    }

    static constexpr int getNumDotProducts() { return 1; }
#endif

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
        send_to_(P_1);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_mult(OECL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif

        send_to_(P_1);
        // return u[player_id] * v[player_id];
        return OECL2_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
    }

    void prepare_reveal_to_all() const { send_to_(P_0); }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
/* for(int t = 0; t < num_players-1; t++) */
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
#if PRE == 1 && (OPT_SHARE == 0 || \
                 SHARE_PREP == 1)  // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
        return SET_ALL_ZERO();
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id == P_2)
        {
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
        if constexpr (id == P_1)
        {
            receive_from_(P_1);
        }
        else if constexpr (id == P_0)
        {
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
#endif
        }
        /* if(id == player_id) */
        /*     return; */
        /* int offset = {id > player_id ? 1 : 0}; */
        /* int player = id - offset; */
        /* for(int i = 0; i < l; i++) */
        /*     receiving_args[player].elements_to_rec[receiving_args[player].rec_rounds -1] += 1; */
    }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul>
    OECL2_init mult_public(const Datatype b, func_mul MULT) const
    {
        return OECL2_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL2_init prepare_mult_public_fixed(const Datatype b,
                                         func_mul MULT,
                                         func_add ADD,
                                         func_sub SUB,
                                         func_trunc TRUNC,
                                         int fractional_bits = FRACTIONAL) const
    {
        return OECL2_init();
    }
    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL2_init prepare_trunc_share(func_mul MULT,
                                   func_add ADD,
                                   func_sub SUB,
                                   func_trunc TRUNC,
                                   int frac_bits = FRACTIONAL) const
    {
        return OECL2_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL2_init prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        Datatype dummy;
        return prepare_mult_public_fixed(dummy, MULT, ADD, SUB, TRUNC);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    OECL2_init prepare_trunc_2k(func_add ADD,
                                func_sub SUB,
                                func_xor XOR,
                                func_and AND,
                                func_trunc trunc,
                                int fractional_bits = FRACTIONAL) const
    {
#if PRE == 0
        receive_from_(P_0);  // send share of bit decomposition of x_0 to P_2
#else
        pre_receive_from_(P_0);
#endif
        send_to_(P_1);
        return OECL2_init();
    }

    template <typename func_add, typename func_sub>
    void complete_trunc_2k(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        send_to_(P_1);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
#if PRE == 1
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
#endif
        receive_from_(P_1);
    }

    void get_random_B2A() {}

    static void prepare_B2A(OECL2_init z[], OECL2_init random_mask[], OECL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            send_to_(P_1);
        }
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template prepare_receive_from<P_0>(SET_ALL_ZERO(), OP_ADD, OP_SUB);
        }
    }

    static void complete_B2A(OECL2_init out[], OECL2_init z[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            receive_from_(P_1);
        }
        for (int i = 0; i < BITLENGTH; i++)
        {
            out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
        }
    }

    static void prepare_A2B_S1(int m, int k, OECL2_init in[], OECL2_init out[]) {}

    static void prepare_A2B_S2(int m, int k, OECL2_init in[], OECL2_init out[]) {}

    static void complete_A2B_S1(int k, OECL2_init out[]) {}
    static void complete_A2B_S2(int k, OECL2_init out[])
    {
        for (int i = 0; i < k; i++)
        {
#if PRE == 1
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
#endif
        }
    }

    void prepare_opt_bit_injection(OECL2_init x[], OECL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            pre_receive_from_(P_0);
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
            receive_from_(P_0);
#endif
            send_to_(P_1);
        }
    }

    void complete_opt_bit_injection() { receive_from_(P_1); }

    void prepare_bit2a(OECL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
#endif
            send_to_(P_1);
        }
    }

    void complete_bit2a() { receive_from_(P_1); }

    void prepare_bit_injection_S1(OECL2_init out[]) {}

    void prepare_bit_injection_S2(OECL2_init out[]) {}

    static void complete_bit_injection_S1(OECL2_init out[]) {}

    static void complete_bit_injection_S2(OECL2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
#if PRE == 1
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
#endif
        }
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_dot3(OECL2_init b, OECL2_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
#endif
        OECL2_init d;
        return d;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_mult3(OECL2_init b, OECL2_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
#else
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
        receive_from_(P_0);
#endif

        send_to_(P_1);
        OECL2_init d;
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_dot4(OECL2_init b, OECL2_init c, OECL2_init d, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
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
#endif
        OECL2_init e;
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_mult4(OECL2_init b, OECL2_init c, OECL2_init d, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
        pre_receive_from_(P_0);
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
#endif

        send_to_(P_1);
        OECL2_init e;
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OECL2_init& r_mk2,
                                 OECL2_init& r_msb,
                                 OECL2_init& c,
                                 OECL2_init& c_prime,
                                 int fractional_bits = FRACTIONAL)
    {
        this->template prepare_receive_from<P_0>(ADD, SUB);
        this->template prepare_receive_from<P_0>(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OECL2_init& r_mk2,
                                  OECL2_init& r_msb,
                                  OECL2_init& c,
                                  OECL2_init& c_prime)
    {
        this->template complete_receive_from<P_0>(ADD, SUB);
        this->template complete_receive_from<P_0>(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OECL2_init prepare_trunc_exact_xmod2t(func_add ADD,
                                          func_sub SUB,
                                          func_xor XOR,
                                          func_and AND,
                                          int fractional_bits = FRACTIONAL) const
    {
        return OECL2_init();
    }

#if USE_CUDA_GEMM > 0
    static void GEMM(OECL2_init* a, OECL2_init* b, OECL2_init* c, int m, int n, int k, bool a_fixed) {}
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const OECL2_init* X,
                        const OECL2_init* W,
                        OECL2_init* Y,
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
