#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL1_init
{
  public:
    OECL1_init() {}

    static OECL1_init public_val(Datatype a) { return OECL1_init(); }

    OECL1_init Not() const { return OECL1_init(); }

    template <typename func_add>
    OECL1_init Add(OECL1_init b, func_add ADD) const
    {
        return OECL1_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_dot(const OECL1_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL1_init();
    }
#if FUSE_DOT != 1
    template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_dot(const OECL1_init b, int i, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL1_init();
    }

    template <typename func_add, typename func_sub>
    void join_dots(OECL1_init c[], func_add ADD, func_sub SUB)
    {
    }

    static constexpr int getNumDotProducts() { return 1; }
#endif

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        send_to_(P_2);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_mult(OECL1_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        send_to_(P_2);
        return OECL1_init();

        // return u[player_id] * v[player_id];
    }
    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        receive_from_(P_2);
    }

    void prepare_reveal_to_all() const { return; }

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
        if constexpr (id == P_1)
        {
            send_to_(P_2);
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
        if constexpr (id == P_2)
        {
            receive_from_(P_2);
        }
#if OPT_SHARE == 0
        else if constexpr (id == P_0)
        {
#if PRE == 1 && SHARE_PREP == 1
            pre_receive_from_(P_0);
#else
            receive_from_(P_0);
#endif
        }
#endif
    }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

#if FUNCTION_IDENTIFIER > 8

    template <typename func_mul>
    OECL1_init mult_public(const Datatype b, func_mul MULT) const
    {
        return OECL1_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL1_init prepare_mult_public_fixed(const Datatype b,
                                         func_mul MULT,
                                         func_add ADD,
                                         func_sub SUB,
                                         func_trunc TRUNC,
                                         int fractional_bits = FRACTIONAL) const
    {
        return OECL1_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL1_init prepare_trunc_share(func_mul MULT,
                                   func_add ADD,
                                   func_sub SUB,
                                   func_trunc TRUNC,
                                   int fractional_bits = FRACTIONAL) const
    {
        return OECL1_init();
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL1_init prepare_div_exp2(const int b,
                                func_mul MULT,
                                func_add ADD,
                                func_sub SUB,
                                func_trunc TRUNC,
                                int fractional_bits = FRACTIONAL) const
    {
        Datatype dummy;
        return prepare_mult_public_fixed(dummy, MULT, ADD, SUB, TRUNC);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    OECL1_init prepare_trunc_2k(func_add ADD,
                                func_sub SUB,
                                func_xor XOR,
                                func_and AND,
                                func_trunc trunc,
                                int fractional_bits = FRACTIONAL) const
    {
        send_to_(P_2);
        return OECL1_init();
    }

    template <typename func_add, typename func_sub>
    void complete_trunc_2k(func_add ADD, func_sub SUB)
    {
        receive_from_(P_2);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_dot_add(OECL1_init a, OECL1_init b, OECL1_init& c, func_add ADD, func_sub SUB, func_mul MULT)
    {
    }
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        send_to_(P_2);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        receive_from_(P_2);
    }

    static void prepare_A2B_S1(int m, int k, OECL1_init in[], OECL1_init out[]) {}

    static void prepare_A2B_S2(int m, int k, OECL1_init in[], OECL1_init out[]) {}

    static void complete_A2B_S1(int k, OECL1_init out[]) {}
    static void complete_A2B_S2(int k, OECL1_init out[]) {}

    void prepare_opt_bit_injection(OECL1_init x[], OECL1_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            send_to_(P_2);
        }
    }

    void complete_opt_bit_injection() { receive_from_(P_2); }

    void prepare_bit2a(OECL1_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            send_to_(P_2);
        }
    }

    void get_random_B2A() {}

    static void prepare_B2A(OECL1_init z[], OECL1_init random_mask[], OECL1_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            send_to_(P_2);
        }
        for (int i = 0; i < BITLENGTH; i++)
            out[i].template prepare_receive_from<P_0>(SET_ALL_ZERO(), OP_ADD, OP_SUB);
    }

    static void complete_B2A(OECL1_init out[], OECL1_init z[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            receive_from_(P_2);
        }
        for (int i = 0; i < BITLENGTH; i++)
            out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
    }

    void complete_bit2a() { receive_from_(P_2); }

    void prepare_bit_injection_S1(OECL1_init out[]) {}

    void prepare_bit_injection_S2(OECL1_init out[]) {}

    static void complete_bit_injection_S1(OECL1_init out[]) {}

    static void complete_bit_injection_S2(OECL1_init out[]) {}

    template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_dot3(OECL1_init b, OECL1_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OECL1_init e;
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_mult3(OECL1_init b, OECL1_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        send_to_(P_2);
        OECL1_init e;
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
        receive_from_(P_2);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_dot4(OECL1_init b, OECL1_init c, OECL1_init d, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        OECL1_init e;
        return e;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_mult4(OECL1_init b, OECL1_init c, OECL1_init d, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        send_to_(P_2);
        OECL1_init e;
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
        receive_from_(P_2);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OECL1_init& r_mk2,
                                 OECL1_init& r_msb,
                                 OECL1_init& c,
                                 OECL1_init& c_prime,
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
                                  OECL1_init& r_mk2,
                                  OECL1_init& r_msb,
                                  OECL1_init& c,
                                  OECL1_init& c_prime)
    {
        this->template complete_receive_from<P_0>(ADD, SUB);
        this->template complete_receive_from<P_0>(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OECL1_init prepare_trunc_exact_xmod2t(func_add ADD,
                                          func_sub SUB,
                                          func_xor XOR,
                                          func_and AND,
                                          int frac_bits = FRACTIONAL) const
    {
        return OECL1_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    static void GEMM(const OECL1_init* a,
                     const OECL1_init* b,
                     OECL1_init* c,
                     int m,
                     int n,
                     int k,
                     func_add ADD,
                     func_sub SUB,
                     func_mul MULT)
    {
    }

#if USE_CUDA_GEMM > 0
    static void GEMM(OECL1_init* a, OECL1_init* b, OECL1_init* c, int m, int n, int k, bool a_fixed) {}
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const OECL1_init* X,
                        const OECL1_init* W,
                        OECL1_init* Y,
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
