#pragma once
#include "../generic_share.hpp"
template <typename Datatype>
class TTP_init
{
  public:
    TTP_init() {}
    TTP_init(Datatype a) {}

    Datatype get_p1() { return SET_ALL_ZERO(); }

    template <typename func_mul>
    TTP_init mult_public(const Datatype b, func_mul MULT) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 TTP_init& r_mk2,
                                 TTP_init& r_msb,
                                 TTP_init& c,
                                 TTP_init& c_prime)
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  TTP_init& r_mk2,
                                  TTP_init& r_msb,
                                  TTP_init& c,
                                  TTP_init& c_prime)
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    TTP_init prepare_trunc_exact_xmod2t(func_add ADD, func_sub SUB, func_xor XOR, func_and AND) const
    {
        return TTP_init();
    }

    void prepare_bit2a(TTP_init out[]) {}

    void complete_bit2a() {}

    void prepare_opt_bit_injection(TTP_init x[], TTP_init out[]) {}

    void complete_opt_bit_injection() {}

    void get_random_B2A() {}

    static void prepare_B2A(TTP_init z[], TTP_init random_mask[], TTP_init out[]) {}

    static void complete_B2A(TTP_init z[], TTP_init out[]) {}

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    TTP_init prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC)
        const
    {
        return TTP_init();
    }
    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    TTP_init prepare_trunc_2k(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub>
    void complete_trunc_2k(func_add ADD, func_sub SUB)
    {
    }

    static TTP_init public_val(Datatype a) { return TTP_init(); }

    TTP_init Not() const { return TTP_init(); }

    template <typename func_add>
    TTP_init Add(TTP_init b, func_add ADD) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_dot(const TTP_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_mult(TTP_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return TTP_init();
    }
    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    void prepare_reveal_to_all() const
    {
#if PARTY == 2 && PROTOCOL != 13
        for (int t = 0; t < num_players - 1; t++)
        {
            send_to_(t);
        }  // add to send buffer
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PARTY != 2 && PROTOCOL != 13
        receive_from_(P_2);
#endif
        return SET_ALL_ZERO();
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
#if PROTOCOL != 13 && PARTY != 2
        if constexpr (id == PSELF)
        {
            send_to_(P_2);
        }
#endif
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
        prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
#if PARTY == 2 && PROTOCOL != 13
        if constexpr (id == PSELF)
            return;
        else
            receive_from_(id);
#endif
    }

    static void send()
    {
#if PROTOCOL != 13
        send_();
#endif
    }
    static void receive()
    {
#if PROTOCOL != 13
        receive_();
#endif
    }
    static void communicate()
    {
#if PROTOCOL != 13
        communicate_();
#endif
    }

    static void finalize(std::string* ips)
    {
#if PROTOCOL != 13
        finalize_(ips);
#endif
    }

    static void prepare_A2B_S1(int m, int k, TTP_init in[], TTP_init out[]) {}

    static void prepare_A2B_S2(int m, int k, TTP_init in[], TTP_init out[]) {}

    static void complete_A2B_S1(int k, TTP_init out[]) {}
    static void complete_A2B_S2(int k, TTP_init out[]) {}

    void prepare_bit_injection_S1(TTP_init out[]) {}

    void prepare_bit_injection_S2(TTP_init out[]) {}

    static void complete_bit_injection_S1(TTP_init out[]) {}

    static void complete_bit_injection_S2(TTP_init out[]) {}

    template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_dot3(TTP_init b, TTP_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_mult3(TTP_init b, TTP_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_dot4(TTP_init b, TTP_init c, TTP_init d, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_mult4(TTP_init b, TTP_init c, TTP_init d, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return TTP_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
    }

    TTP_init relu() const { return TTP_init(); }

#if USE_CUDA_GEMM > 0
    static void GEMM(TTP_init* a, TTP_init* b, TTP_init* c, int m, int n, int k, bool a_fixed) {}
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const TTP_init* X,
                        const TTP_init* W,
                        TTP_init* Y,
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
};
