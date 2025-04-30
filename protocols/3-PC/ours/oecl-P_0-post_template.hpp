#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL0_POST_Share
{
  private:
    Datatype p1;
    Datatype p2;

  public:
    // static constexpr int VALS_PER_SHARE = 2;

    OECL0_POST_Share() {}
    OECL0_POST_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}
    OECL0_POST_Share(Datatype p1) : p1(p1) {}

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate()
    {
        /* #if PRE == 0 */
        communicate_live();
        /* #endif */
    }

    void get_random_B2A() {}

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

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
#if OPT_SHARE == 1 && SHARE_PREP == 0
        if constexpr (id == P_0)
        {
            p1 = val;
            p2 = getRandomVal(P_1);
            send_to_live(P_2, XOR(p1, p2));
        }
#endif
    }

    void prepare_reveal_to_all() const {}

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        return SUB(receive_from_live(P_2), retrieve_output_share());
    }

    static OECL0_POST_Share public_val(Datatype a) { return OECL0_POST_Share(); }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_POST_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
        return OECL0_POST_Share();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_POST_Share prepare_mult_public_fixed(const Datatype b,
                                               func_mul MULT,
                                               func_add ADD,
                                               func_sub SUB,
                                               func_trunc TRUNC,
                                               int fractional_bits = FRACTIONAL) const
    {
        return OECL0_POST_Share();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL0_POST_Share prepare_trunc_share(func_mul MULT,
                                         func_add ADD,
                                         func_sub SUB,
                                         func_trunc TRUNC,
                                         int fractiona_bits = FRACTIONAL) const
    {
        return OECL0_POST_Share();
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    OECL0_POST_Share Not() const { return OECL0_POST_Share(); }

    template <typename func_add>
    OECL0_POST_Share Add(OECL0_POST_Share b, func_add ADD) const
    {
        return OECL0_POST_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_dot_add(OECL0_POST_Share a,
                         OECL0_POST_Share b,
                         OECL0_POST_Share& c,
                         func_add ADD,
                         func_sub SUB,
                         func_mul MULT)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_POST_Share prepare_dot(const OECL0_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL0_POST_Share();
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
    }
    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_POST_Share prepare_mult(OECL0_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL0_POST_Share();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_mul>
    OECL0_POST_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return OECL0_POST_Share();
    }

    static void prepare_A2B_S1(int m, int k, OECL0_POST_Share in[], OECL0_POST_Share out[]) {}

    static void prepare_A2B_S2(int m, int k, OECL0_POST_Share in[], OECL0_POST_Share out[]) {}

    static void complete_A2B_S1(int k, OECL0_POST_Share out[]) {}
    static void complete_A2B_S2(int k, OECL0_POST_Share out[]) {}

    void prepare_opt_bit_injection(OECL0_POST_Share x[], OECL0_POST_Share out[]) {}

    void complete_opt_bit_injection() {}

    void prepare_bit2a(OECL0_POST_Share out[]) {}

    void complete_bit2a() {}

    static void prepare_B2A(OECL0_POST_Share z[], OECL0_POST_Share random_mask[], OECL0_POST_Share out[]) {}

    static void complete_B2A(OECL0_POST_Share z[], OECL0_POST_Share out[]) {}

    void prepare_bit_injection_S1(OECL0_POST_Share out[]) {}

    void prepare_bit_injection_S2(OECL0_POST_Share out[]) {}

    static void complete_bit_injection_S1(OECL0_POST_Share out[]) {}

    static void complete_bit_injection_S2(OECL0_POST_Share out[]) {}

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_POST_Share prepare_dot3(const OECL0_POST_Share b,
                                  const OECL0_POST_Share c,
                                  func_add ADD,
                                  func_sub SUB,
                                  func_mul MULT) const
    {
        return OECL0_POST_Share();
    }
    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_POST_Share prepare_dot4(OECL0_POST_Share b,
                                  OECL0_POST_Share c,
                                  OECL0_POST_Share d,
                                  func_add ADD,
                                  func_sub SUB,
                                  func_mul MULT) const
    {
        return OECL0_POST_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_POST_Share prepare_mult3(OECL0_POST_Share b, OECL0_POST_Share c, func_add ADD, func_sub SUB, func_mul MULT)
        const
    {
        return OECL0_POST_Share();
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL0_POST_Share prepare_mult4(OECL0_POST_Share b,
                                   OECL0_POST_Share c,
                                   OECL0_POST_Share d,
                                   func_add ADD,
                                   func_sub SUB,
                                   func_mul MULT) const
    {
        return OECL0_POST_Share();
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
                                 OECL0_POST_Share& r_mk2,
                                 OECL0_POST_Share& r_msb,
                                 OECL0_POST_Share& c,
                                 OECL0_POST_Share& c_prime,
                                 int frac_bits = FRACTIONAL) const
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OECL0_POST_Share& r_mk2,
                                  OECL0_POST_Share& r_msb,
                                  OECL0_POST_Share& c,
                                  OECL0_POST_Share& c_prime) const
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OECL0_POST_Share prepare_trunc_exact_xmod2t(func_add ADD,
                                                func_sub SUB,
                                                func_xor XOR,
                                                func_and AND,
                                                int fractional_bits = FRACTIONAL) const
    {
        return OECL0_POST_Share();
    }

#if USE_CUDA_GEMM > 0
    static void
    GEMM(OECL0_POST_Share* a, OECL0_POST_Share* b, OECL0_POST_Share* c, int m, int n, int k, bool a_fixed = false)
    {
    }
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const OECL0_POST_Share* X,
                        const OECL0_POST_Share* W,
                        OECL0_POST_Share* Y,
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
