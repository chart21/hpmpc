#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL_MAL3_POST_Share
{
  private:
    Datatype r0;
    Datatype r1;

  public:
    OECL_MAL3_POST_Share() {}
    OECL_MAL3_POST_Share(Datatype r0, Datatype r1) : r0(r0), r1(r1) {}
    OECL_MAL3_POST_Share(Datatype r0) : r0(r0) {}

    void prepare_reveal_to_all() const {}

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PROTOCOL == 8
        Datatype mv = receive_from_live(P_0);
        store_compare_view(P_1, mv);  // verify own value
        Datatype result = SUB(mv, retrieve_output_share());
        result = SUB(result, retrieve_output_share());
        result = SUB(result, retrieve_output_share());
#else
        Datatype result = SUB(receive_from_live(P_0), retrieve_output_share());
        store_compare_view(P_123, retrieve_output_share());
        store_compare_view(P_0123, result);
#endif
        return result;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() { communicate_live(); }

    static OECL_MAL3_POST_Share public_val(Datatype a) { return OECL_MAL3_POST_Share(); }

    template <typename func_mul>
    OECL_MAL3_POST_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL_MAL3_POST_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC)
        const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL_MAL3_POST_Share prepare_mult_public_fixed(const Datatype b,
                                                   func_mul MULT,
                                                   func_add ADD,
                                                   func_sub SUB,
                                                   func_trunc TRUNC,
                                                   int fractional_bits = FRACTIONAL) const
    {
        return OECL_MAL3_POST_Share();
    }
    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    OECL_MAL3_POST_Share prepare_trunc_share(func_mul MULT,
                                             func_add ADD,
                                             func_sub SUB,
                                             func_trunc TRUNC,
                                             int fractional_bits = FRACTIONAL) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    OECL_MAL3_POST_Share Not() const { return OECL_MAL3_POST_Share(); }

    template <typename func_add>
    OECL_MAL3_POST_Share Add(OECL_MAL3_POST_Share b, func_add ADD) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL_MAL3_POST_Share prepare_mult(OECL_MAL3_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL_MAL3_POST_Share prepare_dot(const OECL_MAL3_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return OECL_MAL3_POST_Share();
    }
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void prepare_trunc_2k_inputs(func_add ADD,
                                 func_sub SUB,
                                 func_xor XOR,
                                 func_and AND,
                                 func_trunc trunc,
                                 OECL_MAL3_POST_Share& r_mk2,
                                 OECL_MAL3_POST_Share& r_msb,
                                 OECL_MAL3_POST_Share& c,
                                 OECL_MAL3_POST_Share& c_prime,
                                 int fractional_bits = FRACTIONAL) const
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
    void complete_trunc_2k_inputs(func_add ADD,
                                  func_sub SUB,
                                  func_xor XOR,
                                  func_and AND,
                                  func_trunc trunc,
                                  OECL_MAL3_POST_Share& r_mk2,
                                  OECL_MAL3_POST_Share& r_msb,
                                  OECL_MAL3_POST_Share& c,
                                  OECL_MAL3_POST_Share& c_prime) const
    {
    }

    template <typename func_add, typename func_sub, typename func_xor, typename func_and>
    OECL_MAL3_POST_Share prepare_trunc_exact_xmod2t(func_add ADD,
                                                    func_sub SUB,
                                                    func_xor XOR,
                                                    func_and AND,
                                                    int fractional_bits = FRACTIONAL) const
    {
        return OECL_MAL3_POST_Share();
    }

    static void prepare_A2B_S1(int m, int k, OECL_MAL3_POST_Share in[], OECL_MAL3_POST_Share out[]) {}

    static void prepare_A2B_S2(int m, int k, OECL_MAL3_POST_Share in[], OECL_MAL3_POST_Share out[]) {}

    static void complete_A2B_S1(int k, OECL_MAL3_POST_Share out[]) {}

    static void complete_A2B_S2(int k, OECL_MAL3_POST_Share out[]) {}

    static void prepare_B2A(OECL_MAL3_POST_Share z[], OECL_MAL3_POST_Share random_mask[], OECL_MAL3_POST_Share out[]) {}

    static void complete_B2A(OECL_MAL3_POST_Share z[], OECL_MAL3_POST_Share out[]) {}

    static void complete_B2A2(OECL_MAL3_POST_Share z[], OECL_MAL3_POST_Share out[]) {}

    void prepare_bit2a(OECL_MAL3_POST_Share out[]) {}

    void complete_bit2a() {}

    void prepare_opt_bit_injection(OECL_MAL3_POST_Share x[], OECL_MAL3_POST_Share out[]) {}

    void complete_opt_bit_injection() {}

    void prepare_bit_injection_S1(OECL_MAL3_POST_Share out[]) {}

    void prepare_bit_injection_S2(OECL_MAL3_POST_Share out[]) {}

    static void complete_bit_injection_S1(OECL_MAL3_POST_Share out[]) {}

    static void complete_bit_injection_S2(OECL_MAL3_POST_Share out[]) {}
    void get_random_B2A() {}

#if MULTI_INPUT == 1

    template <typename func_add, typename func_sub, typename func_mul>
    OECL_MAL3_POST_Share prepare_dot3(const OECL_MAL3_POST_Share b,
                                      const OECL_MAL3_POST_Share c,
                                      func_add ADD,
                                      func_sub SUB,
                                      func_mul MULT) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL_MAL3_POST_Share prepare_mult3(const OECL_MAL3_POST_Share b,
                                       const OECL_MAL3_POST_Share c,
                                       func_add ADD,
                                       func_sub SUB,
                                       func_mul MULT) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL_MAL3_POST_Share prepare_dot4(const OECL_MAL3_POST_Share b,
                                      const OECL_MAL3_POST_Share c,
                                      const OECL_MAL3_POST_Share d,
                                      func_add ADD,
                                      func_sub SUB,
                                      func_mul MULT) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OECL_MAL3_POST_Share prepare_mult4(const OECL_MAL3_POST_Share b,
                                       const OECL_MAL3_POST_Share c,
                                       const OECL_MAL3_POST_Share d,
                                       func_add ADD,
                                       func_sub SUB,
                                       func_mul MULT) const
    {
        return OECL_MAL3_POST_Share();
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
    }

#endif

#if USE_CUDA_GEMM > 0
    static void
    GEMM(OECL_MAL3_POST_Share* a, OECL_MAL3_POST_Share* b, OECL_MAL3_POST_Share* c, int m, int n, int k, bool a_fixed)
    {
    }
#endif
#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
    static void CONV_2D(const OECL_MAL3_POST_Share* X,
                        const OECL_MAL3_POST_Share* W,
                        OECL_MAL3_POST_Share* Y,
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
