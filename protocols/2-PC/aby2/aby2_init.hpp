#pragma once
#include "../../beaver_triples.hpp"
#include <cstdint>
#include <functional>
template <typename Datatype>
class ABY2_init
{
  private:
  public:
    ABY2_init() {}

    template <typename func_add>
    void generate_lxly_from_triple(func_add ADD, int num_round = 0) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            num_boolean_triples[num_round]++;
            store_output_share_bool_(num_round);
            store_output_share_bool_(num_round);
            store_output_share_bool_(num_round);
        }
        else
        {
            num_arithmetic_triples[num_round]++;
            store_output_share_arithmetic_(num_round);
            store_output_share_arithmetic_(num_round);
            store_output_share_arithmetic_(num_round);
        }
        if (num_round == 0)
        {
            pre_send_to_(PNEXT);
            pre_send_to_(PNEXT);
        }
        else
        {
            send_in_last_round[PNEXT]++;
            send_in_last_round[PNEXT]++;
        }
    }
    
    template <typename func_add>
    void generate_lxly_triple(func_add ADD, int num_round = 0) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            num_boolean_triples[num_round]++;
            store_output_share_bool_(num_round);
            store_output_share_bool_(num_round);
        }
        else
        {
            num_arithmetic_triples[num_round]++;
            store_output_share_arithmetic_(num_round);
            store_output_share_arithmetic_(num_round);
        }
    }

    template <typename func_mul>
    ABY2_init mult_public(const Datatype b, func_mul MULT) const
    {
        return ABY2_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    ABY2_init prepare_mult_public_fixed(const Datatype b,
                                        func_mul MULT,
                                        func_add ADD,
                                        func_sub SUB,
                                        func_trunc TRUNC,
                                        int fractional_bits = FRACTIONAL) const
    {
#if PARTY == 0
        send_to_(PNEXT);
#else
        pre_send_to_(PNEXT);
#endif
        return ABY2_init();
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    ABY2_init prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
#if PARTY == 0
        send_to_(PNEXT);
#else
        pre_send_to_(PNEXT);
#endif
        return ABY2_init();
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
#if PARTY == 1
        receive_from_(PNEXT);
#else
        store_output_share_();
#endif
    }

    // P_i shares mx - lxi, P_j sets lxj to 0
    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
            send_to_(PNEXT);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
        prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
            receive_from_(id);
    }

    template <typename func_add>
    ABY2_init Add(ABY2_init b, func_add ADD) const
    {
        return ABY2_init();
    }

    void prepare_reveal_to_all() const
    {
        pre_send_to_(PNEXT);
        store_output_share_();
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        return SET_ALL_ZERO();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_mult(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_init c;
        generate_lxly_triple(ADD);
        send_to_(PNEXT);
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_init c;
        generate_lxly_triple(ADD);
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot3(const ABY2_init b, const ABY2_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        store_output_share_ab_(ADD, helper_index);        // rxyz
        generate_lxly_triple(ADD);     // rxy
        generate_lxly_triple(ADD, 1);  // rxyz
        generate_lxly_triple(ADD);     // rxz
        b.generate_lxly_triple(ADD);   // ryz
        return ABY2_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot4(const ABY2_init b,
                           const ABY2_init c,
                           const ABY2_init d,
                           func_add ADD,
                           func_sub SUB,
                           func_mul MULT) const
    {
        store_output_share_ab_(ADD, helper_index);        // xzw
        store_output_share_ab_(ADD, helper_index);        // yzw
        store_output_share_ab_(ADD, helper_index);        // xyz
        store_output_share_ab_(ADD, helper_index);        // xyw
        generate_lxly_triple(ADD);     // xy --> +2 stores
        generate_lxly_triple(ADD);     // zw --> +2 stores
        generate_lxly_triple(ADD, 1);  // xyw
        generate_lxly_triple(ADD, 1);  // xzw
        generate_lxly_triple(ADD, 1);  // yzw
        generate_lxly_triple(ADD, 1);  // xyz
        generate_lxly_triple(ADD, 1);  // xyzw
        generate_lxly_triple(ADD);     // xz
        generate_lxly_triple(ADD);     // xw
        generate_lxly_triple(ADD);     // yz
        generate_lxly_triple(ADD);     // yw
        return ABY2_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_mult3(ABY2_init b, ABY2_init c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_init d = prepare_dot3(b, c, ADD, SUB, MULT);
        d.mask_and_send_dot(ADD, SUB);
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
        complete_mult(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_mult4(ABY2_init b, ABY2_init c, ABY2_init d, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_init e = prepare_dot4(b, c, d, ADD, SUB, MULT);
        e.mask_and_send_dot(ADD, SUB);
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
        complete_mult(ADD, SUB);
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        send_to_(PNEXT);
    }
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        send_to_(PNEXT);
    }

    static void prepare_A2B_S1(int m, int k, ABY2_init in[], ABY2_init out[])
    {
#if PARTY == 0
        for (int i = m; i < k; i++)
        {
            send_to_(PNEXT);
        }
#endif
    }

    static void prepare_A2B_S2(int m, int k, ABY2_init in[], ABY2_init out[])
    {
#if PARTY == 1
        for (int i = m; i < k; i++)
        {
            pre_send_to_(PNEXT);
        }
#endif
    }

    static void complete_A2B_S1(int k, ABY2_init out[])
    {
#if PARTY == 1
        for (int i = 0; i < k; i++)
        {
            receive_from_(PNEXT);
        }
#endif
    }

    static void complete_A2B_S2(int k, ABY2_init out[])
    {
#if PARTY == 0
        for (int i = 0; i < k; i++)
        {
            store_output_share_();
        }
#endif
    }

    void prepare_bit2a(ABY2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            num_arithmetic_triples[0]++;
            generate_lxly_triple(OP_ADD);
            send_to_(PNEXT);
        }
    }

    void complete_bit2a() { receive_from_(PNEXT); }

    void prepare_opt_bit_injection(ABY2_init x[], ABY2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            num_arithmetic_triples[0]++;
            num_arithmetic_triples[1]++;
            generate_lxly_triple(OP_ADD);
            store_output_share_arithmetic_(helper_index);
            store_output_share_arithmetic_(helper_index);
            generate_lxly_triple(OP_ADD,1);
            send_to_(PNEXT);
        }
    }

    void complete_opt_bit_injection() { receive_from_(PNEXT); }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        receive_from_(PNEXT);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        receive_from_(PNEXT);
    }

    static ABY2_init public_val(Datatype a) { return ABY2_init(); }

    ABY2_init Not() const { return ABY2_init(); }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

    static void complete_preprocessing(uint64_t* arithmetic_triple_num,
                                       uint64_t* boolean_triple_num,
                                       uint64_t num_output_shares)
    {
        for (uint64_t j = 0; j < 2; j++)
        {
            communicate_pre_();
            // for (uint64_t i = 0; i < arithmetic_triple_num[j] + boolean_triple_num[j]; i++)
            // {
            //     pre_receive_from_(PNEXT);
            //     pre_receive_from_(PNEXT);
            // }
            if (j == 0)
            {
                for (uint64_t i = 0; i < num_output_shares; i++)
                    pre_receive_from_(PNEXT);
                // for (uint64_t i = 0; i < send_in_last_round[PNEXT]; i++)
                //     pre_send_to_(PNEXT);
            }
        }
#if SKIP_PRE == 1
        return;
#endif
        triple_type.push_back(new uint8_t[arithmetic_triple_num[0] + boolean_triple_num[0] + num_output_shares]);
        triple_type_index.push_back(0);
        triple_type.push_back(new uint8_t[arithmetic_triple_num[1] + boolean_triple_num[1]]);
        triple_type_index.push_back(0);
    }

#if SKIP_PRE == 1
    template <typename func_add, typename func_sub, typename func_mul>
    static void generate_lxly_triple(uint64_t triple_num, func_add ADD, func_sub SUB, func_mul MULT)
    {
    }
#endif
    /* { */
    /* DATATYPE* lxly = new DATATYPE[triple_num]; */
    /* for (uint64_t i = 0; i < triple_num; i++) */
    /* { */
    /*     lxly[i] = SET_ALL_ZERO(); */
    /* } */
    /* if constexpr(std::is_same_v<func_add(), OP_XOR>) */
    /* { */
    /* delete[] preprocessed_outputs_bool; */
    /* preprocessed_outputs_bool[0] = lxly; */
    /* preprocessed_outputs_bool_index[0] = 0; */
    /* preprocessed_outputs_bool_input_index[0] = 0; */
    /* } */
    /* else */
    /* { */
    /* delete[] preprocessed_outputs_arithmetic; */
    /* preprocessed_outputs_arithmetic[0] = lxly; */
    /* preprocessed_outputs_arithmetic_index[0] = 0; */
    /* preprocessed_outputs_arithmetic_input_index[0] = 0; */
    /* } */
    /* } */

    // --- Untested Functions --- TODO: Test

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    ABY2_init prepare_trunc_share(func_mul MULT,
                                  func_add ADD,
                                  func_sub SUB,
                                  func_trunc TRUNC,
                                  int fractional_bits = FRACTIONAL) const
    {
        send_to_(PNEXT);
    }

    void get_random_B2A() {}

#if USE_CUDA_GEMM == 2
    static void CONV_2D(const ABY2_init* X,
                        const ABY2_init* W,
                        ABY2_init* Y,
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
        const int m = out_h * out_w * batchSize;
        const int k = wh * ww * din;
        const int n = dout;
        for (int i = 0; i < m * n * k; i++)
        {
            ABY2_init().generate_lxly_triple(OP_ADD);
        }
    }

#elif USE_CUDA_GEMM == 4

    static void CONV_2D(const ABY2_init* X,
                        const ABY2_init* W,
                        ABY2_init* Y,
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
        const int m = out_h * out_w * batchSize;
        const int k = wh * ww * din;
        const int n = dout;
        for (int i = 0; i < m * n * k; i++)
        {
            ABY2_init().generate_lxly_triple(OP_ADD);
        }
    }
#endif
#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1

    static void GEMM(ABY2_init* a, ABY2_init* b, ABY2_init* c, int m, int n, int k, bool a_fixed = false)
    {
        for (int i = 0; i < m * n * k; i++)
        {
            ABY2_init().generate_lxly_triple(OP_ADD);
        }
    }
#else

    static void GEMM(ABY2_init* a, ABY2_init* b, ABY2_init* c, int m, int n, int k, bool a_fixed = false)
    {
        for (int i = 0; i < m * n * k; i++)
        {
            ABY2_init().generate_lxly_triple(OP_ADD);
        }
    }
#endif
#endif
};
