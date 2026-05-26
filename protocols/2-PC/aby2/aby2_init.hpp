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
        }
        else
        {
            num_arithmetic_triples[num_round]++;
        }
    }
    
    template <typename func_add>
    void generate_lxly2_triple(func_add ADD, int num_round = 0) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            num_ab2_boolean_triples[num_round]++;
        }
        else
        {
            num_ab2_arithmetic_triples[num_round]++;
        }
    }
  #if PROTOCOL ==4 && ROT_PREPROCESSING_OPT ==1
   Datatype get_mask() const
   {
       return SET_ALL_ZERO();
   }
   #endif 
    template <typename func_add>
    ABY2_init zero_add(Datatype assign, func_add ADD) const
    {
        pre_send_to_(PNEXT);
        store_output_share_();
        return ABY2_init();
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
        ABY2_init local_mult_and_trunc(const Datatype b,
                                        func_mul MULT,
                                        func_add ADD,
                                        func_sub SUB,
                                        func_trunc TRUNC,
                                        int fractional_bits = FRACTIONAL) const
        {
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
    
    template <typename func_mul>
    ABY2_init mult_a_known_to_evaluators(const ABY2_init b,
                                                func_mul MULT) const
    {
        return ABY2_init();
    }

    template <typename func_add, typename func_sub>
        void prepare_remask(func_add ADD, func_sub SUB)
        {
            send_to_(PNEXT);
        }

    template <typename func_add, typename func_sub>
        void complete_remask(func_add ADD, func_sub SUB)
        {
            receive_from_(PNEXT);
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
#if SHARE_PREP == 0
        if constexpr (id == PSELF)
        {
            send_to_(PNEXT);
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
#if SHARE_PREP == 0
        if constexpr (id != PSELF)
            receive_from_(id);
#endif
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
        generate_lxly_triple(ADD);
        send_to_(PNEXT);
        return ABY2_init();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_mult(ABY2_init b, Datatype assign, Datatype triple_c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        generate_lxly_triple(ADD);
        send_to_(PNEXT);
        return ABY2_init();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_mult_a_known(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        generate_lxly2_triple(ADD);
        send_to_(PNEXT);
        return ABY2_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        generate_lxly_triple(ADD);
        return ABY2_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot_and_assign(ABY2_init b, Datatype assign, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return ABY2_init();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot_ex_lxly(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return ABY2_init();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot_ex_lxly_a_known(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return ABY2_init();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot_a_known(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        generate_lxly2_triple(ADD);
        return ABY2_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_init prepare_dot_ex_lxly_a_known_pre(ABY2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return ABY2_init();
    }

#if FUSE_CONV_BN_SIM == 1
    template <typename func_add, typename func_sub, typename func_mul>
    void prepare_Conv_BN_Accum(const ABY2_PRE x, Datatype* result, func_add ADD, func_sub SUB, func_mul MULT) const
    {
    }
    
    template <typename func_add, typename func_sub, typename func_mul, typename func_trunc>
    void calculate_conv_bn(const ABY2_PRE mu, const ABY2_PRE sigma, const Datatype* accum, func_add ADD, func_sub SUB, func_mul MULT, func_trunc TRUNC) 
    {
    #if BN2D_TRIPLES == 0 //otherwise triples are generated via Batchnorm triples
        store_output_share_ab(ADD, helper_index);
        //lx lw should be generated via conv triple
        generate_lxly_triple(ADD);     // lw lsigma
        generate_lxly_triple(ADD);     // lx lsigma
        generate_lxly_triple(ADD, 1);  // (lw lx) lsigma
    #endif
        mask_and_send_dot_with_trunc(ADD, SUB, TRUNC);
    }
    
    static int get_conv_bn_size() { return 0; }

#endif

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
    
    template <typename func_add, typename func_sub>
    void mask_and_send_dot_with_triple(func_add ADD, func_sub SUB)
    {
        send_to_(PNEXT);
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot_with_triple(func_add ADD, func_sub SUB, int index)
    {
        send_to_(PNEXT);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        send_to_(PNEXT);
    }
    
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc_with_triple(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        send_to_(PNEXT);
    }
    
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc_with_triple(func_add ADD, func_sub SUB, func_trunc TRUNC, int index)
    {
        send_to_(PNEXT);
    }
    
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_a_known_pre_with_triple_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC, int index)
    {
#if PARTY == 0
        send_to_(PNEXT);
#endif
    }

    static void prepare_A2B_S1(int m, int k, ABY2_init in[], ABY2_init out[])
    {
#if A2B_ONLINE_OPT == 0
#if PARTY == 0
        for (int i = m; i < k; i++)
        {
            send_to_(PNEXT);
        }
#endif
#endif
    }

    static void prepare_A2B_S2(int m, int k, ABY2_init in[], ABY2_init out[])
    {
#if A2B_ONLINE_OPT == 1
        for (int i = m; i < k; i++)
        {
            num_boolean_addition_triples++;
        }
#endif
    }

    static void complete_A2B_S1(int k, ABY2_init out[])
    {
#if A2B_ONLINE_OPT == 0
#if PARTY == 1
        for (int i = 0; i < k; i++)
        {
            receive_from_(PNEXT);
        }
#endif
#endif
    }

    static void complete_A2B_S2(int k, ABY2_init out[])
    {
    }

    void prepare_bit2a(ABY2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            // num_arithmetic_triples[0]++;
            generate_lxly_triple(OP_ADD);
            send_to_(PNEXT);
        }
    }

    void complete_bit2a() { receive_from_(PNEXT); }

    void prepare_opt_bit_injection(ABY2_init x[], ABY2_init out[])
    {
        for (int i = 0; i < BITLENGTH; i++)
        {
            // num_arithmetic_triples[0]++;
            // num_arithmetic_triples[1]++;
#if BIT_INJECTION_PREPROCESSING_OPT == 1 
            num_multiplexer_triples++;
            num_cot_triples++;
#else
            generate_lxly2_triple(OP_ADD);
            store_output_share_arithmetic_(helper_index);
            store_output_share_arithmetic_(helper_index);
            generate_lxly_triple(OP_ADD,1);
#endif
            send_to_(PNEXT);
        }
    }

    void complete_opt_bit_injection() { receive_from_(PNEXT); }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
        receive_from_(PNEXT);
    }
    
    template <typename func_add, typename func_sub>
    void complete_mult_a_known_pre(func_add ADD, func_sub SUB)
    {
#if PARTY == 1
        receive_from_(PNEXT);
#endif
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

    static void complete_preprocessing()
    {
            communicate_pre_();
                for (uint64_t i = 0; i < preprocessed_outputs_index; i++)
                    pre_receive_from_(PNEXT);
            communicate_pre_();
        
#if SKIP_PRE == 1 && LOAD_PREPROCESSING == 0
        return;
#endif
        total_num_boolean_output_triples.push_back(num_boolean_triples[0] + num_ab2_boolean_triples[0] + num_boolean_addition_triples);
        total_num_boolean_output_triples.push_back(num_boolean_triples[1] + num_ab2_boolean_triples[1]);
        total_num_arithmetic_output_triples.push_back(num_arithmetic_triples[0] + num_ab2_arithmetic_triples[0] + num_conv_c_triples + num_fc_c_triples + num_bc2D_c_triples + num_multiplexer_triples + num_cot_triples);
        total_num_arithmetic_output_triples.push_back(num_arithmetic_triples[1] + num_ab2_arithmetic_triples[1]);
        uint64_t total_num_output_triples_round0 = preprocessed_outputs_index + total_num_boolean_output_triples[0] + total_num_arithmetic_output_triples[0];
        uint64_t total_num_output_triples_round1 = total_num_boolean_output_triples[1] + total_num_arithmetic_output_triples[1];
#if SKIP_PRE == 1
        return;
#endif
        triple_type.push_back(new uint8_t[total_num_output_triples_round0]);
        triple_type_index.push_back(0);
        triple_type.push_back(new uint8_t[total_num_output_triples_round1]);
        triple_type_index.push_back(0);
    }

#if SKIP_PRE == 1
    template <typename func_add, typename func_sub, typename func_mul>
    static void generate_lxly_triple(uint64_t triple_num, func_add ADD, func_sub SUB, func_mul MULT)
    {
    }
    template <typename func_add, typename func_sub, typename func_mul>
    static void generate_lxly2_triple(uint64_t triple_num, func_add ADD, func_sub SUB, func_mul MULT)
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

        // T::SetupConv2dTriples(prev_out.data(), kernel.data(), this->output.data(),batch, ic, oc, ih, iw, kh, kw, stride, pad);
    static void SetupConv2dTriples(const ABY2_init* X,
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
                                   int oh,
                                   int ow,
                                   int dilation = 1,
                                   bool ab2 = false)
    {
        conv_triple_params.push_back(ConvolutionParameter(batchSize, inh, inw, din, dout, wh, ww, padding, stride, oh, ow, dilation));
        num_conv_c_triples += batchSize * oh * ow * dout;
    }

    static void SetupFullyConnectedTriples(const ABY2_init* X,
                                   const ABY2_init* W,
                                   ABY2_init* Y,
                                   int batchSize,
                                   int in_feat,
                                   int out_feat,
                                   bool ab2 = true)
    {
        fc_triple_params.push_back(FullyConnectedParameter(batchSize, in_feat, out_feat));
        num_fc_c_triples += batchSize * out_feat;
    }
    
    static void SetupBatchNorm2DTriples(const ABY2_init* X,
                                   const ABY2_init* W,
                                   ABY2_init* Y,
                                   int batchSize,
                                   int ch,
                                   int h,
                                   int w,
                                   bool ab2 = true)
    {
        bc2D_triple_params.push_back(BatchNorm2DParameter(batchSize, ch, h, w));
        num_bc2D_c_triples += batchSize * ch * h * w;
    }



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
                        int dilation = 1,
                        bool ab2 = false)
    {
#if CONV_TRIPLES == 0
        const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
        const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
        const int m = out_h * out_w * batchSize;
        const int k = wh * ww * din;
        const int n = dout;
        if(ab2) 
            for (int i = 0; i < m * n * k; i++)
                ABY2_init().generate_lxly2_triple(OP_ADD);
        else
            for (int i = 0; i < m * n * k; i++)
                ABY2_init().generate_lxly_triple(OP_ADD);
#endif
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
                        int dilation = 1,
                        bool ab2 = false)
    {
#if CONV_TRIPLES == 0
        const int m = out_h * out_w * batchSize;
        const int k = wh * ww * din;
        const int n = dout;
        if(ab2) 
            for (int i = 0; i < m * n * k; i++)
                ABY2_init().generate_lxly2_triple(OP_ADD);
        else
            for (int i = 0; i < m * n * k; i++)
                ABY2_init().generate_lxly_triple(OP_ADD);
#endif
    }
#endif
#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1

    static void GEMM(ABY2_init* a, ABY2_init* b, ABY2_init* c, int m, int n, int k, bool a_fixed = false, bool ab2=false)
    {
    if(ab2)
        for (int i = 0; i < m * n * k; i++)
            ABY2_init().generate_lxly2_triple(OP_ADD);
    else
        for (int i = 0; i < m * n * k; i++)
            ABY2_init().generate_lxly_triple(OP_ADD);
    }
#else

    static void GEMM(ABY2_init* a, ABY2_init* b, ABY2_init* c, int m, int n, int k, bool a_fixed = false, bool ab2=false)
    {
        if(ab2)
            for (int i = 0; i < m * n * k; i++)
                ABY2_init().generate_lxly2_triple(OP_ADD);
        else
        for (int i = 0; i < m * n * k; i++)
            ABY2_init().generate_lxly_triple(OP_ADD);
    }
#endif
#endif
};
