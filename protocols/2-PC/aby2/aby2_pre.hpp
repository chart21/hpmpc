#include "../../beaver_triples.hpp"
#include "../../../core/store_to_file.hpp"
#include <cstdint>
#include <string>
template <typename Datatype>
class ABY2_PRE_Share
{
#define CaseAND 0
#define CaseMult 1
#define CaseBit2A 3
#define CaseBitInjection 4
#define CaseDot3Bool 5
#define CaseDot3Arithmetic 6
#define CaseDot4Bool 7
#define CaseDot4Arithmetic 8
#define CaseMatMulFirstDot 9
#define CaseMatMul 10
#define CaseANDAKnown 11
#define CaseMultAKnown 12
#define CaseScalarBoolAKnown 13
#define CaseScalarArithAKnown 14
#define CaseConv 15
#define CaseBatchNorm2D 16
#define CaseFullyConnected 17
#define CaseTripleAlreadyConsumed 99
#define CaseDefault 2


    using BT = triple<Datatype>;

  private:
    Datatype l;

  public:


    ABY2_PRE_Share() {}
    ABY2_PRE_Share(Datatype l) { this->l = l; }

#if LX_TRIPLES == 1 
    template <typename func_add>
    void generate_lxly_triple(ABY2_PRE_Share b, func_add ADD, int num_round = 0) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
           storeBooleanABTriple(l, b.l); 
        }
        else
        {
           storeArithmeticABTriple(l, b.l);
        }
    }
    
    template <typename func_add>
    static Datatype receive_and_compute_lxly_share(func_add ADD, int num_round = 0)
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
            // return retrieve_output_share_bool(num_round);
            return boolean_triple_c[curr_boolean_triple_index++];
        else
            // return retrieve_output_share_arithmetic(num_round);
            return arithmetic_triple_c[curr_arithmetic_triple_index++];
    }
    
    template <typename func_add>
    void generate_lxly2_triple(ABY2_PRE_Share b, func_add ADD, int num_round = 0) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
           storeBooleanAB2Triple(l, b.l); 
        }
        else
        {
           storeArithmeticAB2Triple(l, b.l);
        }
    }
    
    template <typename func_add>
    static Datatype receive_and_compute_lxly2_share(func_add ADD, int num_round = 0)
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
            // return retrieve_output_share_bool(num_round);
            return boolean_ab2_triple_c[curr_boolean_ab2_triple_index++];
        else
            // return retrieve_output_share_arithmetic(num_round);
            return arithmetic_ab2_triple_c[curr_arithmetic_ab2_triple_index++];
    }

#endif



    template <typename func_mul>
    ABY2_PRE_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return ABY2_PRE_Share(MULT(l, b));
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
            l = getRandomVal(PSELF);
        else
            l = SET_ALL_ZERO();
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {
        prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add>
    ABY2_PRE_Share Add(ABY2_PRE_Share b, func_add ADD) const
    {
        return ABY2_PRE_Share(ADD(l, b.l));
    }

    void prepare_reveal_to_all() const
    {
        pre_send_to_live(PNEXT, l);
        triple_type[0][triple_type_index[0]++] = CaseDefault;  
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        return SET_ALL_ZERO();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            triple_type[0][triple_type_index[0]++] = CaseAND;
        }
        else
        {
            triple_type[0][triple_type_index[0]++] = CaseMult;
        }
        generate_triple(b, ADD);
        return ABY2_PRE_Share(getRandomVal(PSELF));  // new mask
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult_a_known(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            triple_type[0][triple_type_index[0]++] = CaseANDAKnown;
        }
        else
        {
            triple_type[0][triple_type_index[0]++] = CaseMultAKnown;
        }
        generate_ab2_triple(b, ADD);
        return ABY2_PRE_Share(getRandomVal(PSELF));  // new mask
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            triple_type[0][triple_type_index[0]++] = CaseAND;
        }
        else
        {
            triple_type[0][triple_type_index[0]++] = CaseMult;
        }
        generate_triple(b, ADD);
        return ABY2_PRE_Share();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot_ex_lxly(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return ABY2_PRE_Share();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot_ex_lxly_a_known(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return ABY2_PRE_Share();
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot_a_known(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            triple_type[0][triple_type_index[0]++] = CaseANDAKnown;
        }
        else
        {
            triple_type[0][triple_type_index[0]++] = CaseMultAKnown;
        }
        generate_ab2_triple(b, ADD);
        return ABY2_PRE_Share();
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
        l = getRandomVal(PSELF);
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot_with_triple(func_add ADD, func_sub SUB)
    {
        l = getRandomVal(PSELF);
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        l = getRandomVal(PSELF);
    }
    
    template <typename func_add, typename func_sub, typename func_trunc>
    void mask_and_send_dot_with_trunc_with_triple(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
        l = getRandomVal(PSELF);
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    ABY2_PRE_Share prepare_mult_public_fixed(const Datatype b,
                                             func_mul MULT,
                                             func_add ADD,
                                             func_sub SUB,
                                             func_trunc TRUNC,
                                             int fractional_bits = FRACTIONAL) const
    {
#if PARTY == 0
        triple_type[0][triple_type_index[0]++] = CaseDefault;
        return ABY2_PRE_Share(getRandomVal(PSELF));
#else
        auto c = ABY2_PRE_Share(getRandomVal(PSELF));
        pre_send_to_live(
            PNEXT, ADD(c.l, SUB(SET_ALL_ZERO(), TRUNC(MULT(l, b), fractional_bits))));  // Share Trunc -(lv1 * b) + lz
        return c;
#endif
    }

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    ABY2_PRE_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
    {
#if PARTY == 0
        triple_type[0][triple_type_index[0]++] = CaseDefault;
        return ABY2_PRE_Share(getRandomVal(PSELF));
#else
        auto result = l;  // Share Trunc - Trunc(lv1)
        for (int i = 2; i <= b; i *= 2)
            result = OP_TRUNC2(result);
        result = OP_SUB(SET_ALL_ZERO(), result);

        Datatype res_l = getRandomVal(PSELF);
        pre_send_to_live(PNEXT, ADD(result, res_l));
        return ABY2_PRE_Share(res_l);
#endif
    }

    template <typename func_add, typename func_sub>
    void complete_public_mult_fixed(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
    }

    template <typename func_add, typename func_sub, typename func_trunc>
    void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
    {
    }

    static void prepare_A2B_S1(int m, int k, ABY2_PRE_Share in[], ABY2_PRE_Share out[])
    {
#if PARTY == 0
        for (int i = m; i < k; i++)
        {
            out[i - m].l = getRandomVal(PSELF);
        }
#endif
    }

    static void prepare_A2B_S2(int m, int k, ABY2_PRE_Share in[], ABY2_PRE_Share out[])
    {
#if PARTY == 1
        Datatype temp_p1[BITLENGTH];
        for (int i = 0; i < BITLENGTH; i++)
        {
            temp_p1[i] = OP_SUB(SET_ALL_ZERO(), in[i].l);  // set second share to -lv2
        }
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        unorthogonalize_arithmetic(temp_p1, temp2);
        orthogonalize_boolean(temp2, temp_p1);

        for (int i = m; i < k; i++)
        {
            out[i - m].l = getRandomVal(PSELF);
            Datatype out_m = OP_XOR(temp_p1[i], out[i - m].l);
            pre_send_to_live(PNEXT, out_m);
        }
#else
        for (int i = m; i < k; i++)
        {
            triple_type[0][triple_type_index[0]++] = CaseDefault;
        }
#endif
    }

    void prepare_bit2a(ABY2_PRE_Share out[])
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        Datatype lb[BITLENGTH]{0};
        lb[BITLENGTH - 1] = l;
        unorthogonalize_boolean(lb, temp2);
        orthogonalize_arithmetic(temp2, lb);
        for (int i = 0; i < BITLENGTH; i++)
        {
            triple_type[0][triple_type_index[0]++] = CaseBit2A;
#if PARTY == 0
            ABY2_PRE_Share b1{lb[i]};
            ABY2_PRE_Share b2{SET_ALL_ZERO()};
#else
            ABY2_PRE_Share b1{SET_ALL_ZERO()};
            ABY2_PRE_Share b2{lb[i]};
#endif
            b1.generate_triple(
                b2,
                OP_ADD);  // communication can be cut in half if triple of type x(P_0),y(P_1),[z] is used
            out[i].l = getRandomVal(PSELF);
            /* #if PARTY == 0 */
            /*         auto bl = SET_ALL_ZERO(); */
            /*         auto al = lb[i]; */
            /* #else */
            /*         auto bl = lb[i]; */
            /*         auto al = SET_ALL_ZERO(); */
            /* #endif */
            /*         auto lta = OP_ADD(al, t.a); */
            /*         auto ltb = OP_ADD(bl, t.b); //optimization? */
            /*         pre_send_to_live(PNEXT, lta); */
            /*         pre_send_to_live(PNEXT, ltb); */
            /*         auto lxly = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, t.a)), t.c); */
            /*         store_output_share_arithmetic(t.a); */
            /*         store_output_share_arithmetic(bl); */
            /*         store_output_share_arithmetic(lxly); */
            /* out[i].l = getRandomVal(PSELF); */
        }
    }

    void complete_bit2a() {}

    void complete_opt_bit_injection() {}

    void prepare_opt_bit_injection(ABY2_PRE_Share x[], ABY2_PRE_Share out[])
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
        Datatype lb[BITLENGTH]{0};
        lb[BITLENGTH - 1] = l;
        unorthogonalize_boolean(lb, temp2);
        orthogonalize_arithmetic(temp2, lb);
        for (int i = 0; i < BITLENGTH; i++)
        {
            triple_type[0][triple_type_index[0]++] = CaseBitInjection;
            triple_type[1][triple_type_index[1]++] = CaseBitInjection;
#if PARTY == 0
            ABY2_PRE_Share b1{lb[i]};
            ABY2_PRE_Share b2{SET_ALL_ZERO()};
#else
            ABY2_PRE_Share b1{SET_ALL_ZERO()};
            ABY2_PRE_Share b2{lb[i]};
#endif
            b1.generate_triple(
                b2,
                OP_ADD);  // communication can be cut in half if triple of type x(P_0),y(P_1),[z] is used
            store_output_share_arithmetic(lb[i],helper_index);
            store_output_share_arithmetic(x[i].l, helper_index);
            out[i].l = getRandomVal(PSELF);
        }
    }
    static void complete_A2B_S1(int k, ABY2_PRE_Share out[])
    {
#if PARTY == 1
        for (int i = 0; i < k; i++)
        {
            out[i].l = SET_ALL_ZERO();
        }
#endif
    }

    static void complete_A2B_S2(int k, ABY2_PRE_Share out[])
    {
#if PARTY == 0
        for (int i = 0; i < k; i++)
        {
            out[i].l = SET_ALL_ZERO();
        }
#endif
    }

    static ABY2_PRE_Share public_val(Datatype a) { return ABY2_PRE_Share(SET_ALL_ZERO()); }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot3(const ABY2_PRE_Share b,
                                const ABY2_PRE_Share c,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            triple_type[0][triple_type_index[0]++] = CaseDot3Bool;
            triple_type[0][triple_type_index[0]++] = CaseAND;
            triple_type[0][triple_type_index[0]++] = CaseAND;
            triple_type[1][triple_type_index[1]++] = CaseDot3Bool;
        }
        else
        {
            triple_type[0][triple_type_index[0]++] = CaseDot3Arithmetic;
            triple_type[0][triple_type_index[0]++] = CaseMult;
            triple_type[0][triple_type_index[0]++] = CaseMult;
            triple_type[1][triple_type_index[1]++] = CaseDot3Arithmetic;
        }
        store_output_share_ab(c.l, ADD, helper_index);
        generate_triple(b, ADD);    // rxy
        generate_triple(c, ADD);    // rxz
        b.generate_triple(c, ADD);  // ryz
        return ABY2_PRE_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot4(const ABY2_PRE_Share b,
                                const ABY2_PRE_Share c,
                                const ABY2_PRE_Share d,
                                func_add ADD,
                                func_sub SUB,
                                func_mul MULT) const
    {
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            triple_type[0][triple_type_index[0]++] = CaseDot4Bool;   // xy, zw
            triple_type[0][triple_type_index[0]++] = CaseTripleAlreadyConsumed;  // since xy,zw are together, skip next triple
            triple_type[0][triple_type_index[0]++] = CaseAND;   // xz
            triple_type[0][triple_type_index[0]++] = CaseAND;   // xw
            triple_type[0][triple_type_index[0]++] = CaseAND;   // yz
            triple_type[0][triple_type_index[0]++] = CaseAND;   // yw
            triple_type[1][triple_type_index[1]++] = CaseDot4Bool;   // xyzw
        }
        else
        {
            triple_type[0][triple_type_index[0]++] = CaseDot4Arithmetic;   // xy, zw
            triple_type[0][triple_type_index[0]++] = CaseTripleAlreadyConsumed;  // since xy,zw are together, skip next triple 
            triple_type[0][triple_type_index[0]++] = CaseMult;   // xz 
            triple_type[0][triple_type_index[0]++] = CaseMult;   // xw
            triple_type[0][triple_type_index[0]++] = CaseMult;   // yz
            triple_type[0][triple_type_index[0]++] = CaseMult;   // yw
            triple_type[1][triple_type_index[1]++] = CaseDot4Arithmetic;   // xyzw
        }

        store_output_share_ab(l, ADD, helper_index);                   // xzw
        store_output_share_ab(b.l, ADD, helper_index);                 // yzw
        store_output_share_ab(c.l, ADD, helper_index);                 // xyz
        store_output_share_ab(d.l, ADD, helper_index);                 // xyw
        generate_triple(b, ADD);    // xy --> +2 stores
        c.generate_triple(d, ADD);  // zw --> +2 stores
        generate_triple(c, ADD);    // xz
        generate_triple(d, ADD);    // xw
        b.generate_triple(c, ADD);  // yz
        b.generate_triple(d, ADD);  // yw
        return ABY2_PRE_Share();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult3(ABY2_PRE_Share b, ABY2_PRE_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        ABY2_PRE_Share d = prepare_dot3(b, c, ADD, SUB, MULT);
        d.mask_and_send_dot(ADD, SUB);
        return d;
    }

    template <typename func_add, typename func_sub>
    void complete_mult3(func_add ADD, func_sub SUB)
    {
        complete_mult(ADD, SUB);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult4(ABY2_PRE_Share b,
                                 ABY2_PRE_Share c,
                                 ABY2_PRE_Share d,
                                 func_add ADD,
                                 func_sub SUB,
                                 func_mul MULT) const
    {
        ABY2_PRE_Share e = prepare_dot4(b, c, d, ADD, SUB, MULT);
        e.mask_and_send_dot(ADD, SUB);
        return e;
    }

    template <typename func_add, typename func_sub>
    void complete_mult4(func_add ADD, func_sub SUB)
    {
        complete_mult(ADD, SUB);
    }

    ABY2_PRE_Share Not() const { return ABY2_PRE_Share(l); }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate() {}

    static void get_triples_from_file(int tid, uint64_t* arithmetic_triple_num, uint64_t* boolean_triple_num)
    {
        save_triple_file(arithmetic_triple_a, arithmetic_triple_num[tid], arithmetic_triple_b, arithmetic_triple_num[tid], boolean_triple_a, boolean_triple_num[tid], boolean_triple_b, boolean_triple_num[tid], std::to_string(PARTY), "pre");
        Datatype* other_arithmetic_triple_a = new Datatype[arithmetic_triple_num[tid]];
        Datatype* other_arithmetic_triple_b = new Datatype[arithmetic_triple_num[tid]];
        Datatype* other_boolean_triple_a = new Datatype[boolean_triple_num[tid]];
        Datatype* other_boolean_triple_b = new Datatype[boolean_triple_num[tid]];
        load_triple_file(other_arithmetic_triple_a, arithmetic_triple_num[tid], other_arithmetic_triple_b, arithmetic_triple_num[tid], other_boolean_triple_a, boolean_triple_num[tid], other_boolean_triple_b, boolean_triple_num[tid], std::to_string(1 - PARTY), "pre");
        delete_triple_file(std::to_string(1 - PARTY), "pre");
        for (uint64_t i = 0; i < arithmetic_triple_num[tid]; i++)
        {
#if PARTY == 0
            arithmetic_triple_c[i] = OP_SUB( OP_MULT(OP_ADD(arithmetic_triple_a[i], other_arithmetic_triple_a[i]), OP_ADD(arithmetic_triple_b[i], other_arithmetic_triple_b[i])), getRandomVal(PNEXT));
#else 
            arithmetic_triple_c[i] = getRandomVal(PNEXT);
#endif

        }
        delete[] other_arithmetic_triple_a;
        delete[] other_arithmetic_triple_b;
        for (uint64_t i = 0; i < boolean_triple_num[tid]; i++)
        {
#if PARTY == 0
            boolean_triple_c[i] = OP_XOR( OP_AND(OP_XOR(boolean_triple_a[i], other_boolean_triple_a[i]), OP_XOR(boolean_triple_b[i], other_boolean_triple_b[i])), getRandomVal(PNEXT));
#else
            boolean_triple_c[i] = getRandomVal(PNEXT);
#endif
        }
        delete[] other_boolean_triple_a;
        delete[] other_boolean_triple_b;
    }

    
    static void get_ab2_triples_from_file(int tid, uint64_t* arithmetic_ab2_triple_num, uint64_t* boolean_ab2_triple_num)
    {
#if PARTY == 0
        uint64_t arithemtic_b_triple_num = 0;
        uint64_t boolean_b_triple_num = 0;
        uint64_t other_arithmetic_b_triple_num = arithmetic_ab2_triple_num[tid];
        uint64_t other_boolean_b_triple_num = boolean_ab2_triple_num[tid];
        uint64_t arithemtic_a_triple_num = arithmetic_ab2_triple_num[tid];
        uint64_t boolean_a_triple_num = boolean_ab2_triple_num[tid];
        uint64_t other_arithmetic_a_triple_num = 0;
        uint64_t other_boolean_a_triple_num = 0;
#else 
        uint64_t arithemtic_b_triple_num = arithmetic_ab2_triple_num[tid];
        uint64_t boolean_b_triple_num = boolean_ab2_triple_num[tid];
        uint64_t other_arithmetic_b_triple_num = 0;
        uint64_t other_boolean_b_triple_num = 0;
        uint64_t arithemtic_a_triple_num = 0;
        uint64_t boolean_a_triple_num = 0;
        uint64_t other_arithmetic_a_triple_num = arithmetic_ab2_triple_num[tid];
        uint64_t other_boolean_a_triple_num = boolean_ab2_triple_num[tid];
#endif
        save_triple_file(arithmetic_ab2_triple_a, arithemtic_a_triple_num, arithmetic_ab2_triple_b, arithemtic_b_triple_num, boolean_ab2_triple_a, boolean_a_triple_num, boolean_ab2_triple_b, boolean_b_triple_num, std::to_string(PARTY), "pre_ab2");
        Datatype* other_arithmetic_triple_a = new Datatype[other_arithmetic_a_triple_num];
        Datatype* other_arithmetic_triple_b = new Datatype[other_arithmetic_b_triple_num];
        Datatype* other_boolean_triple_a = new Datatype[other_boolean_a_triple_num];
        Datatype* other_boolean_triple_b = new Datatype[other_boolean_b_triple_num];
        load_triple_file(other_arithmetic_triple_a, other_arithmetic_a_triple_num, other_arithmetic_triple_b, other_arithmetic_b_triple_num, other_boolean_triple_a, other_boolean_a_triple_num, other_boolean_triple_b, other_boolean_b_triple_num, std::to_string(1 - PARTY), "pre_ab2");
        delete_triple_file(std::to_string(1 - PARTY), "pre_ab2");
        for (uint64_t i = 0; i < arithmetic_ab2_triple_num[tid]; i++)
        {
#if PARTY == 0
            arithmetic_ab2_triple_c[i] = OP_SUB( OP_MULT(arithmetic_ab2_triple_a[i], other_arithmetic_triple_b[i]), getRandomVal(PNEXT));
#else 
            arithmetic_ab2_triple_c[i] = getRandomVal(PNEXT);
#endif

        }
        delete[] other_arithmetic_triple_a;
        delete[] other_arithmetic_triple_b;
        for (uint64_t i = 0; i < boolean_ab2_triple_num[tid]; i++)
        {
#if PARTY == 0
            boolean_ab2_triple_c[i] = OP_XOR( OP_AND(boolean_ab2_triple_a[i], other_boolean_triple_b[i]), getRandomVal(PNEXT));
#else
            boolean_ab2_triple_c[i] = getRandomVal(PNEXT);
#endif
        }
        delete[] other_boolean_triple_a;
        delete[] other_boolean_triple_b;
    }
        
// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE

template <typename T>
static T im2col_get_pixel_l(const T* im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width)
        return SET_ALL_ZERO();
    return im[col + width * (row + height * channel)];
}


    template <typename T>
static void im2col_l(const T* data_im, int channels, int height, int width, int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c)
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h)
        {
            for (w = 0; w < width_col; ++w)
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_l(data_im, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }
}

    // GEMM_l(X_col, other_conv_triple_w, conv_triple_y[i], m, dout, k, true);
static void GEMM_l(const Datatype* A, const Datatype* B, Datatype* C, int m, int p, int f, bool a_fixed)
{
    const int TILE_SIZE = 64;
for (int i = 0; i < m; i += TILE_SIZE) {
    int i_max = std::min(i + TILE_SIZE, m);
    for (int j = 0; j < f; j += TILE_SIZE) {
        int j_max = std::min(j + TILE_SIZE, f);

        // Initialize tile of C to 0
        for (int ii = i; ii < i_max; ++ii) {
            for (int jj = j; jj < j_max; ++jj) {
                C[ii * f + jj] = SET_ALL_ZERO();
            }
        }

        // Tile over k dimension
        for (int k = 0; k < p; k += TILE_SIZE) {
            int k_max = std::min(k + TILE_SIZE, p);

            // Compute the product for the current tile (i..i_max, j..j_max) with k..k_max
            for (int ii = i; ii < i_max; ++ii) {
                for (int kk = k; kk < k_max; ++kk) {
                    Datatype a = A[ii * p + kk];
                    // Unroll the inner loop over j within the tile
                    for (int jj = j; jj < j_max; ++jj) {
                        Datatype b = B[kk * f + jj];
                        C[ii * f + jj] += a * b;
                    }
                }
            }
        }
    }

}
}

    static void get_conv_ab2_triples_from_file()
    {

        uint64_t curr_x_triple_index = 0;
        uint64_t y_index_counter = 0;
        for (uint64_t i = 0; i < conv_triple_params.size(); i++)
        {
    // const int m = out_h * out_w * batchSize;
    // const int k = wh * ww * din;
    // const int n = dout;
        const int din = conv_triple_params[i].din;
        const int inh = conv_triple_params[i].inh;
        const int inw = conv_triple_params[i].inw;
        const int wh = conv_triple_params[i].wh;
        const int ww = conv_triple_params[i].ww;
        const int stride = conv_triple_params[i].stride;
        const int padding = conv_triple_params[i].padding;
        const int batchSize = conv_triple_params[i].batchSize;
        const int dout = conv_triple_params[i].dout;

        // const int lm = conv_triple_params[i].dout;
        // const int lp = conv_triple_params[i].out_h * conv_triple_params[i].out_w;
        // const int lf = conv_triple_params[i].wh * conv_triple_params[i].wh * conv_triple_params[i].din;
        const int lm = conv_triple_params[i].dout;
        const int lp = wh * ww * din;
        const int lf = conv_triple_params[i].out_h * conv_triple_params[i].out_w * 1;
        const uint64_t w_size = wh * wh * din * dout;
        const uint64_t x_size = batchSize * din * inh * inw;
#if PARTY == 0
        uint64_t own_x_size = 0;
        uint64_t own_w_size = w_size;
        uint64_t other_x_size = x_size;
        uint64_t other_w_size = 0;
#else
        uint64_t own_w_size = 0;
        uint64_t own_x_size = x_size;
        uint64_t other_w_size = w_size;
        uint64_t other_x_size = 0;
#endif


        std::string file_ending = "pre_conv";
        file_ending += std::to_string(i);
        Datatype* nullp = new Datatype[1];
        uint64_t nulls = 1;
        save_triple_file(conv_triple_x[i], own_x_size, conv_triple_w[i], own_w_size, nullp, nulls, nullp, nulls, std::to_string(PARTY), file_ending);
        Datatype* other_conv_triple_x = new Datatype[other_x_size];
        Datatype* other_conv_triple_w = new Datatype[other_w_size];
        load_triple_file(other_conv_triple_x, other_x_size, other_conv_triple_w, other_w_size, nullp, nulls, nullp, nulls, std::to_string(1 - PARTY), file_ending);
        delete_triple_file(std::to_string(1 - PARTY), file_ending);
        delete[] nullp;
        #if PARTY == 1
        for (int n = 0; n < conv_triple_params[i].batchSize; n++)
        {
        auto X_col = new Datatype[lp * lf];
        int x_offset = n * din * inh * inw;
        int y_offset = n * lf * dout;
        im2col_l(conv_triple_x[i] + x_offset, din, inh, inw, wh, stride, padding, X_col);
        GEMM_l(other_conv_triple_w, X_col, conv_triple_y + y_index_counter + y_offset,
                lm, lp, lf, true);
        delete[] X_col;
        }
        #endif
        delete[] other_conv_triple_x;
        delete[] other_conv_triple_w;
        uint64_t y_size = batchSize * lf * dout;
        for (uint64_t j = 0; j < y_size; j++)
        {
#if PARTY == 1
            conv_triple_y[y_index_counter + j] = OP_SUB( conv_triple_y[y_index_counter+ j], getRandomVal(PNEXT));
#else
            conv_triple_y[y_index_counter + j] = getRandomVal(PNEXT);
#endif
        }

        y_index_counter += lf * dout * batchSize;
    }
}

static void get_batchnorm2D_triples_from_file()
    {
        uint64_t curr_x_triple_index = 0;
        uint64_t y_index_counter = 0;
        for (uint64_t i = 0; i < bc2D_triple_params.size(); i++)
        {
    // const int m = out_h * out_w * batchSize;
    // const int k = wh * ww * din;
    // const int n = dout;
        const int ch = bc2D_triple_params[i].ch;
        const int h = bc2D_triple_params[i].h;
        const int w = bc2D_triple_params[i].w;
        const int hw = h * w;
        const uint64_t w_size = bc2D_triple_params[i].w_size_per_batch;
        const uint64_t x_size = bc2D_triple_params[i].batchSize * bc2D_triple_params[i].x_size_per_batch;
        const uint64_t y_size = bc2D_triple_params[i].batchSize * bc2D_triple_params[i].y_size_per_batch;
#if PARTY == 0
        uint64_t own_x_size = 0;
        uint64_t own_w_size = w_size;
        uint64_t other_x_size = x_size;
        uint64_t other_w_size = 0;
#else
        uint64_t own_w_size = 0;
        uint64_t own_x_size = x_size;
        uint64_t other_w_size = w_size;
        uint64_t other_x_size = 0;
#endif

        std::string file_ending = "pre_batchnorm2D";
        file_ending += std::to_string(i);
        Datatype* nullp = new Datatype[1];
        uint64_t nulls = 1;
        save_triple_file(bc2D_triple_w[i], own_w_size, bc2D_triple_x[i], own_x_size, nullp, nulls, nullp, nulls, std::to_string(PARTY), file_ending);
        Datatype* other_bc2D_triple_x = new Datatype[other_x_size];
        Datatype* other_bc2D_triple_w = new Datatype[other_w_size];
        load_triple_file(other_bc2D_triple_w, other_w_size, other_bc2D_triple_x, other_x_size, nullp, nulls, nullp, nulls, std::to_string(1 - PARTY), file_ending);
        delete_triple_file(std::to_string(1 - PARTY), file_ending);
        delete[] nullp;
#if PARTY == 1
        for (int n = 0; n < bc2D_triple_params[i].batchSize; n++)
        {
            int x_offset = n * ch * hw;
            int y_offset = n * ch * hw;
            for (int c = 0; c < ch; c++)
            {
                for (int hw_idx = 0; hw_idx < hw; hw_idx++)
                {
                    bc2D_triple_y[y_index_counter + y_offset + c * hw + hw_idx] = OP_MULT(bc2D_triple_x[i][x_offset + c * hw + hw_idx], other_bc2D_triple_w[c]);
                }
            }
        }
#endif 
        delete[] other_bc2D_triple_w;
        delete[] other_bc2D_triple_x;
        for (uint64_t j = 0; j < y_size; j++)
        {
#if PARTY == 1
            bc2D_triple_y[j + y_index_counter] = OP_SUB( bc2D_triple_y[y_index_counter + j], getRandomVal(PNEXT));
#else
            bc2D_triple_y[j + y_index_counter] = getRandomVal(PNEXT);
#endif
        }

        y_index_counter += y_size;
    }
}

static void get_fc_triples_from_file()
    {
        uint64_t curr_x_triple_index = 0;
        uint64_t y_index_counter = 0;
        for (uint64_t i = 0; i < fc_triple_params.size(); i++)
        {
            const int in_feat = fc_triple_params[i].in_feat;
            const int out_feat = fc_triple_params[i].out_feat;
            const int batchSize = fc_triple_params[i].batchSize;
            const uint64_t x_size = fc_triple_params[i].x_size_per_batch * batchSize;
            const uint64_t w_size = fc_triple_params[i].w_size_per_batch;
            const uint64_t y_size = fc_triple_params[i].y_size_per_batch * batchSize;
#if PARTY == 0
        uint64_t own_x_size = 0;
        uint64_t own_w_size = w_size;
        uint64_t other_x_size = x_size;
        uint64_t other_w_size = 0;
#else
        uint64_t own_w_size = 0;
        uint64_t own_x_size = x_size;
        uint64_t other_w_size = w_size;
        uint64_t other_x_size = 0;
#endif
            std::string file_ending = "pre_fc";
            file_ending += std::to_string(i);
            Datatype* nullp = new Datatype[1];
            uint64_t nulls = 1;
            save_triple_file(fc_triple_w[i], own_w_size, fc_triple_x[i], own_x_size, nullp, nulls, nullp, nulls, std::to_string(PARTY), file_ending);
            Datatype* other_fc_triple_x = new Datatype[other_x_size];
            Datatype* other_fc_triple_w = new Datatype[other_w_size];
            load_triple_file(other_fc_triple_w, other_w_size, other_fc_triple_x, other_x_size, nullp, nulls, nullp, nulls, std::to_string(1 - PARTY), file_ending);
            delete_triple_file(std::to_string(1 - PARTY), file_ending);
            delete[] nullp;
#if PARTY == 1
            for (int n = 0; n < batchSize; n++)
            {
                int x_offset = n * in_feat;
                int y_offset = n * out_feat;
                for (int o = 0; o < out_feat; o++)
                {
                    fc_triple_y[y_index_counter + y_offset + o] = PROMOTE(0);
                    for (int in = 0; in < in_feat; in++)
                    {
                        fc_triple_y[y_index_counter + y_offset + o] = OP_ADD(fc_triple_y[y_index_counter + y_offset + o], OP_MULT(fc_triple_x[i][x_offset + in], other_fc_triple_w[o * in_feat + in]));
                    }
                }
            }
#endif
            delete[] other_fc_triple_x;
            delete[] other_fc_triple_w;

            for (uint64_t j = 0; j < y_size; j++)
            {
#if PARTY == 1
                fc_triple_y[y_index_counter + j] = OP_SUB( fc_triple_y[y_index_counter + j], getRandomVal(PNEXT));
#else
                fc_triple_y[y_index_counter + j] = getRandomVal(PNEXT);
#endif
            }

            y_index_counter += y_size;
        }
    }









    static void complete_preprocessing(std::string ips[], int port, int process_offset)
    {
#if LX_TRIPLES == 1
        init_beaverC(0);
#if FAKE_TRIPLES == 1
        get_triples_from_file(0, num_arithmetic_triples.data(), num_boolean_triples.data());
#else
        generate_beaver_triples(
                ips, port, process_offset, num_arithmetic_triples[0], num_boolean_triples[0], "LXLY");
#endif
        deinit_beaverAB();
        init_beaverAB(1);

        init_beaverAB2C(0);
#if FAKE_TRIPLES == 1
        get_ab2_triples_from_file(0, num_ab2_arithmetic_triples.data(), num_ab2_boolean_triples.data());
        generate_beaver_triples(
                ips, port, process_offset, num_ab2_arithmetic_triples[0], num_ab2_boolean_triples[0], "LXLY2");
#else
        generate_beaver_triples(
                ips, port, process_offset, num_ab2_arithmetic_triples[0], num_ab2_boolean_triples[0], "LXLY2");
#endif
        deinit_beaverAB2();
        init_beaverAB2(1);

        init_ConvC();
#if FAKE_TRIPLES == 1
        get_conv_ab2_triples_from_file();
        generate_beaver_triples(
                ips, port, process_offset, num_conv_c_triples, 0, "CONV");
#else
        generate_beaver_triples(
                ips, port, process_offset, num_conv_c_triples, 0, "CONV");
#endif
        deinit_ConvAB();
#endif

        init_BatchNorm2DC();
#if FAKE_TRIPLES == 1
        get_batchnorm2D_triples_from_file();
        generate_beaver_triples(
                ips, port, process_offset, num_bc2D_c_triples, 0, "BATCHNORM2D");
#else
        generate_beaver_triples(
                ips, port, process_offset, num_bc2D_c_triples, 0, "BATCHNORM2D");
#endif
        deinit_BatchNorm2DAB();

        init_FullyConnectedC();
#if FAKE_TRIPLES == 1
        get_fc_triples_from_file();
        generate_beaver_triples(
                ips, port, process_offset, num_fc_c_triples, 0, "FC");
#else
        generate_beaver_triples(
                ips, port, process_offset, num_fc_c_triples, 0, "FC");
#endif


        deinit_FullyConnectedAB();




        communicate_pre();
        //Trigger OTs, HE
        constexpr int num_rounds = 2;
        Datatype** lxly_a = new Datatype*[num_rounds];
        Datatype** lxly_b = new Datatype*[num_rounds];
        lxly_a[0] = new Datatype[total_num_arithmetic_output_triples[0]];
        lxly_b[0] = new Datatype[total_num_boolean_output_triples[0]];
        uint64_t arithmetic_triple_counter[num_rounds]{0};
        uint64_t boolean_triple_counter[num_rounds]{0};
        

        auto num_triples = total_num_arithmetic_output_triples[0] + total_num_boolean_output_triples[0] + total_preprocessed_outputs;
       
        curr_arithmetic_triple_index = 0;
        curr_boolean_triple_index = 0;
        curr_conv_triple_index = 0;
        curr_fc_triple_index = 0;
        curr_bc2D_triple_index = 0;
        arithmetic_triple_index = 0;
        boolean_triple_index = 0;

        // preprocessed_outputs_bool[0] = boolean_triple_c;
        // preprocessed_outputs_arithmetic[0] = arithmetic_triple_c;
        preprocessed_outputs_bool[1] = new Datatype[preprocessed_outputs_bool_input_index[1]];
        preprocessed_outputs_arithmetic[1] = new Datatype[preprocessed_outputs_arithmetic_input_index[1]];
        preprocessed_outputs_arithmetic_input_index[1] = 0;
        preprocessed_outputs_bool_input_index[1] = 0;

        for (uint64_t i = 0; i < num_triples; i++)
        {

            switch (triple_type[0][i])
            {
                case CaseAND:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_XOR );
                    lxly_b[0][boolean_triple_counter[0]++] = lxly;
                    break;
                }
                case CaseMult:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_ADD );
                    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
                    break;
                }
                case CaseBit2A:
                {
                    /* auto lta = pre_receive_from_live(PNEXT); */
                    /* auto ltb = pre_receive_from_live(PNEXT); */
                    /* auto ta = retrieve_output_share_arithmetic(); */
                    /* auto bl = retrieve_output_share_arithmetic(); */
                    /* auto prev_val = retrieve_output_share_arithmetic(); */
                    /* lxly_a[0][arithmetic_triple_counter[0]++] = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)),
                     * prev_val); */
                    auto lxly = receive_and_compute_lxly_share(
                        OP_ADD);  // preprocessing costs can be cut in half if triple of type x(P_0),y(P_1),[z] is used
                    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
                    break;
                }
                case CaseBitInjection:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_ADD);
                    lxly = OP_SUB(retrieve_output_share_arithmetic(helper_index), OP_ADD(lxly, lxly));
                    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;  // [lb] - 2[lb1lb2]

                    ABY2_PRE_Share al2 = retrieve_output_share_arithmetic(helper_index);       // [la]
                    al2.generate_lxly_triple(lxly, OP_ADD, 1);  // [la] [lb]
                    break;
                }
                case CaseDot3Bool:
                {
                    auto third = retrieve_output_share_bool(helper_index);
                    auto lxly = receive_and_compute_lxly_share(OP_XOR);
                    lxly_b[0][boolean_triple_counter[0]++] = lxly;
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(third, OP_XOR, 1);
                    break;
                }
                case CaseDot3Arithmetic: 
                {
                    auto third = retrieve_output_share_arithmetic(helper_index);
                    auto lxly = receive_and_compute_lxly_share(OP_ADD);
                    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(third, OP_ADD, 1);
                    break;
                }
                case CaseDot4Bool:
                {
                    auto x = retrieve_output_share_bool(helper_index);
                    auto y = retrieve_output_share_bool(helper_index);
                    auto z = retrieve_output_share_bool(helper_index);
                    auto w = retrieve_output_share_bool(helper_index);
                    auto lxly = receive_and_compute_lxly_share(OP_XOR);
                    auto lzlw = receive_and_compute_lxly_share(OP_XOR);
                    lxly_b[0][boolean_triple_counter[0]++] = lxly;
                    lxly_b[0][boolean_triple_counter[0]++] = lzlw;
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(z, OP_XOR, 1);
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(w, OP_XOR, 1);
                    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(x, OP_XOR, 1);
                    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(y, OP_XOR, 1);
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(lzlw, OP_XOR, 1);
                    break;
                }
                case CaseDot4Arithmetic:
                {
                    auto x = retrieve_output_share_arithmetic(helper_index);
                    auto y = retrieve_output_share_arithmetic(helper_index);
                    auto z = retrieve_output_share_arithmetic(helper_index);
                    auto w = retrieve_output_share_arithmetic(helper_index);
                    auto lxly = receive_and_compute_lxly_share(OP_ADD);
                    auto lzlw = receive_and_compute_lxly_share(OP_ADD);
                    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
                    lxly_a[0][arithmetic_triple_counter[0]++] = lzlw;
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(z, OP_ADD, 1);
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(w, OP_ADD, 1);
                    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(x, OP_ADD, 1);
                    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(y, OP_ADD, 1);
                    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(lzlw, OP_ADD, 1);
                    break;
                }
                case CaseMatMulFirstDot:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_ADD);
                    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
                    break;
                }
                case CaseMatMul:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_ADD);
                    lxly_a[0][arithmetic_triple_counter[0] - 1] =
                        OP_ADD(lxly_a[0][arithmetic_triple_counter[0] - 1], lxly);
                    break;
                }
                case CaseANDAKnown:
                {
                    auto lxly2 = receive_and_compute_lxly2_share(OP_XOR);
                    lxly_b[0][boolean_triple_counter[0]++] = lxly2;
                    break;
                }
                case CaseMultAKnown:
                {
                    auto lxly2 = receive_and_compute_lxly2_share(OP_ADD);
                    lxly_a[0][arithmetic_triple_counter[0]++] = lxly2;
                    break;
                }
                case CaseConv:
                {
                    lxly_a[0][arithmetic_triple_counter[0]++] = conv_triple_y[curr_conv_triple_index++];
                    break;
                }
                case CaseBatchNorm2D:
                {
                    lxly_a[0][arithmetic_triple_counter[0]++] = bc2D_triple_y[curr_bc2D_triple_index++];
                    break;
                }
                case CaseFullyConnected:
                {
                    lxly_a[0][arithmetic_triple_counter[0]++] = fc_triple_y[curr_fc_triple_index++];
                    break;
                }
                case CaseTripleAlreadyConsumed:  // Triple already consumed by previous case
                {
                    break;
                }
                default: // e.g.  Public Fixed point multiplication
                {
                    auto l = pre_receive_from_live(PNEXT);
                    store_output_share(l);
                    break;
                }
            }
        }
        arithmetic_triple_index = 0;
        boolean_triple_index = 0;
        delete[] triple_type[0];
        delete[] preprocessed_outputs_bool[0];
        preprocessed_outputs_bool[0] = lxly_b[0];
        /* preprocessed_outputs_bool_index[0] = 0; */
        // preprocessed_outputs_bool_input_index[0] = 0;

        delete[] preprocessed_outputs_arithmetic[0];
        preprocessed_outputs_arithmetic[0] = lxly_a[0];
        /* preprocessed_outputs_arithmetic_index[0] = 0; */
        preprocessed_outputs_arithmetic_input_index[0] = 0;

        /* preprocessed_outputs_bool_index[1] = 0; */
        preprocessed_outputs_bool_input_index[1] = 0;

        /* preprocessed_outputs_arithmetic_index[1] = 0; */
        preprocessed_outputs_arithmetic_input_index[1] = 0;


        communicate_pre();
#if LX_TRIPLES == 1 
        deinit_beaverC();
        deinit_beaverAB2C();
        deinit_ConvC();
        deinit_BatchNorm2DC();
        deinit_FullyConnectedC();
        init_beaverC(1);
#if FAKE_TRIPLES == 1
        get_triples_from_file(1, num_arithmetic_triples.data(), num_boolean_triples.data());
        init_beaverAB2C(1);
        deinit_beaverAB();
        get_ab2_triples_from_file(1, num_ab2_arithmetic_triples.data(), num_ab2_boolean_triples.data());
#else
        generate_beaver_triples(
             ips, port, process_offset, num_arithmetic_triples[1], num_boolean_triples[1], "LXLY");
        init_beaverAB2C(1);
        deinit_beaverAB();
        generate_beaver_triples(
                ips, port, process_offset, num_ab2_arithmetic_triples[1], num_ab2_boolean_triples[1], "LXLY2");
#endif
        deinit_beaverAB2();
#endif

        lxly_a[1] = new Datatype[total_num_arithmetic_output_triples[1]];
        lxly_b[1] = new Datatype[total_num_boolean_output_triples[1]];
        curr_arithmetic_triple_index = 0;
        curr_boolean_triple_index = 0;
        num_triples = total_num_arithmetic_output_triples[1] + total_num_boolean_output_triples[1];
        for (uint64_t i = 0; i < num_triples; i++)
        {
            switch (triple_type[1][i])
            {
                case CaseBitInjection:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_ADD, 1);
                    lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
                    break;
                }
                case CaseDot3Bool:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_XOR, 1);
                    lxly_b[1][boolean_triple_counter[1]++] = lxly;
                    break;
                }
                case CaseDot3Arithmetic:
                {
                    auto lxly = receive_and_compute_lxly_share(OP_ADD, 1);
                    lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
                    break;
                }
                case CaseDot4Bool:
                {
                    auto lxly_lz = receive_and_compute_lxly_share(OP_XOR, 1);
                    auto lxly_lw = receive_and_compute_lxly_share(OP_XOR, 1);
                    auto lzlw_lx = receive_and_compute_lxly_share(OP_XOR, 1);
                    auto lzlw_ly = receive_and_compute_lxly_share(OP_XOR, 1);
                    auto lxly_lzlw = receive_and_compute_lxly_share(OP_XOR, 1);
                    lxly_b[1][boolean_triple_counter[1]++] = lxly_lz;
                    lxly_b[1][boolean_triple_counter[1]++] = lxly_lw;
                    lxly_b[1][boolean_triple_counter[1]++] = lzlw_lx;
                    lxly_b[1][boolean_triple_counter[1]++] = lzlw_ly;
                    lxly_b[1][boolean_triple_counter[1]++] = lxly_lzlw;
                    break;
                }
                case CaseDot4Arithmetic:
                {
                    auto lxly_lz = receive_and_compute_lxly_share(OP_ADD, 1);
                    auto lxly_lw = receive_and_compute_lxly_share(OP_ADD, 1);
                    auto lzlw_lx = receive_and_compute_lxly_share(OP_ADD, 1);
                    auto lzlw_ly = receive_and_compute_lxly_share(OP_ADD, 1);
                    auto lxly_lzlw = receive_and_compute_lxly_share(OP_ADD, 1);
                    lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lz;
                    lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lw;
                    lxly_a[1][arithmetic_triple_counter[1]++] = lzlw_lx;
                    lxly_a[1][arithmetic_triple_counter[1]++] = lzlw_ly;
                    lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lzlw;
                    break;
                }
            }
        }
        delete[] triple_type[1];
        delete[] preprocessed_outputs_bool[1];
        preprocessed_outputs_bool[1] = lxly_b[1];
        /* preprocessed_outputs_bool_index[1] = 0; */
        preprocessed_outputs_bool_input_index[1] = 0;

        delete[] preprocessed_outputs_arithmetic[1];
        preprocessed_outputs_arithmetic[1] = lxly_a[1];
        /* preprocessed_outputs_arithmetic_index[1] = 0; */
        preprocessed_outputs_arithmetic_input_index[1] = 0;

        preprocessed_outputs_bool_index[1] = 0;
        preprocessed_outputs_arithmetic_index[1] = 0;

        delete[] lxly_a;
        delete[] lxly_b;
        deinit_beaverC();
        deinit_beaverAB2C();
        init_srngs();
    }

    // --- Untested Functions --- TODO: Test

    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    ABY2_PRE_Share prepare_trunc_share(func_mul MULT,
                                       func_add ADD,
                                       func_sub SUB,
                                       func_trunc TRUNC,
                                       int fractional_bits = FRACTIONAL) const
    {
        return ABY2_PRE_Share(getRandomVal(PSELF));
    }

    void get_random_B2A() { l = getRandomVal(PSELF); }


    static void SetupConv2dTriples(const ABY2_PRE_Share* X,
                                   const ABY2_PRE_Share* W,
                                   ABY2_PRE_Share* Y,
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
                                   bool ab2 = true)
    {
#if PARTY == 0 || AB2_TRIPLES == 0 // Party0 holds W in plain in AB2 setting
        conv_triple_w[curr_conv_triple_index] = new Datatype[wh * ww * din * dout];
        for (int i = 0; i < wh * ww * din * dout; i++)
            conv_triple_w[curr_conv_triple_index][i] = W[i].l;
#endif

#if PARTY == 1 || AB2_TRIPLES == 0 // Party0 does not need X triples in AB2 setting
        conv_triple_x[curr_conv_triple_index] = new Datatype[batchSize * inh * inw * din];
        for (int i = 0; i < batchSize * inh * inw * din; i++)
            conv_triple_x[curr_conv_triple_index][i] = X[i].l;
#endif

        uint64_t num_conv_triples = conv_triple_params[curr_conv_triple_index].out_h * conv_triple_params[curr_conv_triple_index].out_w * batchSize * dout;
        for(uint64_t i = 0; i < num_conv_triples; i++)
            triple_type[0][triple_type_index[0]++] = CaseConv;
        curr_conv_triple_index++;
    }

    static void SetupFullyConnectedTriples(const ABY2_PRE_Share* X,
                                   const ABY2_PRE_Share* W,
                                   ABY2_PRE_Share* Y,
                                   int batchSize,
                                   int in_feat,
                                   int out_feat,
                                   bool ab2 = true)
    {
        const uint64_t w_size = in_feat * out_feat;
        const uint64_t y_size = out_feat * batchSize;
        const uint64_t x_size = in_feat * batchSize;

#if PARTY == 0 || AB2_TRIPLES == 0 // Party0 holds W in plain in AB2 setting
        fc_triple_w[curr_fc_triple_index] = new Datatype[w_size];
        for (int i = 0; i < w_size; i++)
            fc_triple_w[curr_fc_triple_index][i] = W[i].l;
#endif

#if PARTY == 1 || AB2_TRIPLES == 0 // Party0 does not need X triples in AB2 setting
        fc_triple_x[curr_fc_triple_index] = new Datatype[x_size];
        for (int i = 0; i < x_size; i++)
            fc_triple_x[curr_fc_triple_index][i] = X[i].l;
#endif


        for(uint64_t i = 0; i < y_size; i++)
            triple_type[0][triple_type_index[0]++] = CaseFullyConnected;
        curr_fc_triple_index++;
    }
    
    static void SetupBatchNorm2DTriples(const ABY2_PRE_Share* X,
                                   const ABY2_PRE_Share* W,
                                   ABY2_PRE_Share* Y,
                                   int batchSize,
                                   int ch,
                                   int h,
                                   int w,
                                   bool ab2 = true)
    {
        const uint64_t w_size = ch;
        const uint64_t x_size = ch * h * w * batchSize;
        const uint64_t y_size = ch * h * w * batchSize;
#if PARTY == 0 || AB2_TRIPLES == 0 // Party0 holds W in plain in AB2 setting
        bc2D_triple_w[curr_bc2D_triple_index] = new Datatype[w_size];
        for (int i = 0; i < w_size; i++)
            bc2D_triple_w[curr_bc2D_triple_index][i] = W[i].l;
#endif

#if PARTY == 1 || AB2_TRIPLES == 0 // Party0 does not need X triples in AB2 setting
        bc2D_triple_x[curr_bc2D_triple_index] = new Datatype[x_size];
        for (int i = 0; i < x_size; i++)
            bc2D_triple_x[curr_bc2D_triple_index][i] = X[i].l;
#endif

        for(uint64_t i = 0; i < y_size; i++)
            triple_type[0][triple_type_index[0]++] = CaseBatchNorm2D;
        curr_bc2D_triple_index++;
    }

#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1

    static void GEMM(ABY2_PRE_Share* a, ABY2_PRE_Share* b, ABY2_PRE_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        if (a_fixed == true)
        {
            const int factor = DATTYPE / BITLENGTH;
            if (factor > 1)
                for (int i = 0; i < m * k; i++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    unorthogonalize_arithmetic(&a[i].l, temp, 1);
                    a[i].l = PROMOTE(temp[0]);
                }
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int l = 0; l < k; l++)
                {
                    a[i * k + l].generate_lxly_triple(b[l * n + j], OP_ADD);
                    triple_type[0][triple_type_index[0]++] = CaseMatMul;
                }
                triple_type[0][triple_type_index[0] - k] = CaseMatMulFirstDot;  
            }
        }
    }

#else

    static void GEMM(ABY2_PRE_Share* a, ABY2_PRE_Share* b, ABY2_PRE_Share* c, int m, int n, int k, bool a_fixed = false)
    {
        if (a_fixed == true)
        {
            const int factor = DATTYPE / BITLENGTH;
            if (factor > 1)
                for (int i = 0; i < m * k; i++)
                {
                    alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
                    unorthogonalize_arithmetic(&a[i].l, temp, 1);
                    a[i].l = PROMOTE(temp[0]);
                }
        }
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int l = 0; l < k; l++)
                {
                    a[i * k + l].generate_lxly_triple(b[l * n + j], OP_ADD);
                    triple_type[0][triple_type_index[0]++] = CaseMatMul;
                }
                triple_type[0][triple_type_index[0] - k] = CaseMatMulFirstDot;
            }
        }
    }

#endif
#endif
};

#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4


/* struct CONV2D_args */
/* { */
/*     int batchSize; */
/*     int inh; */
/*     int inw; */
/*     int din; */
/*     int dout; */
/*     int wh; */
/*     int ww; */
/*     int padding; */
/*     int stride; */
/*     int dilation; */
/*     int m; */
/*     int n; */
/*     int k; */
/* }; */

/* std::queue<CONV2D_args> CONV2D_args_queue; */

/* static void COMPLETE_CONV_2D(const ABY2_PRE_Share* X, const ABY2_PRE_Share* W, ABY2_PRE_Share* Y, int batchSize, int
 * inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation = 1) */
/* { */
/* } */

template <typename Datatype>
static void ABY2_PRE_Share<Datatype>::CONV_2D(const ABY2_PRE_Share* X,
                                              const ABY2_PRE_Share* W,
                                              ABY2_PRE_Share* Y,
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
    const int factor = DATTYPE / BITLENGTH;
    const int xSize = inh * inw * din * batchSize;
    const int wSize = wh * ww * din * dout;
    const int ySize = out_h * out_w * dout * batchSize;
    const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
    const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;

    const int m = out_h * out_w * batchSize;
    const int k = wh * ww * din;
    const int n = dout;
    batchSize *= factor;

    X_col = new Datatype[k * m];
    im2col_l(X, din, inh, inw, wh, stride, padding, X_col);
    ABY2_PRE_Share<T>::GEMM(X_col, W, Y, m, n, k, true);
    delete[] X_col;
}
#endif


#if LX_TRIPLES == 0
    template <typename func_add, typename func_sub, typename func_mul>
    void generate_lxly_from_triple_comp_opt(ABY2_PRE_Share b,
                                            func_add ADD,
                                            func_sub SUB,
                                            func_mul MULT,
                                            int num_round = 0) const
    {
        BT t;
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            t = retrieveBooleanTriple<Datatype>();
        }
        else
        {
            t = retrieveArithmeticTriple<Datatype>();
        }
        auto lta = ADD(l, t.a);
        auto ltb = ADD(b.l, t.b);
        pre_send_to_live(PNEXT, lta);
        auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
        store_output_share_ab(lta, ADD, num_round);
        store_output_share_ab(ltb, ADD, num_round);
        store_output_share_ab(b.l, ADD, num_round);
        store_output_share_ab(t.a, ADD, num_round);
        store_output_share_ab(t.c, ADD, num_round);
    }

    template <typename func_add, typename func_sub, typename func_mul>
    static Datatype receive_and_compute_lxly_share_comp_opt(func_add ADD,
                                                            func_sub SUB,
                                                            func_mul MULT,
                                                            int num_round = 0)
    {
        auto lta = OP_ADD(pre_receive_from_live(PNEXT), retrieve_output_share_ab(ADD, num_round));
        auto ltb = OP_ADD(pre_receive_from_live(PNEXT), retrieve_output_share_ab(ADD, num_round));
        auto ta = retrieve_output_share_ab(ADD, num_round);
        auto bl = retrieve_output_share_ab(ADD, num_round);
        auto tc = retrieve_output_share_ab(ADD, num_round);
        return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), tc);
    }
    
    template <typename func_add, typename func_sub, typename func_mul>
    void generate_lxly_from_triple(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT, int num_round = 0) const
    {
        BT t;
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            t = retrieveBooleanTriple<Datatype>();
        }
        else
        {
            t = retrieveArithmeticTriple<Datatype>();
        }
        auto lta = ADD(l, t.a);
        auto ltb = ADD(b.l, t.b);
        pre_send_to_live(PNEXT, lta);
        pre_send_to_live(PNEXT, ltb);
        auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            store_output_share_bool(t.a, num_round);
            store_output_share_bool(b.l, num_round);
            store_output_share_bool(lxly, num_round);
        }
        else
        {
            store_output_share_arithmetic(t.a, num_round);
            store_output_share_arithmetic(b.l, num_round);
            store_output_share_arithmetic(lxly, num_round);
        }
    }
    

    template <typename func_add, typename func_sub, typename func_mul>
    static Datatype receive_and_compute_lxly_share(func_add ADD, func_sub SUB, func_mul MULT, int num_round = 0)
    {
        auto lta = pre_receive_from_live(PNEXT);
        auto ltb = pre_receive_from_live(PNEXT);
        if constexpr (std::is_same_v<func_add(), OP_XOR>)
        {
            auto ta = retrieve_output_share_bool(num_round);
            auto bl = retrieve_output_share_bool(num_round);
            auto prev_val = retrieve_output_share_bool(num_round);
            return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), prev_val);
        }
        else
        {
            auto ta = retrieve_output_share_arithmetic(num_round);
            auto bl = retrieve_output_share_arithmetic(num_round);
            auto prev_val = retrieve_output_share_arithmetic(num_round);
            return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), prev_val);
        }
    }

    //static void complete_preprocessing(uint64_t* arithmetic_triple_num,
    //                                   uint64_t* boolean_triple_num,
    //                                   uint64_t num_output_shares)
    //{
    //    communicate_pre();
    //    //Trigger OTs, HE
    //    constexpr int num_rounds = 2;
    //    Datatype** lxly_a = new Datatype*[num_rounds];
    //    Datatype** lxly_b = new Datatype*[num_rounds];
    //    lxly_a[0] = new Datatype[arithmetic_triple_num[0]];
    //    lxly_b[0] = new Datatype[boolean_triple_num[0]];
    //    uint64_t arithmetic_triple_counter[num_rounds]{0};
    //    uint64_t boolean_triple_counter[num_rounds]{0};
    //    auto num_triples = arithmetic_triple_num[0] + boolean_triple_num[0] + num_output_shares;
    //    preprocessed_outputs_bool[1] = new Datatype[preprocessed_outputs_bool_input_index[1]];
    //    preprocessed_outputs_arithmetic[1] = new Datatype[preprocessed_outputs_arithmetic_input_index[1]];
    //    preprocessed_outputs_arithmetic_input_index[1] = 0;
    //    preprocessed_outputs_bool_input_index[1] = 0;
//#if LX_TRIPLES == 1
    //    Generate_triples(0);
//#endif

    //    for (uint64_t i = 0; i < num_triples; i++)
    //    {


    //        switch (triple_type[0][i])
    //        {
    //            case CaseAND:
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND);
    //                lxly_b[0][boolean_triple_counter[0]++] = lxly;
    //                break;
    //            }
    //            case CaseMult:
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT);
    //                lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    //                break;
    //            }
    //            case CaseBit2A:
    //            {
    //                /* auto lta = pre_receive_from_live(PNEXT); */
    //                /* auto ltb = pre_receive_from_live(PNEXT); */
    //                /* auto ta = retrieve_output_share_arithmetic(); */
    //                /* auto bl = retrieve_output_share_arithmetic(); */
    //                /* auto prev_val = retrieve_output_share_arithmetic(); */
    //                /* lxly_a[0][arithmetic_triple_counter[0]++] = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)),
    //                 * prev_val); */
    //                auto lxly = receive_and_compute_lxly_share(
    //                    OP_ADD,
    //                    OP_SUB,
    //                    OP_MULT);  // preprocessing costs can be cut in half if triple of type x(P_0),y(P_1),[z] is used
    //                lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    //                break;
    //            }
    //            case CaseBitInjection:
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT);
    //                lxly = OP_SUB(retrieve_output_share_arithmetic(helper_index), OP_ADD(lxly, lxly));
    //                lxly_a[0][arithmetic_triple_counter[0]++] = lxly;  // [lb] - 2[lb1lb2]

    //                ABY2_PRE_Share al2 = retrieve_output_share_arithmetic(helper_index);       // [la]
    //                al2.generate_lxly_triple(lxly, OP_ADD, OP_SUB, OP_MULT, 1);  // [la] [lb]
    //                break;
    //            }
    //            case CaseDot3Bool:
    //            {
    //                auto third = retrieve_output_share_bool(helper_index);
    //                auto lxly = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND);
    //                lxly_b[0][boolean_triple_counter[0]++] = lxly;
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(third, OP_XOR, OP_XOR, OP_AND, 1);
    //                break;
    //            }
    //            case CaseDot3Arithmetic: 
    //            {
    //                auto third = retrieve_output_share_arithmetic(helper_index);
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT);
    //                lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(third, OP_ADD, OP_SUB, OP_MULT, 1);
    //                break;
    //            }
    //            case CaseDot4Bool:
    //            {
    //                auto x = retrieve_output_share_bool(helper_index);
    //                auto y = retrieve_output_share_bool(helper_index);
    //                auto z = retrieve_output_share_bool(helper_index);
    //                auto w = retrieve_output_share_bool(helper_index);
    //                auto lxly = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND);
    //                auto lzlw = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND);
    //                lxly_b[0][boolean_triple_counter[0]++] = lxly;
    //                lxly_b[0][boolean_triple_counter[0]++] = lzlw;
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(z, OP_XOR, OP_XOR, OP_AND, 1);
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(w, OP_XOR, OP_XOR, OP_AND, 1);
    //                ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(x, OP_XOR, OP_XOR, OP_AND, 1);
    //                ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(y, OP_XOR, OP_XOR, OP_AND, 1);
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(lzlw, OP_XOR, OP_XOR, OP_AND, 1);
    //                break;
    //            }
    //            case Dot4Arithmetic:
    //            {
    //                auto x = retrieve_output_share_arithmetic(helper_index);
    //                auto y = retrieve_output_share_arithmetic(helper_index);
    //                auto z = retrieve_output_share_arithmetic(helper_index);
    //                auto w = retrieve_output_share_arithmetic(helper_index);
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT);
    //                auto lzlw = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT);
    //                lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    //                lxly_a[0][arithmetic_triple_counter[0]++] = lzlw;
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(z, OP_ADD, OP_SUB, OP_MULT, 1);
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(w, OP_ADD, OP_SUB, OP_MULT, 1);
    //                ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(x, OP_ADD, OP_SUB, OP_MULT, 1);
    //                ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_triple(y, OP_ADD, OP_SUB, OP_MULT, 1);
    //                ABY2_PRE_Share<Datatype>(lxly).generate_lxly_triple(lzlw, OP_ADD, OP_SUB, OP_MULT, 1);
    //                break;
    //            }
    //            case CaseMatMulFirstDot:
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT);
    //                lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    //                break;
    //            }
    //            case CaseMatMul:
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT);
    //                lxly_a[0][arithmetic_triple_counter[0] - 1] =
    //                    OP_ADD(lxly_a[0][arithmetic_triple_counter[0] - 1], lxly);
    //                break;
    //            }
    //            case CaseTripleAlreadyConsumed:  // Triple already consumed by previous case
    //            {
    //                break;
    //            }
    //            default: // e.g.  Public Fixed point multiplication
    //            {
    //                auto l = pre_receive_from_live(PNEXT);
    //                store_output_share(l, helper_index);
    //                break;
    //            }
    //        }
    //    }
    //    delete[] triple_type[0];
    //    delete[] preprocessed_outputs_bool[0];
    //    preprocessed_outputs_bool[0] = lxly_b[0];
    //    /* preprocessed_outputs_bool_index[0] = 0; */
    //    preprocessed_outputs_bool_input_index[0] = 0;

    //    delete[] preprocessed_outputs_arithmetic[0];
    //    preprocessed_outputs_arithmetic[0] = lxly_a[0];
    //    /* preprocessed_outputs_arithmetic_index[0] = 0; */
    //    preprocessed_outputs_arithmetic_input_index[0] = 0;

    //    /* preprocessed_outputs_bool_index[1] = 0; */
    //    preprocessed_outputs_bool_input_index[1] = 0;

    //    /* preprocessed_outputs_arithmetic_index[1] = 0; */
    //    preprocessed_outputs_arithmetic_input_index[1] = 0;


    //    deinit_beaver();
    //    communicate_pre();
    //    //Trigger OTs, HE
    //    GenerateTripples(1);
//#if LX_TRIPLES == 0
    //    lxly_a[1] = new Datatype[arithmetic_triple_num[1]];
    //    lxly_b[1] = new Datatype[boolean_triple_num[1]];
//#endif
    //    num_triples = arithmetic_triple_num[1] + boolean_triple_num[1];
    //    for (uint64_t i = 0; i < num_triples; i++)
    //    {
    //        switch (triple_type[1][i])
    //        {
    //            case CaseBitInjection;
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT, 1);
    //                lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
    //                break;
    //            }
    //            case CaseDot3Bool:
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND, 1);
    //                lxly_b[1][boolean_triple_counter[1]++] = lxly;
    //                break;
    //            }
    //            case CaseDot3Arithmetic:
    //            {
    //                auto lxly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT, 1);
    //                lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
    //                break;
    //            }
    //            case CaseDot4Bool:
    //            {
    //                auto lxly_lz = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND, 1);
    //                auto lxly_lw = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND, 1);
    //                auto lzlw_lx = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND, 1);
    //                auto lzlw_ly = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND, 1);
    //                auto lxly_lzlw = receive_and_compute_lxly_share(OP_XOR, OP_XOR, OP_AND, 1);
    //                lxly_b[1][boolean_triple_counter[1]++] = lxly_lz;
    //                lxly_b[1][boolean_triple_counter[1]++] = lxly_lw;
    //                lxly_b[1][boolean_triple_counter[1]++] = lzlw_lx;
    //                lxly_b[1][boolean_triple_counter[1]++] = lzlw_ly;
    //                lxly_b[1][boolean_triple_counter[1]++] = lxly_lzlw;
    //                break;
    //            }
    //            case CaseDot4Arithmetic:
    //            {
    //                auto lxly_lz = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT, 1);
    //                auto lxly_lw = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT, 1);
    //                auto lzlw_lx = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT, 1);
    //                auto lzlw_ly = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT, 1);
    //                auto lxly_lzlw = receive_and_compute_lxly_share(OP_ADD, OP_SUB, OP_MULT, 1);
    //                lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lz;
    //                lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lw;
    //                lxly_a[1][arithmetic_triple_counter[1]++] = lzlw_lx;
    //                lxly_a[1][arithmetic_triple_counter[1]++] = lzlw_ly;
    //                lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lzlw;
    //                break;
    //            }
    //        }
    //    }
    //    delete[] triple_type[1];
    //    delete[] preprocessed_outputs_bool[1];
    //    preprocessed_outputs_bool[1] = lxly_b[1];
    //    /* preprocessed_outputs_bool_index[1] = 0; */
    //    preprocessed_outputs_bool_input_index[1] = 0;

    //    delete[] preprocessed_outputs_arithmetic[1];
    //    preprocessed_outputs_arithmetic[1] = lxly_a[1];
    //    /* preprocessed_outputs_arithmetic_index[1] = 0; */
    //    preprocessed_outputs_arithmetic_input_index[1] = 0;

    //    preprocessed_outputs_bool_index[1] = 0;
    //    preprocessed_outputs_arithmetic_index[1] = 0;

    //    delete[] lxly_a;
    //    delete[] lxly_b;
    //    init_srngs();
    //}

    //// --- Untested Functions --- TODO: Test

    //template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
    //ABY2_PRE_Share prepare_trunc_share(func_mul MULT,
    //                                   func_add ADD,
    //                                   func_sub SUB,
    //                                   func_trunc TRUNC,
    //                                   int fractional_bits = FRACTIONAL) const
    //{
    //    return ABY2_PRE_Share(getRandomVal(PSELF));
    //}

    //void get_random_B2A() { l = getRandomVal(PSELF); }

//#if USE_CUDA_GEMM > 0
//#if USE_CUDA_GEMM == 1

    //static void GEMM(ABY2_PRE_Share* a, ABY2_PRE_Share* b, ABY2_PRE_Share* c, int m, int n, int k, bool a_fixed = false)
    //{
    //    if (a_fixed == true)
    //    {
    //        const int factor = DATTYPE / BITLENGTH;
    //        if (factor > 1)
    //            for (int i = 0; i < m * k; i++)
    //            {
    //                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
    //                unorthogonalize_arithmetic(&a[i].l, temp, 1);
    //                a[i].l = PROMOTE(temp[0]);
    //            }
    //    }
    //    for (int i = 0; i < m; i++)
    //    {
    //        for (int j = 0; j < n; j++)
    //        {
    //            for (int l = 0; l < k; l++)
    //            {
    //                a[i * k + l].generate_lxly_triple(b[l * n + j], OP_ADD, OP_SUB, OP_MULT);
    //                triple_type[0][triple_type_index[0]++] = CaseMatMul;
    //            }
    //            triple_type[0][triple_type_index[0] - k] = CaseMatMulFirstDot;  
    //        }
    //    }
    //}

//#else

    //static void GEMM(ABY2_PRE_Share* a, ABY2_PRE_Share* b, ABY2_PRE_Share* c, int m, int n, int k, bool a_fixed = false)
    //{
    //    if (a_fixed == true)
    //    {
    //        const int factor = DATTYPE / BITLENGTH;
    //        if (factor > 1)
    //            for (int i = 0; i < m * k; i++)
    //            {
    //                alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
    //                unorthogonalize_arithmetic(&a[i].l, temp, 1);
    //                a[i].l = PROMOTE(temp[0]);
    //            }
    //    }
    //    for (int i = 0; i < m; i++)
    //    {
    //        for (int j = 0; j < n; j++)
    //        {
    //            for (int l = 0; l < k; l++)
    //            {
    //                a[i * k + l].generate_lxly_triple(b[l * n + j], OP_ADD, OP_SUB, OP_MULT);
    //                triple_type[0][triple_type_index[0]++] = CaseMatMul;
    //            }
    //            triple_type[0][triple_type_index[0] - k] = CaseMatMulFirstDot;
    //        }
    //    }
    //}

//#endif
//#endif
//};

//#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
//template <typename T>
//T im2col_get_pixel_l(const T* im, int height, int width, int channels, int row, int col, int channel, int pad)
//{
    //row -= pad;
    //col -= pad;

    //if (row < 0 || col < 0 || row >= height || col >= width)
    //    return 0;
    //return im[col + width * (row + height * channel)];
//}

//// From Berkeley Vision's Caffe!
//// https://github.com/BVLC/caffe/blob/master/LICENSE
//template <typename T>
//void im2col_l(const T* data_im, int channels, int height, int width, int ksize, int stride, int pad, T* data_col)
//{
    //int c, h, w;
    //int height_col = (height + 2 * pad - ksize) / stride + 1;
    //int width_col = (width + 2 * pad - ksize) / stride + 1;

    //int channels_col = channels * ksize * ksize;
    //for (c = 0; c < channels_col; ++c)
    //{
    //    int w_offset = c % ksize;
    //    int h_offset = (c / ksize) % ksize;
    //    int c_im = c / ksize / ksize;
    //    for (h = 0; h < height_col; ++h)
    //    {
    //        for (w = 0; w < width_col; ++w)
    //        {
    //            int im_row = h_offset + h * stride;
    //            int im_col = w_offset + w * stride;
    //            int col_index = (c * height_col + h) * width_col + w;
    //            data_col[col_index] = im2col_get_pixel_l(data_im, height, width, channels, im_row, im_col, c_im, pad);
    //        }
    //    }
    //}
//}

///* struct CONV2D_args */
///* { */
///*     int batchSize; */
///*     int inh; */
///*     int inw; */
///*     int din; */
///*     int dout; */
///*     int wh; */
///*     int ww; */
///*     int padding; */
///*     int stride; */
///*     int dilation; */
///*     int m; */
///*     int n; */
///*     int k; */
///* }; */

///* std::queue<CONV2D_args> CONV2D_args_queue; */

///* static void COMPLETE_CONV_2D(const ABY2_PRE_Share* X, const ABY2_PRE_Share* W, ABY2_PRE_Share* Y, int batchSize, int
 //* inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation = 1) */
///* { */
///* } */

//template <typename Datatype>
//static void ABY2_PRE_Share<Datatype>::CONV_2D(const ABY2_PRE_Share* X,
    //                                          const ABY2_PRE_Share* W,
    //                                          ABY2_PRE_Share* Y,
    //                                          int batchSize,
    //                                          int inh,
    //                                          int inw,
    //                                          int din,
    //                                          int dout,
    //                                          int wh,
    //                                          int ww,
    //                                          int padding,
    //                                          int stride,
    //                                          int dilation = 1)
//{
    //const int factor = DATTYPE / BITLENGTH;
    //const int xSize = inh * inw * din * batchSize;
    //const int wSize = wh * ww * din * dout;
    //const int ySize = out_h * out_w * dout * batchSize;
    //const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
    //const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;

    //const int m = out_h * out_w * batchSize;
    //const int k = wh * ww * din;
    //const int n = dout;
    //batchSize *= factor;

    //X_col = new Datatype[k * m];
    //im2col_l(X, din, inh, inw, wh, stride, padding, X_col);
    //ABY2_PRE_Share<T>::GEMM(X_col, W, Y, m, n, k, true);
    //delete[] X_col;
//}
//#endif


//#if LX_TRIPLES == 0
    //template <typename func_add, typename func_sub, typename func_mul>
    //void generate_lxly_from_triple_comp_opt(ABY2_PRE_Share b,
    //                                        func_add ADD,
    //                                        func_sub SUB,
    //                                        func_mul MULT,
    //                                        int num_round = 0) const
    //{
    //    BT t;
    //    if constexpr (std::is_same_v<func_add(), OP_XOR>)
    //    {
    //        t = retrieveBooleanTriple<Datatype>();
    //    }
    //    else
    //    {
    //        t = retrieveArithmeticTriple<Datatype>();
    //    }
    //    auto lta = ADD(l, t.a);
    //    auto ltb = ADD(b.l, t.b);
    //    pre_send_to_live(PNEXT, lta);
    //    auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
    //    store_output_share_ab(lta, ADD, num_round);
    //    store_output_share_ab(ltb, ADD, num_round);
    //    store_output_share_ab(b.l, ADD, num_round);
    //    store_output_share_ab(t.a, ADD, num_round);
    //    store_output_share_ab(t.c, ADD, num_round);
    //}

    //template <typename func_add, typename func_sub, typename func_mul>
    //static Datatype receive_and_compute_lxly_share_comp_opt(func_add ADD,
    //                                                        func_sub SUB,
    //                                                        func_mul MULT,
    //                                                        int num_round = 0)
    //{
    //    auto lta = OP_ADD(pre_receive_from_live(PNEXT), retrieve_output_share_ab(ADD, num_round));
    //    auto ltb = OP_ADD(pre_receive_from_live(PNEXT), retrieve_output_share_ab(ADD, num_round));
    //    auto ta = retrieve_output_share_ab(ADD, num_round);
    //    auto bl = retrieve_output_share_ab(ADD, num_round);
    //    auto tc = retrieve_output_share_ab(ADD, num_round);
    //    return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), tc);
    //}
    
    //template <typename func_add, typename func_sub, typename func_mul>
    //void generate_lxly_from_triple(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT, int num_round = 0) const
    //{
    //    BT t;
    //    if constexpr (std::is_same_v<func_add(), OP_XOR>)
    //    {
    //        t = retrieveBooleanTriple<Datatype>();
    //    }
    //    else
    //    {
    //        t = retrieveArithmeticTriple<Datatype>();
    //    }
    //    auto lta = ADD(l, t.a);
    //    auto ltb = ADD(b.l, t.b);
    //    pre_send_to_live(PNEXT, lta);
    //    pre_send_to_live(PNEXT, ltb);
    //    auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
    //    if constexpr (std::is_same_v<func_add(), OP_XOR>)
    //    {
    //        store_output_share_bool(t.a, num_round);
    //        store_output_share_bool(b.l, num_round);
    //        store_output_share_bool(lxly, num_round);
    //    }
    //    else
    //    {
    //        store_output_share_arithmetic(t.a, num_round);
    //        store_output_share_arithmetic(b.l, num_round);
    //        store_output_share_arithmetic(lxly, num_round);
    //    }
    //}
    

    //template <typename func_add, typename func_sub, typename func_mul>
    //static Datatype receive_and_compute_lxly_share(func_add ADD, func_sub SUB, func_mul MULT, int num_round = 0)
    //{
    //    auto lta = pre_receive_from_live(PNEXT);
    //    auto ltb = pre_receive_from_live(PNEXT);
    //    if constexpr (std::is_same_v<func_add(), OP_XOR>)
    //    {
    //        auto ta = retrieve_output_share_bool(num_round);
    //        auto bl = retrieve_output_share_bool(num_round);
    //        auto prev_val = retrieve_output_share_bool(num_round);
    //        return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), prev_val);
    //    }
    //    else
    //    {
    //        auto ta = retrieve_output_share_arithmetic(num_round);
    //        auto bl = retrieve_output_share_arithmetic(num_round);
    //        auto prev_val = retrieve_output_share_arithmetic(num_round);
    //        return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), prev_val);
    //    }
    //}
//#endif

#endif
