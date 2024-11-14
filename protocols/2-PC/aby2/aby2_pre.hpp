#include <cstdint>
#include "../../beaver_triples.hpp"
template <typename Datatype>
class ABY2_PRE_Share{
using BT = triple<Datatype>;
private:
Datatype l;
public:
ABY2_PRE_Share()  {}
ABY2_PRE_Share(Datatype l) { this->l = l; }

template <typename func_add, typename func_sub, typename func_mul>
void generate_lxly_from_triple(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT, int num_round=0) const
{
    BT t;
    if constexpr(std::is_same_v<func_add(), OP_XOR>)
    {
        t = retrieveBooleanTriple<Datatype>();
    }
    else
    {
        t = retrieveArithmeticTriple<Datatype>();
    }
    auto lta = ADD(l,t.a);
    auto ltb = ADD(b.l,t.b);
    pre_send_to_live(PNEXT, lta);
    pre_send_to_live(PNEXT, ltb);
    auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
    if constexpr(std::is_same_v<func_add(), OP_XOR>)
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
void generate_lxly_from_triple_comp_opt(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT, int num_round=0) const
{
    BT t;
    if constexpr(std::is_same_v<func_add(), OP_XOR>)
    {
        t = retrieveBooleanTriple<Datatype>();
    }
    else
    {
        t = retrieveArithmeticTriple<Datatype>();
    }
    auto lta = ADD(l,t.a);
    auto ltb = ADD(b.l,t.b);
    pre_send_to_live(PNEXT, lta);
    auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
    store_output_share_ab(lta, ADD, num_round);
    store_output_share_ab(ltb, ADD, num_round);
    store_output_share_ab(b.l, ADD, num_round);
    store_output_share_ab(t.a, ADD, num_round);
    store_output_share_ab(t.c, ADD, num_round);
}

    template <typename func_add, typename func_sub, typename func_mul>
static Datatype receive_and_compute_lxly_share_comp_opt(func_add ADD, func_sub SUB, func_mul MULT, int num_round=0) 
{
    auto lta = OP_ADD(pre_receive_from_live(PNEXT), retrieve_output_share_ab(ADD, num_round));
    auto ltb = OP_ADD(pre_receive_from_live(PNEXT), retrieve_output_share_ab(ADD, num_round));
    auto ta = retrieve_output_share_ab(ADD, num_round);
    auto bl = retrieve_output_share_ab(ADD, num_round);
    auto tc = retrieve_output_share_ab(ADD, num_round);
    return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), tc);
}
    
template <typename func_add, typename func_sub, typename func_mul>
static Datatype receive_and_compute_lxly_share(func_add ADD, func_sub SUB, func_mul MULT, int num_round=0) 
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    if constexpr(std::is_same_v<func_add(), OP_XOR>)
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

template <typename func_mul>
ABY2_PRE_Share mult_public(const Datatype b, func_mul MULT) const
{
    return ABY2_PRE_Share(MULT(l,b));
}

template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
        l = getRandomVal(PSELF);
    else
        l = SET_ALL_ZERO();
}

template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
}

template <typename func_add>
ABY2_PRE_Share Add( ABY2_PRE_Share b, func_add ADD) const
{
    return ABY2_PRE_Share(ADD(l,b.l));
}

void prepare_reveal_to_all() const
{
    pre_send_to_live(PNEXT, l);
    triple_type[0][triple_type_index[0]++] = 2;
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
    return SET_ALL_ZERO();
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    if constexpr(std::is_same_v<func_add(), OP_XOR>)
    {
        triple_type[0][triple_type_index[0]++] = 0;
    }
    else
    {
        triple_type[0][triple_type_index[0]++] = 1;
    }
generate_lxly_from_triple(b, ADD, SUB, MULT);
return ABY2_PRE_Share(getRandomVal(PSELF)); //new mask
}


template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    if constexpr(std::is_same_v<func_add(), OP_XOR>)
    {
        triple_type[0][triple_type_index[0]++] = 0;
    }
    else
    {
        triple_type[0][triple_type_index[0]++] = 1;
    }
    generate_lxly_from_triple(b, ADD, SUB, MULT);
    return ABY2_PRE_Share();
}

/* template <typename func_add, typename func_sub, typename func_mul, typename func_trunc> */
/*     ABY2_PRE_Share prepare_dot_with_trunc(ABY2_PRE_Share b, func_add ADD, func_sub SUB, func_mul MULT, func_trunc TRUNC) const */
/* { */
/*     return prepare_mult(b, ADD, SUB, MULT); */
/* } */

template <typename func_add, typename func_sub>

void mask_and_send_dot(func_add ADD, func_sub SUB)
{
    l = getRandomVal(PSELF);
}
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
    l = getRandomVal(PSELF);
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_PRE_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC, int fractional_bits = FRACTIONAL) const
{
#if PARTY == 0
triple_type[0][triple_type_index[0]++] = 2;
return ABY2_PRE_Share(getRandomVal(PSELF));
#else
auto c = ABY2_PRE_Share(getRandomVal(PSELF));
pre_send_to_live(PNEXT,ADD(c.l, SUB(SET_ALL_ZERO(), TRUNC(MULT(l,b),fractional_bits)))); // Share Trunc -(lv1 * b) + lz
return c;
#endif
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_PRE_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
#if PARTY == 0
    triple_type[0][triple_type_index[0]++] = 2;
    return ABY2_PRE_Share(getRandomVal(PSELF));
#else
    auto result = l; // Share Trunc - Trunc(lv1)
    for(int i = 2; i <= b; i*=2)
        result = OP_TRUNC2(result);
    result = OP_SUB(SET_ALL_ZERO(),result);

    Datatype res_l = getRandomVal(PSELF);
    pre_send_to_live(PNEXT, ADD(result,res_l));
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
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
    }
#endif
}

static void prepare_A2B_S2(int m, int k, ABY2_PRE_Share in[], ABY2_PRE_Share out[])
{
#if PARTY == 1
    Datatype temp_p1[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp_p1[i] = OP_SUB(SET_ALL_ZERO(),in[i].l) ; // set second share to -lv2
    }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp_p1, temp2);
    orthogonalize_boolean(temp2, temp_p1);
    
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
        Datatype out_m = OP_XOR(temp_p1[i],out[i-m].l);
        pre_send_to_live(PNEXT, out_m);
    }
#else
    for(int i = m; i < k; i++)
    {
        triple_type[0][triple_type_index[0]++] = 2;
    }
#endif

}

void prepare_bit2a(ABY2_PRE_Share out[])
{
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    Datatype lb[BITLENGTH]{0};
    lb[BITLENGTH - 1] = l;
    unorthogonalize_boolean(lb, temp2);
    orthogonalize_arithmetic(temp2, lb);
    for(int i = 0; i < BITLENGTH; i++)
    {
        triple_type[0][triple_type_index[0]++] = 3;
#if PARTY == 0
        ABY2_PRE_Share b1{lb[i]};
        ABY2_PRE_Share b2{SET_ALL_ZERO()};
#else
        ABY2_PRE_Share b1{SET_ALL_ZERO()};
        ABY2_PRE_Share b2{lb[i]};
#endif
        b1.generate_lxly_from_triple(b2, OP_ADD, OP_SUB, OP_MULT); // communication can be cut in half if triple of type x(P_0),y(P_1),[z] is used
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

void complete_bit2a()
{
}

void complete_opt_bit_injection()
{
}

void prepare_opt_bit_injection(ABY2_PRE_Share x[], ABY2_PRE_Share out[])
{
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    Datatype lb[BITLENGTH]{0};
    lb[BITLENGTH - 1] = l;
    unorthogonalize_boolean(lb, temp2);
    orthogonalize_arithmetic(temp2, lb);
    for(int i = 0; i < BITLENGTH; i++)
    {
        triple_type[0][triple_type_index[0]++] = 4;
        triple_type[1][triple_type_index[1]++] = 4;
#if PARTY == 0
        ABY2_PRE_Share b1{lb[i]};
        ABY2_PRE_Share b2{SET_ALL_ZERO()};
#else
        ABY2_PRE_Share b1{SET_ALL_ZERO()};
        ABY2_PRE_Share b2{lb[i]};
#endif
        b1.generate_lxly_from_triple(b2, OP_ADD, OP_SUB, OP_MULT); // communication can be cut in half if triple of type x(P_0),y(P_1),[z] is used
        store_output_share_arithmetic(lb[i]);
        store_output_share_arithmetic(x[i].l);
        out[i].l = getRandomVal(PSELF); 
    }
}
static void complete_A2B_S1(int k, ABY2_PRE_Share out[])
{
#if PARTY == 1
    for(int i = 0; i < k; i++)
    {
        out[i].l = SET_ALL_ZERO();
    }
#endif
}

static void complete_A2B_S2(int k, ABY2_PRE_Share out[])
{
#if PARTY == 0
    for(int i = 0; i < k; i++)
    {
        out[i].l = SET_ALL_ZERO();
    }
#endif
}


static ABY2_PRE_Share public_val(Datatype a)
{
    return ABY2_PRE_Share(SET_ALL_ZERO());
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot3(const ABY2_PRE_Share b, const ABY2_PRE_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
if constexpr(std::is_same_v<func_add(), OP_XOR>)
{
    triple_type[0][triple_type_index[0]++] = 5;
    triple_type[0][triple_type_index[0]++] = 0;
    triple_type[0][triple_type_index[0]++] = 0;
    triple_type[1][triple_type_index[1]++] = 5;
}
else
{
    triple_type[0][triple_type_index[0]++] = 6;
    triple_type[0][triple_type_index[0]++] = 1;
    triple_type[0][triple_type_index[0]++] = 1;
    triple_type[1][triple_type_index[1]++] = 6;
}
store_output_share_ab(c.l, ADD);
generate_lxly_from_triple(b, ADD, SUB, MULT); //rxy
generate_lxly_from_triple(c, ADD, SUB, MULT); //rxz
b.generate_lxly_from_triple(c, ADD, SUB, MULT); //ryz
return ABY2_PRE_Share();
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot4(const ABY2_PRE_Share b, const ABY2_PRE_Share c, const ABY2_PRE_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
if constexpr(std::is_same_v<func_add(), OP_XOR>)
{
    triple_type[0][triple_type_index[0]++] = 7; //xy, zw
    triple_type[0][triple_type_index[0]++] = 99; //since xy,zw are together, skip next triple
    triple_type[0][triple_type_index[0]++] = 0; //xz
    triple_type[0][triple_type_index[0]++] = 0; //xw
    triple_type[0][triple_type_index[0]++] = 0; //yz
    triple_type[0][triple_type_index[0]++] = 0; //yw
    triple_type[1][triple_type_index[1]++] = 7; //xyzw

}
else
{
    triple_type[0][triple_type_index[0]++] = 8; //xy, zw
    triple_type[0][triple_type_index[0]++] = 99; //since xy,zw are together, skip next triple
    triple_type[0][triple_type_index[0]++] = 1; //xz
    triple_type[0][triple_type_index[0]++] = 1; //xw
    triple_type[0][triple_type_index[0]++] = 1; //yz
    triple_type[0][triple_type_index[0]++] = 1; //yw
    triple_type[1][triple_type_index[1]++] = 8; //xyzw
}

store_output_share_ab(l, ADD); //xzw
store_output_share_ab(b.l, ADD); //yzw
store_output_share_ab(c.l, ADD); // xyz
store_output_share_ab(d.l, ADD); //xyw
generate_lxly_from_triple(b, ADD, SUB, MULT); //xy --> +2 stores
c.generate_lxly_from_triple(d, ADD, SUB, MULT); //zw --> +2 stores
generate_lxly_from_triple(c, ADD, SUB, MULT); //xz
generate_lxly_from_triple(d, ADD, SUB, MULT); //xw
b.generate_lxly_from_triple(c, ADD, SUB, MULT); //yz
b.generate_lxly_from_triple(d, ADD, SUB, MULT); //yw
return ABY2_PRE_Share();
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult3(ABY2_PRE_Share b, ABY2_PRE_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
    ABY2_PRE_Share d = prepare_dot3(b,c,ADD,SUB,MULT);
    d.mask_and_send_dot(ADD,SUB);
    return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
    complete_mult(ADD,SUB);
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_mult4(ABY2_PRE_Share b, ABY2_PRE_Share c, ABY2_PRE_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
    ABY2_PRE_Share e = prepare_dot4(b,c,d,ADD,SUB,MULT);
    e.mask_and_send_dot(ADD,SUB);
    return e;
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
    complete_mult(ADD,SUB);
}





ABY2_PRE_Share Not() const
{
    return ABY2_PRE_Share(l);
}

static void send()
{
    send_live();
}

static void receive()
{
    receive_live();
}

static void communicate()
{
}


static void complete_preprocessing(uint64_t* arithmetic_triple_num, uint64_t* boolean_triple_num, uint64_t num_output_shares)
{
communicate_pre();
const int num_rounds = 2;
Datatype** lxly_a = new Datatype*[num_rounds];
Datatype** lxly_b = new Datatype*[num_rounds];
lxly_a[0] = new Datatype[arithmetic_triple_num[0]];
lxly_b[0] = new Datatype[boolean_triple_num[0]];
uint64_t arithmetic_triple_counter[num_rounds]{0};
uint64_t boolean_triple_counter[num_rounds]{0};
auto num_triples = arithmetic_triple_num[0] + boolean_triple_num[0] + num_output_shares;
preprocessed_outputs_bool[1] = new Datatype[preprocessed_outputs_bool_input_index[1]];
preprocessed_outputs_arithmetic[1] = new Datatype[preprocessed_outputs_arithmetic_input_index[1]];
preprocessed_outputs_arithmetic_input_index[1] = 0;
preprocessed_outputs_bool_input_index[1] = 0;
for(uint64_t i = 0; i < num_triples; i++)
{
switch(triple_type[0][i])
{
case 0: //AND
{
    auto lxly = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND);
    lxly_b[0][boolean_triple_counter[0]++] = lxly;
    break;
}
case 1: //ADD
{
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    break;
}
case 3: //Bit2A
{
    /* auto lta = pre_receive_from_live(PNEXT); */
    /* auto ltb = pre_receive_from_live(PNEXT); */
    /* auto ta = retrieve_output_share_arithmetic(); */
    /* auto bl = retrieve_output_share_arithmetic(); */
    /* auto prev_val = retrieve_output_share_arithmetic(); */
    /* lxly_a[0][arithmetic_triple_counter[0]++] = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)), prev_val); */
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT); // preprocessing costs can be cut in half if triple of type x(P_0),y(P_1),[z] is used
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    break;
}
case 4: //BitInjection
{
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly = OP_SUB(retrieve_output_share_arithmetic(), OP_ADD(lxly,lxly)); 
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly; // [lb] - 2[lb1lb2] 
    
    ABY2_PRE_Share al2 = retrieve_output_share_arithmetic(); // [la]
    al2.generate_lxly_from_triple(lxly, OP_ADD, OP_SUB, OP_MULT,1); // [la] [lb]
    break;
}
case 5: //Dot3 (bool)
{
    auto third = retrieve_output_share_bool();
    auto lxly = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND);
    lxly_b[0][boolean_triple_counter[0]++] = lxly;
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(third, OP_XOR, OP_XOR, OP_AND, 1);
    break;
}
case 6: //Dot3 (arithemtic)
{
    auto third = retrieve_output_share_arithmetic();
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(third, OP_ADD, OP_SUB, OP_MULT, 1);
    break;
}
case 7: //Dot4 (bool)
{
    auto x = retrieve_output_share_bool();
    auto y = retrieve_output_share_bool();
    auto z = retrieve_output_share_bool();
    auto w = retrieve_output_share_bool();
    auto lxly = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND);
    auto lzlw = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND);
    lxly_b[0][boolean_triple_counter[0]++] = lxly;
    lxly_b[0][boolean_triple_counter[0]++] = lzlw;
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(z, OP_XOR, OP_XOR, OP_AND, 1);
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(w, OP_XOR, OP_XOR, OP_AND, 1);
    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_from_triple(x, OP_XOR, OP_XOR, OP_AND, 1);
    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_from_triple(y, OP_XOR, OP_XOR, OP_AND, 1);
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(lzlw, OP_XOR, OP_XOR, OP_AND, 1);
    break;
}
case 8: //Dot4 (arithemtic)
{
    auto x = retrieve_output_share_arithmetic();
    auto y = retrieve_output_share_arithmetic();
    auto z = retrieve_output_share_arithmetic();
    auto w = retrieve_output_share_arithmetic();
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    auto lzlw = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    lxly_a[0][arithmetic_triple_counter[0]++] = lzlw;
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(z, OP_ADD, OP_SUB, OP_MULT, 1);
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(w, OP_ADD, OP_SUB, OP_MULT, 1);
    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_from_triple(x, OP_ADD, OP_SUB, OP_MULT,1);
    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_from_triple(y, OP_ADD, OP_SUB, OP_MULT, 1);
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(lzlw, OP_ADD, OP_SUB, OP_MULT, 1);
    break;
}
case 9: //MatMul First Dot element
{
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    break;
}
case 10: //MatMul
{
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly_a[0][arithmetic_triple_counter[0]-1] = OP_ADD(lxly_a[0][arithmetic_triple_counter[0]-1], lxly);
    break;
}
case 99: //Triple already consumed by previous case
{
    break;
}
default:
{
    auto l = pre_receive_from_live(PNEXT);
    store_output_share(l);
    break;
}
}
}
delete[] triple_type[0];
delete[] preprocessed_outputs_bool[0];
preprocessed_outputs_bool[0] = lxly_b[0];
/* preprocessed_outputs_bool_index[0] = 0; */
preprocessed_outputs_bool_input_index[0] = 0;

delete[] preprocessed_outputs_arithmetic[0];
preprocessed_outputs_arithmetic[0] = lxly_a[0];
/* preprocessed_outputs_arithmetic_index[0] = 0; */
preprocessed_outputs_arithmetic_input_index[0] = 0;

/* preprocessed_outputs_bool_index[1] = 0; */
preprocessed_outputs_bool_input_index[1] = 0;

/* preprocessed_outputs_arithmetic_index[1] = 0; */
preprocessed_outputs_arithmetic_input_index[1] = 0;


deinit_beaver();
communicate_pre();
lxly_a[1] = new Datatype[arithmetic_triple_num[1]];
lxly_b[1] = new Datatype[boolean_triple_num[1]];
num_triples = arithmetic_triple_num[1] + boolean_triple_num[1];
for(uint64_t i = 0; i < num_triples; i++)
{
    switch(triple_type[1][i])
    {
        case 4:
        {
            auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
            break;
        }
        case 5:
        {
            auto lxly = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND,1);
            lxly_b[1][boolean_triple_counter[1]++] = lxly;
            break;
        }
        case 6:
        {
            auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
            break;
        }
        case 7:
        {
            auto lxly_lz = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND,1);
            auto lxly_lw = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND,1);
            auto lzlw_lx = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND,1);
            auto lzlw_ly = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND,1);
            auto lxly_lzlw = receive_and_compute_lxly_share(OP_XOR,OP_XOR,OP_AND,1);
            lxly_b[1][boolean_triple_counter[1]++] = lxly_lz;
            lxly_b[1][boolean_triple_counter[1]++] = lxly_lw;
            lxly_b[1][boolean_triple_counter[1]++] = lzlw_lx;
            lxly_b[1][boolean_triple_counter[1]++] = lzlw_ly;
            lxly_b[1][boolean_triple_counter[1]++] = lxly_lzlw;
            break;
        }
        case 8:
        {
            auto lxly_lz = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lxly_lw = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lzlw_lx = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lzlw_ly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lxly_lzlw = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
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
init_srngs();
}

// --- Untested Functions --- TODO: Test

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_PRE_Share prepare_trunc_share(func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC, int fractional_bits=FRACTIONAL) const
{
    return ABY2_PRE_Share(getRandomVal(PSELF));
} 

void get_random_B2A()
{
        l = getRandomVal(PSELF);
}



#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1
    

static void GEMM(ABY2_PRE_Share* a, ABY2_PRE_Share* b, ABY2_PRE_Share* c, int m, int n, int k, bool a_fixed = false)
{
    if(a_fixed == true)
    {
        const int factor = DATTYPE/BITLENGTH;
        if(factor > 1)
            for(int i = 0; i < m*k; i++)
            {
                alignas (sizeof(Datatype)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&a[i].l, temp,1);
                a[i].l = PROMOTE(temp[0]);
            }
    }
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int l = 0; l < k; l++)
            {
                a[i * k + l].generate_lxly_from_triple(b[l * n + j], OP_ADD, OP_SUB, OP_MULT);
                triple_type[0][triple_type_index[0]++] = 10;
            }
            triple_type[0][triple_type_index[0]-k] = 9;
        }
    }
}

#else

static void GEMM(ABY2_PRE_Share* a, ABY2_PRE_Share* b, ABY2_PRE_Share* c, int m, int n, int k, bool a_fixed = false)
{
    if(a_fixed == true)
    {
        const int factor = DATTYPE/BITLENGTH;
        if(factor > 1)
            for(int i = 0; i < m*k; i++)
            {
                alignas (sizeof(Datatype)) UINT_TYPE temp[factor];
                unorthogonalize_arithmetic(&a[i].l, temp,1);
                a[i].l = PROMOTE(temp[0]);
            }
    }
    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            for(int l = 0; l < k; l++)
            {
                a[i * k + l].generate_lxly_from_triple(b[l * n + j], OP_ADD, OP_SUB, OP_MULT);
                triple_type[0][triple_type_index[0]++] = 10;
            }
            triple_type[0][triple_type_index[0]-k] = 9;
        }
    }
}
       
#endif
#endif

};

#if USE_CUDA_GEMM == 2 || USE_CUDA_GEMM == 4
template <typename T>
T im2col_get_pixel_l(const T* im, int height, int width, int channels,
                        int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 || row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}

// From Berkeley Vision's Caffe!
// https://github.com/BVLC/caffe/blob/master/LICENSE
template <typename T>
void im2col_l(const T* data_im, int channels, int height, int width,
            int ksize, int stride, int pad, T* data_col)
{
    int c, h, w;
    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel_l(data_im, height, width, channels,
                    im_row, im_col, c_im, pad);
            }
        }
    }
}

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





/* static void COMPLETE_CONV_2D(const ABY2_PRE_Share* X, const ABY2_PRE_Share* W, ABY2_PRE_Share* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation = 1) */
/* { */
/* } */

template <typename Datatype>
static void ABY2_PRE_Share<Datatype>::CONV_2D(const ABY2_PRE_Share* X, const ABY2_PRE_Share* W, ABY2_PRE_Share* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation = 1)
{
    const int factor = DATTYPE/BITLENGTH;
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
