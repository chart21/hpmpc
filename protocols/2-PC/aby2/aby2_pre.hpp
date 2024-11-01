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
    if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
    {
        t = retrieveBooleanTriple<Datatype>();
    }
    else
    {
        t = retrieveArithmeticTriple<Datatype>();
    }
    std::cout << "l, b.l" << int(l) << " " << int(b.l) << std::endl;
    auto lta = ADD(l,t.a);
    auto ltb = ADD(b.l,t.b);
    pre_send_to_live(PNEXT, lta);
    pre_send_to_live(PNEXT, ltb);
    auto lxly = ADD(SUB(MULT(lta, b.l), MULT(ltb, t.a)), t.c);
    if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
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
static Datatype receive_and_compute_lxly_share(func_add ADD, func_sub SUB, func_mul MULT, int num_round=0) 
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share_arithmetic(num_round);
    auto bl = retrieve_output_share_arithmetic(num_round);
    auto prev_val = retrieve_output_share_arithmetic(num_round);
    return ADD(SUB(MULT(lta, bl), MULT(ltb, ta)), prev_val);
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
    if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
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
    if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
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
return ABY2_PRE_Share(getRandomVal(PSELF));
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
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
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
        auto t = retrieveArithmeticTriple<Datatype>();
        triple_type[0][triple_type_index[0]++] = 3;
#if PARTY == 0
        auto bl = SET_ALL_ZERO();
        auto al = lb[i];
#else 
        auto bl = lb[i];
        auto al = SET_ALL_ZERO();
#endif
        auto lta = OP_ADD(al, t.a);
        auto ltb = OP_ADD(bl, t.b); //optimization?
        pre_send_to_live(PNEXT, lta); 
        pre_send_to_live(PNEXT, ltb);
        auto lxly = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, t.a)), t.c);
        store_output_share_arithmetic(t.a);
        store_output_share_arithmetic(bl);
        store_output_share_arithmetic(lxly);
        out[i].l = getRandomVal(PSELF); 
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
        auto t = retrieveArithmeticTriple<Datatype>();
        triple_type[0][triple_type_index[0]++] = 4;
#if PARTY == 0
        auto bl = SET_ALL_ZERO();
        auto al = lb[i];
#else 
        auto bl = lb[i];
        auto al = SET_ALL_ZERO();
#endif
        auto lta = OP_ADD(al, t.a);
        auto ltb = OP_ADD(bl, t.b); //optimization?
        pre_send_to_live(PNEXT, lta); 
        pre_send_to_live(PNEXT, ltb);
        auto lxly = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, t.a)), t.c);
        store_output_share_arithmetic(t.a);
        store_output_share_arithmetic(bl);
        store_output_share_arithmetic(lxly);
        store_output_share_arithmetic(l);
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
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
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
store_output_share_arithmetic(c.l); // rxyz
generate_lxly_from_triple(b, ADD, SUB, MULT); //rxy
generate_lxly_from_triple(c, ADD, SUB, MULT); //rxz
b.generate_lxly_from_triple(c, ADD, SUB, MULT); //ryz
return ABY2_PRE_Share();
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_PRE_Share prepare_dot4(const ABY2_PRE_Share b, const ABY2_PRE_Share c, const ABY2_PRE_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
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

store_output_share_arithmetic(l); //xzw
store_output_share_arithmetic(b.l); //yzw
store_output_share_arithmetic(c.l); // xyz
store_output_share_arithmetic(d.l); //xyw
generate_lxly_from_triple(b, ADD, SUB, MULT); //xy --> +2 stores
c.generate_lxly_from_triple(d, ADD, SUB, MULT); //zw --> +2 stores
generate_lxly_from_triple(c, ADD, SUB, MULT); //xz
generate_lxly_from_triple(d, ADD, SUB, MULT); //xw
b.generate_lxly_from_triple(d, ADD, SUB, MULT); //yz
b.generate_lxly_from_triple(c, ADD, SUB, MULT); //yw
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
std::cout << PARTY << "Num arithmetic, boolean , output shares: " << arithmetic_triple_num[0] << " " << boolean_triple_num[0] << " " << num_output_shares << std::endl;
preprocessed_outputs_bool[1] = new Datatype[preprocessed_outputs_bool_input_index[1]];
preprocessed_outputs_arithmetic[1] = new Datatype[preprocessed_outputs_arithmetic_input_index[1]];
preprocessed_outputs_arithmetic_input_index[1] = 0;
preprocessed_outputs_bool_input_index[1] = 0;
for(uint64_t i = 0; i < num_triples; i++)
{
std::cout << PARTY << "Triple type: " << int(triple_type[0][i]) << std::endl;
switch(triple_type[0][i])
{
case 0: //AND
{
    auto lxly = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND);
    lxly_b[0][boolean_triple_counter[0]++] = lxly;
    break;
}
case 1: //ADD
{
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    std::cout << PARTY << "case 1, index: " << arithmetic_triple_counter[0] << " lxly: " << lxly << std::endl;
    break;
}
case 3: //Bit2A
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share_arithmetic();
    auto bl = retrieve_output_share_arithmetic();
    auto prev_val = retrieve_output_share_arithmetic();
    lxly_a[0][arithmetic_triple_counter[0]++] = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)), prev_val);
    break;
}
case 4: //BitInjection
{
    auto lta = pre_receive_from_live(PNEXT);
    auto ltb = pre_receive_from_live(PNEXT);
    auto ta = retrieve_output_share_arithmetic();
    auto bl = retrieve_output_share_arithmetic();
    auto prev_val = retrieve_output_share_arithmetic();
    auto lxly = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)), prev_val);
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    auto t = retrieveArithmeticTriple<Datatype>();
    triple_type[1][triple_type_index[1]++] = 4;
    auto bl2 = OP_SUB(retrieve_output_share_arithmetic(), OP_ADD(lxly,lxly)); // [lb] - 2[lb1lb2]
    auto al2 = retrieve_output_share_arithmetic();
    auto lta2 = OP_ADD(al2, t.a);
    auto ltb2 = OP_ADD(bl2, t.b); //optimization?
    pre_send_to_live(PNEXT, lta2); 
    pre_send_to_live(PNEXT, ltb2);
    auto lxly2 = OP_ADD(OP_SUB(OP_MULT(lta2, bl2), OP_MULT(ltb2, t.a)), t.c);
    store_output_share_arithmetic(t.a,1);
    store_output_share_arithmetic(bl2,1);
    store_output_share_arithmetic(lxly2,1);
    break;
}
case 5: //Dot3 (bool)
{
    auto third = retrieve_output_share_bool();
    auto lxly = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND);
    lxly_b[0][boolean_triple_counter[0]++] = lxly;
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(third, FUNC_XOR, FUNC_XOR, FUNC_AND, 1);
    break;
}
case 6: //Dot3 (arithemtic)
{
    auto third = retrieve_output_share_arithmetic();
    auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT);
    lxly_a[0][arithmetic_triple_counter[0]++] = lxly;
    std::cout << PARTY << "index: " << arithmetic_triple_counter[0] << " lxly: " << lxly << std::endl;
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(third, OP_ADD, OP_SUB, OP_MULT, 1);
    break;
}
case 7: //Dot4 (bool)
{
    auto x = retrieve_output_share_bool();
    auto y = retrieve_output_share_bool();
    auto z = retrieve_output_share_bool();
    auto w = retrieve_output_share_bool();
    auto lxly = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND);
    auto lzlw = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND);
    lxly_b[0][boolean_triple_counter[0]++] = lxly;
    lxly_b[0][boolean_triple_counter[0]++] = lzlw;
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(z, FUNC_XOR, FUNC_XOR, FUNC_AND, 1);
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(w, FUNC_XOR, FUNC_XOR, FUNC_AND, 1);
    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_from_triple(x, FUNC_XOR, FUNC_XOR, FUNC_AND, 1);
    ABY2_PRE_Share<Datatype>(lzlw).generate_lxly_from_triple(y, FUNC_XOR, FUNC_XOR, FUNC_AND, 1);
    ABY2_PRE_Share<Datatype>(lxly).generate_lxly_from_triple(lzlw, FUNC_XOR, FUNC_XOR, FUNC_AND, 1);
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
            /* auto lta = pre_receive_from_live(PNEXT,1); */
            /* auto ltb = pre_receive_from_live(PNEXT,1); */
            auto lta = pre_receive_from_live(PNEXT);
            auto ltb = pre_receive_from_live(PNEXT);
            auto ta = retrieve_output_share_arithmetic(1);
            auto bl = retrieve_output_share_arithmetic(1);
            auto prev_val = retrieve_output_share_arithmetic(1);
            auto lxly = OP_ADD(OP_SUB(OP_MULT(lta, bl), OP_MULT(ltb, ta)), prev_val);
            lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
        }
        case 5:
        {
            auto lxly = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND,1);
            lxly_b[1][boolean_triple_counter[1]++] = lxly;
            break;
        }
        case 6:
        {
            auto lxly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            lxly_a[1][arithmetic_triple_counter[1]++] = lxly;
            std::cout << PARTY << "case 6, index: " << arithmetic_triple_counter[1] << " lxly: " << lxly << std::endl;
            break;
        }
        case 7:
        {
            auto lxly_lw = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND,1);
            auto lxly_lz = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND,1);
            auto lzlw_lx = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND,1);
            auto lzlw_ly = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND,1);
            auto lxly_lzlw = receive_and_compute_lxly_share(FUNC_XOR,FUNC_XOR,FUNC_AND,1);
            lxly_b[1][boolean_triple_counter[1]++] = lxly_lw;
            lxly_b[1][boolean_triple_counter[1]++] = lxly_lz;
            lxly_b[1][boolean_triple_counter[1]++] = lzlw_lx;
            lxly_b[1][boolean_triple_counter[1]++] = lzlw_ly;
            lxly_b[1][boolean_triple_counter[1]++] = lxly_lzlw;
        }
        case 8:
        {
            auto lxly_lw = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lxly_lz = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lzlw_lx = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lzlw_ly = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            auto lxly_lzlw = receive_and_compute_lxly_share(OP_ADD,OP_SUB,OP_MULT,1);
            lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lw;
            lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lz;
            lxly_a[1][arithmetic_triple_counter[1]++] = lzlw_lx;
            lxly_a[1][arithmetic_triple_counter[1]++] = lzlw_ly;
            lxly_a[1][arithmetic_triple_counter[1]++] = lxly_lzlw;
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

};
