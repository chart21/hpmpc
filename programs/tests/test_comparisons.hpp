#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "test_helper.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "../functions/prob_truncation.hpp"
#include "../functions/exact_truncation.hpp"
#include "../functions/comparisons.hpp"
#include "../functions/Relu.hpp"
#pragma once
#include "../functions/max_min.hpp"
#define RESULTTYPE DATATYPE
#ifndef FUNCTION
#define FUNCTION test_comparisons
#endif
#define TEST_EQZ 0 // [a] == 0 ? [1] : [0]
#define TEST_LTZ 1 // [a] < 0 ? [1] : [0]
#define TEST_MAX_MIN 0 // [A] -> [max(A)] [min(A)]
#define TEST_RELU 0 // [a] > 0 ? [a] : 0
#define TEST_BOOLEAN_ADDITION 0 // [a]^B + [b]^B
#define TEST_A2B_ADD 1 // Test A2B followed by addition when testing boolean addition
#define TEST_A2B 0 // [a]^A -> [a]^B
#define TEST_Bit2A 0 // [a]^b -> [a]^A

#if TEST_EQZ == 1
template<typename Share>
bool test_EQZ()
{
    const int vectorization_factor = DATTYPE/BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
   
    //set shares from public values
    A eqz_output[3];
    A eqz_input[] = {A(0), A(1), A(-1)};
  
    //EQZ
    pack_additive<0, BITLENGTH>(eqz_input, eqz_output, 3, EQZ<0, BITLENGTH, Share, DATATYPE>); //EQZ
   
    //reveal
    alignas(sizeof(DATATYPE)) UINT_TYPE output[3][vectorization_factor];
    reveal_and_store(eqz_output, output, 3);

    //compare
    for(int i = 0; i < vectorization_factor; i++)
    {
        print_compare(1, output[0][i]);
        print_compare(0, output[1][i]);
        print_compare(0, output[2][i]);

        if(output[0][i] != 1 || output[1][i] != 0 || output[2][i] != 0)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_LTZ == 1
template<typename Share>
bool test_LTZ()
{
    const int vectorization_factor = DATTYPE/BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
   
    //set shares from public values
    A ltz_output[6];
    A ltz_input[] = {A(0), A(1), A(-1), A(1), A(-1), A(1)};
  
    //LTZ
    pack_additive<0, BITLENGTH>(ltz_input, ltz_output, 6, LTZ<0, BITLENGTH, Share, DATATYPE>);
   
    //reveal
    alignas(sizeof(DATATYPE)) UINT_TYPE output[6][vectorization_factor];
    reveal_and_store(ltz_output, output, 6);

    //compare
    for(int i = 0; i < vectorization_factor; i++)
    {
        print_compare(0, output[0][i]);
        print_compare(0, output[1][i]);
        print_compare(1, output[2][i]);
        print_compare(0, output[3][i]);
        print_compare(1, output[4][i]);
        print_compare(0, output[5][i]);

        if(output[0][i] != 0 || output[1][i] != 0 || output[2][i] != 1 || output[3][i] != 0 || output[4][i] != 1 || output[5][i] != 0)
        {
            return false;
        }
    }
    return true;
}

#endif

#if TEST_MAX_MIN == 1
    template<typename Share>
bool max_min_test()
{
    Share::communicate();
    const int len = 400;
    const int vectorization_factor = DATTYPE/BITLENGTH;
using A = Additive_Share<DATATYPE, Share>;
//two indepndent arrays of lenth len, we want to find the max and min for each of the two
A a[len];
A b[len];
for(int i = 0; i < len; i++)
{
    a[i] = A(i);
    b[i] = A(len-i);
}
A merged[2*len];
A argmax[len];
A argmin[len];
A max_val[2];
A min_val[2];
A argmax_merged[2*len];
A argmin_merged[2*len];

//merge two lists two achieve both computations in equal communication rounds
for(int i = 0; i < len; i++)
{
    merged[i] = a[i];
    merged[i+len] = b[i];
}

max_min_sint<0,BITLENGTH>(merged, len, max_val, 2, true); //batch size of 2, want max of each independent array

max_min_sint<0,BITLENGTH>(merged, len, min_val, 2, false); //batch size of 2, want min of each independent array
argmax_argmin_sint<0,BITLENGTH>(merged, len, argmax_merged, 2, true);                                            
argmax_argmin_sint<0,BITLENGTH>(merged, len, argmin_merged, 2, false);
//Extract final results
A result[8] = {max_val[0], min_val[0], argmax_merged[len-1], argmin_merged[0], max_val[1], min_val[1], argmax_merged[len+0], argmin_merged[len+len-1]};
alignas(sizeof(DATATYPE)) UINT_TYPE output[8][vectorization_factor];
reveal_and_store(result, output, 8);
for(int i = 0; i < vectorization_factor; i++)
{
    print_compare(len-1, output[0][i]);
    print_compare(0, output[1][i]);
    print_compare(1, output[2][i]);
    print_compare(1, output[3][i]);
    print_compare(len, output[4][i]);
    print_compare(1, output[5][i]);
    print_compare(1, output[6][i]);
    print_compare(1, output[7][i]);
    if(output[0][i] != len-1 || output[1][i] != 0 || output[2][i] != 1 || output[3][i] != 1 || output[4][i] != len || output[5][i] != 1 || output[6][i] != 1 || output[7][i] != 1)
    {
        return false;
    }
}
return true;
}
#endif

#if TEST_RELU == 1
    template<typename Share>
bool test_RELU()
{
    const int vectorization_factor = DATTYPE/BITLENGTH;
    using A = Additive_Share<DATATYPE, Share>;
   
    //set shares from public values
    A relu_output[3];
    A relu_input[] = {A(100), A(11), A(-12)};
  
    //ReLU
    RELU<0, BITLENGTH, Share, DATATYPE>(relu_input, relu_input+3, relu_output);
    //reveal
    alignas(sizeof(DATATYPE)) UINT_TYPE output[3][vectorization_factor];
    reveal_and_store(relu_output, output, 3);

    //compare
    for(int i = 0; i < vectorization_factor; i++)
    {
        print_compare(100, output[0][i]);
        print_compare(11, output[1][i]);
        print_compare(0, output[2][i]);

        if(output[0][i] != 100 || output[1][i] != 11 || output[2][i] != 0)
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_BOOLEAN_ADDITION == 1
template<typename Share>
bool test_boolean_addition()
{
    using S = XOR_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;
    const int vectorization_factor = DATTYPE/BITLENGTH;
    
    //initialize plaintext inputs
    UINT_TYPE a[BITLENGTH*vectorization_factor];
    UINT_TYPE b[BITLENGTH*vectorization_factor];
    for(int i = 0; i < BITLENGTH*vectorization_factor; i++)
    {
        a[i] = i;
        b[i] = i;
    }
    DATATYPE vectorized_input_a[BITLENGTH];
    DATATYPE vectorized_input_b[BITLENGTH];
    

#if TEST_A2B_ADD == 0
    orthogonalize_boolean(a, vectorized_input_a);
    orthogonalize_boolean(b, vectorized_input_b);
    //secret sharing 
    Bitset share_a;
    Bitset share_b;
#else
    orthogonalize_arithmetic(a, vectorized_input_a, BITLENGTH);
    orthogonalize_arithmetic(b, vectorized_input_b, BITLENGTH);
    //secret sharing
    sint share_a;
    sint share_b;
#endif
    Bitset share_c;
    
    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    share_b.template prepare_receive_from<P_1>(vectorized_input_b);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();
    share_b.template complete_receive_from<P_1>();

#if TEST_A2B_ADD == 1
    //A2B
    Bitset s1;
    Bitset s2;

    s1 = Bitset::prepare_A2B_S1(BITLENGTH, (S*) share_a.get_share_pointer());
    s2 = Bitset::prepare_A2B_S2(BITLENGTH, (S*) share_a.get_share_pointer());

    Share::communicate();

    s1.complete_A2B_S1();
    s2.complete_A2B_S2();
    Share::communicate();
#endif

    //boolean addition
    std::vector<BooleanAdder<BITLENGTH,S>> adders;
    
    adders.reserve(1);
    for(int i = 0; i < 1; i++)
    {
#if TEST_A2B_ADD == 0
        adders.emplace_back(share_a, share_b, share_c);
#else
        adders.emplace_back(s1, s2, share_c);
#endif
    }
    while(!adders[0].is_done())
    {
        for(int i = 0; i < 1; i++)
        {
            adders[i].step();
        }
        Share::communicate();
    }
    adders.clear();
    adders.shrink_to_fit();

    //reveal
    UINT_TYPE output[BITLENGTH*vectorization_factor]; 
    share_c.prepare_reveal_to_all();
    Share::communicate();
    share_c.complete_reveal_to_all(output);
#if TEST_A2B_ADD == 1
    UINT_TYPE ortho_output[BITLENGTH]; //TODO: Check, another ortho should not be neccecary!
    unorthogonalize_boolean( (DATATYPE*) output, ortho_output);
    for(int i = 0; i < DATTYPE; i++)
        output[i] = ortho_output[i] * 2;
#endif

    //compare
    for(int i = 0; i < BITLENGTH*vectorization_factor; i++)
        print_compare(a[i] + b[i], output[i]);
    for(int i = 0; i < BITLENGTH*vectorization_factor; i++)
    {
        if(output[i] != (a[i] + b[i]))
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_A2B == 1
template<typename Share>
bool test_A2B()
{

    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;
    const int vectorization_factor = DATTYPE/BITLENGTH;
    
    //initialize plaintext inputs
    UINT_TYPE a[BITLENGTH*vectorization_factor];
    for(int i = 0; i < BITLENGTH*vectorization_factor; i++)
    {
        a[i] = i;
    }
    DATATYPE vectorized_input_a[BITLENGTH];
    orthogonalize_arithmetic(a, vectorized_input_a, BITLENGTH);
    //secret sharing 
    sint share_a;
    share_a.template prepare_receive_from<P_0>(vectorized_input_a);
    Share::communicate();
    share_a.template complete_receive_from<P_0>();
    Share::communicate();

    //A2B
    Bitset s1;
    Bitset s2;

    s1 = Bitset::prepare_A2B_S1(BITLENGTH, (S*) share_a.get_share_pointer());
    s2 = Bitset::prepare_A2B_S2(BITLENGTH, (S*) share_a.get_share_pointer());
    
    Share::communicate();
    
    s1.complete_A2B_S1();
    s2.complete_A2B_S2();
    Share::communicate();
    
    //reveal
    UINT_TYPE a1_output[BITLENGTH*vectorization_factor]; 
    UINT_TYPE a2_output[BITLENGTH*vectorization_factor];
    UINT_TYPE a_output[BITLENGTH*vectorization_factor];
    s1.prepare_reveal_to_all();
    s2.prepare_reveal_to_all();
    /* share_a.prepare_reveal_to_all(); */
    Share::communicate();
    s1.complete_reveal_to_all(a1_output);
    s2.complete_reveal_to_all(a2_output);
   
    //TODO: Check, another ortho should not be neccecary! 
    UINT_TYPE ortho_a1[BITLENGTH]; 
    UINT_TYPE ortho_a2[BITLENGTH];
    unorthogonalize_boolean( (DATATYPE*) a1_output, ortho_a1);
    unorthogonalize_boolean( (DATATYPE*) a2_output, ortho_a2);

    //compare
    for(int i = 0; i < BITLENGTH*vectorization_factor; i++)
        print_compare(a[i], ortho_a1[i] + ortho_a2[i]);
    for(int i = 0; i < BITLENGTH*vectorization_factor; i++)
    {
        if(a[i] != ortho_a1[i] + ortho_a2[i])
        {
            return false;
        }
    }
    return true;
}
#endif

#if TEST_Bit2A == 1
template<typename Share>
bool test_Bit2A()
{

    using S = XOR_Share<DATATYPE, Share>;
    using A = Additive_Share<DATATYPE, Share>;
    using Bitset = sbitset_t<BITLENGTH, S>;
    using sint = sint_t<A>;
    const int vectorization_factor = DATTYPE/BITLENGTH;
    
    S share_a = SET_ALL_ZERO();
    S share_b = SET_ALL_ONE();
    sint output_a;
    sint output_b;

    //Bit2A
    share_a.prepare_bit2a(output_a.get_share_pointer());
    share_b.prepare_bit2a(output_b.get_share_pointer());
    Share::communicate();
    output_a.complete_bit2a();
    output_b.complete_bit2a();
    
    //reveal
    UINT_TYPE ortho_a[DATTYPE];
    UINT_TYPE ortho_b[DATTYPE];
    output_a.prepare_reveal_to_all();
    output_b.prepare_reveal_to_all();
    Share::communicate();
    output_a.complete_reveal_to_all(ortho_a);
    output_b.complete_reveal_to_all(ortho_b);
    //compare
    
    for(int i = 0; i < DATTYPE; i++)
    {
        print_compare(0, ortho_a[i]);
        print_compare(1, ortho_b[i]);
    }
    for(int i = 0; i < DATTYPE; i++)
    {
        if(ortho_a[i] != 0 || ortho_b[i] != 1)
        {
            return false;
        }
    }
    return true;
}
#endif

template<typename Share>
bool test_comparisons(DATATYPE *res)
{
    int num_tests = 0;
    int num_passed = 0;

#if TEST_EQZ == 1
    test_function(num_tests, num_passed, "EQZ", test_EQZ<Share>);
#endif

#if TEST_LTZ == 1
    test_function(num_tests, num_passed, "LTZ", test_LTZ<Share>);
#endif

#if TEST_MAX_MIN == 1
    test_function(num_tests, num_passed, "MAX_MIN", max_min_test<Share>);
#endif

#if TEST_RELU == 1
    test_function(num_tests, num_passed, "RELU", test_RELU<Share>);
#endif

#if TEST_BOOLEAN_ADDITION == 1
    test_function(num_tests, num_passed, "BOOLEAN_ADDITION", test_boolean_addition<Share>);
#endif

#if TEST_A2B == 1
    test_function(num_tests, num_passed, "A2B", test_A2B<Share>);
#endif

#if TEST_Bit2A == 1
    test_function(num_tests, num_passed, "Bit2A", test_Bit2A<Share>);
#endif
    
    print_stats(num_tests, num_passed);
    if(num_tests == num_passed)
    {
        return true;
    }
    return false;
}
