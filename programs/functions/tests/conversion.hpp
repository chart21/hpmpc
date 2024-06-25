template<typename Share>
void argmax_test(DATATYPE* res)
{
using S = XOR_Share<DATATYPE, Share>;
using A = Additive_Share<DATATYPE, Share>;
using sint = sint_t<A>;
const int k = BITLENGTH;
auto a = new sint[NUM_INPUTS];
auto max_output = new S[NUM_INPUTS];
auto min_output = new S[NUM_INPUTS];
for(int i = 0; i < NUM_INPUTS; i++)
        a[i]. template prepare_receive_from<P_2>();
Share::communicate();
for(int i = 0; i < NUM_INPUTS; i++)
        a[i]. template complete_receive_from<P_2>();
Share::communicate();
                /* #if PARTY == 2 */
        /* for(int i = 0; i < NUM_INPUTS; i++) */
            /* std::cout << "a: " << a[i].get_p1() << std::endl; */
        /* #endif */ 
argmax_argmin<k>(a, a+NUM_INPUTS, max_output,true);
argmax_argmin<k>(a, a+NUM_INPUTS, min_output,false);
auto max_val = max_min<k>(a, a+NUM_INPUTS, true);
auto min_val = max_min<k>(a, a+NUM_INPUTS, false);
for(int i = 0; i < NUM_INPUTS; i++)
{
        max_output[i].prepare_reveal_to_all();
        min_output[i].prepare_reveal_to_all();
}
        max_val.prepare_reveal_to_all();
        min_val.prepare_reveal_to_all();
Share::communicate();
auto result_arr = new DATATYPE[2][NUM_INPUTS];
for(int i = 0; i < NUM_INPUTS; i++)
{
        result_arr[0][i] = max_output[i].complete_reveal_to_all();
        result_arr[1][i] = min_output[i].complete_reveal_to_all();
}
auto max_int = NEW( UINT_TYPE[BITLENGTH * sizeof(DATATYPE)/sizeof(UINT_TYPE)]);
auto min_int = NEW( UINT_TYPE[BITLENGTH * sizeof(DATATYPE)/sizeof(UINT_TYPE)]);
max_val.complete_reveal_to_all(max_int);
min_val.complete_reveal_to_all(min_int);
if(current_phase == PHASE_LIVE)
{
#if DATTYPE <= 64
for(int i = 0; i < NUM_INPUTS; i++)
    std::cout << "arg_max: " << "Index: " << i << " Value: " << result_arr[0][i] << std::endl;
for(int i = 0; i < NUM_INPUTS; i++)
    std::cout << "arg_min: " << "Index: " << i << " Value: "<< result_arr[1][i] << std::endl;
#endif
std::cout << "max: " << max_int[0] << std::endl;
std::cout << "min: " << min_int[0] << std::endl;
}
delete[] a;
delete[] max_output;
delete[] min_output;
delete[] result_arr;
}


template<typename Share>
void test_comp_trunc(DATATYPE *res)
{
    using A = Additive_Share<DATATYPE, Share>;
    using sint = sint_t<A>;

    //Syntax for additive shares
    A* input = new A[NUM_INPUTS];
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        if(i % 10 == 0)
            input[i] = PROMOTE(0);
        else if(i % 2 == 0)
            input[i] = PROMOTE(i);
        else
            input[i] = PROMOTE(-i);
    }
    A* ltz_output = new A[NUM_INPUTS];
    A* eqz_output = new A[NUM_INPUTS];
    pack_additive<0, BITLENGTH>(input, ltz_output, NUM_INPUTS, LTZ<0, BITLENGTH, Share, DATATYPE>); //LTZ
    pack_additive<0, BITLENGTH>(input, eqz_output, NUM_INPUTS, EQZ<0, BITLENGTH, Share, DATATYPE>); //EQZ
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        input[i].prepare_reveal_to_all();
        ltz_output[i].prepare_reveal_to_all();
        eqz_output[i].prepare_reveal_to_all();
    } 
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        auto inp = input[i].complete_reveal_to_all();
        auto res = ltz_output[i].complete_reveal_to_all();
        auto res2 = eqz_output[i].complete_reveal_to_all();
        UINT_TYPE uint_inp[DATTYPE/BITLENGTH];
        UINT_TYPE ltz_res[DATTYPE/BITLENGTH];
        UINT_TYPE eqz_res[DATTYPE/BITLENGTH];
        unorthogonalize_arithmetic(&inp, uint_inp, 1);
        unorthogonalize_arithmetic(&res, ltz_res, 1);
        unorthogonalize_arithmetic(&res2, eqz_res, 1);
        for(int j = 0; j < DATTYPE/BITLENGTH; j++)
        {
            std::cout << "Result: " << "input: " << INT_TYPE(uint_inp[j]) << " LTZ: " << ltz_res[j] << " EQZ: " << eqz_res[j] << std::endl;
        }

    }

    /* trunc_pr<A>(input, output, NUM_INPUTS); */
    delete[] input;
    delete[] ltz_output;
    delete[] eqz_output;

    //Syntax for sint
    /* sint* sint_input = new sint[NUM_INPUTS]; */
    /* sint* sint_output = new sint[NUM_INPUTS]; */
    /* LTZ<0, BITLENGTH, Share, DATATYPE>(sint_input, sint_output, NUM_INPUTS); */
    /* EQZ<0, BITLENGTH, Share, DATATYPE>(sint_input, sint_output, NUM_INPUTS); */
    /* trunc_pr<sint>(sint_input, sint_output, NUM_INPUTS); */
    /* delete[] sint_input; */
    /* delete[] sint_output; */
}





