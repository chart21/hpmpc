
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





