
template<typename T>
static void trunc_2k_in_place(T*  val, const int len){
    
    T* r_msb = new T[len];
    T* r_mk2 = new T[len];
    T* c = new T[len];
    T* c_prime = new T[len];
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].prepare_trunc_2k_inputs(r_mk2[i], r_msb[i], c[i], c_prime[i]);
    }
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].complete_trunc_2k_inputs(r_mk2[i], r_msb[i],c[i], c_prime[i]);
    }
    T::communicate();
    T* b = new T[len];
    for(int i = 0; i < len; i++)
        b[i].prepare_XOR(r_msb[i],c[i]);
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        b[i].complete_XOR(r_msb[i],c[i]);
        b[i] = b[i].mult_public(UINT_TYPE(1) << (BITLENGTH - FRACTIONAL - 1));
    }
    T::communicate();
    delete[] c;
    
    for(int i = 0; i < len; i++)
    {
        val[i] = c_prime[i] + b[i] - r_mk2[i];
    }
    delete[] r_mk2;
    delete[] r_msb;
    delete[] c_prime;
    delete[] b;
}

template<typename T>
static void trunc_pr_in_place(T*  val, const int len){
    for(int i = 0; i < len; i++)
    {
        val[i] = val[i].prepare_mult_public_fixed(UINT_TYPE(1)); //multiply by 1 to trigeger truncation
    }
    T::communicate();
    for(int i = 0; i < len; i++)
    {
        val[i].complete_public_mult_fixed();
    }
}

