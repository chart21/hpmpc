/* #include "../../protocols/Additive_Share.hpp" */
#include "../../protocols/Matrix_Share.hpp"
#define RESULTTYPE DATATYPE
#if FUNCTION_IDENTIFIER == 14
#define FUNCTION dot_prod_bench
#endif
void generateElements()
{}

    /* template<typename Share> */
/* void dot_prod_bench(DATATYPE* res) */
/* { */
/*     Share::communicate(); // dummy round */
/*     using M = Matrix_Share<DATATYPE, Share>; */
/*     /1* using M = Additive_Share<DATATYPE, Share>; *1/ */
/*     auto a = new M[NUM_INPUTS]; */
/*     auto b = new M[NUM_INPUTS][NUM_INPUTS]; */
/*     auto c = new M[NUM_INPUTS]; */
/*     Share::communicate(); // dummy round */
/*     for(int i = 0; i < NUM_INPUTS; i++) */
/*     { */
/* #if FUNCTION_IDENTIFIER == 14 */
/*         for(int j = 0; j < NUM_INPUTS; j++) */
/*         { */
/*             c[i] += a[i].prepare_dot(b[i][j]); */
/*         } */
/* #endif */
/*         c[i].mask_and_send_dot_without_trunc(); */
/*     } */
/*     Share::communicate(); */
/*     for(int i = 0; i < NUM_INPUTS; i++) */
/*     { */
/*             c[i].complete_mult_without_trunc(); */
/*     } */

/*     Share::communicate(); */
/*     c[NUM_INPUTS-1].prepare_reveal_to_all(); */
/*     Share::communicate(); */
/*     *res = c[NUM_INPUTS-1].complete_reveal_to_all(); */

/* delete[] a; */
/* delete[] b; */
/* delete[] c; */

/* } */
    template<typename Share>
void dot_prod_bench(DATATYPE* res)
{
    Share::communicate(); // dummy round
    using M = Matrix_Share<DATATYPE, Share>;
    auto a = new M[NUM_INPUTS];
    auto b = new M[NUM_INPUTS][NUM_INPUTS];
    auto c = new M[NUM_INPUTS];
    Share::communicate(); // dummy round
    for(int i = 0; i < NUM_INPUTS; i++)
    {
#if FUNCTION_IDENTIFIER == 14
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            c[i] += a[i] * b[i][j];
        }
#endif
        c[i].mask_and_send_dot();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
            c[i].complete_mult();
    
}

    Share::communicate();
    c[NUM_INPUTS-1].prepare_reveal_to_all();
    Share::communicate();
    *res = c[NUM_INPUTS-1].complete_reveal_to_all();

delete[] a;
delete[] b;
delete[] c;

}

