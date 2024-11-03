#pragma once
#include "../../beaver_triples.hpp"
template <typename Datatype>
class ABY2_ONLINE_Share{
using BT = triple<Datatype>;
private:
Datatype m;
Datatype l;
    public:
ABY2_ONLINE_Share()  {}
ABY2_ONLINE_Share(Datatype x) { this->m = x; }
ABY2_ONLINE_Share(Datatype x, Datatype l) { this->m = x; this->l = l; }



template <typename func_mul>
ABY2_ONLINE_Share mult_public(const Datatype b, func_mul MULT) const
{
    return ABY2_ONLINE_Share(MULT(m,b),MULT(l,b));
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_ONLINE_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC, int fractional_bits = FRACTIONAL) const
{
ABY2_ONLINE_Share c;
#if PARTY == 0
c.m = TRUNC(MULT(SUB(m,l),b),fractional_bits); // Share Trunc(mv1 * b)
#else
c.m = SUB(SET_ALL_ZERO(), TRUNC(MULT(l,b),fractional_bits)); // Share Trunc -(lv1 * b)
#endif
c.l = getRandomVal(PSELF);
c.m = ADD(c.m,c.l);
send_to_live(PNEXT, c.m);
return c;
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_ONLINE_Share prepare_div_exp2(const int b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
#if PARTY == 0
    auto result = MULT(SUB(m,l)); // Share Trunc(mv1)
#else
    auto result = l; // Share Trunc - Trunc(lv1)
#endif
    for(int i = 2; i <= b; i*=2)
        result = OP_TRUNC2(result);

#if PARTY == 1
    result = OP_SUB(SET_ALL_ZERO(),result);
#endif

    ABY2_ONLINE_Share res;
    res.l = getRandomVal(PSELF);
    res.m = ADD(result,res.l);
    send_to_live(PNEXT, res.m);
    return res;
} 


    
    template <typename func_add, typename func_sub>
void complete_public_mult_fixed(func_add ADD, func_sub SUB)
{
    Datatype msg = receive_from_live(PNEXT);
    m = ADD(m,msg); //recv Trunc(mv1 * b) - TRunc(lv1 * b)
}

void prepare_opt_bit_injection(ABY2_ONLINE_Share x[], ABY2_ONLINE_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = m; //convert b0 to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);
    Datatype lb[BITLENGTH]{0};
    lb[BITLENGTH - 1] = l;
    unorthogonalize_boolean(lb, temp2);
    orthogonalize_arithmetic(temp2, lb);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype lb = retrieve_output_share_arithmetic();
        Datatype lalb = retrieve_output_share_arithmetic(1);
        auto xim = x[i].m;
        auto xil = x[i].l;
#if PARTY == 0
        out[i].m = OP_MULT(b0[i],xim);
#else
        out[i].m = SET_ALL_ZERO();
#endif
        out[i].m =
                OP_ADD(
                OP_SUB(out[i].m, // mamb 
                OP_MULT(b0[i], xil) ), // - mb [la]
                OP_MULT( OP_SUB(OP_ADD(b0[i], b0[i]),PROMOTE(1)), // + (2mb -1) 
                OP_SUB(lalb, OP_MULT(xim, lb)) )   ); // ([lalb] - ma [lb]) 
        out[i].l = getRandomVal(PSELF); 
        out[i].m = OP_ADD(out[i].m, out[i].l);
        send_to_live(PNEXT, out[i].m);
    }
}



// P_i shares mx - lxi, P_j sets lxj to 0
template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
    {
        l = getRandomVal(PSELF);
        m = ADD(val,l);
        send_to_live(PNEXT, m);
    }
    else
    {
        l = SET_ALL_ZERO();
    }
}
    
template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
        prepare_receive_from<id>(get_input_live(), ADD, SUB);
    else
        prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id != PSELF)
        m = receive_from_live(id);
}

template <typename func_add>
ABY2_ONLINE_Share Add( ABY2_ONLINE_Share b, func_add ADD) const
{
    return ABY2_ONLINE_Share(ADD(m,b.m),ADD(l,b.l));
}

void prepare_reveal_to_all() const
{
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
    return SUB(m,ADD(retrieve_output_share(),l));
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_mult(ABY2_ONLINE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype lxly;
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
   lxly = retrieve_output_share_bool();
else
   lxly = retrieve_output_share_arithmetic();
ABY2_ONLINE_Share c;
c.l = getRandomVal(PSELF);
#if PARTY == 0
c.m = MULT(m,b.m);
#else
c.m = SET_ALL_ZERO();
#endif
c.m = SUB(c.m, SUB(ADD(MULT(m,b.l),MULT(l,b.m)),ADD(lxly,c.l))); // mx my - (mx[ly] + my[lx] - [lxly] - [lz])
send_to_live(PNEXT, c.m);
return c;
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_dot(ABY2_ONLINE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype lxly;
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
   lxly = retrieve_output_share_bool();
else
   lxly = retrieve_output_share_arithmetic();
ABY2_ONLINE_Share c;
/* c.l = getRandomVal(PSELF); */
#if PARTY == 0
c.m = MULT(m,b.m);
#else
c.m = SET_ALL_ZERO();
#endif
c.m = SUB(c.m, SUB(ADD(MULT(m,b.l),MULT(l,b.m)),lxly)); // mx my - (mx[ly] + my[lx] - [lxly])
return c;
}

/* template <typename func_add, typename func_sub, typename func_mul, typename func_trunc> */
/*     ABY2_ONLINE_Share prepare_dot_with_trunc(ABY2_ONLINE_Share b, func_add ADD, func_sub SUB, func_mul MULT, func_trunc TRUNC) const */
/* { */
/* Datatype lxly; */
/* if constexpr(std::is_same_v<func_add(), FUNC_XOR>) */
/*    lxly = retrieve_output_share_bool(); */
/* else */
/*    lxly = retrieve_output_share_arithmetic(); */
/* ABY2_ONLINE_Share c; */
/* c.l = getRandomVal(PSELF); */
/* #if PARTY == 0 */
/* c.m = MULT(m,b.m); */
/* #else */
/* c.m = SET_ALL_ZERO(); */
/* #endif */
/* c.m = SUB(c.m, SUB(ADD(MULT(m,b.l),MULT(l,b.m)),lxly)); // mx my - (mx[ly] + my[lx] - [lxly]) */
/* c.m = ADD(TRUNC(c.m),c.l); // [c] = [c]^t + [lz] */
/* return c; */
/* } */

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
    l = getRandomVal(PSELF);
    m = ADD(m,l);
    send_to_live(PNEXT, m);
}
    
template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
    l = getRandomVal(PSELF);
#if PARTY == 0
    /* m = ADD(TRUNC(m),l); */
    m = ADD(SUB(SET_ALL_ZERO(),TRUNC(SUB(SET_ALL_ZERO(),m))),l); // whyever this is necessary ...
#else
    m = ADD(TRUNC(m),l);
    /* m = ADD(SUB(TRUNC(m), OP_MULT(OP_SHIFT_LOG_RIGHTF(m, BITLENGTH -1), PROMOTE(UINT_TYPE(1) << (BITLENGTH - 1))))   ,l); // x2^t - (x2 > 1) * 2^l */
#endif
    send_to_live(PNEXT, m);
}
    
    
    template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    Datatype msg = receive_from_live(PNEXT);
    m = ADD(m,msg);
}
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
    complete_mult(ADD,SUB);
}

static void prepare_A2B_S1(int m, int k, ABY2_ONLINE_Share in[], ABY2_ONLINE_Share out[])
{
#if PARTY == 0
    Datatype temp_p1[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp_p1[i] = OP_SUB(in[i].m,in[i].l) ; // set first share to mv - lv1
    }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp_p1, temp2);
    orthogonalize_boolean(temp2, temp_p1);
    
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
        out[i-m].m = FUNC_XOR(temp_p1[i],out[i-m].l);
        send_to_live(PNEXT, out[i-m].m);
    }
#endif
}

static void prepare_A2B_S2(int m, int k, ABY2_ONLINE_Share in[], ABY2_ONLINE_Share out[])
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
        out[i-m].m = FUNC_XOR(temp_p1[i],out[i-m].l);
        send_to_live(PNEXT, out[i-m].m);
    }
#endif
}

static void complete_A2B_S1(int k, ABY2_ONLINE_Share out[])
{
#if PARTY == 1
    for(int i = 0; i < k; i++)
    {
        out[i].m = receive_from_live(PNEXT);
        out[i].l = SET_ALL_ZERO();
    }
#endif
}

static void complete_A2B_S2(int k, ABY2_ONLINE_Share out[])
{
#if PARTY == 0
    for(int i = 0; i < k; i++)
    {
        out[i].m = receive_from_live(PNEXT);
        out[i].l = SET_ALL_ZERO();
    }
#endif
}

void prepare_bit2a(ABY2_ONLINE_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = m; //convert b0 to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);
    Datatype lb[BITLENGTH]{0};
    lb[BITLENGTH - 1] = l;
    unorthogonalize_boolean(lb, temp2);
    orthogonalize_arithmetic(temp2, lb);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype lxly = retrieve_output_share_arithmetic();
#if PARTY == 0
        out[i].m = b0[i];
#else
        out[i].m = SET_ALL_ZERO();
#endif
        out[i].m = OP_SUB(out[i].m, OP_MULT( OP_SUB(OP_ADD(b0[i], b0[i]),PROMOTE(1)), OP_SUB(lb[i], OP_ADD(lxly,lxly)))); // mb -  (2mb - 1) * (lvi - 2 lxly)
        out[i].l = getRandomVal(PSELF); 
        out[i].m = OP_ADD(out[i].m, out[i].l);
        send_to_live(PNEXT, out[i].m);
    }
}

void complete_bit2a()
{
        m = OP_ADD(m, receive_from_live(PNEXT));
}

void complete_opt_bit_injection()
{
        m = OP_ADD(m, receive_from_live(PNEXT));
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_dot3(const ABY2_ONLINE_Share b, const ABY2_ONLINE_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{

Datatype rxy = retrieve_output_share_ab(ADD);
Datatype rxyz = retrieve_output_share_ab(ADD,1);
Datatype rxz = retrieve_output_share_ab(ADD);
Datatype ryz = retrieve_output_share_ab(ADD);
/* Datatype rxy = retrieve_output_share_arithmetic(); */
/* Datatype rxyz = retrieve_output_share_arithmetic(1); */
/* Datatype rxz = retrieve_output_share_arithmetic(); */
/* Datatype ryz = retrieve_output_share_arithmetic(); */

ABY2_ONLINE_Share d;
#if PARTY == 1
d.m = 
    ADD(
        ADD( MULT(m,SUB(ryz,MULT(b.m,c.l))) // a [lb lc] - ab [lc]
        ,(MULT(b.m,SUB(rxz, MULT(c.m,l))))) // + b [la lc] - bc [la]
        ,MULT(c.m,SUB(rxy, MULT(m,b.l)))); // + c [la lb] - ac [lb]
#else
d.m = ADD(                                              //abc +
        ADD( MULT(m,ADD(MULT(b.m,SUB(c.m,c.l)),ryz)) // a [lb lc] - ab [lc]
        ,(MULT(b.m,SUB(rxz, MULT(c.m,l))))) // + b [la lc] - bc [la]
        ,MULT(c.m,SUB(rxy, MULT(m,b.l))));  // + c [la lb] - ac [lb]
#endif
d.m = SUB(d.m, rxyz); // -rxyz
return d;
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_dot4(ABY2_ONLINE_Share b, ABY2_ONLINE_Share c, ABY2_ONLINE_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{

Datatype rxy = retrieve_output_share_ab(ADD);
Datatype rzw = retrieve_output_share_ab(ADD);

Datatype rxyz = retrieve_output_share_ab(ADD,1);
Datatype rxyw = retrieve_output_share_ab(ADD,1);
Datatype rxzw = retrieve_output_share_ab(ADD,1);
Datatype ryzw = retrieve_output_share_ab(ADD,1);
Datatype rxyzw = retrieve_output_share_ab(ADD,1);

Datatype rxz = retrieve_output_share_ab(ADD);
Datatype rxw = retrieve_output_share_ab(ADD);
Datatype ryz = retrieve_output_share_ab(ADD);
Datatype ryw = retrieve_output_share_ab(ADD);

    
/* Datatype rxy = retrieve_output_share_arithmetic(); */
/* Datatype rzw = retrieve_output_share_arithmetic(); */

/* Datatype rxyz = retrieve_output_share_arithmetic(1); */
/* Datatype rxyw = retrieve_output_share_arithmetic(1); */
/* Datatype rxzw = retrieve_output_share_arithmetic(1); */
/* Datatype ryzw = retrieve_output_share_arithmetic(1); */
/* Datatype rxyzw = retrieve_output_share_arithmetic(1); */

/* Datatype rxz = retrieve_output_share_arithmetic(); */
/* Datatype rxw = retrieve_output_share_arithmetic(); */
/* Datatype ryz = retrieve_output_share_arithmetic(); */
/* Datatype ryw = retrieve_output_share_arithmetic(); */

ABY2_ONLINE_Share e;
#if PARTY == 1
e.m =     
                ADD(
                    ADD(
                        MULT(m, SUB( MULT(d.m, SUB(ryz, MULT(b.m,c.l))), ryzw))
                        ,
                            MULT(b.m, ADD( MULT(m, SUB(rzw, MULT(c.m,d.l))), 
                            SUB( MULT(c.m, rxw), rxzw)))
                    ),
                    ADD(
                            
                            MULT(c.m, SUB( MULT(m, SUB(ryw, MULT(d.m,b.l))), rxyw))
                            ,
                    
                        
                            MULT(d.m, ADD( MULT(b.m, SUB(rxz, MULT(c.m,l))),
                            SUB( MULT(c.m, rxy), rxyz)))
                )
            );
#else
e.m = 
                ADD(
                    ADD(
                        MULT(m, SUB( MULT(d.m, ADD(MULT(b.m,SUB(c.m,c.l)),ryz )), ryzw))
                            ,
                            MULT(b.m, ADD( MULT(m, SUB(rzw, MULT(c.m,d.l))), 
                            SUB( MULT(c.m, rxw), rxzw)))
                        )
                    ,
                    ADD(
                            MULT(c.m, SUB( MULT(m, SUB(ryw, MULT(d.m,b.l))), rxyw))
                            ,
                            MULT(d.m, ADD( MULT(b.m, SUB(rxz, MULT(c.m,l))),
                            SUB( MULT(c.m, rxy), rxyz)))
                        )
                               
                ); // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw

#endif
e.m = ADD(e.m, rxyzw);
return e;
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_mult3(ABY2_ONLINE_Share b, ABY2_ONLINE_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
    ABY2_ONLINE_Share d = prepare_dot3(b,c,ADD,SUB,MULT);
    d.mask_and_send_dot(ADD,SUB);
    return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
    complete_mult(ADD,SUB);
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_mult4(ABY2_ONLINE_Share b, ABY2_ONLINE_Share c, ABY2_ONLINE_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
    ABY2_ONLINE_Share e = prepare_dot4(b,c,d,ADD,SUB,MULT);
    e.mask_and_send_dot(ADD,SUB);
    return e;
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
    complete_mult(ADD,SUB);
}

static ABY2_ONLINE_Share public_val(Datatype a)
{
    return ABY2_ONLINE_Share(a,SET_ALL_ZERO());
}

ABY2_ONLINE_Share Not() const
{
    return ABY2_ONLINE_Share(NOT(m),l);
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
    communicate_live();
}

// --- Untested Functions --- TODO: Test

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_ONLINE_Share prepare_trunc_share(func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC, int fractional_bits=FRACTIONAL) const
{
#if PARTY == 0
    auto result = TRUNC(SUB(m,l),fractional_bits); // Share Trunc(mv - lv1)
#else
    auto result = OP_SUB(SET_ALL_ZERO(), TRUNC(l,fractional_bits)); // Share Trunc - Trunc(lv1)
#endif
    ABY2_ONLINE_Share res;
    res.l = getRandomVal(PSELF);
    res.m = ADD(result,res.l);
    send_to_live(PNEXT, res.m);
    return res;
} 

void get_random_B2A()
{
        m = SET_ALL_ZERO();
        l = getRandomVal(PSELF);
}


#if USE_CUDA_GEMM == 2
static void CONV_2D(const ABY2_ONLINE_Share* X, const ABY2_ONLINE_Share* W, ABY2_ONLINE_Share* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation = 1){
    const int factor = DATTYPE/BITLENGTH;
    const int xSize = inh * inw * din * batchSize;
    const int wSize = wh * ww * din * dout;
    const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
    const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
    const int ySize = out_h * out_w * dout * batchSize;
    batchSize *= factor; 

    UINT_TYPE* x_p1 = new UINT_TYPE[factor * xSize];
    UINT_TYPE* x_p2 = new UINT_TYPE[factor * xSize];
    UINT_TYPE* w_p1 = new UINT_TYPE[wSize]; // W is always constant
    UINT_TYPE* w_p2 = new UINT_TYPE[wSize];
    UINT_TYPE* y_p1 = new UINT_TYPE[factor * ySize];
    UINT_TYPE* y_p1_2 = new UINT_TYPE[factor * ySize];

    for(int i = 0; i< xSize; i++){
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
        #if PARTY == 0
        auto tempml = OP_SUB(X[i].m,X[i].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
        #else
        unorthogonalize_arithmetic(&X[i].m, temp, 1);
        #endif
        for(int j = 0; j < factor; j++)
            x_p1[j * xSize + i] = temp[j];

        unorthogonalize_arithmetic(&X[i].l, temp, 1);
        for(int j = 0; j < factor; j++)
            x_p2[j * xSize + i] = temp[j];
    }

    for(int i = 0; i< wSize; i++){
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
        #if PARTY == 0
        auto tempml = OP_SUB(W[i].m,W[i].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
        #else
        unorthogonalize_arithmetic(&W[i].m, temp, 1);
        #endif
        w_p1[i] = temp[0];
        unorthogonalize_arithmetic(&W[i].l, temp, 1);
        w_p2[i] = temp[0];
    }
#if PARTY == 0 // (mx - lx0) (mw - lw0) - l0w0
    conv2d_cutlass(x_p1, w_p1, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
    conv2d_cutlass(x_p2, w_p2, y_p1_2, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
#else // - (mx lw1) - (mw - lx1)
    conv2d_cutlass(x_p1, w_p2, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
    conv2d_cutlass(x_p2, w_p1, y_p1_2, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
#endif

    for(int i = 0; i< ySize; i++){
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
        for(int j = 0; j < factor; j++)
#if PARTY == 0
            temp[j] = y_p1[j * ySize + i] - y_p1_2[j * ySize + i];
#else
            temp[j] = - y_p1[j * ySize + i] + y_p1_2[j * ySize + i];
#endif
            orthogonalize_arithmetic(temp, &Y[i].m, 1);
            auto lxly = retrieve_output_share_arithmetic();
            Y[i].m = OP_SUB(Y[i].m, lxly);
    }

    delete[] x_p1;
    delete[] x_p2;
    delete[] w_p1;
    delete[] w_p2;
    delete[] y_p1;
    delete[] y_p1_2;

}

#elif USE_CUDA_GEMM == 4

static void CONV_2D(const ABY2_ONLINE_Share* X, const ABY2_ONLINE_Share* W, ABY2_ONLINE_Share* Y, int batchSize, int inh, int inw, int din, int dout, int wh, int ww, int padding, int stride, int dilation = 1){
    const int factor = DATTYPE/BITLENGTH;
    const int xSize = inh * inw * din * batchSize;
    const int wSize = wh * ww * din * dout;
    const int out_h = (inh + 2 * padding - wh - (wh - 1) * (dilation - 1)) / stride + 1;
    const int out_w = (inw + 2 * padding - ww - (ww - 1) * (dilation - 1)) / stride + 1;
    const int ySize = out_h * out_w * dout * batchSize;
    batchSize *= factor; 

    alignas(sizeof(Datatype)) UINT_TYPE* x_p1 = new UINT_TYPE[factor * xSize];
    alignas(sizeof(Datatype)) UINT_TYPE* x_p2 = new UINT_TYPE[factor * xSize];
    alignas(sizeof(Datatype)) UINT_TYPE* w_p1 = new UINT_TYPE[wSize]; // W is always constant
    alignas(sizeof(Datatype)) UINT_TYPE* w_p2 = new UINT_TYPE[wSize];
    alignas(sizeof(Datatype)) UINT_TYPE* y_p1 = new UINT_TYPE[factor * ySize];
    alignas(sizeof(Datatype)) UINT_TYPE* y_p1_2 = new UINT_TYPE[factor * ySize];

    for(int i = 0; i< xSize; i++){
#if PARTY == 0
        auto tempml = OP_SUB(X[i].m,X[i].l);
        unorthogonalize_arithmetic(&tempml, x_p1 + i * factor, 1);
#else
        unorthogonalize_arithmetic(&X[i].m, x_p1 + i * factor, 1);
#endif
        unorthogonalize_arithmetic(&X[i].l, x_p2 + i * factor, 1);
    }

    for(int i = 0; i< wSize; i++){
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
#if PARTY == 0
        auto tempml = OP_SUB(W[i].m,W[i].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
#else
        unorthogonalize_arithmetic(&W[i].m, temp, 1);
#endif
        w_p1[i] = temp[0];
        unorthogonalize_arithmetic(&W[i].l, temp, 1);
        w_p2[i] = temp[0];
    }
#if PARTY == 0
    conv2d_cutlass(x_p1, w_p1, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
    conv2d_cutlass(x_p2, w_p2, y_p1_2, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
#else
    conv2d_cutlass(x_p1, w_p2, y_p1, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
    conv2d_cutlass(x_p2, w_p1, y_p1_2, batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
#endif

    for(int i = 0; i< ySize; i++){
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
        for(int j = 0; j < factor; j++)
#if PARTY == 0
            temp[j] = y_p1[j * ySize + i] - y_p1_2[j * ySize + i];
#else
            temp[j] = - y_p1[j * ySize + i] - y_p1_2[j * ySize + i];
#endif
        orthogonalize_arithmetic(temp, &Y[i].m, 1);
        auto lxly = retrieve_output_share_arithmetic();
        Y[i].m = OP_SUB(Y[i].m, lxly);
    }

    delete[] x_p1;
    delete[] x_p2;
    delete[] w_p1;
    delete[] w_p2;
    delete[] y_p1;
    delete[] y_p1_2;

}
#endif
#if USE_CUDA_GEMM > 0
#if USE_CUDA_GEMM == 1
    

static void GEMM(ABY2_ONLINE_Share* a, ABY2_ONLINE_Share* b, ABY2_ONLINE_Share* c, int m, int n, int k, bool a_fixed = false)
{
    const int factor = DATTYPE / BITLENGTH;
    const int a_size = m * k;    
    const int b_size = k * n;
    const int c_size = m * n;
    UINT_TYPE* p1;
    UINT_TYPE* p2;
    if(a_fixed)
    {
        p1 = new UINT_TYPE[a_size];
        p2 = new UINT_TYPE[a_size];
    }
    else
    {
        p1 = new UINT_TYPE[factor * a_size];
        p2 = new UINT_TYPE[factor * a_size];
    }
    UINT_TYPE* bp1 = new UINT_TYPE[factor * b_size];
    UINT_TYPE* bp2 = new UINT_TYPE[factor * b_size];
    UINT_TYPE* cp1_1 = new UINT_TYPE[factor * c_size];
    UINT_TYPE* cp1_2 = new UINT_TYPE[factor * c_size];


    for(int i = 0; i < a_size; i++)
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
#if PARTY == 0
        auto tempml = OP_SUB(a[i].m,a[i].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
#else
        unorthogonalize_arithmetic(&a[i].m, temp, 1);
#endif
        if(a_fixed)
        {
            p1[i] = temp[0];

        }
        else
            for(int j = 0; j < factor; j++)
                p1[j*a_size + i] = temp[j];
        unorthogonalize_arithmetic(&a[i].l, temp, 1);
        if(a_fixed)
        {
            p2[i] = temp[0];
        }
        else
            for(int j = 0; j < factor; j++)
                p2[j*a_size + i] = temp[j];
    }

    for(int i = 0; i < b_size; i++)
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
#if PARTY == 0
        auto tempml = OP_SUB(b[i].m,b[i].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
#else
        unorthogonalize_arithmetic(&b[i].m, temp, 1);
#endif
        for(int j = 0; j < factor; j++)
            bp1[j * b_size + i] = temp[j];
        unorthogonalize_arithmetic(&b[i].l, temp, 1);
        for(int j = 0; j < factor; j++)
            bp2[j * b_size + i] = temp[j];
    }


    for (int i = 0; i < factor; i++)
    {
        if(a_fixed)
        {
#if PARTY == 0
            gemm_cutlass(m,n,k,p1, &bp1[i * b_size], &cp1_1[i * c_size]);
            gemm_cutlass(m,n,k,p2, &bp2[i * b_size], &cp1_2[i * c_size]);
#else
            gemm_cutlass(m,n,k,p1, &bp2[i * b_size], &cp1_1[i * c_size]);
            gemm_cutlass(m,n,k,p2, &bp1[i * b_size], &cp1_2[i * c_size]);
#endif
        }
    
        else
        {
#if PARTY == 0
            gemm_cutlass(m,n,k, &p1[i * a_size], &bp1[i * b_size], &cp1_1[i * c_size]);
            gemm_cutlass(m,n,k, &p2[i * a_size], &bp2[i * b_size], &cp1_2[i * c_size]);
#else
            gemm_cutlass(m,n,k, &p1[i * a_size], &bp2[i * b_size], &cp1_1[i * c_size]);
            gemm_cutlass(m,n,k, &p2[i * a_size], &bp1[i * b_size], &cp1_2[i * c_size]);
#endif
        }
    }

    for (int j = 0; j < c_size; j++)
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
        for (int i = 0; i < factor; i++)
#if PARTY == 0
            temp[i] = cp1_1[i * c_size + j] - cp1_1[i * c_size + j];
#else
            temp[i] = - cp1_1[i * c_size + j] - cp1_1[i * c_size + j];
#endif
        orthogonalize_arithmetic(temp, &c[j].m, 1);
        auto lxly = retrieve_output_share_arithmetic();
        c[j].m = OP_SUB(c[j].m, lxly);
    }

    delete[] p1;
    delete[] p2;
    delete[] bp1;
    delete[] bp2;
    delete[] cp1_1;
    delete[] cp1_2;
}
#else
    

static void GEMM(ABY2_ONLINE_Share* a, ABY2_ONLINE_Share* b, ABY2_ONLINE_Share* c, int m, int n, int k, bool a_fixed = false)
{
    const int factor = DATTYPE / BITLENGTH;
    const int a_size = m * k;    
    const int b_size = k * n;
    const int c_size = m * n;
    UINT_TYPE* p1;
    UINT_TYPE* p2;
    if(a_fixed)
    {
        p1 = new UINT_TYPE[a_size];
        p2 = new UINT_TYPE[a_size];
    }
    else
    {
        p1 = new UINT_TYPE[factor * a_size];
        p2 = new UINT_TYPE[factor * a_size];
    }
    UINT_TYPE* bp1 = new UINT_TYPE[factor * b_size];
    UINT_TYPE* bp2 = new UINT_TYPE[factor * b_size];
    UINT_TYPE* cp1_1 = new UINT_TYPE[factor * c_size];
    UINT_TYPE* cp1_2 = new UINT_TYPE[factor * c_size];


    for(int i = 0; i < a_size; i++)
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
#if PARTY == 0
        auto tempml = OP_SUB(a[i].m,a[i].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
#else
        unorthogonalize_arithmetic(&a[i].m, temp, 1);
#endif
        if(a_fixed)
        {
            p1[i] = temp[0];

        }
        else
            for(int j = 0; j < factor; j++)
                p1[j*a_size + i] = temp[j];
        unorthogonalize_arithmetic(&a[i].l, temp, 1);
        if(a_fixed)
        {
            p2[i] = temp[0];
        }
        else
            for(int j = 0; j < factor; j++)
                p2[j*a_size + i] = temp[j];
    }

if(a_fixed)
{

    for(int i = 0; i < k; i++)
    {
        for(int j = 0; j < n; j++)
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
#if PARTY == 0
        auto tempml = OP_SUB(b[i * n + j].m,b[i * n + j].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
#else
        unorthogonalize_arithmetic(&b[i * n + j].m, temp, 1);
#endif
        for(int l = 0; l < factor; l++)
            bp1[i*n*factor + l*n + j] = temp[l];
        unorthogonalize_arithmetic(&b[i * n + j].l, temp, 1);
        for(int l = 0; l < factor; l++)
            bp2[i*n*factor + l*n + j] = temp[l];
    }
    }
#if PARTY == 0
            gemm_cutlass(m,n*factor,k,p1, bp1, cp1_1);
            gemm_cutlass(m,n*factor,k,p2, bp2, cp1_2);
#else
            gemm_cutlass(m,n*factor,k,p1, bp2, cp1_1);
            gemm_cutlass(m,n*factor,k,p2, bp1, cp1_2);
#endif

    for(int i = 0; i < m; i++)
    {
        for(int j = 0; j < n; j++)
        {
            alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
            for(int l = 0; l < factor; l++)
#if PARTY == 0
                temp[l] = cp1_1[i*n*factor + l*n + j] - cp1_2[i*n*factor + l*n + j];
#else
                temp[l] = - cp1_1[i*n*factor + l*n + j] - cp1_2[i*n*factor + l*n + j];
#endif
            orthogonalize_arithmetic(temp, &c[i * n + j].m, 1);
        }
    }

}
else
{
    for(int i = 0; i < b_size; i++)
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
#if PARTY == 0
        auto tempml = OP_SUB(b[i].m,b[i].l);
        unorthogonalize_arithmetic(&tempml, temp, 1);
#else
        unorthogonalize_arithmetic(&b[i].m, temp, 1);
#endif
        for(int j = 0; j < factor; j++)
            bp1[j * b_size + i] = temp[j];
        unorthogonalize_arithmetic(&b[i].l, temp, 1);
        for(int j = 0; j < factor; j++)
            bp2[j * b_size + i] = temp[j];
    }


    for (int i = 0; i < factor; i++)
    {
#if PARTY == 0
        gemm_cutlass(m,n,k, &p1[i * a_size], &bp1[i * b_size], &cp1_1[i * c_size]);
        gemm_cutlass(m,n,k, &p2[i * a_size], &bp2[i * b_size], &cp1_2[i * c_size]);
#else
            gemm_cutlass(m,n,k, &p1[i * a_size], &bp2[i * b_size], &cp1_1[i * c_size]);
            gemm_cutlass(m,n,k, &p2[i * a_size], &bp1[i * b_size], &cp1_2[i * c_size]);
#endif
    }

    for (int j = 0; j < c_size; j++)
    {
        alignas(sizeof(Datatype)) UINT_TYPE temp[factor];
        for (int i = 0; i < factor; i++)
#if PARTY == 0
            temp[i] = cp1_1[i * c_size + j] - cp1_2[i * c_size + j];
#else
            temp[i] = - cp1_1[i * c_size + j] - cp1_2[i * c_size + j];
#endif
        orthogonalize_arithmetic(temp, &c[j].m, 1);
        auto lxly = retrieve_output_share_arithmetic();
        c[j].m = OP_SUB(c[j].m, lxly);
    }

}
    delete[] p1;
    delete[] p2;
    delete[] bp1;
    delete[] bp2;
    delete[] cp1_1;
    delete[] cp1_2;
}
#endif
#endif




};

