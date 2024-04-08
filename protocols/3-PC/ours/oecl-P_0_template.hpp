#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE OECL0_Share
/* #define VALS_PER_SHARE 2 */
#define SHARE OECL0_Share



template <typename Datatype>
class OECL0_Share 
{
private:
    Datatype p1;
    Datatype p2;
    public:
    //static constexpr int VALS_PER_SHARE = 2;

    OECL0_Share() {}
    OECL0_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}
    OECL0_Share(Datatype p1) : p1(p1) {}

    static OECL0_Share public_val(Datatype a)
{
    return OECL0_Share(SET_ALL_ZERO(),SET_ALL_ZERO());
}



    template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
OECL0_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
#if TRUNC_THEN_MULT == 1
    auto result = MULT(TRUNC(ADD(p1,p2)),b);
#else
    auto result = MULT(ADD(p1,p2),b);
#endif
    OECL0_Share res;
    res.p2 = getRandomVal(P_1);
#if TRUNC_THEN_MULT == 1
    res.p1 = SUB(result,res.p2);
#else
    res.p1 = SUB(TRUNC(result),res.p2);
#endif

#if PRE == 1
    pre_send_to_live(P_2, res.p1);
#else
    send_to_live(P_2, res.p1);
#endif
    return res;
} 

    template <typename func_add, typename func_sub>
void complete_public_mult_fixed(func_add ADD, func_sub SUB)
{
}

OECL0_Share Not() const
{
   return OECL0_Share(p1,p2);
}

template <typename func_add>
OECL0_Share Add(OECL0_Share b, func_add ADD) const
{
   return OECL0_Share(ADD(p1,b.p1),ADD(p2,b.p2));
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_dot_add(OECL0_Share a, OECL0_Share b , OECL0_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c.p1 = ADD(c.p1, SUB( MULT(a.p1,b.p1), MULT( SUB(a.p1,a.p2), SUB(b.p1,b.p2)  )) );
}

template <typename func_add, typename func_sub, typename func_mul>
OECL0_Share prepare_dot(const OECL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OECL0_Share c;
/* #if FRACTIONAL > 0 */
/* c.p1 = SUB( MULT(p1,b.p1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  )); // -> -e = x2y2 - (x1-x2)(y1-y2) = x1 y2 +  x2 y+1 - x1_y1 */
c.p1 = SUB( MULT( SUB(p1,p2), SUB(b.p1,b.p2)), MULT(p1,b.p1)  ); // -> e = (x1-x2)(y1-y2) - x2y2 = x1 y1 - x1 y2 - x2 y1
/* #else */
/* c.p1 = SUB( MULT(p1,b.p1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  )); // e = x2y2 - (x1-x2)(y1-y2) */

/* #endif */
return c;
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
Datatype r01 = getRandomVal(P_1);
Datatype z1 = getRandomVal(P_1);
Datatype z2 = getRandomVal(P_2);
#if PRE == 1
pre_send_to_live(P_2, SUB(r01,p1));
#else
send_to_live(P_2, SUB(r01,p1));
#endif
    p1 = z2;
    p2 = z1;
}
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}

    /* template <typename func_add, typename func_sub, typename func_trunc> */
/* void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
/* { */
/* Datatype maskP_1 = getRandomVal(P_1); */
/* Datatype maskP_1_2 = getRandomVal(P_1); */
/* Datatype maskP_2 = getRandomVal(P_2); */

/* p1 = ADD( TRUNC(ADD(ADD(p1,maskP_1),maskP_2)), maskP_1_2); // (e + r0,1 + r0,2)^t + r0,1_2 */
/* p2 = SUB(SET_ALL_ZERO(),maskP_1_2); // - r0,1_2 */


/* #if PRE == 1 */
/* pre_send_to_live(P_2, p1); */
/* #else */
/* send_to_live(P_2, p1); */
/* #endif */
/* } */
    
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
Datatype maskP_1 = getRandomVal(P_1);
Datatype maskP_1_2 = getRandomVal(P_1);
Datatype maskP_2 = getRandomVal(P_2);

p1 = SUB( TRUNC(ADD(ADD(p1,maskP_1),maskP_2)), maskP_1_2); // (e + r0,1 + r0,2)^t - z_1
p2 = maskP_1_2; // z_1


#if PRE == 1
pre_send_to_live(P_2, p1);
#else
send_to_live(P_2, p1);
#endif
}



template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_mult(OECL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype maskP_1 = getRandomVal(P_1);
/* Datatype maskP_1_2 = getRandomVal(P_1); */
/* Datatype maskP_2 = getRandomVal(P_2); */
#if PRE == 1
pre_send_to_live(P_2, SUB( ADD(MULT(p1,b.p1),maskP_1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  ))); 
#else
send_to_live(P_2, SUB( ADD(MULT(p1,b.p1),maskP_1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  ))); 
#endif
// for arithmetic circuits this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 + b.p2)
return OECL0_Share(getRandomVal(P_2),getRandomVal(P_1));
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}

void prepare_reveal_to_all() const
{
        #if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
    pre_send_to_live(P_1, p1);
    pre_send_to_live(P_2, p2);
    #else
    send_to_live(P_1, p1);
    send_to_live(P_2, p2);
#endif
}    



template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share(p2);
#endif
#if PRE == 1
    return p1;
#else
return SUB(receive_from_live(P_2),p2);
#endif

}

template <int id,typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
if constexpr(id == P_0)
{
#if OPT_SHARE == 1
    p2 = getRandomVal(P_1); // r0,1
    p1 = SUB(SET_ALL_ZERO(), ADD(val,p2)); // share -(a + r0,1)
    #if PRE == 1 && SHARE_PREP == 1
        pre_send_to_live(P_2, p1); // share -(a + r0,1) to P_2
    #else
        send_to_live(P_2, p1);
    #endif
#else
    p1 = getRandomVal(P_2); // P_1 does not need to the share -> thus not srng but 2 -> with updated share conversion it needs it
    p2 = getRandomVal(P_1);
    Datatype input = val;
    #if PRE == 1
    pre_send_to_live(P_1, ADD(p1,input));
    pre_send_to_live(P_2, ADD(p2,input));
    #else
    send_to_live(P_1, ADD(p1,input));
    send_to_live(P_2, ADD(p2,input));
    #endif
#endif
}
else if constexpr(id == P_1){
    p1 = SET_ALL_ZERO();
    p2 = getRandomVal(P_1);
}
else if constexpr(id == P_2)// id ==2
{
    p1 = getRandomVal(P_2);
    p2 = SET_ALL_ZERO();
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
}

template <typename func_mul>
OECL0_Share mult_public(const Datatype b, func_mul MULT) const
{
    return OECL0_Share(MULT(p1,b),MULT(p2,b));
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
#if PRE == 0
    communicate_live();
#endif
}

void get_random_B2A()
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        p1 = getRandomVal(P_2);
        p2 = getRandomVal(P_1);
    }
}

static void prepare_A2B_S1(int m, int k, OECL0_Share in[], OECL0_Share out[])
{
    for(int i = m; i < k; i++)
    {
        /* out[i].p1 = getRandomVal(P_2); // set share to r0,2 */ 
        out[i-m].p1 = SET_ALL_ZERO(); // set share to 0
        out[i-m].p2 = SET_ALL_ZERO(); // set other share to 0
    }
}


/* static void get_dabit(OECL0_Share bool_out, OECL0_Share arith_out[]) */
/* { */
/*     Datatype temp[BITLENGTH]{0}; */
/*     bool_out.p1 = getRandomVal(P_2); */
/*     bool_out.p2 = getRandomVal(P_1); */
/*     temp[BITLENGTH - 1] = FUNC_XOR(bool_out.p1,bool_out.p2); */
/*     alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE]; */
/*     unorthogonalize_boolean(temp, temp2); */
/*     orthogonalize_arithmetic(temp2, temp); */
/*     for(int i = 0; i < BITLENGTH; i++) */
/*     { */
/*         arith_out[i].p2 = getRandomVal(P_1); // set second share to r0,1 */
/*         arith_out[i].p1 = OP_SUB(SET_ALL_ZERO(), OP_ADD(temp[i], out[i].p2)) ; // set first share to -(x0 + r0,1) */
/*         #if PRE == 1 */
/*             pre_send_to_live(P_2, arith_out[i].p1); //  - (x0 + r0,1) to P_2 */
/*         #else */
/*             send_to_live(P_2, arith_out[i].p1); // - (x0 + r0,1) to P_2 */
/*         #endif */
        
/*     } */
/* } */

static void prepare_A2B_S2(int m, int k, OECL0_Share in[], OECL0_Share out[])
{
    //convert share x0 to boolean
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = OP_SUB(SET_ALL_ZERO(), OP_ADD(in[j].p1, in[j].p2)); // set share to -x0
        }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp, temp2);
    orthogonalize_boolean(temp2, temp);
    /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
    /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */

    for(int i = m; i < k; i++)
    {
            out[i-m].p2 = getRandomVal(P_1); // set second share to r0,1
            out[i-m].p1 = FUNC_XOR(temp[i],out[i-m].p2); // set first share to -x0 xor r0,1
            #if PRE == 1
                pre_send_to_live(P_2, out[i-m].p1); // -x0 xor r0,1 to P_2
            #else
                send_to_live(P_2, out[i-m].p1); // -x0 xor r0,1 to P_2
            #endif
    } 
            /* out[0].p1 = FUNC_NOT(out[0].p1);// change sign bit -> -x0 xor r0,1 to x0 xor r0,1 */
}

static void complete_A2B_S1(int k, OECL0_Share out[])
{

}
static void complete_A2B_S2(int k, OECL0_Share out[])
{

}


static void prepare_B2A( OECL0_Share z[], OECL0_Share random_mask[], OECL0_Share out[])
{
    // 1. Reveal z to P_1 and P_2
    for(int i = 0; i < BITLENGTH; i++)
    {
        /* send_to_live(P_1, z[i].p1); */
        /* send_to_live(P_2, z[i].p2); */
        z[i].p1 = SET_ALL_ZERO(); //set mask to 0 since it is reveald
        z[i].p2 = SET_ALL_ZERO(); //set mask to 0 since it is reveald
    }
    // 2. Share random mask
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = FUNC_XOR(random_mask[j].p1, random_mask[j].p2); // set share to r01 xor r02
        }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(temp, temp2);
    orthogonalize_arithmetic(temp2, temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].template prepare_receive_from<P_0>(temp[i], OP_ADD, OP_SUB);
    } 


}

static void complete_B2A(OECL0_Share z[], OECL0_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
        out[i] = z[i].Add(out[i], OP_SUB); // calculate z - randmon mask
    }

}

void prepare_opt_bit_injection(OECL0_Share x[], OECL0_Share out[])
{
    Datatype y0[BITLENGTH]{0};
    y0[BITLENGTH - 1] = FUNC_XOR(p1,p2); //convert b to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(y0, temp2);
    orthogonalize_arithmetic(temp2, y0);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype r01 = getRandomVal(P_1);
        Datatype r01_2 = getRandomVal(P_1);
        Datatype m00 = OP_SUB(y0[i], r01);
        Datatype m01 = OP_SUB(OP_MULT(OP_ADD(x[i].p1,x[i].p2), y0[i]), r01_2);
#if PRE == 1
        pre_send_to_live(P_2, m00);
        pre_send_to_live(P_2, m01);
#else
        send_to_live(P_2, m00);
        send_to_live(P_2, m01);
#endif
        out[i].p1 = getRandomVal(P_2); // set share to z_2
        out[i].p2 = getRandomVal(P_1); // set other share to z_1
    }
}

void complete_opt_bit_injection()
{
}

void prepare_bit2a(OECL0_Share out[])
{
    Datatype temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = FUNC_XOR(p1,p2); // convert y_0 to an arithmetic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(temp, temp2);
    orthogonalize_arithmetic(temp2, temp);

    for(int i = 0; i < BITLENGTH; i++)
    {
#if PRE == 1
        pre_send_to_live(P_2, OP_SUB(temp[i], getRandomVal(P_1))); // send y_0 - r_0,1 to P_2
#else
        send_to_live(P_2, OP_SUB(temp[i], getRandomVal(P_1))); // send y_0 - r_0,1 to P_2
#endif
        out[i].p1 = getRandomVal(P_2); // set share to z_2
        out[i].p2 = getRandomVal(P_1); // set other share to z_1
    }
}

void complete_bit2a()
{
}

void prepare_bit_injection_S1(OECL0_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = SET_ALL_ZERO(); // set share to 0
        out[i].p2 = SET_ALL_ZERO(); // set other share to 0
    }
}

void prepare_bit_injection_S2(OECL0_Share out[])
{
    Datatype temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = FUNC_XOR(p1,p2);
    /* unorthogonalize_boolean(temp,(UINT_TYPE*)temp); */
    /* orthogonalize_arithmetic((UINT_TYPE*) temp,  temp); */
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(temp, temp2);
    orthogonalize_arithmetic(temp2, temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p2 = getRandomVal(P_1); // set second share to r0,1 
        out[i].p1 = OP_SUB(SET_ALL_ZERO(), OP_ADD(temp[i], out[i].p2)) ; // set first share to -(x0 + r0,1)
        #if PRE == 1
            pre_send_to_live(P_2, out[i].p1); //  - (x0 + r0,1) to P_2
        #else
            send_to_live(P_2, out[i].p1); // - (x0 + r0,1) to P_2
        #endif
        
    }
}



static void complete_bit_injection_S1(OECL0_Share out[])
{
    
}

static void complete_bit_injection_S2(OECL0_Share out[])
{


}

template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_dot3(const OECL0_Share b, const OECL0_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype x0 = ADD(p1,p2);
Datatype y0 = ADD(b.p1,b.p2);
Datatype z0 = ADD(c.p1,c.p2);
Datatype mxy = SUB(MULT(x0,y0),getRandomVal(P_1));
Datatype mxz = SUB(MULT(x0,z0),getRandomVal(P_1));
Datatype myz = SUB(MULT(y0,z0),getRandomVal(P_1));
Datatype mxyz = MULT(MULT(x0,y0),z0);
#if PRE == 1
pre_send_to_live(P_2, mxy);
pre_send_to_live(P_2, mxz);
pre_send_to_live(P_2, myz);
#else
send_to_live(P_2, mxy);
send_to_live(P_2, mxz);
send_to_live(P_2, myz);
#endif
// for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 + b.p2)
return OECL0_Share(mxyz,SET_ALL_ZERO());
}
template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_dot4(OECL0_Share b, OECL0_Share c, OECL0_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype x0 = ADD(p1,p2);
Datatype y0 = ADD(b.p1,b.p2);
Datatype z0 = ADD(c.p1,c.p2);
Datatype w0 = ADD(d.p1,d.p2);
Datatype xy = MULT(x0,y0);
Datatype xz = MULT(x0,z0);
Datatype xw = MULT(x0,w0);
Datatype yz = MULT(y0,z0);
Datatype yw = MULT(y0,w0);
Datatype zw = MULT(z0,w0);
Datatype mxy = SUB(xy,getRandomVal(P_1));
Datatype mxz = SUB(xz,getRandomVal(P_1));
Datatype mxw = SUB(xw,getRandomVal(P_1));
Datatype myz = SUB(yz,getRandomVal(P_1));
Datatype myw = SUB(yw,getRandomVal(P_1));
Datatype mzw = SUB(zw,getRandomVal(P_1));
Datatype mxyz = SUB(MULT(xy,z0),getRandomVal(P_1));
Datatype mxyw = SUB(MULT(xy,w0),getRandomVal(P_1));
Datatype mxzw = SUB(MULT(xz,w0),getRandomVal(P_1));
Datatype myzw = SUB(MULT(yz,w0),getRandomVal(P_1));
Datatype mxyzw = MULT(xy,zw);
mxyzw = SUB(SET_ALL_ZERO(),mxyzw); // trick do be comptatible with 2PC dot product
#if PRE == 1
pre_send_to_live(P_2, mxy);
pre_send_to_live(P_2, mxz);
pre_send_to_live(P_2, mxw);
pre_send_to_live(P_2, myz);
pre_send_to_live(P_2, myw);
pre_send_to_live(P_2, mzw);
pre_send_to_live(P_2, mxyz);
pre_send_to_live(P_2, mxyw);
pre_send_to_live(P_2, mxzw);
pre_send_to_live(P_2, myzw);
/* pre_send_to_live(P_2, mxyzw); */
#else
send_to_live(P_2, mxy);
send_to_live(P_2, mxz);
send_to_live(P_2, mxw);
send_to_live(P_2, myz);
send_to_live(P_2, myw);
send_to_live(P_2, mzw);
send_to_live(P_2, mxyz);
send_to_live(P_2, mxyw);
send_to_live(P_2, mxzw);
send_to_live(P_2, myzw);
/* send_to_live(P_2, mxyzw); */
#endif
// for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 + b.p2)
return OECL0_Share(mxyzw,SET_ALL_ZERO());
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_mult3(OECL0_Share b, OECL0_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype x0 = ADD(p1,p2);
Datatype y0 = ADD(b.p1,b.p2);
Datatype z0 = ADD(c.p1,c.p2);
Datatype mxy = SUB(MULT(x0,y0),getRandomVal(P_1));
Datatype mxz = SUB(MULT(x0,z0),getRandomVal(P_1));
Datatype myz = SUB(MULT(y0,z0),getRandomVal(P_1));
Datatype mxyz = SUB(MULT(MULT(x0,y0),z0),getRandomVal(P_1));
#if PRE == 1
pre_send_to_live(P_2, mxy);
pre_send_to_live(P_2, mxz);
pre_send_to_live(P_2, myz);
pre_send_to_live(P_2, mxyz);
#else
send_to_live(P_2, mxy);
send_to_live(P_2, mxz);
send_to_live(P_2, myz);
send_to_live(P_2, mxyz);
#endif
// for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 + b.p2)
return OECL0_Share(getRandomVal(P_2),getRandomVal(P_1));
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){}

template <typename func_add, typename func_sub, typename func_mul>
    OECL0_Share prepare_mult4(OECL0_Share b, OECL0_Share c, OECL0_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype x0 = ADD(p1,p2);
Datatype y0 = ADD(b.p1,b.p2);
Datatype z0 = ADD(c.p1,c.p2);
Datatype w0 = ADD(d.p1,d.p2);
Datatype xy = MULT(x0,y0);
Datatype xz = MULT(x0,z0);
Datatype xw = MULT(x0,w0);
Datatype yz = MULT(y0,z0);
Datatype yw = MULT(y0,w0);
Datatype zw = MULT(z0,w0);
Datatype mxy = SUB(xy,getRandomVal(P_1));
Datatype mxz = SUB(xz,getRandomVal(P_1));
Datatype mxw = SUB(xw,getRandomVal(P_1));
Datatype myz = SUB(yz,getRandomVal(P_1));
Datatype myw = SUB(yw,getRandomVal(P_1));
Datatype mzw = SUB(zw,getRandomVal(P_1));
Datatype mxyz = SUB(MULT(xy,z0),getRandomVal(P_1));
Datatype mxyw = SUB(MULT(xy,w0),getRandomVal(P_1));
Datatype mxzw = SUB(MULT(xz,w0),getRandomVal(P_1));
Datatype myzw = SUB(MULT(yz,w0),getRandomVal(P_1));
Datatype mxyzw = SUB(MULT(xy,zw),getRandomVal(P_1));
#if PRE == 1
pre_send_to_live(P_2, mxy);
pre_send_to_live(P_2, mxz);
pre_send_to_live(P_2, mxw);
pre_send_to_live(P_2, myz);
pre_send_to_live(P_2, myw);
pre_send_to_live(P_2, mzw);
pre_send_to_live(P_2, mxyz);
pre_send_to_live(P_2, mxyw);
pre_send_to_live(P_2, mxzw);
pre_send_to_live(P_2, myzw);
pre_send_to_live(P_2, mxyzw);
#else
send_to_live(P_2, mxy);
send_to_live(P_2, mxz);
send_to_live(P_2, mxw);
send_to_live(P_2, myz);
send_to_live(P_2, myw);
send_to_live(P_2, mzw);
send_to_live(P_2, mxyz);
send_to_live(P_2, mxyw);
send_to_live(P_2, mxzw);
send_to_live(P_2, myzw);
send_to_live(P_2, mxyzw);
#endif
// for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 + b.p2)
return OECL0_Share(getRandomVal(P_2),getRandomVal(P_1));
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL0_Share& r_mk2, OECL0_Share& r_msb, OECL0_Share& c, OECL0_Share& c_prime) const{
    /* Datatype rmk2 = (ADD(p1,p2) << 1) >> (FRACTIONAL + 1); */
    /* Datatype rmsb = ADD(p1,p2) >> (BITLENGTH - 1); */
    Datatype rmk2 = OP_SHIFT_LOG_RIGHT<FRACTIONAL+1>( OP_SHIFT_LEFT<1>(ADD(p1,p2)) );
    Datatype rmsb = OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(ADD(p1,p2));
    /* Datatype rmk2 = ADD(p1,p2); */    
    /* Datatype rmsb = ADD(p1,p2); */

    /* Datatype rmk2 = (ADD(p1,p2) << 1) >> (FRACTIONAL + 1); */
    /* Datatype rmsb = ADD(p1,p2) >> (BITLENGTH - 1); */
    r_mk2.prepare_receive_from<PSELF>(rmk2, ADD, SUB);
    r_msb.prepare_receive_from<PSELF>(rmsb, ADD, SUB);
    /* r_mk2.template prepare_receive_from<PSELF>(400, ADD, SUB); */
    /* r_msb.template prepare_receive_from<PSELF>(600, ADD, SUB); */
    c.p1 = SET_ALL_ZERO();
    c.p2 = SET_ALL_ZERO();
    c_prime.p1 = SET_ALL_ZERO();
    c_prime.p2 = SET_ALL_ZERO();
}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL0_Share& r_mk2, OECL0_Share& r_msb, OECL0_Share& c, OECL0_Share& c_prime) const{
    r_mk2.template complete_receive_from<PSELF>(ADD, SUB);
    r_msb.template complete_receive_from<PSELF>(ADD, SUB);
}



};

