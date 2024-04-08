#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL1_Share
{
Datatype p1;
Datatype p2;
public:
OECL1_Share() {}
OECL1_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}
OECL1_Share(Datatype p1) : p1(p1) {}

static OECL1_Share public_val(Datatype a)
{
    return OECL1_Share(a,SET_ALL_ZERO());
}

OECL1_Share Not() const
{
   return OECL1_Share(NOT(p1),p2);
}

template <typename func_add>
OECL1_Share Add(OECL1_Share b, func_add ADD) const
{
   return OECL1_Share(ADD(p1,b.p1),ADD(p2,b.p2));
}
    
template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
OECL1_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
#if TRUNC_THEN_MULT == 1
    auto result = MULT(TRUNC(ADD(p1,p2)),b);
#else
    auto result = TRUNC(MULT(ADD(p1,p2),b));
#endif

    OECL1_Share res;
    res.p2 = getRandomVal(P_0);
    res.p1 = SUB(result,res.p2);
    return res;
} 
    
    template <typename func_add, typename func_sub>
void complete_public_mult_fixed(func_add ADD, func_sub SUB)
{
}


/* template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc> */
/* OECL1_Share prepare_trunc_2k(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc) const{ */
/*     Datatype c = ADD(p1,p2); //open c = x + r */
/*     Datatype c_prime = trunc(c); */
/*     UINT_TYPE maskValue = (1 << (BITLENGTH-FRACTIONAL-1)) - 1; */ 
/*     Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue */
/*     // Apply the mask using bitwise AND */
/*     c_prime = AND(c_prime, mask); //mod 2^k-m-1 */
/*     c_prime = c_prime % (1 << (BITLENGTH-FRACTIONAL-1)); */
/*     Datatype r = getRandomVal(P_0); */
/*     /1* Datatype b = r >> (BITLENGTH-1); // msb (Only one party xors with c) *1/ */
/*     Datatype b = XOR(r >> (BITLENGTH-1), c >> (BITLENGTH-1)); // xor msbs */
/*     Datatype m = SUB( b << (BITLENGTH-FRACTIONAL-1), (r << (1)) >> (FRACTIONAL+1)); */
/*     c_prime = ADD(c_prime,m); */
/*     Datatype z = getRandomVal(P_0); */
/*     send_to_live(P_2,ADD(m,z)); */
/*     return OECL1_Share(c_prime,z); */
/* } */


    template <typename func_mul>
OECL1_Share mult_public(const Datatype b, func_mul MULT) const
{
    return OECL1_Share(MULT(p1,b),MULT(p2,b));
}



    template <typename func_add, typename func_sub, typename func_mul>
void prepare_dot_add(OECL1_Share a, OECL1_Share b , OECL1_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c.p1 = ADD(c.p1, ADD(MULT(a.p1,b.p2), MULT(b.p1,a.p2)));
}
    template <typename func_add, typename func_sub, typename func_mul>
OECL1_Share prepare_dot(const OECL1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OECL1_Share c;
c.p1 = ADD(MULT(p1,b.p2), MULT(b.p1,p2)); // ab_2, e_1 = x1 y2 + x2 y1 -> since substraction: e_1 = - x1 y2 - x2 y1
return c;
}

template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
    p1 = ADD(getRandomVal(P_0), p1);
    p2 = getRandomVal(P_0);
    send_to_live(P_2,SUB(p1,p2));
}
    
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
Datatype maskP_1 = getRandomVal(P_0);

p1 = SUB(p1, maskP_1); // a2y1 + b2x1 - r0,1
p2 = getRandomVal(P_0); // z_1

send_to_live(P_2, p1);


}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
p1 = SUB( TRUNC( SUB(receive_from_live(P_2),p1)), p2 ); // (ab + e + r01 + r0,2)^T + r0,1_2
/* p2 = p2; // z_1 */
}


    /* template <typename func_add, typename func_sub, typename func_trunc> */
/* void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
/* { */
/* Datatype maskP_1 = getRandomVal(P_0); */

/* p1 = SUB(p1, maskP_1); // - ab_1 - e_1 - r0,1 */
/* p2 = getRandomVal(P_0); // r0,1_2 */

/* send_to_live(P_2, p1); */


/* } */

    /* template <typename func_add, typename func_sub, typename func_trunc> */
/* void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
/* { */
/* p1 = ADD( TRUNC( SUB(receive_from_live(P_2),p1)), p2 ); // (ab + e + r01 + r0,2)^T + r0,1_2 */
/* p2 = SUB(SET_ALL_ZERO(), p2); // - r0,1_2 */
/* } */


template <typename func_add, typename func_sub, typename func_mul>
    OECL1_Share prepare_mult(OECL1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
/* OECL1_Share c; */
Datatype cp1 = ADD(getRandomVal(P_0), ADD(MULT(p1,b.p2), MULT(b.p1,p2))); //remove P_1_mask, then (a+ra)rl + (b+rb)rr 
Datatype cp2 = getRandomVal(P_0); //generate P_1_2 mask
send_to_live(P_2,SUB(cp1,cp2)); 
return OECL1_Share(cp1,cp2);
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
p1 = SUB(receive_from_live(P_2),p1);
}

void prepare_reveal_to_all() const
{
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
return SUB(p1, pre_receive_from_live(P_0));
#else
return SUB(p1, receive_from_live(P_0));
#endif

}




template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
if constexpr(id == P_0)
{
        p2 = getRandomVal(P_0);
        p1 = SUB(SET_ALL_ZERO(), p2); // set p1 to - r0,1
}
else if constexpr(id == P_1)
{
    p1 = val;
    p2 = getRandomVal(P_0);
    send_to_live(P_2,ADD(p1,p2));
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    
#if OPT_SHARE == 0
if constexpr(id == P_0)
{
    #if PRE == 1 && SHARE_PREP == 1
        p1 = pre_receive_from_live(P_0);
    #else
        p1 = receive_from_live(P_0);
    #endif
}
#endif 
if constexpr(id == P_2)
{
p1 = receive_from_live(P_2);
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

void get_random_B2A()
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        p1 = getRandomVal(P_0);
        p2 = p1;
    }
}


static void prepare_A2B_S1(int m, int k, OECL1_Share in[], OECL1_Share out[])
{
    Datatype temp_p1[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp_p1[i] = OP_ADD(in[i].p1,in[i].p2) ; // set first share to a+x_0
    }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp_p1, temp2);
    orthogonalize_boolean(temp2, temp_p1);
    /* unorthogonalize_arithmetic(temp_p1, (UINT_TYPE*) temp_p1); */
    /* orthogonalize_boolean((UINT_TYPE*) temp_p1, temp_p1); */
    
    for(int i = m; i < k; i++)
    {
        out[i-m].p1 = temp_p1[i];
        out[i-m].p2 = SET_ALL_ZERO(); // set other share to 0
    }

}


static void prepare_A2B_S2(int m, int k, const OECL1_Share in[], OECL1_Share out[])
{

    for(int i = m; i < k; i++)
    {
        out[i-m].p1 = getRandomVal(P_0); 
        out[i-m].p2 = out[i-m].p1; // set both shares to r0,1
    } 
}
static void complete_A2B_S1(int k, OECL1_Share out[])
{
}

static void complete_A2B_S2( int k, OECL1_Share out[])
{
}

static void prepare_B2A( OECL1_Share z[], OECL1_Share random_mask[], OECL1_Share out[])
{
    // 1. Reveal z to P_1 and P_2
    for(int i = 0; i < BITLENGTH; i++)
    {
        /* send_to_live(P_1, z[i].p1); */
        send_to_live(P_2, z[i].p2); //reveal z to P_2
        z[i].p2 = SET_ALL_ZERO(); //set mask to 0 since it is reveald
    }
    // 2. Share random mask
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].template prepare_receive_from<P_0>(OP_ADD, OP_SUB);
    } 


}

static void complete_B2A(OECL1_Share z_bool[], OECL1_Share out[])
{
    Datatype z[BITLENGTH]; 
    for(int i = 0; i < BITLENGTH; i++)
        z[i] = FUNC_XOR(z_bool[i].p1, receive_from_live(P_2));
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(z, temp2);
    orthogonalize_arithmetic(temp2, z);

    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
        out[i] = public_val(z[i]).Add(out[i], OP_SUB); // z - r
    }

}

void prepare_opt_bit_injection(OECL1_Share a[], OECL1_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = FUNC_XOR(p1,p2); //convert b to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype a0 = OP_ADD(a[i].p1,a[i].p2);
        Datatype r01 = getRandomVal(P_0);
        Datatype r01_2 = getRandomVal(P_0);
        Datatype z1 = getRandomVal(P_0);
        Datatype m1 = OP_SUB(OP_ADD(b0[i], b0[i]), PROMOTE(1));
        m1 = OP_MULT(m1, OP_SUB(r01_2, OP_MULT(a0, r01)));
        m1 = OP_SUB(m1, OP_MULT(b0[i], a[i].p2));
        send_to_live(P_2, OP_ADD(m1, z1));
        out[i].p1 = OP_ADD(m1,OP_MULT(a0, b0[i]));
        out[i].p2 = z1;
    }
}

void complete_opt_bit_injection()
{
        p1 = OP_ADD(p1, receive_from_live(P_2));
}
void prepare_bit2a(OECL1_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = FUNC_XOR(p1,p2); // convert b_0 to an arithmetic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);

    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype r01 = getRandomVal(P_0);
        Datatype m1_prime = OP_SUB(r01, OP_MULT(OP_ADD(b0[i], b0[i]), r01));
        out[i].p1 = OP_ADD(m1_prime, b0[i]); // set share to m2' + b_0
        out[i].p2 = getRandomVal(P_0); // set other share to z_1
        send_to_live(P_2, OP_ADD( m1_prime, out[i].p2));
    }
}

void complete_bit2a()
{
    p1 = OP_ADD(p1, receive_from_live(P_2));
}




void prepare_bit_injection_S1(OECL1_Share out[])
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
        out[i].p1 = temp[i];// set share to b xor x_0
        out[i].p2 = SET_ALL_ZERO(); // set other share to 0
    }
}

void prepare_bit_injection_S2(OECL1_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p2 = getRandomVal(P_0); // set second share to r0,1
        out[i].p1 = OP_SUB(SET_ALL_ZERO(), out[i].p2) ; // set first share -r0,1
    }
}

static void complete_bit_injection_S1(OECL1_Share out[])
{
    
}

static void complete_bit_injection_S2(OECL1_Share out[])
{


}

template <typename func_add, typename func_sub, typename func_mul>
    OECL1_Share prepare_dot3(const OECL1_Share b, const OECL1_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype rxy = getRandomVal(P_0);
Datatype rxz = getRandomVal(P_0);
Datatype ryz = getRandomVal(P_0);
/* Datatype rxyz = getRandomVal(P_0); */

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);

OECL1_Share d;
d.p1 = ADD(
        ADD( MULT(a0,SUB(ryz,MULT(b0,c.p2)))
        ,(MULT(b0,SUB(rxz, MULT(c0,p2)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.p2)))); // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.p2 = SET_ALL_ZERO();
d.p1 = SUB(d.p2, d.p1);
return d;
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL1_Share prepare_mult3(OECL1_Share b, OECL1_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype rxy = getRandomVal(P_0);
Datatype rxz = getRandomVal(P_0);
Datatype ryz = getRandomVal(P_0);
Datatype rxyz = getRandomVal(P_0);

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);

OECL1_Share d;
d.p1 = SUB(ADD(
        ADD( MULT(a0,SUB(ryz,MULT(b0,c.p2)))
        ,(MULT(b0,SUB(rxz, MULT(c0,p2)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.p2)))), rxyz); // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.p2 = getRandomVal(P_0);
send_to_live(P_2, ADD(d.p1,d.p2));
return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
p1 = ADD(p1, receive_from_live(P_2));
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL1_Share prepare_dot4(OECL1_Share b, OECL1_Share c, OECL1_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype rxy = getRandomVal(P_0);
Datatype rxz = getRandomVal(P_0);
Datatype rxw = getRandomVal(P_0);
Datatype ryz = getRandomVal(P_0);
Datatype ryw = getRandomVal(P_0);
Datatype rzw = getRandomVal(P_0);
Datatype rxyz = getRandomVal(P_0);
Datatype rxyw = getRandomVal(P_0);
Datatype rxzw = getRandomVal(P_0);
Datatype ryzw = getRandomVal(P_0);

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);
Datatype d0 = ADD(d.p1,d.p2);



OECL1_Share e;
e.p1 =        
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(ryz, MULT(b0,c.p2))), ryzw))
                        ,
                            MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.p2))), 
                            SUB( MULT(c0, rxw), rxzw)))
                    ),
                    ADD(
                            
                            MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.p2))), rxyw))
                            ,
                    
                        
                            MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,p2))),
                            SUB( MULT(c0, rxy), rxyz)))
                )
            );
         // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.p2 = SET_ALL_ZERO();
e.p1 = SUB(e.p2, e.p1);
return e;
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL1_Share prepare_mult4(OECL1_Share b, OECL1_Share c, OECL1_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype rxy = getRandomVal(P_0);
Datatype rxz = getRandomVal(P_0);
Datatype rxw = getRandomVal(P_0);
Datatype ryz = getRandomVal(P_0);
Datatype ryw = getRandomVal(P_0);
Datatype rzw = getRandomVal(P_0);
Datatype rxyz = getRandomVal(P_0);
Datatype rxyw = getRandomVal(P_0);
Datatype rxzw = getRandomVal(P_0);
Datatype ryzw = getRandomVal(P_0);
Datatype rxyzw = getRandomVal(P_0);

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);
Datatype d0 = ADD(d.p1,d.p2);



OECL1_Share e;
e.p1 =        
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(ryz, MULT(b0,c.p2))), ryzw))
                        ,
                            MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.p2))), 
                            SUB( MULT(c0, rxw), rxzw)))
                    ),
                    ADD(
                            
                            ADD(rxyzw, MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.p2))), rxyw)))
                            ,
                    
                        
                            MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,p2))),
                            SUB( MULT(c0, rxy), rxyz)))
                )
            );
         // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.p2 = getRandomVal(P_0);
send_to_live(P_2, ADD(e.p1,e.p2));

return e;
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
p1 = ADD(p1, receive_from_live(P_2));
}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL1_Share& r_mk2, OECL1_Share& r_msb, OECL1_Share& c, OECL1_Share& c_prime) const{
    r_mk2.template prepare_receive_from<P_0>(ADD, SUB);
    r_msb.template prepare_receive_from<P_0>(ADD, SUB);
    
    Datatype c_dat_prime = trunc(ADD(p1,p2));
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH-FRACTIONAL-1)) - 1;
    Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue
    // Apply the mask using bitwise AND
    c_dat_prime = AND(c_dat_prime, mask); //mod 2^k-m-1
    /* Datatype c_dat = ADD(p1,p2) >> (BITLENGTH - 1); */
    Datatype c_dat = OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(ADD(p1,p2));
    c = OECL1_Share(c_dat, SET_ALL_ZERO());
    c_prime = OECL1_Share(c_dat_prime, SET_ALL_ZERO());
    
    /* c_prime.p1 = trunc(ADD(p1,p2)); */
    /* c_prime.p2 = SET_ALL_ZERO(); */
    /* UINT_TYPE maskValue = (1 << (BITLENGTH-FRACTIONAL-1)) - 1; */
    /* Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue */
    /* // Apply the mask using bitwise AND */
    /* c_prime.p1 = AND(c_prime.p1, mask); //mod 2^k-m-1 */
    
    /* c.p1 = ADD(p1,p2) >> (BITLENGTH - 1); //open c = x + r */
    /* c.p2 = SET_ALL_ZERO(); */
}


template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL1_Share& r_mk2, OECL1_Share& r_msb, OECL1_Share& c, OECL1_Share& c_prime) const{
    r_mk2.template complete_receive_from<P_0>(ADD, SUB);
    r_msb.template complete_receive_from<P_0>(ADD, SUB);
}



/* template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc> */
/* void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL1_Share& r_mk2, OECL1_Share& r_msb, OECL1_Share& c, OECL1_Share& c_prime) const{ */
/*     r_mk2.prepare_receive_from<P_0>(ADD, SUB); */
/*     r_msb.prepare_receive_from<P_0>(ADD, SUB); */
    
/*     Datatype c_dat_prime = trunc(ADD(p1,p2)); */
/*     UINT_TYPE maskValue = (1 << (BITLENGTH-FRACTIONAL-1)) - 1; */
/*     Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue */
/*     // Apply the mask using bitwise AND */
/*     c_dat_prime = AND(c_dat_prime, mask); //mod 2^k-m-1 */
/*     Datatype c_dat = ADD(p1,p2) >> (BITLENGTH - 1); */
/*     c = OECL1_Share(c_dat, SET_ALL_ZERO()); */
/*     c_prime = OECL1_Share(c_dat_prime, SET_ALL_ZERO()); */
    
/*     /1* c_prime.p1 = trunc(ADD(p1,p2)); *1/ */
/*     /1* c_prime.p2 = SET_ALL_ZERO(); *1/ */
/*     /1* UINT_TYPE maskValue = (1 << (BITLENGTH-FRACTIONAL-1)) - 1; *1/ */
/*     /1* Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue *1/ */
/*     /1* // Apply the mask using bitwise AND *1/ */
/*     /1* c_prime.p1 = AND(c_prime.p1, mask); //mod 2^k-m-1 *1/ */
    
/*     /1* c.p1 = ADD(p1,p2) >> (BITLENGTH -1) ; //open c = x + r *1/ */
/*     /1* c.p2 = SET_ALL_ZERO(); *1/ */
/* } */

/* template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc> */
/* void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL1_Share& r_mk2, OECL1_Share& r_msb, OECL1_Share& c, OECL1_Share& c_prime) const{ */
/*     r_mk2.complete_receive_from<P_0>(ADD, SUB); */
/*     r_msb.complete_receive_from<P_0>(ADD, SUB); */
/* } */



};
