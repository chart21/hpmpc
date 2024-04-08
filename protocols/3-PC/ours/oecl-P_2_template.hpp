#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL2_Share
{
Datatype p1;
Datatype p2;
public:
OECL2_Share() {}
OECL2_Share(Datatype p1) : p1(p1) {}
OECL2_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}

static OECL2_Share public_val(Datatype a)
{
    return OECL2_Share(a,SET_ALL_ZERO());
}
    


OECL2_Share Not() const
{
    return OECL2_Share(NOT(p1),p2);
}

template <typename func_add>
OECL2_Share Add(OECL2_Share b, func_add ADD) const
{
    return OECL2_Share(ADD(p1,b.p1),ADD(p2,b.p2));
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
OECL2_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
    OECL2_Share res;
#if TRUNC_THEN_MULT == 1
    res.p1 = MULT(TRUNC(ADD(p1,p2)),b);
#else
    res.p1 = TRUNC(MULT(ADD(p1,p2),b));
#endif
    return res;
} 
    
    template <typename func_add, typename func_sub>
void complete_public_mult_fixed( func_add ADD, func_sub SUB)
{
#if PRE == 1
    p2 = pre_receive_from_live(P_0);
#else
    p2 = receive_from_live(P_0);
#endif
    p1 = SUB(p1,p2);
}
    
    template <typename func_mul>
OECL2_Share mult_public(const Datatype b, func_mul MULT) const
{
    return OECL2_Share(MULT(p1,b),MULT(p2,b));
}


    template <typename func_add, typename func_sub, typename func_mul>
void prepare_dot_add(OECL2_Share a, OECL2_Share b , OECL2_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c.p1 = ADD(c.p1, MULT(a.p1,b.p1));
}    
    template <typename func_add, typename func_sub, typename func_mul>
OECL2_Share prepare_dot( const OECL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OECL2_Share c;
c.p1 = MULT(p1,b.p1); // ab_2 + e_2, e_2 = x1 y_1 
return c;
}

template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
#if PRE == 1
    p1 = ADD(pre_receive_from_live(P_0), p1);
#else
    p1 = ADD(receive_from_live(P_0), p1);
#endif
    p2 = getRandomVal(P_0);
    send_to_live(P_1,ADD(p1,p2));
}
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{

p1 = ADD(p1, getRandomVal(P_0)); // a1b1 + r_0,2
send_to_live(P_1, p1); 
}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
#if PRE == 1
p2 = pre_receive_from_live(P_0); // (e + r0,1 + r0,2)^T - r_0,1
#else
p2 = receive_from_live(P_0); // (e + r0,1 + r0,2)^T - r_0,1
#endif
p1 = SUB(TRUNC( SUB(p1,receive_from_live(P_1))),p2); // [m2 -m1]^t - m^0
}
    
    /* template <typename func_add, typename func_sub, typename func_trunc> */
/* void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
/* { */

/* p1 = ADD(p1, getRandomVal(P_0)); // ab_2 + e_2 + r0,2 */
/* send_to_live(P_1, p1); // ab_2 + e_2 + r0,2 */
/* } */

    /* template <typename func_add, typename func_sub, typename func_trunc> */
/* void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC) */
/* { */
/* #if PRE == 1 */
/* p2 = pre_receive_from_live(P_0); // (e + r0,1 + r0,2)^T + r0,1_2 */
/* #else */
/* p2 = receive_from_live(P_0); // (e + r0,1 + r0,2)^T + r0,1_2 */
/* #endif */
/* p1 = TRUNC( SUB(p1,receive_from_live(P_1))); // (ab + e + r0,1 + r0,2)^T */ 
/* p1 = SUB(p1, p2); // - [ ( (e + r0,1 + r0,2)^T + r0,1_2 ) ] */
/* } */
template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult(OECL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
/* OECL2_Share c; */
Datatype cp2 = getRandomVal(P_0); // P_2 mask for P_1
#if PRE == 1
Datatype cp1 = ADD(pre_receive_from_live(P_0), MULT(p1,b.p1)); // P_0_message + (a+rr) (b+rl)
#else
Datatype cp1 = ADD(receive_from_live(P_0), MULT(p1,b.p1)); // P_0_message + (a+rr) (b+rl)
#endif

send_to_live(P_1, ADD(cp1,cp2)); 
return OECL2_Share(cp1,cp2); // (a+rr) (b+rl)
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
p1 = SUB(p1, receive_from_live(P_1)); 
}


void prepare_reveal_to_all() const
{
send_to_live(P_0, p1);
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


    template <int id,typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
if constexpr(id == P_2)
{
    p1 = val;
    p2 = getRandomVal(P_0);
    /* p1 = getRandomVal(0); *1/ */
    send_to_live(P_1, ADD(p1,p2));
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P_0)
{
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
        p2 = pre_receive_from_live(P_0);
#else
        p2 = receive_from_live(P_0);
#endif
        p1 = SUB(SET_ALL_ZERO(), p2); // set own share to - - (a + r0,1)
}
else if constexpr(id == P_1)
{
p1 = receive_from_live(P_1);
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


//higher level functions

static void prepare_B2A( OECL2_Share z[], OECL2_Share random_mask[], OECL2_Share out[])
{
    // 1. Reveal z to P_1 and P_2
    for(int i = 0; i < BITLENGTH; i++)
    {
        /* send_to_live(P_1, z[i].p1); */
        send_to_live(P_1, z[i].p2); //reveal z to P_2
        z[i].p2 = SET_ALL_ZERO(); //set mask to 0 since it is reveald
    }
    // 2. Share random mask
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].template prepare_receive_from<P_0>(OP_ADD, OP_SUB);
    } 


}

static void complete_B2A(OECL2_Share z_bool[], OECL2_Share out[])
{
    Datatype z[BITLENGTH]; 
    for(int i = 0; i < BITLENGTH; i++)
        z[i] = FUNC_XOR(z_bool[i].p1, receive_from_live(P_1));
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(z, temp2);
    orthogonalize_arithmetic(temp2, z);

    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].template complete_receive_from<P_0>(OP_ADD, OP_SUB);
        out[i] = public_val(z[i]).Add(out[i], OP_SUB); // z - r
    }

}




static void prepare_A2B_S1(int m, int k, OECL2_Share in[], OECL2_Share out[])
{
    //convert share a + x1 to boolean
    Datatype temp[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp[i] = OP_ADD(in[i].p1,in[i].p2); // set share to a + x_0
    }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp, temp2);
    orthogonalize_boolean(temp2, temp);
    /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
    /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */
    for(int i = m; i < k; i++)
    {
        out[i-m].p1 = temp[i];
        out[i-m].p2 = SET_ALL_ZERO();
    }
}

static void prepare_A2B_S2(int m, int k, OECL2_Share in[], OECL2_Share out[])
{
}

static void complete_A2B_S1(int k, OECL2_Share out[])
{
}

static void complete_A2B_S2(int k, OECL2_Share out[])
{
    for(int i = 0; i < k; i++)
    {
#if PRE == 1
        out[i].p1 = pre_receive_from_live(P_0);
#else
        out[i].p1 = receive_from_live(P_0);
#endif
        out[i].p2 = out[i].p1; // set both shares to -x0 xor r0,1
    }
        /* out[0].p2 = FUNC_NOT(out[0].p2);// change sign bit -> -x0 xor r0,1 to x0 xor r0,1 */
}

void prepare_opt_bit_injection(OECL2_Share a[], OECL2_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = FUNC_XOR(p1,p2); //convert b to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype a0 = OP_ADD(a[i].p1,a[i].p2);
        Datatype z2 = getRandomVal(P_0);
#if PRE == 1
        Datatype m00 = pre_receive_from_live(P_0);
        Datatype m01 = pre_receive_from_live(P_0);
#else
        Datatype m00 = receive_from_live(P_0);
        Datatype m01 = receive_from_live(P_0);
#endif
        Datatype m1 = OP_SUB(OP_ADD(b0[i], b0[i]), PROMOTE(1));
        m1 = OP_MULT(m1, OP_SUB(m01, OP_MULT(a0, m00)));
        m1 = OP_SUB(m1, OP_MULT(b0[i], a[i].p2));
        send_to_live(P_1, OP_ADD(m1, z2));
        out[i].p1 = OP_ADD(m1,OP_MULT(a0, b0[i]));
        out[i].p2 = z2;
    }
}

void complete_opt_bit_injection()
{
        p1 = OP_ADD(p1, receive_from_live(P_1));
}

void prepare_bit2a(OECL2_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = FUNC_XOR(p1,p2); // convert b_0 to an arithmetic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);

    for(int i = 0; i < BITLENGTH; i++)
    {
#if PRE == 1
        Datatype m0 = pre_receive_from_live(P_0);
#else
        Datatype m0 = receive_from_live(P_0);
#endif
        Datatype m2_prime = OP_SUB(m0, OP_MULT(OP_ADD(b0[i], b0[i]), m0));
        out[i].p1 = OP_ADD(m2_prime, b0[i]); // set share to m2' + b_0
        out[i].p2 = getRandomVal(P_0); // set other share to z_1
        send_to_live(P_1, OP_ADD(m2_prime, out[i].p2));
    }
}

void complete_bit2a()
{
    p1 = OP_ADD(p1, receive_from_live(P_1));
}


void prepare_bit_injection_S1(OECL2_Share out[])
{
    Datatype temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = FUNC_XOR(p1,p2);
    /* unorthogonalize_boolean(temp,(UINT_TYPE*)temp); */
    /* orthogonalize_arithmetic((UINT_TYPE*) temp,  temp); */
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(temp, temp2);
    orthogonalize_arithmetic(temp2, temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].p1 = temp[i];// set share to b xor x_0
        out[i].p2 = SET_ALL_ZERO(); // set other share to 0
    }
}

void prepare_bit_injection_S2( OECL2_Share out[])
{
}

static void complete_bit_injection_S1(OECL2_Share out[])
{
    
}

static void complete_bit_injection_S2(OECL2_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
     #if PRE == 1
        out[i].p2 = pre_receive_from_live(P_0);
        #else
        out[i].p2 = receive_from_live(P_0);
        #endif
        out[i].p1 = OP_SUB(SET_ALL_ZERO(), out[i].p2); // set first share to x0 + r0,1
    }


}

template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_dot3(const OECL2_Share b, const OECL2_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PRE == 1
Datatype rxy = pre_receive_from_live(P_0);
Datatype rxz = pre_receive_from_live(P_0);
Datatype ryz = pre_receive_from_live(P_0);
#else
Datatype rxy = receive_from_live(P_0);
Datatype rxz = receive_from_live(P_0);
Datatype ryz = receive_from_live(P_0);
#endif

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);

OECL2_Share d;
d.p1 = ADD(
        ADD( MULT(a0,ADD(MULT(b0,SUB(c0,c.p2)),ryz))
        ,(MULT(b0,SUB(rxz, MULT(c0,p2)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.p2)))); // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.p2 = SET_ALL_ZERO();
return d;
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult3(OECL2_Share b, OECL2_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PRE == 1
Datatype rxy = pre_receive_from_live(P_0);
Datatype rxz = pre_receive_from_live(P_0);
Datatype ryz = pre_receive_from_live(P_0);
Datatype rxyz = pre_receive_from_live(P_0);
#else
Datatype rxy = receive_from_live(P_0);
Datatype rxz = receive_from_live(P_0);
Datatype ryz = receive_from_live(P_0);
Datatype rxyz = receive_from_live(P_0);
#endif

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);

OECL2_Share d;
d.p1 = SUB(ADD(
        ADD( MULT(a0,ADD(MULT(b0,SUB(c0,c.p2)),ryz))
        ,(MULT(b0,SUB(rxz, MULT(c0,p2)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.p2)))), rxyz); // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.p2 = getRandomVal(P_0);
send_to_live(P_1, ADD(d.p1,d.p2));
return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
p1 = ADD(p1, receive_from_live(P_1));
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_dot4(OECL2_Share b, OECL2_Share c, OECL2_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PRE == 1
Datatype rxy = pre_receive_from_live(P_0);
Datatype rxz = pre_receive_from_live(P_0);
Datatype rxw = pre_receive_from_live(P_0);
Datatype ryz = pre_receive_from_live(P_0);
Datatype ryw = pre_receive_from_live(P_0);
Datatype rzw = pre_receive_from_live(P_0);
Datatype rxyz = pre_receive_from_live(P_0);
Datatype rxyw = pre_receive_from_live(P_0);
Datatype rxzw = pre_receive_from_live(P_0);
Datatype ryzw = pre_receive_from_live(P_0);
#else
Datatype rxy = receive_from_live(P_0);
Datatype rxz = receive_from_live(P_0);
Datatype rxw = receive_from_live(P_0);
Datatype ryz = receive_from_live(P_0);
Datatype ryw = receive_from_live(P_0);
Datatype rzw = receive_from_live(P_0);
Datatype rxyz = receive_from_live(P_0);
Datatype rxyw = receive_from_live(P_0);
Datatype rxzw = receive_from_live(P_0);
Datatype ryzw = receive_from_live(P_0);
#endif

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);
Datatype d0 = ADD(d.p1,d.p2);



OECL2_Share e;
e.p1 = 
          
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, ADD(MULT(b0,SUB(c0,c.p2)),ryz )), ryzw))
                            ,
                            MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.p2))), 
                            SUB( MULT(c0, rxw), rxzw)))
                        )
                    ,
                    ADD(
                            MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.p2))), rxyw))
                            ,
                            MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,p2))),
                            SUB( MULT(c0, rxy), rxyz)))
                        )
                               
                ); // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.p2 = SET_ALL_ZERO();
return e;
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL2_Share prepare_mult4(OECL2_Share b, OECL2_Share c, OECL2_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PRE == 1
Datatype rxy = pre_receive_from_live(P_0);
Datatype rxz = pre_receive_from_live(P_0);
Datatype rxw = pre_receive_from_live(P_0);
Datatype ryz = pre_receive_from_live(P_0);
Datatype ryw = pre_receive_from_live(P_0);
Datatype rzw = pre_receive_from_live(P_0);
Datatype rxyz = pre_receive_from_live(P_0);
Datatype rxyw = pre_receive_from_live(P_0);
Datatype rxzw = pre_receive_from_live(P_0);
Datatype ryzw = pre_receive_from_live(P_0);
Datatype rxyzw = pre_receive_from_live(P_0);
#else
Datatype rxy = receive_from_live(P_0);
Datatype rxz = receive_from_live(P_0);
Datatype rxw = receive_from_live(P_0);
Datatype ryz = receive_from_live(P_0);
Datatype ryw = receive_from_live(P_0);
Datatype rzw = receive_from_live(P_0);
Datatype rxyz = receive_from_live(P_0);
Datatype rxyw = receive_from_live(P_0);
Datatype rxzw = receive_from_live(P_0);
Datatype ryzw = receive_from_live(P_0);
Datatype rxyzw = receive_from_live(P_0);
#endif

Datatype a0 = ADD(p1,p2);
Datatype b0 = ADD(b.p1,b.p2);
Datatype c0 = ADD(c.p1,c.p2);
Datatype d0 = ADD(d.p1,d.p2);



OECL2_Share e;
e.p1 = 
          
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, ADD(MULT(b0,SUB(c0,c.p2)),ryz )), ryzw))
                            ,
                            MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.p2))), 
                            SUB( MULT(c0, rxw), rxzw)))
                        )
                    ,
                    ADD(
                            ADD(MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.p2))), rxyw)), rxyzw)
                            ,
                            MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,p2))),
                            SUB( MULT(c0, rxy), rxyz)))
                        )
                               
                ); // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.p2 = getRandomVal(P_0);
send_to_live(P_1, ADD(e.p1,e.p2));

return e;
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
p1 = ADD(p1, receive_from_live(P_1));
}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL2_Share& r_mk2, OECL2_Share& r_msb, OECL2_Share& c, OECL2_Share& c_prime) const{
    r_mk2.template prepare_receive_from<P_0>(ADD, SUB);
    r_msb.template prepare_receive_from<P_0>(ADD, SUB);
    
    Datatype c_dat_prime = trunc(ADD(p1,p2));
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH-FRACTIONAL-1)) - 1;
    Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue
    // Apply the mask using bitwise AND
    c_dat_prime = AND(c_dat_prime, mask); //mod 2^k-m-1
    /* Datatype c_dat = ADD(p1,p2) >> (BITLENGTH - 1); */
    Datatype c_dat = OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(ADD(p1,p2));
    c = OECL2_Share(c_dat, SET_ALL_ZERO());
    c_prime = OECL2_Share(c_dat_prime, SET_ALL_ZERO());
    
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
void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OECL2_Share& r_mk2, OECL2_Share& r_msb, OECL2_Share& c, OECL2_Share& c_prime) const{
    r_mk2.template complete_receive_from<P_0>(ADD, SUB);
    r_msb.template complete_receive_from<P_0>(ADD, SUB);
}



};


