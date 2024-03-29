#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL1_Share
{

private:
    Datatype v;
    Datatype r;
#if PROTOCOL == 11 || FRACTIONAL > 0 || MULTI_INPUT == 1
    Datatype m;
#endif

public:
    
OEC_MAL1_Share() {}
OEC_MAL1_Share(Datatype v, Datatype r) : v(v), r(r) {}
#if MULTI_INPUT == 1
OEC_MAL1_Share(Datatype v, Datatype r, Datatype m) : v(v), r(r), m(m) {}
#endif
OEC_MAL1_Share(Datatype v) : v(v) {}


static OEC_MAL1_Share public_val(Datatype a)
{
    #if MULTI_INPUT == 1
    return OEC_MAL1_Share(a,SET_ALL_ZERO(),SET_ALL_ZERO());
    #else
    return OEC_MAL1_Share(a,SET_ALL_ZERO());
    #endif
}

template <typename func_mul>
OEC_MAL1_Share mult_public(const Datatype b, func_mul MULT) const
{
#if MULTI_INPUT == 1
    return OEC_MAL1_Share(MULT(v,b),MULT(r,b),MULT(m,b));
#else
    return OEC_MAL1_Share(MULT(v,b),MULT(r,b));
#endif

}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
OEC_MAL1_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
    OEC_MAL1_Share res;
#if TRUNC_THEN_MULT == 1
    res.v = MULT(TRUNC(v),b);
#else
    res.v = TRUNC(MULT(v,b));
#endif
#if MULTI_INPUT == 1
    res.m = getRandomVal(P_123);
    store_compare_view(P_0,ADD(res.v,res.m)); // compare v*b + r123 with P_0
#else
    store_compare_view(P_0,ADD(res.v,getRandomVal(P_123))); // compare v*b + r123 with P_0
#endif
    res.r = getRandomVal(P_013);
    return res;
} 
    
    template <typename func_add, typename func_sub>
void complete_public_mult_fixed( func_add ADD, func_sub SUB)
{
}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OEC_MAL1_Share& r_mk2, OEC_MAL1_Share& r_msb, OEC_MAL1_Share& c, OEC_MAL1_Share& c_prime) const{
    Datatype c_dat_prime = trunc(v);
    UINT_TYPE maskValue = (UINT_TYPE(1) << (BITLENGTH-FRACTIONAL-1)) - 1;
    Datatype mask = PROMOTE(maskValue); // Set all elements to maskValue
    c_dat_prime = AND(c_dat_prime, mask); //mod 2^k-m-1
    Datatype c_dat = OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(v);
    c = OEC_MAL1_Share(c_dat, SET_ALL_ZERO());
    c_prime = OEC_MAL1_Share(c_dat_prime, SET_ALL_ZERO());
   
#if MULTI_INPUT == 1
    r_mk2.m = SET_ALL_ZERO();
    r_msb.m = SET_ALL_ZERO();
    c.m = getRandomVal(P_123);
    c_prime.m = getRandomVal(P_123);
    store_compare_view(P_0, ADD(c_dat,c.m));
    store_compare_view(P_0, ADD(c_dat_prime,c_prime.m));
#else
    store_compare_view(P_0, ADD(c_dat,getRandomVal(P_123)));
    store_compare_view(P_0, ADD(c_dat_prime,getRandomVal(P_123)));
#endif

    r_mk2.v = SET_ALL_ZERO();
    r_mk2.r = getRandomVal(P_013);
    r_msb.v = SET_ALL_ZERO(); 
    r_msb.r = getRandomVal(P_013);
}


template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OEC_MAL1_Share& r_mk2, OEC_MAL1_Share& r_msb, OEC_MAL1_Share& c, OEC_MAL1_Share& c_prime) const{

}



OEC_MAL1_Share Not() const
{
   #if MULTI_INPUT == 1
   return OEC_MAL1_Share(NOT(v),r,m);
   #else
   return OEC_MAL1_Share(NOT(v),r);
   #endif
}

template <typename func_add>
OEC_MAL1_Share Add(OEC_MAL1_Share b, func_add ADD) const
{
    #if MULTI_INPUT == 1
    return OEC_MAL1_Share(ADD(v,b.v),ADD(r,b.r),ADD(m,b.m));
    #else
   return OEC_MAL1_Share(ADD(v,b.v),ADD(r,b.r));
    #endif
}



template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_Share prepare_mult(OEC_MAL1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
/* Datatype cr = XOR(getRandomVal(P_013),getRandomVal(P_123)); */
/* c.r = SUB(getRandomVal(P_013),getRandomVal(P_123)); */
OEC_MAL1_Share c;
c.r = getRandomVal(P_013);
Datatype r124 = getRandomVal(P_013);
/* Datatype r234 = getRandomVal(P_123); //used for veryfying m3' sent by P_3 -> probably not needed -> for verification needed */
c.v = ADD( ADD(MULT(v,b.r), MULT(b.v,r))  , r124);  
/* Datatype m_2 = XOR(c.v, c.r); */
send_to_live(P_2,c.v);

/* Datatype m3_prime = XOR( XOR(r234,cr) , AND( XOR(v,r) ,XOR(b.v,b.r))); //computationally wise more efficient to verify ab instead of m_3 prime */

/* store_compare_view(P_0,m3_prime); */
/* c.m = ADD(c.v,getRandomVal(P_123)); */
Datatype a1b1 = MULT(v,b.v);
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P_0,ADD(a1b1,getRandomVal(P_123))); // compare a1b1 + r123_2 with P_0
#endif
/* c.v = XOR( AND(      XOR(v,r) , XOR(b.v,b.r) ) , c.v); */
#if PROTOCOL == 11
c.m = ADD(c.v,getRandomVal(P_123)); // m_2 + r234_2 store to compareview later
#endif

c.v = SUB( a1b1,c.v);
return c;
}

template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL1_Share prepare_dot(const OEC_MAL1_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL1_Share c;
c.r = ADD(MULT(v,b.r), MULT(b.v,r)); // a_0 y_1 + b_0 x_1
c.v = MULT(v, b.v); // a0b0
return c;
}

#if FRACTIONAL > 0

    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
m = SUB(r, getRandomVal(P_013));// a_0 y_1 + b_0 x_1 - r_0,1,3   
r = getRandomVal(P_013); // z_1
send_to_live(P_2, m); 

}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
m = ADD(m, receive_from_live(P_2)); // v^1,2 = m^1 + m^2
v = TRUNC(SUB(v, m)); // [a_0 b_0 - v^1,2]^t
#if PROTOCOL == 11
store_compare_view(P_0, ADD(m,getRandomVal(P_123))); // compare m1 + m2 + r123 with P_0
#else
store_compare_view(P_012,ADD(m, getRandomVal(P_123))); // v^1,2 + r_1,2,3
#endif
#if MULTI_INPUT == 1
m = getRandomVal(P_123); // w
store_compare_view(P_0,ADD(v,m)); // compare c0w with P_0
#else
store_compare_view(P_0,ADD(v, getRandomVal(P_123))); // c_0 + w
#endif
}

#endif

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
Datatype cr = getRandomVal(P_013);
Datatype r124 = getRandomVal(P_013);
/* Datatype r234 = getRandomVal(P_123); //used for veryfying m3' sent by P_3 -> probably not needed -> for verification needed */
r = ADD( r  , r124); // a_0 y_1 + b_0 x_1
/* Datatype m_2 = XOR(c.v, c.r); */
send_to_live(P_2,r);

#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P_0,ADD(v,getRandomVal(P_123))); // compare a0b0 + r123_2 with P_0
#endif
#if PROTOCOL == 11
m = ADD(r,getRandomVal(P_123)); // m_2 + r234_2 store to compareview later
#endif

v = SUB( v,r);
r = cr;

}
template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
Datatype m_3 = receive_from_live(P_2);
v = SUB(v, m_3);

/* c.m = XOR(c.m,m_3); */
/* Datatype cm = XOR(c.m,m_3); */

#if PROTOCOL == 11
store_compare_view(P_0,ADD(m,m_3)); // compare m_2 + m_3 + r234_2
#if MULTI_INPUT == 1
m = getRandomVal(P_123); // w
store_compare_view(P_0,ADD(v,m)); // w
#else
store_compare_view(P_0,ADD(v,getRandomVal(P_123))); //compare ab + c1 + r234_1
#endif
#else
#if MULTI_INPUT == 1
m = getRandomVal(P_123); // w
store_compare_view(P_012,ADD(v,m)); // w
#else
store_compare_view(P_012,ADD(v,getRandomVal(P_123))); //compare ab + c1 + r234_1
#endif
#endif
}


void prepare_reveal_to_all() const
{
return;
}    

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
Datatype r = receive_from_live(P_0);
Datatype result = SUB(v, r);
store_compare_view(P_123, r);
store_compare_view(P_0123, result);
return result;
}



template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    
    Datatype x_0 = getRandomVal(P_013);
    Datatype u = getRandomVal(P_123);
    r = x_0; //  = x_1, x_2 = 0
    v = ADD(val,x_0);
    send_to_live(P_0,ADD(v,u));
    send_to_live(P_2,ADD(v,u));
    #if MULTI_INPUT == 1
    m = u;
    #endif

}
else if constexpr(id == P_0)
{
    r = getRandomVal(P_013);
    v = SET_ALL_ZERO();
    // u = 0
    #if MULTI_INPUT == 1
    m = SET_ALL_ZERO();
    #endif

}
else if constexpr(id == P_2)
{
    r = SET_ALL_ZERO();
    v = getRandomVal(P_123); //u
    #if MULTI_INPUT == 1
    m = v;
    #endif
    
  

}
else if constexpr(id == P_3)
{
    r = getRandomVal(P_013); //x1
    v = getRandomVal(P_123); //u
    #if MULTI_INPUT == 1
    m = v;
    #endif

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
{
            #if PRE == 1
        Datatype val;
        if(id == P_3)
            val = pre_receive_from_live(P_3);
        else
            val = receive_from_live(id);
    #else
    Datatype val = receive_from_live(id);
    #endif

    if constexpr(id != P_0)
            store_compare_view(P_0,val);
    if constexpr(id != P_2)
            store_compare_view(P_2,val);
    v = SUB(val,v); // convert locally to a + x_0
}
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

static void prepare_A2B_S1(int m, int k, OEC_MAL1_Share in[], OEC_MAL1_Share out[])
{
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = in[j].v; //a0 
        }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp, temp2);
    orthogonalize_boolean(temp2, temp);
    /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
    /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */
    for (int j = m; j < k; j++)
    {
        out[j-m].r = SET_ALL_ZERO(); // set share to 0
        out[j-m].v = temp[j];
    #if MULTI_INPUT == 1
    out[j-m].m = getRandomVal(P_123);
    store_compare_view(P_0, FUNC_XOR(out[j-m].v, out[j-m].m));
    #else
    store_compare_view(P_0, FUNC_XOR(out[j-m].v, getRandomVal(P_123))); 
    #endif
    }

}


static void prepare_A2B_S2(int m, int k, OEC_MAL1_Share in[], OEC_MAL1_Share out[])
{
    for(int i = m; i < k; i++)
    {
        out[i-m].r = getRandomVal(P_013);
        out[i-m].v = SET_ALL_ZERO();
        #if MULTI_INPUT == 1
        out[i-m].m = SET_ALL_ZERO();
        #endif
    }
}

static void complete_A2B_S1(int k, OEC_MAL1_Share out[])
{
}

static void complete_A2B_S2(int k, OEC_MAL1_Share out[])
{

}

void prepare_bit2a(OEC_MAL1_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = v; //convert b0 to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);
    Datatype b0v[BITLENGTH]{0};
    b0v[BITLENGTH - 1] = FUNC_XOR(v,m); //convert b0v to an arithemtic value
    unorthogonalize_boolean(b0v, temp2);
    orthogonalize_arithmetic(temp2, b0v);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype r013 = getRandomVal(P_013);
         
        Datatype tmp = OP_SUB(r013, OP_MULT(OP_ADD(b0[i], b0[i]), r013));
        Datatype out_r = getRandomVal(P_013);
        Datatype m1 = OP_ADD(out_r, tmp);
        send_to_live(P_2, m1); //m1
       
        Datatype r123 = getRandomVal(P_123);
        
        tmp = OP_SUB(r123, OP_MULT(OP_ADD(b0v[i], b0v[i]), r123));
        out[i].m = getRandomVal(P_123);
        store_compare_view(P_0, OP_ADD(out[i].m, tmp)); //m20
        
        out[i].r = out_r;
        out[i].v = OP_ADD(b0[i], m1);
        
    }
}

void complete_bit2a()
{
        Datatype m21 = receive_from_live(P_2);
        v = OP_ADD(v, m21);
        store_compare_view(P_012, OP_ADD(v, m));
}


void prepare_opt_bit_injection(OEC_MAL1_Share a[], OEC_MAL1_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = v; //convert b0 to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);
    Datatype b0v[BITLENGTH]{0};
    b0v[BITLENGTH - 1] = FUNC_XOR(v,m); //convert b0v to an arithemtic value
    unorthogonalize_boolean(b0v, temp2);
    orthogonalize_arithmetic(temp2, b0v);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype r013 = getRandomVal(P_013);
        Datatype r013_2 = getRandomVal(P_013);
         
        Datatype tmp = OP_SUB(OP_ADD(b0[i], b0[i]), PROMOTE(1));
        tmp = OP_MULT(tmp, OP_SUB(r013_2, OP_MULT(a[i].v, r013)));
        tmp = OP_SUB(tmp, OP_MULT(b0[i], a[i].r));
        Datatype out_r = getRandomVal(P_013);
        Datatype m1 = OP_ADD(out_r, tmp);
        send_to_live(P_2, m1); //m1
       
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);
        Datatype a0u = OP_ADD(a[i].v, a[i].m); // set share to a_0 + u
        
        tmp = OP_SUB(OP_ADD(b0v[i], b0v[i]), PROMOTE(1));
        tmp = OP_MULT(tmp, OP_SUB(r123_2, OP_MULT(a0u, r123)));
        tmp = OP_SUB(tmp, OP_MULT(b0v[i], a[i].m));
        out[i].m = getRandomVal(P_123);
        store_compare_view(P_0, OP_ADD(out[i].m, tmp)); //m20
        
        out[i].r = out_r;
        out[i].v = OP_ADD(OP_MULT(a[i].v, b0[i]), m1);
        
    }
}

void complete_opt_bit_injection()
{
        Datatype m21 = receive_from_live(P_2);
        v = OP_ADD(v, m21);
        store_compare_view(P_012, OP_ADD(v, m));
}

void prepare_bit_injection_S1(OEC_MAL1_Share out[])
{
    Datatype temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = v;
    /* unorthogonalize_boolean(temp, (UINT_TYPE*) temp); */
    /* orthogonalize_arithmetic((UINT_TYPE*) temp, temp); */
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(temp, temp2);
    orthogonalize_arithmetic(temp2, temp);
    for (int j = 0; j < BITLENGTH; j++)
    {
        out[j].v = temp[j]; 
        out[j].r = SET_ALL_ZERO(); // set share to 0
    #if MULTI_INPUT == 1
    out[j].m = getRandomVal(P_123);
    store_compare_view(P_0, OP_ADD(out[j].v, out[j].m));
    #else
    store_compare_view(P_0, OP_ADD(out[j].v, getRandomVal(P_123))); 
    #endif
    }
}

void prepare_bit_injection_S2(OEC_MAL1_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].r = getRandomVal(P_013);
        out[i].v = SET_ALL_ZERO();
        #if MULTI_INPUT == 1
        out[i].m = SET_ALL_ZERO();
        #endif
    }
}

static void complete_bit_injection_S1(OEC_MAL1_Share out[])
{
    
}

static void complete_bit_injection_S2(OEC_MAL1_Share out[])
{
}

#if MULTI_INPUT == 1

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_Share prepare_dot3(const OEC_MAL1_Share b, const OEC_MAL1_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = getRandomVal(P_013);
Datatype mxz = getRandomVal(P_013);
Datatype myz = getRandomVal(P_013);
Datatype sxy = ADD(mxy, ADD(ADD(MULT(m,b.m),MULT(r,b.m)),MULT(m,b.r)));
Datatype sxz = ADD(mxz, ADD(ADD(MULT(m,c.m),MULT(r,c.m)),MULT(m,c.r)));
Datatype syz = ADD(myz, ADD(ADD(MULT(b.m,c.m),MULT(b.r,c.m)),MULT(b.m,c.r)));
/* Datatype sxyz = ADD(ADD(mxyz,ADD(MULT(mxy,c.m),MULT(MULT(m,b.m),c.r))),MULT(MULT(m,b.m),c.m)); */
/* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, */ 
/*                     MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), */ 
/*         ADD(ADD( MULT(MULT(m,b.m),c.r), ADD(MULT(MULT(m,c.m),b.r),MULT(MULT(m,b.m),c.r))), MULT(MULT(m,b.m),c.m))); */
/* Datatype sxyz = */
/* ADD(MULT(MULT(m,b.m),c.m),  ADD(mxyz, */
/* ADD( */
/*     ADD(ADD(MULT(mxy,c.m),MULT(mxz,b.m)),MULT(myz,m)), */
/*     ADD(ADD(MULT(MULT(m,b.m),c.r),MULT(MULT(m,c.m),b.r)),MULT(MULT(b.m,c.m),r)) */
/*     ))); */
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(c.m,(ADD(b.r,b.m)))))),
        MULT(b.m,(ADD(mxz,MULT(c.r,m))))
        ),
        MULT(c.m,(ADD(mxy,MULT(r,b.m))))
    );

Datatype a0 = ADD(v,m);
Datatype b0 = ADD(b.v,b.m);
Datatype c0 = ADD(c.v,c.m);
Datatype rxy = getRandomVal(P_123);
Datatype rxz = getRandomVal(P_123);
Datatype ryz = getRandomVal(P_123);
Datatype ar = ADD(r,m);
Datatype br = ADD(b.r,b.m);
Datatype cr = ADD(c.r,c.m);
OEC_MAL1_Share d;
d.r = SUB(ADD(
        ADD( MULT(a0,SUB(syz, MULT(b0,cr)))
        ,(MULT(b0,SUB(sxz, MULT(c0,ar)))))
        ,MULT(c0,SUB(sxy, MULT(a0,br)))), sxyz); // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.v = ADD(
        ADD( MULT(a0,SUB(ryz,MULT(b0,c.m)))
        ,(MULT(b0,SUB(rxz, MULT(c0,m)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.m)))); // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz

d.r = SUB(d.v, d.r); // hack for mask_and_send_dot
return d;
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_Share prepare_mult3(const OEC_MAL1_Share b, const OEC_MAL1_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = getRandomVal(P_013);
Datatype mxz = getRandomVal(P_013);
Datatype myz = getRandomVal(P_013);
Datatype mxyz = getRandomVal(P_013);
Datatype sxy = ADD(mxy, ADD(ADD(MULT(m,b.m),MULT(r,b.m)),MULT(m,b.r)));
Datatype sxz = ADD(mxz, ADD(ADD(MULT(m,c.m),MULT(r,c.m)),MULT(m,c.r)));
Datatype syz = ADD(myz, ADD(ADD(MULT(b.m,c.m),MULT(b.r,c.m)),MULT(b.m,c.r)));
/* Datatype sxyz = ADD(ADD(mxyz,ADD(MULT(mxy,c.m),MULT(MULT(m,b.m),c.r))),MULT(MULT(m,b.m),c.m)); */
/* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, */ 
/*                     MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), */ 
/*         ADD(ADD( MULT(MULT(m,b.m),c.r), ADD(MULT(MULT(m,c.m),b.r),MULT(MULT(m,b.m),c.r))), MULT(MULT(m,b.m),c.m))); */
/* Datatype sxyz = */
/* ADD(MULT(MULT(m,b.m),c.m),  ADD(mxyz, */
/* ADD( */
/*     ADD(ADD(MULT(mxy,c.m),MULT(mxz,b.m)),MULT(myz,m)), */
/*     ADD(ADD(MULT(MULT(m,b.m),c.r),MULT(MULT(m,c.m),b.r)),MULT(MULT(b.m,c.m),r)) */
/*     ))); */
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(c.m,(ADD(b.r,b.m)))))),
        MULT(b.m,(ADD(mxz,MULT(c.r,m))))
        ),
    ADD(
        MULT(c.m,(ADD(mxy,MULT(r,b.m)))),
        mxyz 
        )
    );

Datatype a0 = ADD(v,m);
Datatype b0 = ADD(b.v,b.m);
Datatype c0 = ADD(c.v,c.m);
Datatype rxy = getRandomVal(P_123);
Datatype rxz = getRandomVal(P_123);
Datatype ryz = getRandomVal(P_123);
Datatype rxyz = getRandomVal(P_123);
Datatype ar = ADD(r,m);
Datatype br = ADD(b.r,b.m);
Datatype cr = ADD(c.r,c.m);
OEC_MAL1_Share d;
d.v = SUB(ADD(
        ADD( MULT(a0,SUB(syz, MULT(b0,cr)))
        ,(MULT(b0,SUB(sxz, MULT(c0,ar)))))
        ,MULT(c0,SUB(sxy, MULT(a0,br)))), sxyz); // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
Datatype m20 = SUB(ADD(
        ADD( MULT(a0,SUB(ryz,MULT(b0,c.m)))
        ,(MULT(b0,SUB(rxz, MULT(c0,m)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.m)))), rxyz); // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.m = getRandomVal(P_123);
d.r = getRandomVal(P_013);
d.v = ADD(d.v,d.r);
store_compare_view(P_0, ADD(m20, d.m));
send_to_live(P_2,d.v);
return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
Datatype m21 = receive_from_live(P_2);
v = ADD(v,m21);
store_compare_view(P_012, ADD(v,m)); //compare d_0 s
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_Share prepare_dot4(const OEC_MAL1_Share b,const  OEC_MAL1_Share c, const OEC_MAL1_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = getRandomVal(P_013);
Datatype mxz = getRandomVal(P_013);
Datatype mxw = getRandomVal(P_013);
Datatype myz = getRandomVal(P_013);
Datatype myw = getRandomVal(P_013);
Datatype mzw = getRandomVal(P_013);
Datatype mxyz = getRandomVal(P_013);
Datatype mxyw = getRandomVal(P_013);
Datatype mxzw = getRandomVal(P_013);
Datatype myzw = getRandomVal(P_013);

Datatype sxy = ADD(mxy, ADD(ADD(MULT(m,b.m),MULT(r,b.m)),MULT(m,b.r)));
Datatype sxz = ADD(mxz, ADD(ADD(MULT(m,c.m),MULT(r,c.m)),MULT(m,c.r)));
Datatype sxw = ADD(mxw, ADD(ADD(MULT(m,d.m),MULT(r,d.m)),MULT(m,d.r)));
Datatype syz = ADD(myz, ADD(ADD(MULT(b.m,c.m),MULT(b.r,c.m)),MULT(b.m,c.r)));
Datatype syw = ADD(myw, ADD(ADD(MULT(b.m,d.m),MULT(b.r,d.m)),MULT(b.m,d.r)));
Datatype szw = ADD(mzw, ADD(ADD(MULT(c.m,d.m),MULT(c.r,d.m)),MULT(c.m,d.r)));
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(c.m,(ADD(b.r,b.m)))))),
        MULT(b.m,(ADD(mxz,MULT(c.r,m))))
        ),
    ADD(
        MULT(c.m,(ADD(mxy,MULT(r,b.m)))),
        mxyz 
        )
    );
Datatype sxzw =
ADD(
    ADD(
        MULT(m,(ADD(mzw,MULT(d.m,(ADD(c.r,c.m)))))),
        MULT(c.m,(ADD(mxw,MULT(d.r,m))))
        ),
    ADD(
        MULT(d.m,(ADD(mxz,MULT(r,c.m)))),
        mxzw 
        )
    );
Datatype syzw =
ADD(
    ADD(
        MULT(b.m,(ADD(mzw,MULT(d.m,(ADD(c.r,c.m)))))),
        MULT(c.m,(ADD(myw,MULT(d.r,b.m))))
        ),
    ADD(
        MULT(d.m,(ADD(myz,MULT(b.r,c.m)))),
        myzw 
        )
    );
Datatype sxyw =
ADD(
    ADD(
        MULT(m,(ADD(myw,MULT(d.m,(ADD(b.r,b.m)))))),
        MULT(b.m,(ADD(mxw,MULT(d.r,m))))
        ),
    ADD(
        MULT(d.m,(ADD(mxy,MULT(r,b.m)))),
        mxyw 
        )
    );



/* Datatype sxyz = ADD(ADD(mxyz,ADD(MULT(mxy,c.m),MULT(MULT(m,b.m),c.r))),MULT(MULT(m,b.m),c.m)); */
/* Datatype sxzw = ADD(ADD(mxzw,ADD(MULT(mxz,d.m),MULT(MULT(m,c.m),d.r))),MULT(MULT(m,c.m),d.m)); */
/* Datatype syzw = ADD(ADD(myzw,ADD(MULT(myz,d.m),MULT(MULT(b.m,c.m),d.r))),MULT(MULT(b.m,c.m),d.m)); */
/* Datatype sxyw = ADD(ADD(mxyw,ADD(MULT(mxy,d.m),MULT(MULT(m,b.m),d.r))),MULT(MULT(m,b.m),d.m)); */
/* Datatype sxyzw = ADD(mxyzw,ADD(MULT(mxyz,d.m),ADD(MULT(mxy,c.r),ADD(MULT(MULT(m,b.m),MULT(c.m,d.r)),MULT(MULT(m,b.m),MULT(c.m,d.m)))))); */
Datatype sxyzw =
                ADD(
                    ADD(
                        MULT(m, ADD( MULT(d.m, ADD(myz, MULT(b.m,c.r))), myzw))
                        ,
                        MULT(b.m, ADD( MULT(m, ADD(mzw, MULT(c.m,d.r))), 
                            ADD( MULT(c.m, mxw), mxzw)))
                       )
                    ,
                    ADD(
                            MULT(c.m, ADD( MULT(m, ADD(myw, MULT(d.m,b.r))), mxyw)),
                            MULT(d.m, ADD( MULT(b.m, ADD(mxz, MULT(c.m,r))),
                                ADD( MULT(c.m, mxy), mxyz)))
                    ));

Datatype a0 = ADD(v,m);
Datatype b0 = ADD(b.v,b.m);
Datatype c0 = ADD(c.v,c.m);
Datatype d0 = ADD(d.v,d.m);
Datatype rxy = getRandomVal(P_123);
Datatype rxz = getRandomVal(P_123);
Datatype rxw = getRandomVal(P_123);
Datatype ryz = getRandomVal(P_123);
Datatype ryw = getRandomVal(P_123);
Datatype rzw = getRandomVal(P_123);
Datatype rxyz = getRandomVal(P_123);
Datatype rxyw = getRandomVal(P_123);
Datatype rxzw = getRandomVal(P_123);
Datatype ryzw = getRandomVal(P_123);
Datatype ar = ADD(r,m);
Datatype br = ADD(b.r,b.m);
Datatype cr = ADD(c.r,c.m);
Datatype dr = ADD(d.r,d.m);
OEC_MAL1_Share e;
e.r = 
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(syz, MULT(b0,cr))), syzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(szw, MULT(c0,dr))), 
                            SUB( MULT(c0, sxw), sxzw)))
                       )
                    ,
                    ADD(
                            ADD(sxyzw, MULT(c0, SUB( MULT(a0, SUB(syw, MULT(d0,br))), sxyw))),
                            MULT(d0, ADD( MULT(b0, SUB(sxz, MULT(c0,ar))),
                                SUB( MULT(c0, sxy), sxyz)))
                    ));
            
                /* ADD( */
                /*     ADD( */
                /*         MULT(a0, SUB( MULT(d0, ADD(MULT(b0,SUB(c0,cr)),syz )), syzw)) */
                /*         , */
                /*         MULT(b0, ADD( MULT(a0, SUB(szw, MULT(c0,dr))), */ 
                /*             SUB( MULT(c0, sxy), sxzw))) */
                       
                /*         ) */
                /*     , */
                /*     ADD( */
                /*         ADD(sxyzw, MULT(c0, SUB( MULT(a0, SUB(syw, MULT(d0,br))), sxyw))) */
                /*         , */
                /*         MULT(d0, ADD( MULT(b0, SUB(sxz, MULT(c0,ar))), */
                /*             SUB( MULT(c0, sxy), sxyz))) */
                /*         ) */
                /*     ); // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw */
e.v = 
            
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(ryz, MULT(b0,c.m))), ryzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.m))), 
                            SUB( MULT(c0, rxw), rxzw)))
                       )
                    ,
                    ADD(
                            MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.m))), rxyw)),
                            MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,m))),
                                SUB( MULT(c0, rxy), rxyz)))
                        )
        ); // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.r = SUB(e.v, e.r); // hack for mask_and_send_dot
return e;
}


template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_Share prepare_mult4(const OEC_MAL1_Share b,const  OEC_MAL1_Share c, const OEC_MAL1_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = getRandomVal(P_013);
Datatype mxz = getRandomVal(P_013);
Datatype mxw = getRandomVal(P_013);
Datatype myz = getRandomVal(P_013);
Datatype myw = getRandomVal(P_013);
Datatype mzw = getRandomVal(P_013);
Datatype mxyz = getRandomVal(P_013);
Datatype mxyw = getRandomVal(P_013);
Datatype mxzw = getRandomVal(P_013);
Datatype myzw = getRandomVal(P_013);
Datatype mxyzw = getRandomVal(P_013);

Datatype sxy = ADD(mxy, ADD(ADD(MULT(m,b.m),MULT(r,b.m)),MULT(m,b.r)));
Datatype sxz = ADD(mxz, ADD(ADD(MULT(m,c.m),MULT(r,c.m)),MULT(m,c.r)));
Datatype sxw = ADD(mxw, ADD(ADD(MULT(m,d.m),MULT(r,d.m)),MULT(m,d.r)));
Datatype syz = ADD(myz, ADD(ADD(MULT(b.m,c.m),MULT(b.r,c.m)),MULT(b.m,c.r)));
Datatype syw = ADD(myw, ADD(ADD(MULT(b.m,d.m),MULT(b.r,d.m)),MULT(b.m,d.r)));
Datatype szw = ADD(mzw, ADD(ADD(MULT(c.m,d.m),MULT(c.r,d.m)),MULT(c.m,d.r)));
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(c.m,(ADD(b.r,b.m)))))),
        MULT(b.m,(ADD(mxz,MULT(c.r,m))))
        ),
    ADD(
        MULT(c.m,(ADD(mxy,MULT(r,b.m)))),
        mxyz 
        )
    );
Datatype sxzw =
ADD(
    ADD(
        MULT(m,(ADD(mzw,MULT(d.m,(ADD(c.r,c.m)))))),
        MULT(c.m,(ADD(mxw,MULT(d.r,m))))
        ),
    ADD(
        MULT(d.m,(ADD(mxz,MULT(r,c.m)))),
        mxzw 
        )
    );
Datatype syzw =
ADD(
    ADD(
        MULT(b.m,(ADD(mzw,MULT(d.m,(ADD(c.r,c.m)))))),
        MULT(c.m,(ADD(myw,MULT(d.r,b.m))))
        ),
    ADD(
        MULT(d.m,(ADD(myz,MULT(b.r,c.m)))),
        myzw 
        )
    );
Datatype sxyw =
ADD(
    ADD(
        MULT(m,(ADD(myw,MULT(d.m,(ADD(b.r,b.m)))))),
        MULT(b.m,(ADD(mxw,MULT(d.r,m))))
        ),
    ADD(
        MULT(d.m,(ADD(mxy,MULT(r,b.m)))),
        mxyw 
        )
    );



/* Datatype sxyz = ADD(ADD(mxyz,ADD(MULT(mxy,c.m),MULT(MULT(m,b.m),c.r))),MULT(MULT(m,b.m),c.m)); */
/* Datatype sxzw = ADD(ADD(mxzw,ADD(MULT(mxz,d.m),MULT(MULT(m,c.m),d.r))),MULT(MULT(m,c.m),d.m)); */
/* Datatype syzw = ADD(ADD(myzw,ADD(MULT(myz,d.m),MULT(MULT(b.m,c.m),d.r))),MULT(MULT(b.m,c.m),d.m)); */
/* Datatype sxyw = ADD(ADD(mxyw,ADD(MULT(mxy,d.m),MULT(MULT(m,b.m),d.r))),MULT(MULT(m,b.m),d.m)); */
/* Datatype sxyzw = ADD(mxyzw,ADD(MULT(mxyz,d.m),ADD(MULT(mxy,c.r),ADD(MULT(MULT(m,b.m),MULT(c.m,d.r)),MULT(MULT(m,b.m),MULT(c.m,d.m)))))); */
Datatype sxyzw =
                ADD(
                    ADD(
                        MULT(m, ADD( MULT(d.m, ADD(myz, MULT(b.m,c.r))), myzw))
                        ,
                        MULT(b.m, ADD( MULT(m, ADD(mzw, MULT(c.m,d.r))), 
                            ADD( MULT(c.m, mxw), mxzw)))
                       )
                    ,
                    ADD(
                            ADD(mxyzw, MULT(c.m, ADD( MULT(m, ADD(myw, MULT(d.m,b.r))), mxyw))),
                            MULT(d.m, ADD( MULT(b.m, ADD(mxz, MULT(c.m,r))),
                                ADD( MULT(c.m, mxy), mxyz)))
                    ));

Datatype a0 = ADD(v,m);
Datatype b0 = ADD(b.v,b.m);
Datatype c0 = ADD(c.v,c.m);
Datatype d0 = ADD(d.v,d.m);
Datatype rxy = getRandomVal(P_123);
Datatype rxz = getRandomVal(P_123);
Datatype rxw = getRandomVal(P_123);
Datatype ryz = getRandomVal(P_123);
Datatype ryw = getRandomVal(P_123);
Datatype rzw = getRandomVal(P_123);
Datatype rxyz = getRandomVal(P_123);
Datatype rxyw = getRandomVal(P_123);
Datatype rxzw = getRandomVal(P_123);
Datatype ryzw = getRandomVal(P_123);
Datatype rxyzw = getRandomVal(P_123);
Datatype ar = ADD(r,m);
Datatype br = ADD(b.r,b.m);
Datatype cr = ADD(c.r,c.m);
Datatype dr = ADD(d.r,d.m);
OEC_MAL1_Share e;
e.v = 
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(syz, MULT(b0,cr))), syzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(szw, MULT(c0,dr))), 
                            SUB( MULT(c0, sxw), sxzw)))
                       )
                    ,
                    ADD(
                            ADD(sxyzw, MULT(c0, SUB( MULT(a0, SUB(syw, MULT(d0,br))), sxyw))),
                            MULT(d0, ADD( MULT(b0, SUB(sxz, MULT(c0,ar))),
                                SUB( MULT(c0, sxy), sxyz)))
                    ));
            
                /* ADD( */
                /*     ADD( */
                /*         MULT(a0, SUB( MULT(d0, ADD(MULT(b0,SUB(c0,cr)),syz )), syzw)) */
                /*         , */
                /*         MULT(b0, ADD( MULT(a0, SUB(szw, MULT(c0,dr))), */ 
                /*             SUB( MULT(c0, sxy), sxzw))) */
                       
                /*         ) */
                /*     , */
                /*     ADD( */
                /*         ADD(sxyzw, MULT(c0, SUB( MULT(a0, SUB(syw, MULT(d0,br))), sxyw))) */
                /*         , */
                /*         MULT(d0, ADD( MULT(b0, SUB(sxz, MULT(c0,ar))), */
                /*             SUB( MULT(c0, sxy), sxyz))) */
                /*         ) */
                /*     ); // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw */
Datatype m20 = 
            
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(ryz, MULT(b0,c.m))), ryzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.m))), 
                            SUB( MULT(c0, rxw), rxzw)))
                       )
                    ,
                    ADD(
                            ADD(rxyzw, MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.m))), rxyw))),
                            MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,m))),
                                SUB( MULT(c0, rxy), rxyz)))
                        )
        ); // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.m = getRandomVal(P_123);
e.r = getRandomVal(P_013);
e.v = ADD(e.v,e.r);
store_compare_view(P_0, ADD(m20, e.m)); // + s
send_to_live(P_2,e.v);
return e;
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
Datatype m21 = receive_from_live(P_2);
v = ADD(v,m21);
store_compare_view(P_012, ADD(v,m)); //compare d_0 s
}

#endif

};

