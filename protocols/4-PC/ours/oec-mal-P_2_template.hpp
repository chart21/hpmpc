#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL2_Share
{
    private:
    Datatype v;
    Datatype r;
#if PROTOCOL == 11 || FRACTIONAL > 0 || MULTI_INPUT == 1
    Datatype m;
#endif

public:
    
OEC_MAL2_Share() {}
OEC_MAL2_Share(Datatype v, Datatype r) : v(v), r(r) {}
#if MULTI_INPUT == 1
OEC_MAL2_Share(Datatype v, Datatype r, Datatype m) : v(v), r(r), m(m) {}
#endif
OEC_MAL2_Share(Datatype v) : v(v) {}

OEC_MAL2_Share public_val(Datatype a)
{
    #if MULTI_INPUT == 1
    return OEC_MAL2_Share(a,SET_ALL_ZERO(),SET_ALL_ZERO());
    #else
    return OEC_MAL2_Share(a,SET_ALL_ZERO());
    #endif
}

OEC_MAL2_Share Not() const
{
#if MULTI_INPUT == 1
    return OEC_MAL2_Share(NOT(v),r,m);
#else
   return OEC_MAL2_Share(NOT(v),r);
#endif
}

template <typename func_add>
OEC_MAL2_Share Add(OEC_MAL2_Share b, func_add ADD) const
{
#if MULTI_INPUT == 1
    return OEC_MAL2_Share(ADD(v,b.v),ADD(r,b.r),ADD(m,b.m));
#else
   return OEC_MAL2_Share(ADD(v,b.v),ADD(r,b.r));
#endif
}


template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_mult(OEC_MAL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    //1* get Random val
    OEC_MAL2_Share c;
    c.r = getRandomVal(P_023);
   /* c.r = ADD(getRandomVal(P_023), getRandomVal(P_123)); */
   /* Datatype r234 = getRandomVal(P_123); */
   /* Datatype r234_1 = */
   /*     getRandomVal(P_123); // Probably sufficient to only generate with P_3 -> */
   Datatype r234_2 =
       getRandomVal(P_123); // Probably sufficient to only generate with P_3 ->
                           // Probably not because of verification
/* c.r = getRandomVal(P_3); */
#if PROTOCOL == 12
#if PRE == 1
   Datatype o1 = pre_receive_from_live(P_3);
#else
   Datatype o1 = receive_from_live(P_3);
#endif
   store_compare_view(P_0, o1);
#else
   Datatype o1 = receive_from_live(P_0);
   store_compare_view(P_3, o1);
#endif
   c.v = ADD(SUB(MULT(v, b.r),o1), MULT(b.v, r));
   send_to_live(P_1, c.v);


   /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */
   Datatype a1b1 = MULT(v, b.v);

#if PROTOCOL == 10 || PROTOCOL == 12
   /* Datatype m3_prime = XOR(XOR(r234, c.r), c.v); */
   /* Datatype m3_prime = XOR(XOR(r234, c.r), AND(XOR(a.v, a.r), XOR(b.v, b.r))); */
   /* send_to_live(P_0, ADD(r234 ,ADD(c.v,c.r))); */
   send_to_live(P_0, ADD(a1b1,r234_2));
#endif
#if PROTOCOL == 11
c.m = ADD(c.v, r234_2); // store m_2 + m_3 + r_234_2 to send P_0 later
#endif
   c.v = SUB(a1b1, c.v);
   /* c.m = ADD(c.m, r234); */
return c;
}

template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL2_Share prepare_dot(const OEC_MAL2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL2_Share c;
c.r = ADD(MULT(v, b.r), MULT(b.v, r)); // a0 y_2 + b_0 x_2
c.v = MULT(v, b.v); // a0b0
return c;
}

#if FRACTIONAL > 0

    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
r = SUB(r, getRandomVal(P_023));// a_0 y_2 + b_0 x_2 - r_0,2,3   
send_to_live(P_1, r); 

}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
r = ADD(r, receive_from_live(P_1)); // v^1,2 = m^1 + m^2
v = TRUNC(SUB(v, r)); // [a_0 b_0 - v^1,2]^t



#if PROTOCOL == 11
send_to_live(P_0, ADD(r,getRandomVal(P_123))); // send m1 + m2 + r123 to P_0
#else
store_compare_view(P_012,ADD(r, getRandomVal(P_123))); // v^1,2 + r_1,2,3
#endif


#if MULTI_INPUT == 1
m = getRandomVal(P_123); // w
send_to_live(P_0,ADD(v, m)); // c_0 + w
#else
send_to_live(P_0,ADD(v, getRandomVal(P_123))); // c_0 + w
#endif

#if PROTOCOL == 12
#if PRE == 1
r = pre_receive_from_live(P_3); // z_2 = m0
#else
r = receive_from_live(P_3); // z_2 = m0 
#endif
store_compare_view(P_0, r); // compare view of m0
#else
#if PRE == 1
r = pre_receive_from_live(P_0); // z_2 = m0
#else
r = receive_from_live(P_0); // z_2 = m0 
#endif
store_compare_view(P_3, r); // compare view of m0
#endif

}

#endif

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
Datatype cr = getRandomVal(P_023);
   /* c.r = ADD(getRandomVal(P_023), getRandomVal(P_123)); */
   /* Datatype r234 = getRandomVal(P_123); */
   /* Datatype r234_1 = */
   /*     getRandomVal(P_123); // Probably sufficient to only generate with P_3 -> */
   Datatype r234_2 =
       getRandomVal(P_123); // Probably sufficient to only generate with P_3 ->
                           // Probably not because of verification
/* c.r = getRandomVal(P_3); */
#if PROTOCOL == 12
#if PRE == 1
   Datatype o1 = pre_receive_from_live(P_3);
#else
   Datatype o1 = receive_from_live(P_3);
#endif
   store_compare_view(P_0, o1);
#else
   Datatype o1 = receive_from_live(P_0);
   store_compare_view(P_3, o1);
#endif
   r = SUB(r,o1);
   send_to_live(P_1, r); // if one party xored with v (m20) here for multi input and gates it should cancel out 


   /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */

#if PROTOCOL == 10 || PROTOCOL == 12
   send_to_live(P_0, ADD(v,r234_2));
#endif
#if PROTOCOL == 11
m = ADD(r, r234_2); // store m_2 + m_3 + r_234_2 to send P_0 later
#endif
   v = SUB(v, r); // For multi input AND gates, we should not add those two -> possible hack: both parties Add v to r -> v cancels out
   /* c.m = ADD(c.m, r234); */
r = cr;

}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
Datatype m2 = receive_from_live(P_1);
v = SUB(v, m2);
/* c.m = XOR(c.m, m2); */
/* Datatype cm = XOR(c.m, m2); */
#if PROTOCOL == 11
send_to_live(P_0, ADD(m, m2)); // let P_0 verify m_2 XOR m_3, obtain m_2 + m_3 + r_234_2
#if MULTI_INPUT == 1
m = getRandomVal(P_123);
send_to_live(P_0, ADD(v, m)); // send c_0 + w to P_0
#else
send_to_live(P_0, ADD(v,getRandomVal(P_123))); // let P_0 obtain ab + c0 + w
#endif
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
#if MULTI_INPUT == 1
m = getRandomVal(P_123);
store_compare_view(P_012, ADD(v, m)); // send c_0 + w to P_0
#else
store_compare_view(P_012, ADD(getRandomVal(P_123), v)); // compare ab + c1 + r234_1
#endif
#endif
/* store_compare_view(P_0, c.v); */
}


void prepare_reveal_to_all()
{
}

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
Datatype r0 = receive_from_live(P_0);
Datatype result = SUB(v, r0);
store_compare_view(P_123, r0);
store_compare_view(P_0123, result);
return result;
}


template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    Datatype x_0 = getRandomVal(P_023);
    Datatype u = getRandomVal(P_123);
    r = x_0; //  = x_2, x_1 = 0
    v = ADD(get_input_live(),x_0);
    send_to_live(P_0,ADD(v,u));
    send_to_live(P_1,ADD(v,u));
#if MULTI_INPUT == 1
    m = u;
#endif
}
else if constexpr(id == P_0)
{
    r = getRandomVal(P_023);
    v = SET_ALL_ZERO();
    // u = 0
#if MULTI_INPUT == 1
    m = SET_ALL_ZERO();
#endif
}
else if constexpr(id == P_1)
{
    r = SET_ALL_ZERO();
    v = getRandomVal(P_123); //u
#if MULTI_INPUT == 1
    m = v;
#endif
}
else if constexpr(id == P_3)
{
    r = getRandomVal(P_023); //x2
    v = getRandomVal(P_123); //u
#if MULTI_INPUT == 1
    m = v;
#endif
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id != PSELF)
{
    
    #if PRE == 1
        Datatype val;
        if constexpr(id == P_3)
            val = pre_receive_from_live(P_3);
        else
            val = receive_from_live(id);
    #else
    Datatype val = receive_from_live(id);
    #endif
    if constexpr(id != P_0)
            store_compare_view(P_0,val);
    if constexpr(id != P_1)
            store_compare_view(P_1,val);
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

static void prepare_A2B_S1(int k, OEC_MAL2_Share in[], OEC_MAL2_Share out[])
{
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = in[j].v; //a0 
        }
    unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp);
    orthogonalize_boolean((UINT_TYPE*) temp, temp);
    for (int j = 0; j < k; j++)
    {
            out[j].v = temp[j];
            out[j].r = SET_ALL_ZERO(); // set share to 0
            #if MULTI_INPUT == 1
            out[j].m = getRandomVal(P_123);
            send_to_live(P_0, FUNC_XOR(out[j].v, out[j].m)); 
            #else
            send_to_live(P_0, FUNC_XOR(out[j].v, getRandomVal(P_123))); 
            #endif
    }

}


static void prepare_A2B_S2(int k, OEC_MAL2_Share in[], OEC_MAL2_Share out[])
{
    for(int i = 0; i < k; i++)
    {
        out[i].v = SET_ALL_ZERO();
        #if MULTI_INPUT == 1
        out[i].m = SET_ALL_ZERO();
        #endif
    }
}

static void complete_A2B_S1(int k, OEC_MAL2_Share out[])
{
}

static void complete_A2B_S2(int k, OEC_MAL2_Share out[])
{
    for(int i = 0; i < k; i++)
    {
    #if PROTOCOL != 12
        #if PRE == 0
        Datatype m0 = receive_from_live(P_0);
        #else
        Datatype m0 = pre_receive_from_live(P_0);
        #endif
        store_compare_view(P_3, m0);
    #else
        #if PRE == 0
        Datatype m0 = receive_from_live(P_3);
        #else
        Datatype m0 = pre_receive_from_live(P_3);
        #endif
        store_compare_view(P_0, m0);
    #endif

        out[i].r = m0;
    }

}

void prepare_bit_injection_S1(OEC_MAL2_Share out[])
{
    Datatype temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = v;
    unorthogonalize_boolean(temp, (UINT_TYPE*) temp);
    orthogonalize_arithmetic((UINT_TYPE*) temp, temp);
    for (int j = 0; j < BITLENGTH; j++)
    {
        out[j].r = SET_ALL_ZERO(); // set share to 0
        out[j].v = temp[j];
            #if MULTI_INPUT == 1
            out[j].m = getRandomVal(P_123);
            send_to_live(P_0, OP_ADD(out[j].v, out[j].m)); 
            #else
            send_to_live(P_0, OP_ADD(out[j].v, getRandomVal(P_123))); 
            #endif
    }
}

void prepare_bit_injection_S2(OEC_MAL2_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].v = SET_ALL_ZERO();
        #if MULTI_INPUT == 1
        out[i].m = SET_ALL_ZERO();
        #endif
    }
}

static void complete_bit_injection_S1(OEC_MAL2_Share out[])
{
    
}

static void complete_bit_injection_S2(OEC_MAL2_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        #if PROTOCOL != 12
        #if PRE == 0
        Datatype m0 = receive_from_live(P_0);
        #else
        Datatype m0 = pre_receive_from_live(P_0);
        #endif
        store_compare_view(P_3, m0);
#else
        #if PRE == 0
        Datatype m0 = receive_from_live(P_3);
        #else
        Datatype m0 = pre_receive_from_live(P_3);
        #endif
        store_compare_view(P_0, m0);
#endif
        out[i].r = OP_SUB(SET_ALL_ZERO(), m0);
    }
}

#if MULTI_INPUT == 1

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_dot3(const OEC_MAL2_Share b, const OEC_MAL2_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PROTOCOL == 12
#if PRE == 1
Datatype mxy = pre_receive_from_live(P_3);
Datatype mxz = pre_receive_from_live(P_3);
Datatype myz = pre_receive_from_live(P_3);
#else
Datatype mxy = receive_from_live(P_3);
Datatype mxz = receive_from_live(P_3);
Datatype myz = receive_from_live(P_3);
#endif
store_compare_view(P_0, mxy);
store_compare_view(P_0, mxz);
store_compare_view(P_0, myz);
#else
Datatype mxy = receive_from_live(P_0);
Datatype mxz = receive_from_live(P_0);
Datatype myz = receive_from_live(P_0);
store_compare_view(P_3, mxy);
store_compare_view(P_3, mxz);
store_compare_view(P_3, myz);
#endif
Datatype sxy = ADD(mxy,ADD(MULT(r,b.m),MULT(m,b.r)));
Datatype sxz = ADD(mxz,ADD(MULT(r,c.m),MULT(m,c.r)));
Datatype syz = ADD(myz,ADD(MULT(b.r,c.m),MULT(b.m,c.r)));
/* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), MULT(MULT(m,b.m),c.r)); */
/* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, */ 
/*                     MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), */ 
/*         ADD( MULT(MULT(m,b.m),c.r), ADD(MULT(MULT(m,c.m),b.r),MULT(MULT(m,b.m),c.r)))); */
/* Datatype sxyz = */
/* ADD(mxyz, */
/* ADD( */
/*     ADD(ADD(MULT(mxy,c.m),MULT(mxz,b.m)),MULT(myz,m)), */
/*     ADD(ADD(MULT(MULT(m,b.m),c.r),MULT(MULT(m,c.m),b.r)),MULT(MULT(b.m,c.m),r)) */
/*     )); */
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(b.r,c.m)))),
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
OEC_MAL2_Share d;
d.r = SUB(ADD(
        ADD( MULT(a0, ADD(syz, MULT(b0,SUB(c0,c.r))))
        ,(MULT(b0,SUB(sxz, MULT(c0,r)))))
        ,MULT(c0,SUB(sxy, MULT(a0,b.r)))), sxyz); // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.v = ADD(
        ADD( MULT(a0,SUB(ryz,MULT(b0,c.m)))
        ,(MULT(b0,SUB(rxz, MULT(c0,m)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.m)))); // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
/* d.r = ADD(d.r, d.v); // hack for mask_and_send_dot */
d.r = SUB(SET_ALL_ZERO(), d.r); // hack for mask_and_send_dot

return d;
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_mult3(const OEC_MAL2_Share b, const OEC_MAL2_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PROTOCOL == 12
#if PRE == 1
Datatype mxy = pre_receive_from_live(P_3);
Datatype mxz = pre_receive_from_live(P_3);
Datatype myz = pre_receive_from_live(P_3);
Datatype mxyz = pre_receive_from_live(P_3);
#else
Datatype mxy = receive_from_live(P_3);
Datatype mxz = receive_from_live(P_3);
Datatype myz = receive_from_live(P_3);
Datatype mxyz = receive_from_live(P_3);
#endif
store_compare_view(P_0, mxy);
store_compare_view(P_0, mxz);
store_compare_view(P_0, myz);
store_compare_view(P_0, mxyz);
#else
Datatype mxy = receive_from_live(P_0);
Datatype mxz = receive_from_live(P_0);
Datatype myz = receive_from_live(P_0);
Datatype mxyz = receive_from_live(P_0);
store_compare_view(P_3, mxy);
store_compare_view(P_3, mxz);
store_compare_view(P_3, myz);
store_compare_view(P_3, mxyz);
#endif
Datatype sxy = ADD(mxy,ADD(MULT(r,b.m),MULT(m,b.r)));
Datatype sxz = ADD(mxz,ADD(MULT(r,c.m),MULT(m,c.r)));
Datatype syz = ADD(myz,ADD(MULT(b.r,c.m),MULT(b.m,c.r)));
/* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), MULT(MULT(m,b.m),c.r)); */
/* Datatype sxyz = ADD(ADD(ADD(ADD(mxyz, */ 
/*                     MULT(mxy,c.m)),MULT(mxz,b.m)),MULT(myz,m)), */ 
/*         ADD( MULT(MULT(m,b.m),c.r), ADD(MULT(MULT(m,c.m),b.r),MULT(MULT(m,b.m),c.r)))); */
/* Datatype sxyz = */
/* ADD(mxyz, */
/* ADD( */
/*     ADD(ADD(MULT(mxy,c.m),MULT(mxz,b.m)),MULT(myz,m)), */
/*     ADD(ADD(MULT(MULT(m,b.m),c.r),MULT(MULT(m,c.m),b.r)),MULT(MULT(b.m,c.m),r)) */
/*     )); */
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(b.r,c.m)))),
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
OEC_MAL2_Share d;
d.v = SUB(ADD(
        ADD( MULT(a0, ADD(syz, MULT(b0,SUB(c0,c.r))))
        ,(MULT(b0,SUB(sxz, MULT(c0,r)))))
        ,MULT(c0,SUB(sxy, MULT(a0,b.r)))), sxyz); // a0(b0(c0 + ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
Datatype m20 = SUB(ADD(
        ADD( MULT(a0,SUB(ryz,MULT(b0,c.m)))
        ,(MULT(b0,SUB(rxz, MULT(c0,m)))))
        ,MULT(c0,SUB(rxy, MULT(a0,b.m)))), rxyz); // a0(b0(ryz-z1) + b0(rxz- c0 x1) + c0(rxy- a0 y1)) - rxyz
d.m = getRandomVal(P_123);
d.r = getRandomVal(P_023);
d.v = ADD(d.v,d.r);
send_to_live(P_0, ADD(m20,d.m)); // + s
send_to_live(P_1,d.v);
return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
Datatype m21 = receive_from_live(P_1);
v = ADD(v,m21);
store_compare_view(P_012, ADD(v,m)); //compare d_0 s
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_dot4(const OEC_MAL2_Share b, const OEC_MAL2_Share c, const OEC_MAL2_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PROTOCOL == 12
#if PRE == 1
Datatype mxy = pre_receive_from_live(P_3);
Datatype mxz = pre_receive_from_live(P_3);
Datatype mxw = pre_receive_from_live(P_3);
Datatype myz = pre_receive_from_live(P_3);
Datatype myw = pre_receive_from_live(P_3);
Datatype mzw = pre_receive_from_live(P_3);
Datatype mxyz = pre_receive_from_live(P_3);
Datatype mxyw = pre_receive_from_live(P_3);
Datatype mxzw = pre_receive_from_live(P_3);
Datatype myzw = pre_receive_from_live(P_3);
#else
Datatype mxy = receive_from_live(P_3);
Datatype mxz = receive_from_live(P_3);
Datatype mxw = receive_from_live(P_3);
Datatype myz = receive_from_live(P_3);
Datatype myw = receive_from_live(P_3);
Datatype mzw = receive_from_live(P_3);
Datatype mxyz = receive_from_live(P_3);
Datatype mxyw = receive_from_live(P_3);
Datatype mxzw = receive_from_live(P_3);
Datatype myzw = receive_from_live(P_3);
#endif
store_compare_view(P_0, mxy);
store_compare_view(P_0, mxz);
store_compare_view(P_0, mxw);
store_compare_view(P_0, myz);
store_compare_view(P_0, myw);
store_compare_view(P_0, mzw);
store_compare_view(P_0, mxyz);
store_compare_view(P_0, mxyw);
store_compare_view(P_0, mxzw);
store_compare_view(P_0, myzw);
#else
Datatype mxy = receive_from_live(P_0);
Datatype mxz = receive_from_live(P_0);
Datatype mxw = receive_from_live(P_0);
Datatype myz = receive_from_live(P_0);
Datatype myw = receive_from_live(P_0);
Datatype mzw = receive_from_live(P_0);
Datatype mxyz = receive_from_live(P_0);
Datatype mxyw = receive_from_live(P_0);
Datatype mxzw = receive_from_live(P_0);
Datatype myzw = receive_from_live(P_0);
store_compare_view(P_3, mxy);
store_compare_view(P_3, mxz);
store_compare_view(P_3, mxw);
store_compare_view(P_3, myz);
store_compare_view(P_3, myw);
store_compare_view(P_3, mzw);
store_compare_view(P_3, mxyz);
store_compare_view(P_3, mxyw);
store_compare_view(P_3, mxzw);
store_compare_view(P_3, myzw);
#endif
Datatype sxy = ADD(mxy,ADD(MULT(r,b.m),MULT(m,b.r)));
Datatype sxz = ADD(mxz,ADD(MULT(r,c.m),MULT(m,c.r)));
Datatype sxw = ADD(mxw,ADD(MULT(r,d.m),MULT(m,d.r)));
Datatype syz = ADD(myz,ADD(MULT(b.r,c.m),MULT(b.m,c.r)));
Datatype syw = ADD(myw,ADD(MULT(b.r,d.m),MULT(b.m,d.r)));
Datatype szw = ADD(mzw,ADD(MULT(c.r,d.m),MULT(c.m,d.r)));
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(b.r,c.m)))),
        MULT(b.m,(ADD(mxz,MULT(c.r,m))))
        ),
    ADD(
        MULT(c.m,(ADD(mxy,MULT(r,b.m)))),
        mxyz 
        )
    );
Datatype sxyw =
ADD(
    ADD(
        MULT(m,(ADD(myw,MULT(b.r,d.m)))),
        MULT(b.m,(ADD(mxw,MULT(d.r,m))))
        ),
    ADD(
        MULT(d.m,(ADD(mxy,MULT(r,b.m)))),
        mxyw 
        )
    );
Datatype sxzw =
ADD(
    ADD(
        MULT(m,(ADD(mzw,MULT(c.r,d.m)))),
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
        MULT(b.m,(ADD(mzw,MULT(c.r,d.m)))),
        MULT(c.m,(ADD(myw,MULT(d.r,b.m))))
        ),
    ADD(
        MULT(d.m,(ADD(myz,MULT(b.r,c.m)))),
        myzw 
        )
    );
Datatype sxyzw =
                ADD(
                    ADD(
                        MULT(m, ADD( MULT(d.m, ADD(MULT(b.m,ADD(c.m,c.r)),myz )), myzw))
                        ,
                        MULT(b.m, ADD( MULT(m, ADD(mzw, MULT(c.m,d.r))), 
                            ADD( MULT(c.m, mxw), mxzw))))
                    ,
                    ADD(
                        MULT(c.m, ADD( MULT(m, ADD(myw, MULT(d.m,b.r))), mxyw))
                        ,
                        MULT(d.m, ADD( MULT(b.m, ADD(mxz, MULT(c.m,r))),
                            ADD( MULT(c.m, mxy), mxyz)))
                    )
        ); 

/* Datatype sxyz = ADD(ADD(mxyz,MULT(mxy,c.m)), MULT(MULT(m,b.m),c.r)); */
/* Datatype sxyw = ADD(ADD(mxyw,MULT(mxy,d.m)), MULT(MULT(m,b.m),d.r)); */
/* Datatype sxzw = ADD(ADD(mxzw,MULT(mxz,d.m)), MULT(MULT(m,c.m),d.r)); */
/* Datatype syzw = ADD(ADD(myzw,MULT(myz,d.m)), MULT(MULT(m,b.m),d.r)); */
/* Datatype sxyzw = ADD(  ADD(  ADD(mxyzw,MULT(mxyz,d.m))  , MULT(mxy,c.r)), ADD(MULT(MULT(m,b.m),c.m),d.r)); */

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
OEC_MAL2_Share e;
e.r = 
     
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, ADD(MULT(b0,SUB(c0,c.r)),syz )), syzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(szw, MULT(c0,d.r))), 
                            SUB( MULT(c0, sxw), sxzw))))
                    ,
                    ADD(
                        ADD( sxyzw, MULT(c0, SUB( MULT(a0, SUB(syw, MULT(d0,b.r))), sxyw)))
                        ,
                        MULT(d0, ADD( MULT(b0, SUB(sxz, MULT(c0,r))),
                            SUB( MULT(c0, sxy), sxyz)))
                    )
        ); // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.v = 
            
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(ryz, MULT(b0,c.m))), ryzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.m))), 
                            SUB( MULT(c0, rxw), rxzw))))
                    ,
                    ADD(
                        MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.m))), rxyw)),
                        MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,m))),
                            SUB( MULT(c0, rxy), rxyz)))
                    )
            
                ); // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
/* e.r = ADD(e.r, e.v); // hack for mask_and_send_dot */
e.r = SUB(SET_ALL_ZERO(), e.r); // hack for mask_and_send_dot
return e;
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_Share prepare_mult4(const OEC_MAL2_Share b, const OEC_MAL2_Share c, const OEC_MAL2_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PROTOCOL == 12
#if PRE == 1
Datatype mxy = pre_receive_from_live(P_3);
Datatype mxz = pre_receive_from_live(P_3);
Datatype mxw = pre_receive_from_live(P_3);
Datatype myz = pre_receive_from_live(P_3);
Datatype myw = pre_receive_from_live(P_3);
Datatype mzw = pre_receive_from_live(P_3);
Datatype mxyz = pre_receive_from_live(P_3);
Datatype mxyw = pre_receive_from_live(P_3);
Datatype mxzw = pre_receive_from_live(P_3);
Datatype myzw = pre_receive_from_live(P_3);
Datatype mxyzw = pre_receive_from_live(P_3);
#else
Datatype mxy = receive_from_live(P_3);
Datatype mxz = receive_from_live(P_3);
Datatype mxw = receive_from_live(P_3);
Datatype myz = receive_from_live(P_3);
Datatype myw = receive_from_live(P_3);
Datatype mzw = receive_from_live(P_3);
Datatype mxyz = receive_from_live(P_3);
Datatype mxyw = receive_from_live(P_3);
Datatype mxzw = receive_from_live(P_3);
Datatype myzw = receive_from_live(P_3);
Datatype mxyzw = receive_from_live(P_3);
#endif
store_compare_view(P_0, mxy);
store_compare_view(P_0, mxz);
store_compare_view(P_0, mxw);
store_compare_view(P_0, myz);
store_compare_view(P_0, myw);
store_compare_view(P_0, mzw);
store_compare_view(P_0, mxyz);
store_compare_view(P_0, mxyw);
store_compare_view(P_0, mxzw);
store_compare_view(P_0, myzw);
store_compare_view(P_0, mxyzw);
#else
Datatype mxy = receive_from_live(P_0);
Datatype mxz = receive_from_live(P_0);
Datatype mxw = receive_from_live(P_0);
Datatype myz = receive_from_live(P_0);
Datatype myw = receive_from_live(P_0);
Datatype mzw = receive_from_live(P_0);
Datatype mxyz = receive_from_live(P_0);
Datatype mxyw = receive_from_live(P_0);
Datatype mxzw = receive_from_live(P_0);
Datatype myzw = receive_from_live(P_0);
Datatype mxyzw = receive_from_live(P_0);
store_compare_view(P_3, mxy);
store_compare_view(P_3, mxz);
store_compare_view(P_3, mxw);
store_compare_view(P_3, myz);
store_compare_view(P_3, myw);
store_compare_view(P_3, mzw);
store_compare_view(P_3, mxyz);
store_compare_view(P_3, mxyw);
store_compare_view(P_3, mxzw);
store_compare_view(P_3, myzw);
store_compare_view(P_3, mxyzw);
#endif
Datatype sxy = ADD(mxy,ADD(MULT(r,b.m),MULT(m,b.r)));
Datatype sxz = ADD(mxz,ADD(MULT(r,c.m),MULT(m,c.r)));
Datatype sxw = ADD(mxw,ADD(MULT(r,d.m),MULT(m,d.r)));
Datatype syz = ADD(myz,ADD(MULT(b.r,c.m),MULT(b.m,c.r)));
Datatype syw = ADD(myw,ADD(MULT(b.r,d.m),MULT(b.m,d.r)));
Datatype szw = ADD(mzw,ADD(MULT(c.r,d.m),MULT(c.m,d.r)));
Datatype sxyz =
ADD(
    ADD(
        MULT(m,(ADD(myz,MULT(b.r,c.m)))),
        MULT(b.m,(ADD(mxz,MULT(c.r,m))))
        ),
    ADD(
        MULT(c.m,(ADD(mxy,MULT(r,b.m)))),
        mxyz 
        )
    );
Datatype sxyw =
ADD(
    ADD(
        MULT(m,(ADD(myw,MULT(b.r,d.m)))),
        MULT(b.m,(ADD(mxw,MULT(d.r,m))))
        ),
    ADD(
        MULT(d.m,(ADD(mxy,MULT(r,b.m)))),
        mxyw 
        )
    );
Datatype sxzw =
ADD(
    ADD(
        MULT(m,(ADD(mzw,MULT(c.r,d.m)))),
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
        MULT(b.m,(ADD(mzw,MULT(c.r,d.m)))),
        MULT(c.m,(ADD(myw,MULT(d.r,b.m))))
        ),
    ADD(
        MULT(d.m,(ADD(myz,MULT(b.r,c.m)))),
        myzw 
        )
    );
Datatype sxyzw =
                ADD(
                    ADD(
                        MULT(m, ADD( MULT(d.m, ADD(MULT(b.m,ADD(c.m,c.r)),myz )), myzw))
                        ,
                        MULT(b.m, ADD( MULT(m, ADD(mzw, MULT(c.m,d.r))), 
                            ADD( MULT(c.m, mxw), mxzw))))
                    ,
                    ADD(
                        ADD( mxyzw, MULT(c.m, ADD( MULT(m, ADD(myw, MULT(d.m,b.r))), mxyw)))
                        ,
                        MULT(d.m, ADD( MULT(b.m, ADD(mxz, MULT(c.m,r))),
                            ADD( MULT(c.m, mxy), mxyz)))
                    )
        ); 

/* Datatype sxyz = ADD(ADD(mxyz,MULT(mxy,c.m)), MULT(MULT(m,b.m),c.r)); */
/* Datatype sxyw = ADD(ADD(mxyw,MULT(mxy,d.m)), MULT(MULT(m,b.m),d.r)); */
/* Datatype sxzw = ADD(ADD(mxzw,MULT(mxz,d.m)), MULT(MULT(m,c.m),d.r)); */
/* Datatype syzw = ADD(ADD(myzw,MULT(myz,d.m)), MULT(MULT(m,b.m),d.r)); */
/* Datatype sxyzw = ADD(  ADD(  ADD(mxyzw,MULT(mxyz,d.m))  , MULT(mxy,c.r)), ADD(MULT(MULT(m,b.m),c.m),d.r)); */

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
OEC_MAL2_Share e;
e.v = 
     
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, ADD(MULT(b0,SUB(c0,c.r)),syz )), syzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(szw, MULT(c0,d.r))), 
                            SUB( MULT(c0, sxw), sxzw))))
                    ,
                    ADD(
                        ADD( sxyzw, MULT(c0, SUB( MULT(a0, SUB(syw, MULT(d0,b.r))), sxyw)))
                        ,
                        MULT(d0, ADD( MULT(b0, SUB(sxz, MULT(c0,r))),
                            SUB( MULT(c0, sxy), sxyz)))
                    )
        ); // a0(d0(b0(c0 - z1) + ryz) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
Datatype m20 = 
            
                ADD(
                    ADD(
                        MULT(a0, SUB( MULT(d0, SUB(ryz, MULT(b0,c.m))), ryzw))
                        ,
                        MULT(b0, ADD( MULT(a0, SUB(rzw, MULT(c0,d.m))), 
                            SUB( MULT(c0, rxw), rxzw))))
                    ,
                    ADD(
                        ADD(rxyzw, MULT(c0, SUB( MULT(a0, SUB(ryw, MULT(d0,b.m))), rxyw))),
                        MULT(d0, ADD( MULT(b0, SUB(rxz, MULT(c0,m))),
                            SUB( MULT(c0, rxy), rxyz)))
                    )
            
                ); // a0(d0(ryz-b0z1) - ryzw) + b0(a0(rzw-c0w1) + c0rxy - rxzw) + c0(a0(ryw-d0y1) - rxyw) + d0(b0(rxz-c0x1) + c0rxy - rxyz) + rxyzw
e.m = getRandomVal(P_123);
e.r = getRandomVal(P_023);
e.v = ADD(e.v,e.r);
send_to_live(P_0, ADD(m20, e.m)); // + s
send_to_live(P_1,e.v);
return e;
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
Datatype m21 = receive_from_live(P_1);
v = ADD(v,m21);
store_compare_view(P_012, ADD(v,m)); //compare d_0 s
}

#endif

};
