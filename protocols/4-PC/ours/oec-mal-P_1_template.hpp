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


OEC_MAL1_Share public_val(Datatype a)
{
    #if MULTI_INPUT == 1
    return OEC_MAL1_Share(a,SET_ALL_ZERO(),SET_ALL_ZERO());
    #else
    return OEC_MAL1_Share(a,SET_ALL_ZERO());
    #endif
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
store_compare_view(P_0,ADD(a1b1,getRandomVal(P_123_2))); // compare a1b1 + r123_2 with P_0
#endif
/* c.v = XOR( AND(      XOR(v,r) , XOR(b.v,b.r) ) , c.v); */
#if PROTOCOL == 11
c.m = ADD(c.v,getRandomVal(P_123_2)); // m_2 + r234_2 store to compareview later
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
store_compare_view(P_0,ADD(v,getRandomVal(P_123_2))); // compare a0b0 + r123_2 with P_0
#endif
#if PROTOCOL == 11
m = ADD(r,getRandomVal(P_123_2)); // m_2 + r234_2 store to compareview later
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


void prepare_reveal_to_all()
{
return;
}    

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
Datatype r = receive_from_live(P_0);
Datatype result = SUB(v, r);
store_compare_view(P_123, r);
store_compare_view(P_0123, result);
return result;
}



template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    
    Datatype x_0 = getRandomVal(P_013);
    Datatype u = getRandomVal(P_123);
    r = x_0; //  = x_1, x_2 = 0
    v = ADD(get_input_live(),x_0);
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

static void prepare_A2B_S1(OEC_MAL1_Share in[], OEC_MAL1_Share out[])
{
        Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = in[j].v; //a0 
        }
    unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp);
    orthogonalize_boolean((UINT_TYPE*) temp, temp);
    for (int j = 0; j < BITLENGTH; j++)
    {
        out[j].r = SET_ALL_ZERO(); // set share to 0
        out[j].v = temp[j];
    #if MULTI_INPUT == 1
    out[j].m = getRandomVal(P_123);
    store_compare_view(P_0, FUNC_XOR(out[j].v, out[j].m));
    #else
    store_compare_view(P_0, FUNC_XOR(out[j].v, getRandomVal(P_123))); 
    #endif
    }

}


static void prepare_A2B_S2(OEC_MAL1_Share in[], OEC_MAL1_Share out[])
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

static void complete_A2B_S1(OEC_MAL1_Share out[])
{
}

static void complete_A2B_S2(OEC_MAL1_Share out[])
{

}

void prepare_bit_injection_S1(OEC_MAL1_Share out[])
{
    Datatype temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = v;
    unorthogonalize_boolean(temp, (UINT_TYPE*) temp);
    orthogonalize_arithmetic((UINT_TYPE*) temp, temp);
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
Datatype rxy = getRandomVal(P_123_2);
Datatype rxz = getRandomVal(P_123_2);
Datatype ryz = getRandomVal(P_123_2);
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
Datatype rxy = getRandomVal(P_123_2);
Datatype rxz = getRandomVal(P_123_2);
Datatype ryz = getRandomVal(P_123_2);
Datatype rxyz = getRandomVal(P_123_2);
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
d.m = getRandomVal(P_123_2);
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
Datatype rxy = getRandomVal(P_123_2);
Datatype rxz = getRandomVal(P_123_2);
Datatype rxw = getRandomVal(P_123_2);
Datatype ryz = getRandomVal(P_123_2);
Datatype ryw = getRandomVal(P_123_2);
Datatype rzw = getRandomVal(P_123_2);
Datatype rxyz = getRandomVal(P_123_2);
Datatype rxyw = getRandomVal(P_123_2);
Datatype rxzw = getRandomVal(P_123_2);
Datatype ryzw = getRandomVal(P_123_2);
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
Datatype rxy = getRandomVal(P_123_2);
Datatype rxz = getRandomVal(P_123_2);
Datatype rxw = getRandomVal(P_123_2);
Datatype ryz = getRandomVal(P_123_2);
Datatype ryw = getRandomVal(P_123_2);
Datatype rzw = getRandomVal(P_123_2);
Datatype rxyz = getRandomVal(P_123_2);
Datatype rxyw = getRandomVal(P_123_2);
Datatype rxzw = getRandomVal(P_123_2);
Datatype ryzw = getRandomVal(P_123_2);
Datatype rxyzw = getRandomVal(P_123_2);
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
e.m = getRandomVal(P_123_2);
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

