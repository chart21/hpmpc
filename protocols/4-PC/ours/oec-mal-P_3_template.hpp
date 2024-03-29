#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE OEC_MAL3_Share
template <typename Datatype>
class OEC_MAL3_Share
{
private:
    Datatype r0;
    Datatype r1;
public:
    
OEC_MAL3_Share() {}
OEC_MAL3_Share(Datatype r0, Datatype r1) : r0(r0), r1(r1) {}
OEC_MAL3_Share(Datatype r0) : r0(r0) {}


    

static OEC_MAL3_Share public_val(Datatype a)
{
    return OEC_MAL3_Share(SET_ALL_ZERO(),SET_ALL_ZERO());
}

template <typename func_mul>
OEC_MAL3_Share mult_public(const Datatype b, func_mul MULT) const
{
    return OEC_MAL3_Share(MULT(r0,b),MULT(r1,b));
}
    
template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
OEC_MAL3_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
#if TRUNC_THEN_MULT == 1
    auto result = MULT(TRUNC(r1),b);
#else
    auto result = TRUNC(MULT(r1,b));
#endif
    auto rand_val = getRandomVal(P_013);
    auto val = SUB(result,rand_val);
#if PROTOCOL == 12
#if PRE == 1
    pre_send_to_live(P_2, val);
#else
    send_to_live(P_2, val);
#endif
#else
    store_compare_view(P_2, val);
#endif
    
    return OEC_MAL3_Share(getRandomVal(P_123),result);
} 
    
template <typename func_add, typename func_sub>
void complete_public_mult_fixed( func_add ADD, func_sub SUB)
{
}


OEC_MAL3_Share Not() const
{
   return *this;
}

template <typename func_add>
OEC_MAL3_Share Add(OEC_MAL3_Share b, func_add ADD) const
{
   return OEC_MAL3_Share(ADD(r0,b.r0),ADD(r1,b.r1));
}



template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_mult(OEC_MAL3_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL3_Share c;
c.r1 = ADD(getRandomVal(P_023),getRandomVal(P_013)); // x1 

/* Datatype r124 = getRandomVal(P_013); // used for verification */
/* Datatype r234 = getRandomVal(P_123); // Probably sufficient to only generate with P_2(-> P_3 in paper) -> no because of verification */

/* Datatype o1 = ADD( x1y1, getRandomVal(P_013)); */
Datatype o1 = ADD(c.r1, ADD(MULT(r1,b.r1), getRandomVal(P_013)));

#if PROTOCOL == 11
/* Datatype o4 = ADD(SUB(SUB(x1y1, MULT(a.r0,b.r1)) ,MULT(a.r1,b.r0)),getRandomVal(P_123)); // r123_2 */
/* Datatype o4 = ADD(SUB(MULT(a.r1, SUB(b.r0,b.r1)) ,MULT(b.r1,a.r0)),getRandomVal(P_123)); // r123_2 */
Datatype o4 = ADD(SUB(MULT(r1, SUB(b.r1,b.r0)) ,MULT(b.r1,r0)),getRandomVal(P_123)); // r123_2
#else
Datatype o4 = ADD(SUB(MULT(r1, SUB(b.r1,b.r0)) ,MULT(b.r1,r0)),getRandomVal(P_123)); // r123_2
#endif
c.r0 = o4;
/* Datatype o4 = ADD( SUB( MULT(a.r0,b.r1) ,MULT(a.r1,b.r0)),getRandomVal(P_123)); */
/* o4 = XOR(o4,o1); //computationally easier to let P_3 do it here instead of P_0 later */
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_2, o1);
#else
send_to_live(P_2, o1);
#endif
#else
store_compare_view(P_2, o1);
#endif


return c;
}

template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL3_Share prepare_dot(const OEC_MAL3_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL3_Share c;
c.r1 = MULT(r1,b.r1); // store o_1
c.r0 = SUB(MULT(r1, SUB(b.r1,b.r0)) ,MULT(b.r1,r0)); // store o_4
return c;
}
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
Datatype r0123 = ADD(getRandomVal(P_013),getRandomVal(P_023));
r1 = TRUNC(SUB(r0123, r1)); // z_0 = [r_0,1,3 + r_0,2,3 - x_0 y_0]^t
r0 = SUB(r0, r0123);

#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_2, SUB(r1,getRandomVal(P_013))); // compare m^0 - z_1
#else
send_to_live(P_2, SUB(r1,getRandomVal(P_013))); // compare m^0 - z_1
#endif
#else
store_compare_view(P_2, SUB(r1,getRandomVal(P_013))); // compare m^0 - z_1
#endif
}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
r0 = ADD(r0, getRandomVal(P_123));  //mask v^3
#if PROTOCOL == 11
store_compare_view(P_0, r0);  // v^3 = .. - r_0,1,2 - r_0,2,3 + r_1,2,3 TODO: Recent change: Verify
#else
#if PRE == 1
pre_send_to_live(P_0, r0);  // m^3 = .. - r_0,1,2 - r_0,2,3 + r_1,2,3 TODO: Recent change: Verify
#else
send_to_live(P_0, r0);  // m^3 = .. - r_0,1,2 - r_0,2,3 + r_1,2,3 TODO: Recent change: Verify
#endif
#endif
r0 = getRandomVal(P_123); // w
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{

Datatype rc1 = ADD(getRandomVal(P_023),getRandomVal(P_013)); // x0 

Datatype o1 = ADD(rc1, ADD(r1, getRandomVal(P_013)));
#if PROTOCOL == 11
Datatype o4 = ADD(r0,getRandomVal(P_123)); // r123_2
#else
Datatype o4 = ADD(r0,getRandomVal(P_123)); // - w + r123
#endif
r0 = o4;
r1 = rc1;

#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_2, o1);
#else
send_to_live(P_2, o1);
#endif
#else
store_compare_view(P_2, o1);
#endif



}



template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
Datatype rc0 = getRandomVal(P_123); // w
#if PROTOCOL == 11
Datatype o4 = r0;
#else
Datatype o4 = SUB(r0,rc0);
#endif
#if PROTOCOL == 10 || PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_0, o4);
#else
send_to_live(P_0, o4);
#endif
#elif PROTOCOL == 11
store_compare_view(P_0,o4);
#endif
r0 = rc0;
}


template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OEC_MAL3_Share& r_mk2, OEC_MAL3_Share& r_msb, OEC_MAL3_Share& c, OEC_MAL3_Share& c_prime) const{
    Datatype rmk2 = OP_SHIFT_LOG_RIGHT<FRACTIONAL+1>( OP_SHIFT_LEFT<1>(r1) );
    Datatype rmsb = OP_SHIFT_LOG_RIGHT<BITLENGTH-1>(r1);
    
    r_mk2.r0 = SET_ALL_ZERO();
    r_mk2.r1 = SUB(SET_ALL_ZERO(), rmk2);
    r_msb.r0 = SET_ALL_ZERO();
    r_msb.r1 = SUB(SET_ALL_ZERO(), rmsb);
#if PROTOCOL == 12
#if PRE == 1
    pre_send_to_live(P_2, SUB(r_mk2.r1, getRandomVal(P_013)));
    pre_send_to_live(P_2, SUB(r_msb.r1, getRandomVal(P_013)));
#else
    send_to_live(P_2, SUB(r_mk2.r1, getRandomVal(P_013)));
    send_to_live(P_2, SUB(r_msb.r1, getRandomVal(P_013)));
#endif
#else
    store_compare_view(P_2, SUB(r_mk2.r1, getRandomVal(P_013)));
    store_compare_view(P_2, SUB(r_msb.r1, getRandomVal(P_013)));
#endif

    c.r0 = getRandomVal(P_123);
    c.r1 = SET_ALL_ZERO();
    c_prime.r0 = getRandomVal(P_123);
    c_prime.r1 = SET_ALL_ZERO();
}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OEC_MAL3_Share& r_mk2, OEC_MAL3_Share& r_msb, OEC_MAL3_Share& c, OEC_MAL3_Share& c_prime) const{
}


void prepare_reveal_to_all() const
{
#if PRE == 1
    pre_send_to_live(P_0, r0);
#else
    send_to_live(P_0, r0);
#endif
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
#if PRE == 0
Datatype result = SUB(receive_from_live(P_0),r0);
store_compare_view(P_123, r1);
store_compare_view(P_0123, result);
return result;
#else
#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share(r0);
store_output_share(r1);
#endif
return r0;
#endif


}



template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    Datatype v = val;
    Datatype x_1 = getRandomVal(P_013);
    Datatype x_2 = getRandomVal(P_023);
    Datatype u = getRandomVal(P_123);

    r0 = u;
    r1 = ADD(x_1,x_2);
    Datatype complete_masked = ADD(v, ADD(r1, r0));
    #if PRE == 1
    pre_send_to_live(P_0, complete_masked);
    pre_send_to_live(P_1, complete_masked);
    pre_send_to_live(P_2, complete_masked);
    #else
    send_to_live(P_0, complete_masked);
    send_to_live(P_1, complete_masked);
    send_to_live(P_2, complete_masked);
    #endif
     
}
else if constexpr(id == P_0)
{
    r0 = SET_ALL_ZERO();
    r1 = ADD(getRandomVal(P_013),getRandomVal(P_023));
    
}
else if constexpr(id == P_1)
{
    r0 = getRandomVal(P_123);
    r1 = getRandomVal(P_013);
    
}
else if constexpr(id == P_2)
{
    r0 = getRandomVal(P_123);
    r1 = getRandomVal(P_023);
    
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

static void prepare_A2B_S1(int m, int k, OEC_MAL3_Share in[], OEC_MAL3_Share out[])
{
    for(int i = m; i < k; i++)
    {
        out[i-m].r0 = getRandomVal(P_123); // r123
        out[i-m].r1 = SET_ALL_ZERO(); // set share to 0
    }
}


static void prepare_A2B_S2(int m, int k ,OEC_MAL3_Share in[], OEC_MAL3_Share out[])
{
    //convert share x0 to boolean
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = OP_SUB(SET_ALL_ZERO(), in[j].r1); // set share to -x0
        }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp, temp2);
    orthogonalize_boolean(temp2, temp);
    /* unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp); */
    /* orthogonalize_boolean((UINT_TYPE*) temp, temp); */

    for(int i = m; i < k; i++)
    {
            out[i-m].r0 = SET_ALL_ZERO(); 
            out[i-m].r1 = temp[i]; 
            #if PROTOCOL != 12
            store_compare_view(P_2, FUNC_XOR(temp[i], getRandomVal(P_013))); // compare -x0 xor r0,1 with $P_2
            #else
            #if PRE == 1
                pre_send_to_live(P_2, FUNC_XOR(temp[i], getRandomVal(P_013))); // -x0 xor r0,1 to P_2
            #else
                send_to_live(P_2, FUNC_XOR(temp[i], getRandomVal(P_013))); // -x0 xor r0,1 to P_2
            #endif
            #endif
    } 
}

static void complete_A2B_S1(int k, OEC_MAL3_Share out[])
{
}

static void complete_A2B_S2(int k, OEC_MAL3_Share out[])
{

}

void prepare_bit2a(OEC_MAL3_Share out[])
{
    Datatype y0[BITLENGTH]{0};
    y0[BITLENGTH - 1] = r1; //convert y0 to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(y0, temp2);
    orthogonalize_arithmetic(temp2, y0);
    Datatype y0v[BITLENGTH]{0};
    y0v[BITLENGTH - 1] = FUNC_XOR(r1,r0); //convert y_0 xor v to an arithemtic value
    unorthogonalize_boolean(y0v, temp2);
    orthogonalize_arithmetic(temp2, y0v);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype r013 = getRandomVal(P_013);
        Datatype m00 = OP_SUB(y0[i], r013);
#if PROTOCOL == 12
#if PRE == 1
        pre_send_to_live(P_2, m00);
#else
        send_to_live(P_2, m00);
#endif
#else
        store_compare_view(P_2, m00);
#endif
        
        Datatype r123 = getRandomVal(P_123);
        Datatype m30 = OP_SUB(y0v[i], r123);

#if PRE == 1
        pre_send_to_live(P_0, m30);
#else
        send_to_live(P_0, m30);
#endif 
        out[i].r0 = getRandomVal(P_123);
        out[i].r1 = OP_ADD(getRandomVal(P_013),getRandomVal(P_023));
        
    }
}

void complete_bit2a()
{
}

void prepare_opt_bit_injection(OEC_MAL3_Share x[], OEC_MAL3_Share out[])
{
    Datatype y0[BITLENGTH]{0};
    y0[BITLENGTH - 1] = r1; //convert y0 to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(y0, temp2);
    orthogonalize_arithmetic(temp2, y0);
    Datatype y0v[BITLENGTH]{0};
    y0v[BITLENGTH - 1] = FUNC_XOR(r1,r0); //convert y_0 xor v to an arithemtic value
    unorthogonalize_boolean(y0v, temp2);
    orthogonalize_arithmetic(temp2, y0v);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype r013 = getRandomVal(P_013);
        Datatype r013_2 = getRandomVal(P_013);
        Datatype m00 = OP_SUB(y0[i], r013);
        Datatype m01 = OP_SUB(OP_MULT(x[i].r1, y0[i]), r013_2);
#if PROTOCOL == 12
#if PRE == 1
        pre_send_to_live(P_2, m00);
        pre_send_to_live(P_2, m01);
#else
        send_to_live(P_2, m00);
        send_to_live(P_2, m01);
#endif
#else
        store_compare_view(P_2, m00);
        store_compare_view(P_2, m01);
#endif
        
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);
        Datatype m30 = OP_SUB(y0v[i], r123);
        Datatype m31 = OP_SUB(OP_MULT(OP_ADD(x[i].r0,x[i].r1), y0v[i]), r123_2);

#if PRE == 1
        pre_send_to_live(P_0, m30);
        pre_send_to_live(P_0, m31);
#else
        send_to_live(P_0, m30);
        send_to_live(P_0, m31);
#endif 
        out[i].r0 = getRandomVal(P_123);
        out[i].r1 = OP_ADD(getRandomVal(P_013),getRandomVal(P_023));
        
    }
}

void complete_opt_bit_injection()
{
}


void prepare_bit_injection_S1(OEC_MAL3_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].r0 = getRandomVal(P_123); // r123
        out[i].r1 = SET_ALL_ZERO(); // set share to 0
    }
}

void prepare_bit_injection_S2(OEC_MAL3_Share out[])
{
    Datatype temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = r1;
    /* unorthogonalize_boolean(temp,(UINT_TYPE*)temp); */
    /* orthogonalize_arithmetic((UINT_TYPE*) temp,  temp); */
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(temp, temp2);
    orthogonalize_arithmetic(temp2, temp);
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i].r0 = SET_ALL_ZERO(); //w = 0
        out[i].r1 = OP_SUB(SET_ALL_ZERO(), temp[i]) ; // z_0 = - x_0
            #if PROTOCOL != 12
            store_compare_view(P_2, OP_ADD(temp[i], getRandomVal(P_013))); // compare -x0 xor r0,1 with $P_2
            #else
            #if PRE == 1
                pre_send_to_live(P_2, OP_ADD(temp[i], getRandomVal(P_013))); // -x0 xor r0,1 to P_2
            #else
                send_to_live(P_2, OP_ADD(temp[i], getRandomVal(P_013))); // -x0 xor r0,1 to P_2
            #endif
            #endif
        
    }
}

static void complete_bit_injection_S1(OEC_MAL3_Share out[])
{
    
}

static void complete_bit_injection_S2(OEC_MAL3_Share out[])
{
}

#if MULTI_INPUT == 1

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_dot3(const OEC_MAL3_Share b, const OEC_MAL3_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = SUB(MULT(r1,b.r1),getRandomVal(P_013));
Datatype mxz = SUB(MULT(r1,c.r1),getRandomVal(P_013));
Datatype myz = SUB(MULT(b.r1,c.r1),getRandomVal(P_013));
Datatype mxyz = MULT(MULT(r1,b.r1),c.r1);
mxyz = SUB( SET_ALL_ZERO(), mxyz); // trick to be compatible with dot2
Datatype ax = ADD(r0,r1);
Datatype by = ADD(b.r0,b.r1);
Datatype cz = ADD(c.r0,c.r1);
Datatype m3xy = SUB(MULT(ax,by),getRandomVal(P_123));
Datatype m3xz = SUB(MULT(ax,cz),getRandomVal(P_123));
Datatype m3yz = SUB(MULT(by,cz),getRandomVal(P_123));
Datatype m3xyz = MULT(MULT(ax,by),cz);
/* m3xyz = SUB( SET_ALL_ZERO(), m3xyz); // trick to be compatible with dot2 */
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_0, m3xy);
pre_send_to_live(P_0, m3xz);
pre_send_to_live(P_0, m3yz);
pre_send_to_live(P_2, mxy);
pre_send_to_live(P_2, mxz);
pre_send_to_live(P_2, myz);
#else
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3yz);
send_to_live(P_2, mxy);
send_to_live(P_2, mxz);
send_to_live(P_2, myz);
#endif
#else
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3yz);
store_compare_view(P_2, mxy);
store_compare_view(P_2, mxz);
store_compare_view(P_2, myz);
#endif
OEC_MAL3_Share d;
d.r0 = m3xyz;
d.r1 = mxyz;
return d;
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_mult3(const OEC_MAL3_Share b, const OEC_MAL3_Share c, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = SUB(MULT(r1,b.r1),getRandomVal(P_013));
Datatype mxz = SUB(MULT(r1,c.r1),getRandomVal(P_013));
Datatype myz = SUB(MULT(b.r1,c.r1),getRandomVal(P_013));
Datatype mxyz = SUB(MULT(MULT(r1,b.r1),c.r1),getRandomVal(P_013));
Datatype ax = ADD(r0,r1);
Datatype by = ADD(b.r0,b.r1);
Datatype cz = ADD(c.r0,c.r1);
Datatype m3xy = SUB(MULT(ax,by),getRandomVal(P_123));
Datatype m3xz = SUB(MULT(ax,cz),getRandomVal(P_123));
Datatype m3yz = SUB(MULT(by,cz),getRandomVal(P_123));
Datatype m3xyz = SUB(MULT(MULT(ax,by),cz),getRandomVal(P_123));
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_0, m3xy);
pre_send_to_live(P_0, m3xz);
pre_send_to_live(P_0, m3yz);
pre_send_to_live(P_0, m3xyz);
pre_send_to_live(P_2, mxy);
pre_send_to_live(P_2, mxz);
pre_send_to_live(P_2, myz);
pre_send_to_live(P_2, mxyz);
#else
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3yz);
send_to_live(P_0, m3xyz);
send_to_live(P_2, mxy);
send_to_live(P_2, mxz);
send_to_live(P_2, myz);
send_to_live(P_2, mxyz);
#endif
#else
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3yz);
send_to_live(P_0, m3xyz);
store_compare_view(P_2, mxy);
store_compare_view(P_2, mxz);
store_compare_view(P_2, myz);
store_compare_view(P_2, mxyz);
#endif
OEC_MAL3_Share d;
d.r0 = getRandomVal(P_123);
d.r1 = ADD(getRandomVal(P_013),getRandomVal(P_023));
return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_dot4(const OEC_MAL3_Share b, const OEC_MAL3_Share c, const OEC_MAL3_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = SUB(MULT(r1,b.r1),getRandomVal(P_013));
Datatype mxz = SUB(MULT(r1,c.r1),getRandomVal(P_013));
Datatype mxw = SUB(MULT(r1,d.r1),getRandomVal(P_013));
Datatype myz = SUB(MULT(b.r1,c.r1),getRandomVal(P_013));
Datatype myw = SUB(MULT(b.r1,d.r1),getRandomVal(P_013));
Datatype mzw = SUB(MULT(c.r1,d.r1),getRandomVal(P_013));
Datatype mxyz = SUB(MULT(MULT(r1,b.r1),c.r1),getRandomVal(P_013));
Datatype mxyw = SUB(MULT(MULT(r1,b.r1),d.r1),getRandomVal(P_013));
Datatype mxzw = SUB(MULT(MULT(r1,c.r1),d.r1),getRandomVal(P_013));
Datatype myzw = SUB(MULT(MULT(b.r1,c.r1),d.r1),getRandomVal(P_013));
Datatype mxyzw = MULT(MULT(r1,b.r1),MULT(c.r1,d.r1));
Datatype ax = ADD(r0,r1);
Datatype by = ADD(b.r0,b.r1);
Datatype cz = ADD(c.r0,c.r1);
Datatype dw = ADD(d.r0,d.r1);
Datatype m3xy = SUB(MULT(ax,by),getRandomVal(P_123));
Datatype m3xz = SUB(MULT(ax,cz),getRandomVal(P_123));
Datatype m3xw = SUB(MULT(ax,dw),getRandomVal(P_123));
Datatype m3yz = SUB(MULT(by,cz),getRandomVal(P_123));
Datatype m3yw = SUB(MULT(by,dw),getRandomVal(P_123));
Datatype m3zw = SUB(MULT(cz,dw),getRandomVal(P_123));
Datatype m3xyz = SUB(MULT(MULT(ax,by),cz),getRandomVal(P_123));
Datatype m3xyw = SUB(MULT(MULT(ax,by),dw),getRandomVal(P_123));
Datatype m3xzw = SUB(MULT(MULT(ax,cz),dw),getRandomVal(P_123));
Datatype m3yzw = SUB(MULT(MULT(by,cz),dw),getRandomVal(P_123));
Datatype m3xyzw = MULT(MULT(ax,by),MULT(cz,dw));
m3xyzw = SUB( SET_ALL_ZERO(), m3xyzw); // trick to be compatible with dot2
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_0, m3xy);
pre_send_to_live(P_0, m3xz);
pre_send_to_live(P_0, m3xw);
pre_send_to_live(P_0, m3yz);
pre_send_to_live(P_0, m3yw);
pre_send_to_live(P_0, m3zw);
pre_send_to_live(P_0, m3xyz);
pre_send_to_live(P_0, m3xyw);
pre_send_to_live(P_0, m3xzw);
pre_send_to_live(P_0, m3yzw);
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
#else
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3xw);
send_to_live(P_0, m3yz);
send_to_live(P_0, m3yw);
send_to_live(P_0, m3zw);
send_to_live(P_0, m3xyz);
send_to_live(P_0, m3xyw);
send_to_live(P_0, m3xzw);
send_to_live(P_0, m3yzw);
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
#endif
#else
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3xw);
send_to_live(P_0, m3yz);
send_to_live(P_0, m3yw);
send_to_live(P_0, m3zw);
send_to_live(P_0, m3xyz);
send_to_live(P_0, m3xyw);
send_to_live(P_0, m3xzw);
send_to_live(P_0, m3yzw);
store_compare_view(P_2, mxy);
store_compare_view(P_2, mxz);
store_compare_view(P_2, mxw);
store_compare_view(P_2, myz);
store_compare_view(P_2, myw);
store_compare_view(P_2, mzw);
store_compare_view(P_2, mxyz);
store_compare_view(P_2, mxyw);
store_compare_view(P_2, mxzw);
store_compare_view(P_2, myzw);
#endif
OEC_MAL3_Share e;
e.r0 = m3xyzw;
e.r1 = mxyzw;
return e;

}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_Share prepare_mult4(const OEC_MAL3_Share b, const OEC_MAL3_Share c, const OEC_MAL3_Share d, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype mxy = SUB(MULT(r1,b.r1),getRandomVal(P_013));
Datatype mxz = SUB(MULT(r1,c.r1),getRandomVal(P_013));
Datatype mxw = SUB(MULT(r1,d.r1),getRandomVal(P_013));
Datatype myz = SUB(MULT(b.r1,c.r1),getRandomVal(P_013));
Datatype myw = SUB(MULT(b.r1,d.r1),getRandomVal(P_013));
Datatype mzw = SUB(MULT(c.r1,d.r1),getRandomVal(P_013));
Datatype mxyz = SUB(MULT(MULT(r1,b.r1),c.r1),getRandomVal(P_013));
Datatype mxyw = SUB(MULT(MULT(r1,b.r1),d.r1),getRandomVal(P_013));
Datatype mxzw = SUB(MULT(MULT(r1,c.r1),d.r1),getRandomVal(P_013));
Datatype myzw = SUB(MULT(MULT(b.r1,c.r1),d.r1),getRandomVal(P_013));
Datatype mxyzw = SUB(MULT(MULT(r1,b.r1),MULT(c.r1,d.r1)),getRandomVal(P_013));
Datatype ax = ADD(r0,r1);
Datatype by = ADD(b.r0,b.r1);
Datatype cz = ADD(c.r0,c.r1);
Datatype dw = ADD(d.r0,d.r1);
Datatype m3xy = SUB(MULT(ax,by),getRandomVal(P_123));
Datatype m3xz = SUB(MULT(ax,cz),getRandomVal(P_123));
Datatype m3xw = SUB(MULT(ax,dw),getRandomVal(P_123));
Datatype m3yz = SUB(MULT(by,cz),getRandomVal(P_123));
Datatype m3yw = SUB(MULT(by,dw),getRandomVal(P_123));
Datatype m3zw = SUB(MULT(cz,dw),getRandomVal(P_123));
Datatype m3xyz = SUB(MULT(MULT(ax,by),cz),getRandomVal(P_123));
Datatype m3xyw = SUB(MULT(MULT(ax,by),dw),getRandomVal(P_123));
Datatype m3xzw = SUB(MULT(MULT(ax,cz),dw),getRandomVal(P_123));
Datatype m3yzw = SUB(MULT(MULT(by,cz),dw),getRandomVal(P_123));
Datatype m3xyzw = SUB(MULT(MULT(ax,by),MULT(cz,dw)),getRandomVal(P_123));
#if PROTOCOL == 12
#if PRE == 1
pre_send_to_live(P_0, m3xy);
pre_send_to_live(P_0, m3xz);
pre_send_to_live(P_0, m3xw);
pre_send_to_live(P_0, m3yz);
pre_send_to_live(P_0, m3yw);
pre_send_to_live(P_0, m3zw);
pre_send_to_live(P_0, m3xyz);
pre_send_to_live(P_0, m3xyw);
pre_send_to_live(P_0, m3xzw);
pre_send_to_live(P_0, m3yzw);
pre_send_to_live(P_0, m3xyzw);
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
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3xw);
send_to_live(P_0, m3yz);
send_to_live(P_0, m3yw);
send_to_live(P_0, m3zw);
send_to_live(P_0, m3xyz);
send_to_live(P_0, m3xyw);
send_to_live(P_0, m3xzw);
send_to_live(P_0, m3yzw);
send_to_live(P_0, m3xyzw);
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
#else
send_to_live(P_0, m3xy);
send_to_live(P_0, m3xz);
send_to_live(P_0, m3xw);
send_to_live(P_0, m3yz);
send_to_live(P_0, m3yw);
send_to_live(P_0, m3zw);
send_to_live(P_0, m3xyz);
send_to_live(P_0, m3xyw);
send_to_live(P_0, m3xzw);
send_to_live(P_0, m3yzw);
send_to_live(P_0, m3xyzw);
store_compare_view(P_2, mxy);
store_compare_view(P_2, mxz);
store_compare_view(P_2, mxw);
store_compare_view(P_2, myz);
store_compare_view(P_2, myw);
store_compare_view(P_2, mzw);
store_compare_view(P_2, mxyz);
store_compare_view(P_2, mxyw);
store_compare_view(P_2, mxzw);
store_compare_view(P_2, myzw);
store_compare_view(P_2, mxyzw);
#endif
OEC_MAL3_Share e;
e.r0 = getRandomVal(P_123);
e.r1 = ADD(getRandomVal(P_013),getRandomVal(P_023));
return e;

}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
}

#endif

};
