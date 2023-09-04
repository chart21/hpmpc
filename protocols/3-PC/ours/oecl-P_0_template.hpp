#pragma once
#include "oecl_base.hpp"
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
    static constexpr int VALS_PER_SHARE = 2;

    OECL0_Share() {}
    OECL0_Share(Datatype p1, Datatype p2) : p1(p1), p2(p2) {}
    OECL0_Share(Datatype p1) : p1(p1) {}


    

OECL0_Share public_val(Datatype a)
{
    return OECL0_Share(SET_ALL_ZERO(),SET_ALL_ZERO());
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
#if FRACTIONAL > 0
/* c.p1 = SUB( MULT(p1,b.p1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  )); // -> -e = x2y2 - (x1-x2)(y1-y2) = x1 y2 +  x2 y+1 - x1_y1 */
c.p1 = SUB( MULT( SUB(p1,p2), SUB(b.p1,b.p2)), MULT(p1,b.p1)  ); // -> e = (x1-x2)(y1-y2) - x2y2 = x1 y1 - x1 y2 - x2 y1
#else
c.p1 = SUB( MULT(p1,b.p1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  )); // e = x2y2 - (x1-x2)(y1-y2)

#endif
return c;
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
DATATYPE maskP_1 = getRandomVal(P_1);
DATATYPE maskP_1_2 = getRandomVal(P_1);
DATATYPE maskP_2 = getRandomVal(P_2);
#if PRE == 1
pre_send_to_live(P_2, ADD(p1,maskP_1));
#else
send_to_live(P_2, ADD(p1,maskP_1));
#endif
    p1 = maskP_2;
    p2 = maskP_1_2;
}
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}

    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
DATATYPE maskP_1 = getRandomVal(P_1);
DATATYPE maskP_1_2 = getRandomVal(P_1);
DATATYPE maskP_2 = getRandomVal(P_2);

p1 = ADD( TRUNC(ADD(ADD(p1,maskP_1),maskP_2)), maskP_1_2); // (e + r0,1 + r0,2)^t + r0,1_2
p2 = SUB(SET_ALL_ZERO(),maskP_1_2); // - r0,1_2


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
Datatype maskP_1_2 = getRandomVal(P_1);
Datatype maskP_2 = getRandomVal(P_2);
#if PRE == 1
pre_send_to_live(P_2, SUB( ADD(MULT(p1,b.p1),maskP_1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  ))); 
#else
send_to_live(P_2, SUB( ADD(MULT(p1,b.p1),maskP_1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  ))); 
#endif
// for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 + b.p2)
return OECL0_Share(maskP_2,maskP_1_2);
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}

void prepare_reveal_to_all()
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
Datatype complete_Reveal(func_add ADD, func_sub SUB)
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
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P_0)
{
#if OPT_SHARE == 1
    p2 = getRandomVal(P_1); // r0,1
    p1 = SUB(SET_ALL_ZERO(), ADD(get_input_live(),p2)); // share -(a + r0,1)
    #if PRE == 1 && SHARE_PREP == 1
        pre_send_to_live(P_2, p1); // share -(a + r0,1) to P_2
    #else
        send_to_live(P_2, p1);
    #endif
#else
    p1 = getRandomVal(P_2); // P_1 does not need to the share -> thus not srng but 2 -> with updated share conversion it needs it
    p2 = getRandomVal(P_1);
    Datatype input = get_input_live();
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

static void prepare_A2B_S1(OECL0_Share in[], OECL0_Share out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        /* out[i].p1 = getRandomVal(P_2); // set share to r0,2 */ 
        out[i].p1 = SET_ALL_ZERO(); // set share to 0
        out[i].p2 = SET_ALL_ZERO(); // set other share to 0
    }
}


static void prepare_A2B_S2(OECL0_Share in[], OECL0_Share out[])
{
    //convert share x0 to boolean
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = OP_SUB(SET_ALL_ZERO(), OP_ADD(in[j].p1, in[j].p2)); // set share to -x0
        }
    unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp);
    orthogonalize_boolean((UINT_TYPE*) temp, temp);

    for(int i = 0; i < BITLENGTH; i++)
    {
            out[i].p2 = getRandomVal(P_1); // set second share to r0,1
            out[i].p1 = FUNC_XOR(temp[i],out[i].p2); // set first share to -x0 xor r0,1
            #if PRE == 1
                pre_send_to_live(P_2, out[i].p1); // -x0 xor r0,1 to P_2
            #else
                send_to_live(P_2, out[i].p1); // -x0 xor r0,1 to P_2
            #endif
    } 
            /* out[0].p1 = FUNC_NOT(out[0].p1);// change sign bit -> -x0 xor r0,1 to x0 xor r0,1 */
}

static void complete_A2B_S1(OECL0_Share out[])
{

}
static void complete_A2B_S2(OECL0_Share out[])
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
    DATATYPE temp[BITLENGTH]{0};
    temp[BITLENGTH - 1] = FUNC_XOR(p1,p2);
    unorthogonalize_boolean(temp,(UINT_TYPE*)temp);
    orthogonalize_arithmetic((UINT_TYPE*) temp,  temp);
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



};

