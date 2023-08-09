#pragma once
#include "oecl_base.hpp"
#define PRE_SHARE OECL0_Share
/* #define VALS_PER_SHARE 2 */
#define SHARE OECL0_Share



template <typename Datatype>
class OECL0_Share 
{
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
    OECL0_Share prepare_mult(OECL0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype maskP1 = getRandomVal(P1);
Datatype maskP1_2 = getRandomVal(P1);
Datatype maskP2 = getRandomVal(P2);
#if PRE == 1
pre_send_to_live(P2, SUB( ADD(MULT(p1,b.p1),maskP1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  ))); 
#else
send_to_live(P2, SUB( ADD(MULT(p1,b.p1),maskP1), MULT( SUB(p1,p2), SUB(b.p1,b.p2)  ))); 
#endif
// for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: p1 b.p1 + (p1 + p2) (b.p1 + b.p2)
return OECL0_Share(maskP1,maskP1_2);
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}

void prepare_reveal_to_all()
{
        #if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
    pre_send_to_live(P1, p1);
    pre_send_to_live(P2, p2);
    #else
    send_to_live(P1, p1);
    send_to_live(P2, p2);
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
return SUB(receive_from_live(P2),p2);
#endif

}

template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P0)
{
#if OPT_SHARE == 1
    p1 = SUB(SET_ALL_ZERO(), get_input_live());
    p2 = getRandomVal(P1);
    #if PRE == 1 && SHARE_PREP == 1
        pre_send_to_live(P2, SUB(p2,p1));
    #else
        send_to_live(P2, SUB(p2,p1));
    #endif
#else
    p1 = getRandomVal(P0); // P1 does not need to the share -> thus not srng but 2 
    p2 = getRandomVal(P1);
    Datatype input = get_input_live();
    #if PRE == 1
    pre_send_to_live(P1, ADD(p1,input));
    pre_send_to_live(P2, ADD(p2,input));
    #else
    send_to_live(P1, ADD(p1,input));
    send_to_live(P2, ADD(p2,input));
    #endif
#endif
}
else if constexpr(id == P1){
    p1 = SET_ALL_ZERO();
    p2 = getRandomVal(P1);
}
else if constexpr(id == P2)// id ==2
{
    p1 = getRandomVal(P2);
    p2 = SET_ALL_ZERO();
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    return;
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
        out[i].p1 = getRandomVal(P2); // set share to r0,2 
        out[i].p2 = SET_ALL_ZERO(); // set other share to 0
    }
}


static void prepare_A2B_S2(Datatype in[], Datatype out[])
{
    //convert share a + x1 to boolean
    Datatype temp[BITLENGTH];
        for (int j = 0; j < BITLENGTH; j++)
        {
            temp[j] = SET_ALL_ZERO() -  in[j].p2; // set share to -x1
        }
    unorthogonalize_arithmetic(temp, (UINT_TYPE*) temp);
    orthogonalize_boolean((UINT_TYPE*) temp, temp);

    for(int i = 0; i < BITLENGTH; i++)
    {
            out[i].p0 = SET_ALL_ZERO(); // set first share to 0
            out[i].p1 = temp[i]; // set second share to -x1
    } 
}


static void prepare_A2B(Datatype in[], Datatype out[])
{
    prepare_A2B_S1(in, out);
    prepare_A2B_S2(in, out);
}

static void complete_A2B(Datatype in[], Datatype out[])
{
}
}
;

