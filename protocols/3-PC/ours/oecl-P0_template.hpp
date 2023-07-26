#pragma once
#include "oecl_base.hpp"
#define PRE_SHARE OECL_Share
class OECL0
{
bool optimized_sharing;
public:
OECL0(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

OECL_Share public_val(DATATYPE a)
{
    return OECL_Share(SET_ALL_ZERO(),SET_ALL_ZERO());
}

OECL_Share Not(OECL_Share a)
{
   return a;
}

template <typename func_add>
OECL_Share Add(OECL_Share a, OECL_Share b, func_add ADD)
{
    a.p1 = ADD(a.p1,b.p1);
    a.p2 = ADD(a.p2,b.p2);
   return a;
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(OECL_Share a, OECL_Share b, OECL_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
/* DATATYPE rl = getRandomVal(0); */
/* DATATYPE rr = getRandomVal(0); */

/* DATATYPE rx = getRandomVal(0); */
/* DATATYPE ry = getRandomVal(1); */
/* DATATYPE maskP1 = XOR(a.p1,b.p1); */
/* DATATYPE maskP2 = XOR(a.p2,b.p2); */

DATATYPE maskP1 = getRandomVal(P1);
DATATYPE maskP1_2 = getRandomVal(P1);
DATATYPE maskP2 = getRandomVal(P2);
#if PRE == 1
pre_send_to_live(P2, SUB( ADD(MULT(a.p2,b.p2),maskP1), MULT( SUB(a.p1,b.p2), SUB(a.p2,b.p1)  ))); 
#else
send_to_live(P2, SUB( ADD(MULT(a.p2,b.p2),maskP1), MULT( SUB(a.p1,b.p2), SUB(a.p2,b.p1)  ))); 
#endif
// for arithmetic circuikts this will be more efficient to reduce mult from 3 to 2: a.p1 b.p1 + (a.p1 + a.p2) (b.p1 + b.p2)
c.p1 = maskP2;
c.p2 = maskP1_2;
}

template <typename func_add, typename func_sub>
void complete_mult(OECL_Share &c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(OECL_Share a)
{
        #if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
    pre_send_to_live(P1, a.p1);
    pre_send_to_live(P2, a.p2);
    #else
    send_to_live(P1, a.p1);
    send_to_live(P2, a.p2);
#endif
}    



template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(OECL_Share a, func_add ADD, func_sub SUB)
{
#if PRE == 1
    return a.p1;
#else
return SUB(receive_from_live(P2),a.p2);
#endif
}


OECL_Share* alloc_Share(int l)
{
    return new OECL_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(OECL_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P0)
{
#if OPT_SHARE == 1
    for(int i = 0; i < l; i++)
    {
    a[i].p1 = get_input_live();
    a[i].p2 = getRandomVal(P1);
    #if PRE == 1 && SHARE_PREP == 1
        pre_send_to_live(P2, ADD(a[i].p1,a[i].p2));
    #else
        send_to_live(P2, ADD(a[i].p1,a[i].p2));
    #endif
    }

#else
    for(int i = 0; i < l; i++)
    {
    a[i].p1 = getRandomVal(P0); // P1 does not need to the share -> thus not srng but 2 
    a[i].p2 = getRandomVal(P1);
    DATATYPE input = get_input_live();
    #if PRE == 1
    pre_send_to_live(P1, ADD(a[i].p1,input));
    pre_send_to_live(P2, ADD(a[i].p2,input));
    #else
    send_to_live(P1, ADD(a[i].p1,input));
    send_to_live(P2, ADD(a[i].p2,input));
    #endif
    }
#endif
}
else if(id == P1){
for(int i = 0; i < l; i++)
    {
    a[i].p1 = SET_ALL_ZERO();
    a[i].p2 = getRandomVal(P1);
    }


}
else if(id == P2)// id ==2
{
    for(int i = 0; i < l; i++)
    {
    a[i].p1 = getRandomVal(P2);
    a[i].p2 = SET_ALL_ZERO();
    }

}
}

template <typename func_add, typename func_sub>
void complete_receive_from(OECL_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
    return;
}





void send()
{
    send_live();
}

void receive()
{
    receive_live();
}

void communicate()
{
#if PRE == 0
    communicate_live();
#endif
}

};
