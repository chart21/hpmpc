#pragma once
#include "oecl_base.hpp"
class OECL1
{
bool optimized_sharing;
public:
OECL1(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

OECL_Share public_val(DATATYPE a)
{
    return OECL_Share(a,SET_ALL_ZERO());
}

OECL_Share Not(OECL_Share a)
{
    a.p1 = NOT(a.p1);
   return a;
}

template <typename func_add>
OECL_Share Add(OECL_Share a, OECL_Share b, func_add ADD)
{
   return OECL_Share(ADD(a.p1,b.p1),ADD(a.p2,b.p2));
}



template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(OECL_Share a, OECL_Share b , OECL_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
c.p1 = ADD(getRandomVal(P0), ADD(MULT(a.p1,b.p2), MULT(b.p1,a.p2))); //remove P1_mask, then (a+ra)rl + (b+rb)rr 
c.p2 = getRandomVal(P0); //generate P1_2 mask
send_to_live(P2,SUB(c.p1,c.p2)); 
}

template <typename func_add, typename func_sub>
void complete_mult(OECL_Share &c, func_add ADD, func_sub SUB)
{
c.p1 = SUB(receive_from_live(P2),c.p1);
}

void prepare_reveal_to_all(OECL_Share a)
{
return;
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(OECL_Share a, func_add ADD, func_sub SUB)
{
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
return SUB(a.p1, pre_receive_from_live(P0));
#else
return SUB(a.p1, receive_from_live(P0));
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
    for(int i = 0; i < l; i++)
    {
        a[i].p2 = getRandomVal(P0);
    }
}
else if(id == P1)
{
for(int i = 0; i < l; i++)
{
    a[i].p1 = get_input_live();
    a[i].p2 = getRandomVal(P0);
    send_to_live(P2,ADD(a[i].p1,a[i].p2));
}
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(OECL_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P0)
{

#if OPT_SHARE == 1
    for(int i = 0; i < l; i++)
    {
        a[i].p1 = SET_ALL_ZERO(); 
    }
#else
    for(int i = 0; i < l; i++)
    {
    #if PRE == 1 && SHARE_PREP == 1
        a[i].p1 = pre_receive_from_live(P0);
    #else
        a[i].p1 = receive_from_live(P0);
    #endif
    }
#endif 
}
else if(id == P2)
{
for(int i = 0; i < l; i++)
{
a[i].p1 = receive_from_live(P2);
a[i].p2 = SET_ALL_ZERO();
}
}

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
    communicate_live();
}

};
