#pragma once
#include "oecl_base.hpp"
class OECL0_POST
{
bool optimized_sharing;
public:
OECL0_POST(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

OECL_Share public_val(DATATYPE a)
{
    return OECL_Share();
}

OECL_Share Not(OECL_Share a)
{
   return a;
}

template <typename func_add>
OECL_Share Add(OECL_Share a, OECL_Share b, func_add ADD)
{
   return a;
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(OECL_Share a, OECL_Share b, OECL_Share &c, func_add ADD, func_sub SUB, func_mul MUL)
{
/* DATATYPE rl = getRandomVal(0); */
/* DATATYPE rr = getRandomVal(0); */

/* DATATYPE rx = getRandomVal(0); */
/* DATATYPE ry = getRandomVal(1); */
/* DATATYPE maskP_1 = XOR(a.p1,b.p1); */
/* DATATYPE maskP_2 = XOR(a.p2,b.p2); */

}

template <typename func_add, typename func_sub>
void complete_mult(OECL_Share &c, func_add ADD, func_sub SUB)
{
}

void prepare_reveal_to_all(OECL_Share a)
{
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(OECL_Share a, func_add ADD, func_sub SUB)
{
return SUB(receive_from_live(P_2),retrieve_output_share());
}


OECL_Share* alloc_Share(int l)
{
    return new OECL_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(OECL_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
#if OPT_SHARE == 1 && SHARE_PREP == 0
if(id == P_0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].p1 = get_input_live();
    a[i].p2 = getRandomVal(P_1);
    send_to_live(P_2, XOR(a[i].p1,a[i].p2));
    }
}
#endif
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
    communicate_live();
}

};
