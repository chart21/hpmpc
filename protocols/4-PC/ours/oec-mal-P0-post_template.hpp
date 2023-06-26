#pragma once
#include "../../3-PC/ours/oecl_base.hpp"
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

// Receive sharing of ~XOR(a,b) locally
OECL_Share Xor(OECL_Share a, OECL_Share b)
{
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(OECL_Share a, OECL_Share b, OECL_Share &c)
{
/* DATATYPE rl = getRandomVal(0); */
/* DATATYPE rr = getRandomVal(0); */

/* DATATYPE rx = getRandomVal(0); */
/* DATATYPE ry = getRandomVal(1); */
/* DATATYPE maskP1 = XOR(a.p1,b.p1); */
/* DATATYPE maskP2 = XOR(a.p2,b.p2); */

}

void complete_and(OECL_Share &c)
{
}

void prepare_reveal_to_all(OECL_Share a)
{
}    



DATATYPE complete_Reveal(OECL_Share a)
{
return XOR(a.p2, receive_from_live(P2));
}


OECL_Share* alloc_Share(int l)
{
    return new OECL_Share[l];
}



void prepare_receive_from(OECL_Share a[], int id, int l)
{
#if OPT_SHARE == 1 && SHARE_PREP == 0
if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].p1 = get_input_live();
    a[i].p2 = getRandomVal(P1);
    send_to_live(P2, XOR(a[i].p1,a[i].p2));
    }
}
#endif
}

void complete_receive_from(OECL_Share a[], int id, int l)
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
