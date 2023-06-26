#pragma once
#include "../sharemind/sharemind_base.hpp"
//#include "oecl_base.hpp"
#define SHARE DATATYPE
class OECL2
{
bool optimized_sharing;
public:
OECL2(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

XOR_Share public_val(DATATYPE a)
{
    return a;
}

XOR_Share Not(XOR_Share a)
{
    return NOT(a);
}

// Receive sharing of ~XOR(a,b) locally
XOR_Share Xor(XOR_Share a, XOR_Share b)
{
   return XOR(a,b);
}



void prepare_and(XOR_Share a, XOR_Share b, XOR_Share &c)
{

XOR_Share ap1 = getRandomVal(P0); // P2 mask for P1

#if PRE == 1
c = XOR(pre_receive_from_live(P0), AND(a,b)); // P0_message + (a+rr) (b+rl)
#else
c = XOR(receive_from_live(P0), AND(a,b)); // P0_message + (a+rr) (b+rl)
#endif

send_to_live(P1, XOR(ap1,c)); 
}

void complete_and(XOR_Share &c)
{
c = XOR(c, receive_from_live(P1)); 
}

void prepare_reveal_to_all(XOR_Share a)
{
send_to_live(P0, a);
}


DATATYPE complete_Reveal(XOR_Share a)
{
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
return XOR(a, pre_receive_from_live(P0));
#else
return XOR(a, receive_from_live(P0));
#endif
}

XOR_Share* alloc_Share(int l)
{
    return new XOR_Share[l];
}


void prepare_receive_from(XOR_Share a[], int id, int l)
{
if(id == P2)
{
for(int i = 0; i < l; i++)
{
    a[i] = get_input_live();     
    /* a[i].p1 = getRandomVal(0); *1/ */
    send_to_live(P1, XOR(getRandomVal(P0),a[i]));
}
}
}

void complete_receive_from(XOR_Share a[], int id, int l)
{
if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
        a[i] = pre_receive_from_live(P0);
#else
        a[i] = receive_from_live(P0);
#endif
    }
}
else if(id == P1)
{
for(int i = 0; i < l; i++)
{
a[i] = receive_from_live(P1);
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
