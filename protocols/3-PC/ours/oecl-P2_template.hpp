#pragma once
#include "../sharemind/sharemind_base.hpp"
/* #include "../../../datatypes/k_bitset.hpp" */
/* #include "../../../datatypes/k_sint.hpp" */
//#include "oecl_base.hpp"
#define SHARE DATATYPE
#define VALS_PER_SHARE 1
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

template <typename func_add>
XOR_Share Add(XOR_Share a, XOR_Share b, func_add ADD)
{
    return ADD(a,b);
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(XOR_Share a, XOR_Share b, XOR_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{

XOR_Share ap1 = getRandomVal(P0); // P2 mask for P1

#if PRE == 1
c = ADD(pre_receive_from_live(P0), MULT(a,b)); // P0_message + (a+rr) (b+rl)
#else
c = ADD(receive_from_live(P0), MULT(a,b)); // P0_message + (a+rr) (b+rl)
#endif

send_to_live(P1, ADD(ap1,c)); 
}

template <typename func_add, typename func_sub>
void complete_mult(XOR_Share &c, func_add ADD, func_sub SUB)
{
c = SUB(c, receive_from_live(P1)); 
}


void prepare_reveal_to_all(XOR_Share a)
{
send_to_live(P0, a);
}

template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(XOR_Share a, func_add ADD, func_sub SUB)
{
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
return SUB(a, pre_receive_from_live(P0));
#else
return SUB(a, receive_from_live(P0));
#endif
}

XOR_Share* alloc_Share(int l)
{
    return new XOR_Share[l];
}

template <typename func_add, typename func_sub>
void prepare_receive_from(XOR_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P2)
{
for(int i = 0; i < l; i++)
{
    a[i] = get_input_live();     
    /* a[i].p1 = getRandomVal(0); *1/ */
    send_to_live(P1, ADD(getRandomVal(P0),a[i]));
}
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(XOR_Share a[], int id, int l, func_add ADD, func_sub SUB)
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


//higher level functions


void A2B_S1(DATATYPE in[], DATATYPE out[])
{
    //convert share a + x1 to boolean
    unorthogonalize_arithmetic(in, (UINT_TYPE*) out);
    orthogonalize_boolean((UINT_TYPE*) out, out);
    for(int i = 0; i < BITLENGTH; i++)
    {
        send_to_live(P1, XOR(out[i],getRandomVal(P0))); // send all bits a + x_1 XOR r_0,2 to P1
    }
}

void A2B_S2(DATATYPE out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        out[i] = SET_ALL_ZERO();
    }
}

void prepare_A2B(DATATYPE in[], DATATYPE out[])
{
    A2B_S1(in, out);
    A2B_S2(out);
}

void complete_A2B(DATATYPE out[])
{
}

};

