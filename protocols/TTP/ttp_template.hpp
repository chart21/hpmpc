#pragma once
#include "../generic_share.hpp"
#define SHARE DATATYPE

class TTP
{
bool input_srngs;
    public:
TTP(bool use_srngs) {input_srngs = use_srngs;}

DATATYPE share(DATATYPE a)
{
send_to_live(P2, a);
return a;
}




void share(DATATYPE a[], int length)
{
if(PARTY != 2)
{
for(int l = 0; l < length; l++)
    a[l] = share(a[l]);
}
}

XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return NOT(a);
}

// Receive sharing of ~XOR(a,b) locally
DATATYPE Xor(DATATYPE a, DATATYPE b)
{
   return XOR(a, b);
}



void prepare_and(XOR_Share a, XOR_Share b, XOR_Share &c)
{
c = AND(a,b);
/* #if VERIFY_BUFFER > 0 */
/* for (int i = 0; i < VERIFY_BUFFER; i++) */
/* { */
/*     send_to_live(PNEXT, c); */

/* } */
/* #endif */
}
// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(XOR_Share &c)
{
/* #if VERIFY_BUFFER > 0 */
/* for (int i = 0; i < VERIFY_BUFFER; i++) */
/* { */
/*    c = receive_from_live(PPREV); */

/* } */
/* #endif */
}

void prepare_reveal_to_all(DATATYPE a)
{
    if(PARTY == 2){
    for(int t = 0; t < num_players-1; t++) // for other protocols, sending buffers may be different for each player
        send_to_live(t, a);
    }   
}



DATATYPE complete_Reveal(DATATYPE a)
{
DATATYPE result = a;
if(PARTY != 2)
{
    result = receive_from_live(P2);
}

return result;
}




void prepare_receive_from(DATATYPE a[], int id, int l)
{
if(id == PSELF && PARTY != 2)
{
    for(int s = 0; s < l; s++)
    {
        share(get_input_live());
    }
}
}

void complete_receive_from(DATATYPE a[], int id, int l)
{

#if PARTY == 2
if(id == P2)
{
    for(int s = 0; s < l; s++)
    {
        a[s] = get_input_live();
    }
}
else
{
for (int i = 0; i < l; i++) {
    a[i] = receive_from_live(id);
}
}
#endif
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}



void finalize()
{

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
