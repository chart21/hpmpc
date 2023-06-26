#pragma once
#include "../generic_share.hpp"
#define INIT_SHARE DATATYPE

class TTP_init
{
bool input_srngs;
public:
TTP_init(bool use_srngs) {input_srngs = use_srngs;}




DATATYPE share_SRNG(DATATYPE a)
{
    return a;
}

XOR_Share receive_share_SRNG(int player)
{
XOR_Share dummy;
return dummy;
}

DATATYPE share(DATATYPE a)
{
send_to_(P2);
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
   return a;
}

// Receive sharing of ~XOR(a,b) locally
DATATYPE Xor(DATATYPE a, DATATYPE b)
{
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(DATATYPE &a, DATATYPE &b, DATATYPE &c)
{
/* #if VERIFY_BUFFER < 0 */
/* for (int i = 0; i < VERIFY_BUFFER; i++) */
/* { */
/*     send_to_(PNEXT); */

/* } */
/* #endif */
//return u[player_id] * v[player_id];
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(DATATYPE c)
{
/* #if VERIFY_BUFFER > 0 */
/* for (int i = 0; i < VERIFY_BUFFER; i++) */
/* { */
/*     receive_from_(PPREV); */

/* } */
/* #endif */
}

void prepare_reveal_to_all(DATATYPE a)
{
    if(PARTY == 2)
    {
        for(int t = 0; t < num_players-1; t++) 
        {
            send_to_(t);
        }//add to send buffer
    
    }
}    


// These functions need to be somewhere else

DATATYPE complete_Reveal(DATATYPE a)
{
if(PARTY != 2)
{
    receive_from_(P2);
}
return a;
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}

void receive_from_comm(DATATYPE a[], int id, int l)
{
if(id == PSELF)
{
    return;
}
if(PARTY == 2)
{
        for (int i = 0; i < l; i++) {
receive_from_(id);
        }
    }
}

void receive_from(DATATYPE a[], int id, int l)
{
receive_from_comm(a, id, l);
}

void complete_receive_from_comm(DATATYPE a[], int id, int l)
{
receive_from_comm(a, id, l);
}


void prepare_receive_from_comm(DATATYPE a[], int id, int l)
{

if(id == PSELF && PARTY != 2)
{
    for(int i = 0; i < l; i++) 
    send_to_(P2);
}
}
void prepare_receive_from(DATATYPE a[], int id, int l)
{
prepare_receive_from_comm(a, id, l);
}


void complete_receive_from(DATATYPE a[], int id, int l)
{
complete_receive_from_comm(a, id, l);
}

void send()
{
send_();
}
void receive()
{
    receive_();
}
void communicate()
{
communicate_();
}

void finalize(std::string* ips)
{
    finalize_(ips);
}


};
