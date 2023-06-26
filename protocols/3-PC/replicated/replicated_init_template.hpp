#pragma once
#include "../../generic_share.hpp"
class Replicated_init{
bool input_srngs;
    public:
Replicated_init(bool use_srngs) {input_srngs = use_srngs;}



XOR_Share share_SRNG(DATATYPE a)
{
send_to_(pnext);
send_to_(pprev);
XOR_Share dummy;
return dummy;
}

XOR_Share receive_share_SRNG(int player)
{
receive_from_(player);
XOR_Share s;
return s;
}



void share(XOR_Share a[], int length)
{

    for(int l = 0; l < length; l++)
    {
    share_SRNG(player_input[share_buffer[player_id]]);  
}
}


void prepare_receive_from(XOR_Share a[], int id, int l)
{
    if(id == PSELF)
        share(a,l);
}

void complete_receive_from(XOR_Share a[], int id, int l)
{
if(id == PSELF)
    return;
for (int i = 0; i < l; i++) {
    a[i] = receive_share_SRNG(id);

}
}


// Receive sharing of ~XOR(a,b) locally
XOR_Share Xor(XOR_Share a, XOR_Share b)
{
    return a; 
}

XOR_Share Xor_pub(XOR_Share a, DATATYPE b)
{
    return a; 
}

XOR_Share public_val(DATATYPE a)
{
    XOR_Share dummy;
    return dummy; 
}

XOR_Share Not(XOR_Share a)
{
    return a; 
}


//prepare AND -> send real value a&b to other P
void prepare_and(XOR_Share a, XOR_Share b, XOR_Share &c)
{
send_to_(pnext);
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(XOR_Share &c)
{
receive_from_(pprev);

}

void prepare_reveal_to_all(XOR_Share a)
{
    send_to_(pnext);
    //add to send buffer
}    



DATATYPE complete_Reveal(XOR_Share a)
{
DATATYPE result;
receive_from_(pprev);
return result;
}


XOR_Share* alloc_Share(int l)
{
    return new XOR_Share[l];
}

void receive_from_SRNG(XOR_Share a[], int id, int l)
{
if(id == PSELF)
{
for (int i = 0; i < l; i++) {
    DATATYPE dummy;
  a[i] = share_SRNG(dummy);  
}
}
else{
for (int i = 0; i < l; i++) {
    a[i] = receive_share_SRNG(id);
}
}
}
void receive_from(XOR_Share a[], int id, int l)
{
if(id == PSELF)
{
    return;
}
else{
for (int i = 0; i < l; i++) {
receive_from_(id);
}
}
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
