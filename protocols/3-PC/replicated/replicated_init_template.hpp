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


template <typename func_add, typename func_sub>
void prepare_receive_from(XOR_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
    if(id == PSELF)
        share(a,l);
}

template <typename func_add, typename func_sub>
void complete_receive_from(XOR_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
    return;
for (int i = 0; i < l; i++) {
    a[i] = receive_share_SRNG(id);

}
}

template <typename func_add>
XOR_Share Add(XOR_Share a, XOR_Share b, func_add ADD)
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

template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(XOR_Share a, XOR_Share b, XOR_Share &c, func_add ADD, func_sub SUB, func_mul MUL)
{
send_to_(pnext);
}

template <typename func_add, typename func_sub>
void complete_mult(XOR_Share &c, func_add ADD, func_sub SUB)
{
receive_from_(pprev);

}


void prepare_reveal_to_all(XOR_Share a)
{
    send_to_(pnext);
    //add to send buffer
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(XOR_Share a, func_add ADD, func_sub SUB)
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
