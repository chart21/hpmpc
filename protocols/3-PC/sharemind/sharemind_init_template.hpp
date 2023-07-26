#pragma once
#include "sharemind_base.hpp"
class Sharemind_init
{
bool input_srngs;
public:
Sharemind_init(bool use_srngs) {input_srngs = use_srngs;}

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
send_to_(pnext);
send_to_(pprev);
/* DATATYPE dummy; */
/* return dummy; */
return a;
}


void share(DATATYPE a[], int length)
{
if(input_srngs == true)
{
    return;
}
else{
    for(int l = 0; l < length; l++)
        a[l] = share(a[l]);
}                                      //
}
XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return a;
}

template <typename func_add>
DATATYPE Add(DATATYPE a, DATATYPE b, func_add ADD)
{
   return a;
}



void reshare(DATATYPE a, DATATYPE u[])
{
 
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c, func_add ADD, func_sub SUB, func_mul MUL)
{
send_to_(pnext);
send_to_(pprev);
}

template <typename func_add, typename func_sub>
void complete_mult(DATATYPE &c, func_add ADD, func_sub SUB)
{
receive_from_(pnext);
receive_from_(pprev);
}


void prepare_reveal_to_all(DATATYPE a)
{
    for(int t = 0; t < num_players-1; t++) 
    {
        send_to_(t);
    }//add to send buffer
}    


//reveal to specific player
void prepare_reveal_to(DATATYPE a, int id)
{
    if(PARTY != id)
    {
        send_to_(id);
    //add to send buffer
}
}
// These functions need to be somewhere else

template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
    for(int t = 0; t < num_players-1; t++) 
    {
        receive_from_(t);
    }
return a;
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}

void receive_from_SRNG(XOR_Share a[], int id, int l)
{
}
void receive_from_comm(DATATYPE a[], int id, int l)
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


void receive_from(DATATYPE a[], int id, int l)
{
if(input_srngs == true)
{
    receive_from_SRNG(a, id, l);
}
else
{
receive_from_comm(a, id, l);
}
}

template <typename func_add, typename func_sub>
void complete_receive_from_comm(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
    return;
else{
for (int i = 0; i < l; i++) {
    receive_from_(id);
}
}
}


void prepare_receive_from_comm(DATATYPE a[], int id, int l)
{
if(id == PSELF)
{
for (int i = 0; i < l; i++) {
    send_to_(pprev);
    send_to_(pnext);
    }
}
else {

return;
    }
}

template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(input_srngs == true)
{
    return; //no sending needed
}
else
{
prepare_receive_from_comm(a, id, l);
}
}


template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(input_srngs == true)
{
    return;
}
else
{
complete_receive_from_comm(a, id, l, ADD, SUB);
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
