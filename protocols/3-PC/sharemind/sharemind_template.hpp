#pragma once
#include "sharemind_base.hpp"
#define SHARE Workaround_Share

class Sharemind
{
bool input_srngs;
    public:
Sharemind(bool use_srngs) {input_srngs = use_srngs;}
Workaround_Share share_SRNG(DATATYPE a)
{
DATATYPE s[3]; //last share always belongs to player itself
s[pprev] = getRandomVal(pprev);
s[pnext] = getRandomVal(pnext);
s[2] = XOR(s[pprev],s[pnext]);
s[2] = XOR(a,s[2]);
return Workaround_Share(s[2]);
}

Workaround_Share receive_share_SRNG(int player)
{

    return Workaround_Share(getRandomVal(player));
}

Workaround_Share share(Workaround_Share a)
{
DATATYPE s[3]; //last share always belongs to player itself
s[pprev] = getRandomVal(num_players -1);
s[pnext] = getRandomVal(num_players -1);
s[2] = XOR(s[pprev],s[pnext]);
s[2] = XOR(a.val_,s[2]);
send_to_live(pprev, s[pprev]);
send_to_live(pnext, s[pnext]);
return Workaround_Share(s[2]);
      }





void share(Workaround_Share a[], int length)
{
if(input_srngs == true)
{
    return;
}
else
{
for(int l = 0; l < length; l++)
    a[l] = share(a[l]);
}
                                           //
}
Workaround_Share public_val(DATATYPE a)
{
    return Workaround_Share(a);
}

Workaround_Share Not(Workaround_Share a)
{
   return NOT(a.val_);
}

// Receive sharing of ~XOR(a,b) locally
Workaround_Share Xor(Workaround_Share a, Workaround_Share b)
{
   return XOR(a.val_, b.val_);
}



void reshare(DATATYPE a, DATATYPE u[])
{
u[pprev] = getRandomVal(pprev);
u[pnext] = getRandomVal(pnext);
u[2] = XOR(u[pprev],u[pnext]);
u[2] = XOR(a,u[2]);
 
}
//prepare AND -> send real value a&b to other P
void prepare_and(Workaround_Share a, Workaround_Share b, Workaround_Share &c)
{
DATATYPE u[3];
DATATYPE v[3];
reshare(a.val_,u);
reshare(b.val_,v);
send_to_live(pnext, u[2]);
send_to_live(pprev, v[2]);
c.val_ = AND(u[2],v[2]);
c.helper_ = v[2];
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(Workaround_Share &c)
{

DATATYPE u_p = receive_from_live(pprev);
DATATYPE v_n = receive_from_live(pnext);
/* DATATYPE u_i = a; */
DATATYPE v_i = c.helper_;
c.val_ = XOR (c.val_,   XOR ( AND(u_p,v_i) , AND(u_p,v_n) ));
}

void prepare_reveal_to_all(Workaround_Share a)
{
    for(int t = 0; t < num_players-1; t++) // for other protocols, sending buffers may be different for each player
        send_to_live(t, a.val_);
    //add to send buffer
}    


// These functions need to be somewhere else

DATATYPE complete_Reveal(Workaround_Share a)
{
DATATYPE result = a.val_;
for(int t = 0; t < num_players-1; t++) // for other protocols, sending buffers may be different for each player
    result = XOR(result,receive_from_live(t));
return result;
}


void receive_from_SRNG(Workaround_Share a[], int id, int l)
{
if(id == PSELF)
{
for (int i = 0; i < l; i++) {
  a[i] = get_input_live();
}
}
else{
for (int i = 0; i < l; i++) {
    a[i] = receive_share_SRNG(id);
}
}
}
void receive_from_comm(Workaround_Share a[], int id, int l)
{
if(id == PSELF)
{

    return; // input is already set in sharing phase

}
else{
for (int i = 0; i < l; i++) {
a[i] = receive_from_live(id);
}
}
}

void receive_from(Workaround_Share a[], int id, int l)
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


void prepare_receive_from_comm(Workaround_Share a[], int id, int length)
{
if(id == PSELF)
{
    for(int l = 0; l < length; l++)
    {
        a[l] = get_input_live();
    }
}
}

void prepare_receive_from(Workaround_Share a[], int id, int l)
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

void complete_receive_from_comm(Workaround_Share a[], int id, int l)
{

if(id != PSELF)
{
    receive_from_comm(a,id,l);
}
}

void complete_receive_from(Workaround_Share a[], int id, int l)
{
if(input_srngs == true)
{
    receive_from_SRNG(a,id,l);
}
else
{
complete_receive_from_comm(a, id, l);
}
}



Workaround_Share* alloc_Share(int l)
{
    return new Workaround_Share[l];
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
