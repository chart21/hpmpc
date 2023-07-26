#pragma once
#include "sharemind_base.hpp"
#define SHARE Workaround_Share

class Sharemind
{
bool input_srngs;
    public:
Sharemind(bool use_srngs) {input_srngs = use_srngs;}

template <typename func_add, typename func_sub>
Workaround_Share share_SRNG(DATATYPE a, func_add ADD, func_sub SUB)
{
DATATYPE s[3]; //last share always belongs to player itself
s[pprev] = getRandomVal(pprev);
s[pnext] = getRandomVal(pnext);
s[2] = SUB(s[pprev],s[pnext]);
s[2] = ADD(a,s[2]);
return Workaround_Share(s[2]);
}

Workaround_Share receive_share_SRNG(int player)
{

    return Workaround_Share(getRandomVal(player));
}

template <typename func_add, typename func_sub>
Workaround_Share share(Workaround_Share a, func_add ADD, func_sub SUB)
{
DATATYPE s[3]; //last share always belongs to player itself
s[pprev] = getRandomVal(num_players -1);
s[pnext] = getRandomVal(num_players -1);
s[2] = SUB(s[pprev],s[pnext]);
s[2] = ADD(a.val_,s[2]);
send_to_live(pprev, s[pprev]);
send_to_live(pnext, s[ADD]);
return Workaround_Share(s[2]);
      }





template <typename func_add, typename func_sub>
void share(Workaround_Share a[], int length, func_add ADD, func_sub SUB)
{
if(input_srngs == true)
{
    return;
}
else
{
for(int l = 0; l < length; l++)
    a[l] = share(a[l], ADD, SUB);
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

template <typename func_add>
Workaround_Share Add(Workaround_Share a, Workaround_Share b, func_add ADD)
{
   return ADD(a.val_, b.val_);
}


template <typename func_add, typename func_sub>
void reshare(DATATYPE a, DATATYPE u[], func_add ADD, func_sub SUB)
{
u[pprev] = getRandomVal(pprev);
u[pnext] = getRandomVal(pnext);
u[2] = SUB(u[pprev],u[pnext]);
u[2] = ADD(a,u[2]);
 
}

template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Workaround_Share a, Workaround_Share b, Workaround_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
DATATYPE u[3];
DATATYPE v[3];
reshare(a.val_,u, ADD, SUB);
reshare(b.val_,v, ADD, SUB);
send_to_live(pnext, u[2]);
send_to_live(pprev, v[2]);
c.val_ = MULT(u[2],v[2]);
c.helper_ = v[2];
}

template <typename func_add, typename func_sub>
void complete_mult(Workaround_Share &c, func_add ADD, func_sub SUB)
{

DATATYPE u_p = receive_from_live(pprev);
DATATYPE v_n = receive_from_live(pnext);
/* DATATYPE u_i = a; */
DATATYPE v_i = c.helper_;
c.val_ = ADD (c.val_,   ADD ( AND(u_p,v_i) , AND(u_p,v_n) ));
}


void prepare_reveal_to_all(Workaround_Share a)
{
    for(int t = 0; t < num_players-1; t++) // for other protocols, sending buffers may be different for each player
        send_to_live(t, a.val_);
    //add to send buffer
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Workaround_Share a, func_add ADD, func_sub SUB)
{
DATATYPE result = a.val_;
for(int t = 0; t < num_players-1; t++) // for other protocols, sending buffers may be different for each player
    result = SUB(result,receive_from_live(t));
return result;
}


template <typename func_add, typename func_sub>
void receive_from_SRNG(Workaround_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
{
for (int i = 0; i < l; i++) {
  a[i] = SUB( get_input_live(), ADD(getRandomVal(PPREV),getRandomVal(PNEXT)) );
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

template <typename func_add, typename func_sub>
void receive_from(Workaround_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(input_srngs == true)
{
    receive_from_SRNG(a, id, l, ADD, SUB);
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

template <typename func_add, typename func_sub>
void prepare_receive_from(Workaround_Share a[], int id, int l, func_add ADD, func_sub SUB)
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

template <typename func_add, typename func_sub>
void complete_receive_from(Workaround_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(input_srngs == true)
{
    receive_from_SRNG(a,id,l, ADD, SUB);
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
