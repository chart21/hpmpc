#pragma once
#include "replicated_base.hpp"
class Replicated{
bool input_srngs;
    public:
Replicated(bool use_srngs) {input_srngs = use_srngs;}
Share share_SRNG(DATATYPE a)
{
Share s[3];
s[pprev].x = getRandomVal(pprev);
s[pnext].x = getRandomVal(pnext);
s[2].x = XOR(s[pprev].x,s[pnext].x);

s[pprev].a = XOR(s[pnext].x,a); //xi + x(i-1) + a
s[pnext].a = XOR(s[2].x,a); //xi + x(i-1) + a
s[2].a = XOR(s[pprev].x,a); //xi + x(i-1) + a

send_to_live(pprev, s[pprev].a);
send_to_live(pnext, s[pnext].a);

return s[2];


}

void receive_share_SRNG(Share &s, int player)
{
s.a = receive_from_live(player);
}

Share share(DATATYPE a)
{
Share s[3];
s[pprev].x = getRandomVal(pprev);
s[pnext].x = getRandomVal(pnext);
s[2].x = XOR(s[pprev].x,s[pnext].x);

s[pprev].a = XOR(s[pnext].x,a); //xi + x(i-1) + a
s[pnext].a = XOR(s[2].x,a); //xi + x(i-1) + a
s[2].a = XOR(s[pprev].x,a); //xi + x(i-1) + a

send_to_live(pprev, s[pprev].a);
send_to_live(pnext, s[pnext].a);

return s[2];
      }





void share(Share a[], int length)
{
    for(int l = 0; l < length; l++)
    {
    a[l] = share_SRNG(get_input_live());
    }
}


void receive_from_SRNG(Share a[], int id, int l)
{
if(id == PSELF)
{
for (int i = 0; i < l; i++) {
  a[i] = share_SRNG(get_input_live());
}
}
else{
for (int i = 0; i < l; i++) {
    receive_share_SRNG(a[i], id);
}
}
}

void receive_from(Share a[], int id, int l)
{
if(id == PSELF)
{
    return;
}
else{
for (int i = 0; i < l; i++) {
Share s;
s.x = getRandomVal(id);
s.a = receive_from_live(id);
a[i] = s;
}
}
}

void generate_SRNG(Share a[], int id, int length)
{
    for(int l = 0; l < length; l++)
    {
        a[l].x = getRandomVal(id);
    }
}

void prepare_receive_from(Share a[], int id, int l)
{
    if(id == PSELF)
        share(a,l);
    else
        generate_SRNG(a,id,l);
}

void complete_receive_from(Share a[], int id, int l)
{
if(id == PSELF)
    return;
for (int i = 0; i < l; i++) {
    receive_share_SRNG(a[i],id);

}
}

// Receive sharing of ~XOR(a,b) locally
Share Xor(Share a, Share b)
{
    Share result;
    result.x = XOR(a.x,b.x);
    result.a = XOR(a.a,b.a);
    return result; 
}

Share Xor_pub(Share a, DATATYPE b)
{
    Share result;
    result.x = a.x;
    result.a = XOR(a.a,b);
    return result; 
}

Share public_val(DATATYPE a)
{
    Share result;
    result.x = SET_ALL_ZERO();
    result.a = a;
    return result; 
}

Share Not(Share a)
{
    Share result;
    result.x = a.x;
    result.a = NOT(a.a);
    return result; 
}


void reshare(DATATYPE a, DATATYPE u[])
{
//u[pprev] = getRandomVal(pprev); //replace with SRNG
//u[pnext] = getRandomVal(pnext);
u[pprev] = getRandomVal(pprev);
u[pnext] = getRandomVal(pnext);

/* u[pprev] = SET_ALL_ONE(); */
/* u[pnext] = SET_ALL_ONE(); */
/* if(player_id == 0) */
/*     u[pnext] = SET_ALL_ZERO(); */
/* if(player_id == 1) */
/*     u[pprev] = SET_ALL_ZERO(); */

    /* u[pprev] = SET_ALL_ZERO(); //replace with SRNG */
/* u[pnext] = SET_ALL_ONE(); */
u[2] = XOR(u[pprev],u[pnext]);
u[2] = XOR(a,u[2]);
 
}
//prepare AND -> send real value a&b to other P
void prepare_and(Share a, Share b, Share &c)
{
DATATYPE corr = XOR( getRandomVal(pprev), getRandomVal(pnext) );
DATATYPE r =  XOR( XOR(  AND(a.x,b.x), AND(a.a,b.a) ) , corr);  
c.a = r; //used to access value in complete and 
send_to_live(pnext, r);
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(Share &c)
{
c.x = XOR(c.a, receive_from_live(pprev));
}

void prepare_reveal_to_all(Share a)
{
    send_to_live(pnext, a.x);
}    


//reveal to specific player
void prepare_reveal_to(DATATYPE a, int id)
{
    if(PSELF != id)
    {
        send_to_live(id, a);
}
}

DATATYPE complete_Reveal(Share a)
{
DATATYPE result;
result = XOR(a.a,receive_from_live(pprev));
return result;
}

Share* alloc_Share(int l)
{
    return new Share[l];
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
