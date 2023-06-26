#pragma once
#include "astra_base.hpp"
class ASTRA2
{
bool optimized_sharing;
public:
ASTRA2(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

Evaluator_Share public_val(DATATYPE a)
{
    return Evaluator_Share(a,SET_ALL_ZERO());
}

Evaluator_Share Not(Evaluator_Share a)
{
    a.mv = NOT(a.mv);
   return a;
}

// Receive sharing of ~XOR(a,b) locally
Evaluator_Share Xor(Evaluator_Share a, Evaluator_Share b)
{
   return Evaluator_Share(XOR(a.mv,b.mv),XOR(a.lv,b.lv));
}



//prepare AND -> send real value a&b to other P
void prepare_and(Evaluator_Share a, Evaluator_Share b, Evaluator_Share &c)
{
DATATYPE yz2 = getRandomVal(P0); //yz1
DATATYPE yxy2 = receive_from_live(P0); 
c.mv = XOR( AND(a.mv,b.mv), XOR( XOR(  XOR( AND(a.mv,b.lv), AND(b.mv, a.lv) ), yz2 ), yxy2)); 
send_to_live(P1,c.mv); 
c.lv = yz2;
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(Evaluator_Share &c)
{
// a.p2 already set in last round
c.mv = XOR(c.mv, receive_from_live(P1)); 
}

void prepare_reveal_to_all(Evaluator_Share a)
{
send_to_live(P0,a.mv);
}    



DATATYPE complete_Reveal(Evaluator_Share a)
{
return XOR(a.mv, receive_from_live(0));
}


Evaluator_Share* alloc_Share(int l)
{
return new Evaluator_Share[l];
}


void prepare_receive_from(Evaluator_Share a[], int id, int l)
{
if(id == P0)
{
    if(optimized_sharing == false)
    {
    for(int i = 0; i < l; i++)
    {
        a[i].lv = getRandomVal(P0);
    }


    }
}
else if(id == P2) // -> lv = lv1, lv2=0
{
for(int i = 0; i < l; i++)
{
    a[i].lv = getRandomVal(P0);
    a[i].mv = XOR(get_input_live(),a[i].lv);
    send_to_live(P1,a[i].mv);
}
}
}

void complete_receive_from(Evaluator_Share a[], int id, int l)
{
if(id == P0)
{
if(optimized_sharing == true) // (0,a+yx1)
{
    for(int i = 0; i < l; i++)
    {
        a[i].mv = SET_ALL_ZERO(); //Check options 
        a[i].lv = receive_from_live(P0);
    }
}
else{
    for(int i = 0; i < l; i++)
    {
        a[i].mv = receive_from_live(P0);
    }
    
}
}
else if(id == P1)
{
for(int i = 0; i < l; i++)
{
a[i].mv = receive_from_live(P1); 
a[i].lv = SET_ALL_ZERO();
}
}

/* int offset = {id > player_id ? 1 : 0}; */
/* int player = id - offset; */
/* for(int i = 0; i < l; i++) */
/* { */
/* a[i] = receiving_args[player].received_elements[rounds-1][share_buffer[player]]; */
/* share_buffer[player] +=1; */
/* } */
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
