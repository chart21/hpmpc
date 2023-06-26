#pragma once
#include "astra_base.hpp"
class ASTRA1
{
bool optimized_sharing;
public:
ASTRA1(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

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
DATATYPE yz1 = getRandomVal(P0); //yz1
DATATYPE yxy1 = getRandomVal(P0); 
c.mv = XOR( XOR(  XOR( AND(a.mv,b.lv), AND(b.mv, a.lv) ), yz1 ), yxy1); 
c.lv = yz1;
send_to_live(P2,c.mv);
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(Evaluator_Share &c)
{
// a.p2 already set in last round
c.mv = XOR(c.mv, receive_from_live(P2));
}

void prepare_reveal_to_all(Evaluator_Share a)
{
return;
}    



DATATYPE complete_Reveal(Evaluator_Share a)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */

return XOR(a.mv, receive_from_live(P0));
}





Evaluator_Share* alloc_Share(int l)
{
    return new Evaluator_Share[l];
}


void prepare_receive_from(Evaluator_Share a[], int id, int l)
{
if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
        a[i].lv = getRandomVal(P0);
    }
}
else if(id == P1) // -> lv = lv2, lv1=0
{
for(int i = 0; i < l; i++)
{
    a[i].lv = getRandomVal(P0);
    a[i].mv = XOR(get_input_live(),a[i].lv);
    send_to_live(P2,a[i].mv);
}
}
}

void complete_receive_from(Evaluator_Share a[], int id, int l)
{
if(id == P0)
{
if(optimized_sharing == true) // -> 0,lv1
{
    for(int i = 0; i < l; i++)
    {
        a[i].mv = SET_ALL_ZERO(); //check options

    }
}
else{
    for(int i = 0; i < l; i++)
    {
        a[i].mv = receive_from_live(P0);
    }
    
}
}
else if(id == P2)
{
for(int i = 0; i < l; i++)
{
a[i].mv = receive_from_live(P2);
a[i].lv = SET_ALL_ZERO();
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

};
