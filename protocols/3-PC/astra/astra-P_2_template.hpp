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

template <typename func_add>
Evaluator_Share Add(Evaluator_Share a, Evaluator_Share b, func_add ADD)
{
   return Evaluator_Share(ADD(a.mv,b.mv),ADD(a.lv,b.lv));
}



template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Evaluator_Share a, Evaluator_Share b, Evaluator_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
DATATYPE yz2 = getRandomVal(P_0); //yz1
DATATYPE yxy2 = receive_from_live(P_0); 
c.mv = ADD( ADD( SUB( MULT(a.mv,b.mv), ADD( MULT(a.mv,b.lv), MULT(b.mv, a.lv) )) , yz2 ), yxy2); 
send_to_live(P_1,c.mv); 
c.lv = yz2;
}

template <typename func_add, typename func_sub>
void complete_mult(Evaluator_Share &c, func_add ADD, func_sub SUB)
{
c.mv = ADD(c.mv, receive_from_live(P_1)); 
}

void prepare_reveal_to_all(Evaluator_Share a)
{
send_to_live(P_0,a.mv);
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Evaluator_Share a, func_add ADD, func_sub SUB)
{
return SUB(a.mv, receive_from_live(P_0));
}


Evaluator_Share* alloc_Share(int l)
{
return new Evaluator_Share[l];
}

template <typename func_add, typename func_sub>
void prepare_receive_from(Evaluator_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P_0)
{
    if(optimized_sharing == false)
    {
    for(int i = 0; i < l; i++)
    {
        a[i].lv = getRandomVal(P_0);
    }


    }
}
else if(id == P_2) // -> lv = lv1, lv2=0
{
for(int i = 0; i < l; i++)
{
    a[i].lv = getRandomVal(P_0);
    a[i].mv = ADD(get_input_live(),a[i].lv);
    send_to_live(P_1,a[i].mv);
}
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(Evaluator_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P_0)
{
if(optimized_sharing == true) // (0,a+yx1)
{
    for(int i = 0; i < l; i++)
    {
        a[i].mv = SET_ALL_ZERO(); //Check options 
        a[i].lv = receive_from_live(P_0);
    }
}
else{
    for(int i = 0; i < l; i++)
    {
        a[i].mv = receive_from_live(P_0);
    }
    
}
}
else if(id == P_1)
{
for(int i = 0; i < l; i++)
{
a[i].mv = receive_from_live(P_1); 
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
