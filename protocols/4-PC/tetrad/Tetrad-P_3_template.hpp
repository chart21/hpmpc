#pragma once
#include "tetrad_base.hpp"
#define PRE_SHARE Dealer_Share
class Tetrad3
{
bool optimized_sharing;
public:
Tetrad3(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

Dealer_Share public_val(DATATYPE a)
{
    return Dealer_Share(SET_ALL_ZERO(),SET_ALL_ZERO(),SET_ALL_ZERO());
}

Dealer_Share Not(Dealer_Share a)
{
   return a;
}

// Receive sharing of ~XOR(a,b) locally
template <typename func_add>
Dealer_Share Add(Dealer_Share a, Dealer_Share b, func_add ADD)
{
    a.l1 = ADD(a.l1,b.l1);
    a.l2 = ADD(a.l2,b.l2);
    a.l3 = ADD(a.l3,b.l3);
   return a;
}



template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Dealer_Share a, Dealer_Share b, Dealer_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
DATATYPE y1ab = ADD( ADD(AND(a.l1,b.l3),AND(a.l3,b.l1)), AND(a.l3,a.l3));
DATATYPE y2ab = ADD( ADD(AND(a.l2,b.l3),AND(a.l3,b.l2)), AND(a.l2,a.l2));
DATATYPE y3ab = ADD( ADD(AND(a.l1,b.l2),AND(a.l2,b.l1)), AND(a.l1,a.l1));
DATATYPE u1 = getRandomVal(P_013);
DATATYPE u2 = getRandomVal(P_023);
DATATYPE r = SUB(y3ab, ADD(u1,u2));
Dealer_Share q;

DATATYPE s = getRandomVal(P_123);
DATATYPE w = ADD(s, ADD(y1ab,y2ab));
send_to_live(P_0, w);
//q:
c.l2 = getRandomVal(P_013); //lambda2
c.l1 = SUB(SET_ALL_ZERO(), ADD(r,c.l2));  //lambda1 
c.l3 = SET_ALL_ZERO(); //lambda3
send_to_live(P_2, c.l2);


}

template <typename func_add, typename func_sub>
void complete_mult(Dealer_Share &c, func_add ADD, func_sub SUB)
{
    Dealer_Share p; 
    p.l1 = SET_ALL_ZERO(); //lambda1
    p.l2 = SET_ALL_ZERO(); //lambda2
    p.l3 = getRandomVal(P_123); //lambda3

    //o = p + q
    c.l1 = ADD(c.l1,p.l1);
    c.l2 = ADD(c.l2,p.l2);
    c.l3 = ADD(c.l3,p.l3);
}



void prepare_reveal_to_all(Dealer_Share a)
{
    send_to_live(P_2, a.l1);
    send_to_live(P_1, a.l2);
    send_to_live(P_0, a.l3);
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Dealer_Share a, func_add ADD, func_sub SUB)
{

//receive lambda3 from P_3
DATATYPE mv = receive_from_live(P_0);
DATATYPE result = SUB(mv, a.l3);
result = SUB(result, a.l1);
result = SUB(result, a.l2);
store_compare_view(P_1, mv); //verify own value

return result;
}


Dealer_Share* alloc_Share(int l)
{
    return new Dealer_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(Dealer_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    DATATYPE mv = get_input_live();
    a[i].l1 = getRandomVal(P_013); //l1
    a[i].l2 = getRandomVal(P_023);
    a[i].l3 = getRandomVal(P_123);
    mv = ADD( ADD(mv, a[i].l3), ADD(a[i].l1,a[i].l2));
    send_to_live(P_0, mv);
    send_to_live(P_1, mv);
    send_to_live(P_2, mv);

    
    
    } 
}
else if(id == P_1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l1 = getRandomVal(P_013);
    a[i].l2 = SET_ALL_ZERO();
    a[i].l3 = getRandomVal(P_123);
    }
}
else if(id == P_2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l1 = SET_ALL_ZERO();
    a[i].l2 = getRandomVal(P_023);
    a[i].l3 = getRandomVal(P_123);
    }
}
else if(id == P_0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l1 = getRandomVal(P_013);
    a[i].l2 = getRandomVal(P_023); 
    a[i].l3 = SET_ALL_ZERO();
    }
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(Dealer_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
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
/* #if PRE == 0 */
    communicate_live();
/* #endif */
}

};
