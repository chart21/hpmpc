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
Dealer_Share Xor(Dealer_Share a, Dealer_Share b)
{
    a.l1 = XOR(a.l1,b.l1);
    a.l2 = XOR(a.l2,b.l2);
    a.l3 = XOR(a.l3,b.l3);
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(Dealer_Share a, Dealer_Share b, Dealer_Share &c)
{
DATATYPE y1ab = XOR( XOR(AND(a.l1,b.l3),AND(a.l3,b.l1)), AND(a.l3,a.l3));
DATATYPE y2ab = XOR( XOR(AND(a.l2,b.l3),AND(a.l3,b.l2)), AND(a.l2,a.l2));
DATATYPE y3ab = XOR( XOR(AND(a.l1,b.l2),AND(a.l2,b.l1)), AND(a.l1,a.l1));
DATATYPE u1 = getRandomVal(P013);
DATATYPE u2 = getRandomVal(P023);
DATATYPE r = XOR(y3ab, XOR(u1,u2));
Dealer_Share q;

DATATYPE s = getRandomVal(P123);
DATATYPE w = XOR(s, XOR(y1ab,y2ab));
send_to_live(P0, w);
//q:
c.l1 = getRandomVal(P013); //lambda1
c.l2 = XOR(r,c.l1);  //lambda2 
c.l3 = SET_ALL_ZERO(); //lambda3
send_to_live(P2, c.l2);


}

void complete_and(Dealer_Share &c)
{
    Dealer_Share p; 
    p.l1 = SET_ALL_ZERO(); //lambda1
    p.l2 = SET_ALL_ZERO(); //lambda2
    p.l3 = getRandomVal(P123); //lambda3

    //o = p + q
    c.l1 = XOR(c.l1,p.l1);
    c.l2 = XOR(c.l2,p.l2);
    c.l3 = XOR(c.l3,p.l3);
}



void prepare_reveal_to_all(Dealer_Share a)
{
    send_to_live(P2, a.l1);
    send_to_live(P1, a.l2);
    send_to_live(P0, a.l3);
}    



DATATYPE complete_Reveal(Dealer_Share a)
{

//receive lambda3 from P3
DATATYPE mv = receive_from_live(P0);
DATATYPE result = XOR(mv, a.l3);
result = XOR(result, a.l1);
result = XOR(result, a.l2);
store_compare_view(P1, mv); //verify own value

return result;
}


Dealer_Share* alloc_Share(int l)
{
    return new Dealer_Share[l];
}



void prepare_receive_from(Dealer_Share a[], int id, int l)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    DATATYPE mv = get_input_live();
    a[i].l1 = getRandomVal(P013); //l1
    a[i].l2 = getRandomVal(P023);
    a[i].l3 = getRandomVal(P123);
    mv = XOR( XOR(mv, a[i].l3), XOR(a[i].l1,a[i].l2));
    send_to_live(P0, mv);
    send_to_live(P1, mv);
    send_to_live(P2, mv);

    
    
    } 
}
else if(id == P1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l1 = getRandomVal(P013);
    a[i].l2 = SET_ALL_ZERO();
    a[i].l3 = getRandomVal(P123);
    }
}
else if(id == P2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l1 = SET_ALL_ZERO();
    a[i].l2 = getRandomVal(P023);
    a[i].l3 = getRandomVal(P123);
    }
}
else if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l1 = getRandomVal(P013);
    a[i].l2 = getRandomVal(P023); 
    a[i].l3 = SET_ALL_ZERO();
    }
}
}
void complete_receive_from(Dealer_Share a[], int id, int l)
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
