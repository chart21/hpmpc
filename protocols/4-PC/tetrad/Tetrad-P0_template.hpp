#pragma once
#include "tetrad_base.hpp"
#define PRE_SHARE Tetrad_Share
class Tetrad0
{
bool optimized_sharing;
public:
Tetrad0(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

Tetrad_Share public_val(DATATYPE a)
{
    return Tetrad_Share(a,SET_ALL_ZERO(),SET_ALL_ZERO());
}

Tetrad_Share Not(Tetrad_Share a)
{
   return Tetrad_Share(NOT(a.mv),a.l0,a.l1);
}

// Receive sharing of ~XOR(a,b) locally
Tetrad_Share Xor(Tetrad_Share a, Tetrad_Share b)
{
    a.mv = XOR(a.mv,b.mv);
    a.l0 = XOR(a.l0,b.l0);
    a.l1 = XOR(a.l1,b.l1);
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(Tetrad_Share a, Tetrad_Share b, Tetrad_Share &c)
{
DATATYPE y3ab = XOR( XOR(AND(a.l0,b.l1),AND(a.l1,b.l0)), AND(a.l0,a.l0));
DATATYPE u1 = getRandomVal(P013);
DATATYPE u2 = getRandomVal(P023);
DATATYPE r = XOR(y3ab, XOR(u1,u2));
Tetrad_Share q;

//q:
c.mv = SET_ALL_ZERO();
c.l0 = getRandomVal(P013); //lambda1
c.l1 = XOR(r,c.l0);  //lambda2 
store_compare_view(P2, c.l1); // verify if P2 gets correct value from P3


DATATYPE v = XOR( AND( XOR(a.l0,a.l1), b.mv) , AND ( XOR(b.l0,b.l1), a.mv));
v = XOR(XOR(v,u1),u2);
c.mv = v; //Trick, can be set to zero lateron
}

void complete_and(Tetrad_Share &c)
{
    DATATYPE w = receive_from_live(P3);
    c.mv = XOR(c.mv,w);
    c.mv = SET_ALL_ZERO(); //restore actual value of c.mv

    Tetrad_Share p; 
    p.l0 = SET_ALL_ZERO(); //lambda1
    p.l1 = SET_ALL_ZERO(); //lambda2
    p.mv = receive_from_live(P2);
    store_compare_view(P1,p.mv);
    
    store_compare_view(P012,c.mv);

    //o = p + q
    c.mv = XOR(c.mv,p.mv);
    c.l0 = XOR(c.l0,p.l0);
    c.l1 = XOR(c.l1,p.l1);
}



void prepare_reveal_to_all(Tetrad_Share a)
{
#if PRE == 0
    send_to_live(P3, a.mv);
#endif
}    



DATATYPE complete_Reveal(Tetrad_Share a)
{

//receive lambda3 from P3
DATATYPE l3 = receive_from_live(P3);
DATATYPE result = XOR(a.mv, l3);
result = XOR(result, a.l0);
result = XOR(result, a.l1);
store_compare_view(P1, l3); //verify own value

store_compare_view(P1, a.l1);  // verify others
store_compare_view(P2, a.l0); 
return result;
}


Tetrad_Share* alloc_Share(int l)
{
    return new Tetrad_Share[l];
}



void prepare_receive_from(Tetrad_Share a[], int id, int l)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    a[i].mv = get_input_live();
    a[i].l0 = getRandomVal(P013); //l1
    a[i].l1 = getRandomVal(P023);
    DATATYPE l3 = SET_ALL_ZERO();
    a[i].mv = XOR( XOR(a[i].mv, l3), XOR(a[i].l0,a[i].l1));
    send_to_live(P1, a[i].mv);
    send_to_live(P2, a[i].mv);

    
    
    } 
}
else if(id == P1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = getRandomVal(P013);
    a[i].l1 = SET_ALL_ZERO();
    }
}
else if(id == P2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = SET_ALL_ZERO();
    a[i].l1 = getRandomVal(P023);
    }
}
else if(id == P3)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = getRandomVal(P013);
    a[i].l1 = getRandomVal(P023);
    }
}
}
void complete_receive_from(Tetrad_Share a[], int id, int l)
{
if(id != PSELF)
{
    for(int i = 0; i < l; i++)
    {
    a[i].mv = receive_from_live(id);
    }

        if(id != P1)
            for(int i = 0; i < l; i++)
                store_compare_view(P1,a[i].mv);
        if(id != P2)
            for(int i = 0; i < l; i++)
                store_compare_view(P2,a[i].mv);


    
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
/* #if PRE == 0 */
    communicate_live();
/* #endif */
}

};
