#pragma once
#include "tetrad_base.hpp"
#define PRE_SHARE Tetrad_Share
class Tetrad2
{
bool optimized_sharing;
public:
Tetrad2(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

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


DATATYPE y2ab = XOR( XOR(AND(a.l0,b.l1),AND(a.l1,b.l0)), AND(a.l0,a.l0));
DATATYPE u2 = getRandomVal(P023);

//q:
c.mv = SET_ALL_ZERO();
c.l0 = receive_from_live(P3);
store_compare_view(P0, c.l0);
c.l1 = SET_ALL_ZERO();  //lambda3

DATATYPE s = getRandomVal(P123);

DATATYPE y1 = XOR( XOR(AND(a.l0,b.mv),AND(a.mv,b.l0)), XOR(y2ab,u2));
DATATYPE y3 = XOR(AND(a.l1,b.mv),AND(a.mv,b.l1));
send_to_live(P1, y1);
DATATYPE z_r = XOR( XOR(y1, y3), AND(a.mv,b.mv));

//Trick to store values neede later
c.storage = c.l0;
c.l0 = y1;
c.l1 = s;
c.mv = z_r;

}

void complete_and(Tetrad_Share &c)
{
    DATATYPE y2 = receive_from_live(P1);
    DATATYPE v = XOR(XOR(c.l0,c.l1),y2); // y1 + y2 + s for verification
    store_compare_view(P012, v);
    c.mv = XOR(c.mv, y2);

    //p:
    DATATYPE pl1 = SET_ALL_ZERO(); // known by all
    DATATYPE pl2 = SET_ALL_ZERO(); // known by all
    DATATYPE pl3 = getRandomVal(P123); //hide from P0
    DATATYPE pmv = XOR(c.mv,pl3);                                   
    send_to_live(P0, pmv);



    //o = p + q
    c.mv = pmv;
    c.l0 = c.storage; //lambda2
    c.l1 = pl3; //lambda3
}



void prepare_reveal_to_all(Tetrad_Share a)
{
}    



DATATYPE complete_Reveal(Tetrad_Share a)
{

                              
DATATYPE lambda1 = receive_from_live(P3);
store_compare_view(P0, lambda1); //get help from P0 to veriy
DATATYPE result = XOR(a.mv, lambda1);
result = XOR(result, a.l0);
result = XOR(result, a.l1);
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
    a[i].l0 = getRandomVal(P023); //l2
    a[i].l1 = getRandomVal(P123); //l3
    DATATYPE l2 = SET_ALL_ZERO();
    a[i].mv = XOR( XOR(a[i].mv, l2), XOR(a[i].l0,a[i].l1));
    send_to_live(P0, a[i].mv);
    send_to_live(P1, a[i].mv);

    
    
    } 
}
else if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = getRandomVal(P023);
    a[i].l1 = SET_ALL_ZERO();
    }
}
else if(id == P1)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = SET_ALL_ZERO();
    a[i].l1 = getRandomVal(P123);
    }
}
else if(id == P3)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = getRandomVal(P023);
    a[i].l1 = getRandomVal(P123);
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

        if(id != P0)
            for(int i = 0; i < l; i++)
                store_compare_view(P0,a[i].mv);
        if(id != P1)
            for(int i = 0; i < l; i++)
                store_compare_view(P1,a[i].mv);


    
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
