#pragma once
#include "tetrad_base.hpp"
#define PRE_SHARE Tetrad_Share
class Tetrad1
{
bool optimized_sharing;
public:
Tetrad1(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

Tetrad_Share public_val(DATATYPE a)
{
    return Tetrad_Share(a,SET_ALL_ZERO(),SET_ALL_ZERO());
}

Tetrad_Share Not(Tetrad_Share a)
{
   return Tetrad_Share(NOT(a.mv),a.l0,a.l1);
}

template <typename func_add>
Tetrad_Share Add(Tetrad_Share a, Tetrad_Share b, func_add ADD)
{
    a.mv = ADD(a.mv,b.mv);
    a.l0 = ADD(a.l0,b.l0);
    a.l1 = ADD(a.l1,b.l1);
   return a;
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Tetrad_Share a, Tetrad_Share b, Tetrad_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{


DATATYPE y1ab = ADD( ADD(MULT(a.l0,b.l1),MULT(a.l1,b.l0)), MULT(a.l1,a.l1));
DATATYPE u1 = getRandomVal(P_013);

//q:
c.mv = SET_ALL_ZERO();
c.l0 = getRandomVal(P_013); //lambda1
c.l1 = SET_ALL_ZERO();  //lambda3

DATATYPE s = getRandomVal(P_123);

DATATYPE y2 = SUB( ADD(y1ab,u1), ADD(AND(a.l0,b.mv),AND(a.mv,b.l0))) ;
DATATYPE y3 = SUB(SET_ALL_ZERO(),ADD(MULT(a.l1,b.mv),MULT(a.mv,b.l1)));
send_to_live(P_2, y2);
DATATYPE z_r = ADD( ADD(y2, y3), MULT(a.mv,b.mv));

//Trick to store values neede later
c.storage = c.l0;
c.l0 = y2;
c.l1 = s;
c.mv = z_r;

}

template <typename func_add, typename func_sub>
void complete_mult(Tetrad_Share &c, func_add ADD, func_sub SUB)
{
    DATATYPE y1 = receive_from_live(P_2);
    DATATYPE v = ADD(ADD(c.l0,c.l1),y1); // y1 + y2 + s for verification
    store_compare_view(P_012, v);
    c.mv = ADD(c.mv, y1);

    //p:
    /* DATATYPE pl1 = SET_ALL_ZERO(); // known by all */
    /* DATATYPE pl2 = SET_ALL_ZERO(); // known by all */
    DATATYPE pl3 = getRandomVal(P_123); //hide from P_0
    DATATYPE pmv = ADD(c.mv,pl3);                                   //
    store_compare_view(P_0, pmv);



    //o = p + q
    c.mv = pmv;
    c.l0 = c.storage; //lambda1
    c.l1 = pl3; //lambda3
}



void prepare_reveal_to_all(Tetrad_Share a)
{
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Tetrad_Share a, func_add ADD, func_sub SUB)
{

//receive lambda3 from P_3
store_compare_view(P_0, a.l0); //help P_0 verifying
store_compare_view(P_3, a.mv); //help P_3 verifying
                              
DATATYPE lambda2 = receive_from_live(P_3);
store_compare_view(P_0, lambda2); //get help from P_0 to veriy
DATATYPE result = SUB(a.mv, lambda2);
result = SUB(result, a.l0);
result = SUB(result, a.l1);
return result;
}


Tetrad_Share* alloc_Share(int l)
{
    return new Tetrad_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(Tetrad_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
    a[i].mv = get_input_live();
    a[i].l0 = getRandomVal(P_013); //l1
    a[i].l1 = getRandomVal(P_123); //l3
    DATATYPE l2 = SET_ALL_ZERO();
    a[i].mv = ADD( ADD(a[i].mv, l2), ADD(a[i].l0,a[i].l1));
    send_to_live(P_0, a[i].mv);
    send_to_live(P_2, a[i].mv);

    
    
    } 
}
else if(id == P_0)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = getRandomVal(P_013);
    a[i].l1 = SET_ALL_ZERO();
    }
}
else if(id == P_2)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = SET_ALL_ZERO();
    a[i].l1 = getRandomVal(P_123);
    }
}
else if(id == P_3)
{
    for(int i = 0; i < l; i++)
    {
    a[i].l0 = getRandomVal(P_013);
    a[i].l1 = getRandomVal(P_123);
    }
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(Tetrad_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id != PSELF)
{
    for(int i = 0; i < l; i++)
    {
    a[i].mv = receive_from_live(id);
    }

        if(id != P_0)
            for(int i = 0; i < l; i++)
                store_compare_view(P_0,a[i].mv);
        if(id != P_2)
            for(int i = 0; i < l; i++)
                store_compare_view(P_2,a[i].mv);


    
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
