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

template <typename func_add>
Tetrad_Share Add(Tetrad_Share a, Tetrad_Share b, func_add ADD)
{
    a.mv = ADD(a.mv,b.mv);
    a.l0 = ADD(a.l0,b.l0);
    a.l1 = ADD(a.l1,b.l1);
   return a;
}



//prepare AND -> send real value a&b to other P
template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Tetrad_Share a, Tetrad_Share b, Tetrad_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
DATATYPE y3ab = ADD( ADD(MULT(a.l0,b.l1),MULT(a.l1,b.l0)), MULT(a.l0,a.l0));
DATATYPE u1 = getRandomVal(P013);
DATATYPE u2 = getRandomVal(P023);
DATATYPE r = SUB(y3ab, ADD(u1,u2));
Tetrad_Share q;

//q:
c.mv = SET_ALL_ZERO();
c.l1 = getRandomVal(P013); //lambda2
c.l0 = SUB(SET_ALL_ZERO(),ADD(r,c.l1));  //lambda1
store_compare_view(P2, c.l0); // verify if P2 gets correct value from P3

DATATYPE v = ADD(u1,u2);

v = SUB( v, ADD( MULT( ADD(a.l0,a.l1), b.mv) , MULT ( ADD(b.l0,b.l1), a.mv)));
c.mv = v; //Trick, can be set to zero later on
}

template <typename func_add, typename func_sub>
void complete_mult(Tetrad_Share &c, func_add ADD, func_sub SUB)
{
    DATATYPE w = receive_from_live(P3);
    c.mv = ADD(c.mv,w);
    store_compare_view(P012,c.mv);
    c.mv = SET_ALL_ZERO(); //restore actual value of c.mv

    Tetrad_Share p; 
    p.l0 = SET_ALL_ZERO(); //lambda1
    p.l1 = SET_ALL_ZERO(); //lambda2
    p.mv = receive_from_live(P2);
    store_compare_view(P1,p.mv);
    

    //o = p + q
    c.mv = ADD(c.mv,p.mv);
    c.l0 = ADD(c.l0,p.l0);
    c.l1 = ADD(c.l1,p.l1);
}



void prepare_reveal_to_all(Tetrad_Share a)
{
#if PRE == 0
    send_to_live(P3, a.mv);
#endif
}    



template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Tetrad_Share a, func_add ADD, func_sub SUB)
{
    DATATYPE l3 = receive_from_live(P3);
    DATATYPE result = SUB(a.mv, l3);
    result = SUB(result, a.l0);
    result = SUB(result, a.l1);
    store_compare_view(P1, l3); //verify own value

    store_compare_view(P1, a.l1);  // verify others
    store_compare_view(P2, a.l0); 
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
    a[i].l0 = getRandomVal(P013); //l1
    a[i].l1 = getRandomVal(P023);
    DATATYPE l3 = SET_ALL_ZERO();
    a[i].mv = ADD( ADD(a[i].mv, l3), ADD(a[i].l0,a[i].l1));
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

template <typename func_add, typename func_sub>
void complete_receive_from(Tetrad_Share a[], int id, int l, func_add ADD, func_sub SUB)
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
