#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Tetrad2_Share
{

private:
    DATATYPE mv;
    DATATYPE l0;
    DATATYPE l1;
    DATATYPE storage; // used for storing results needed later
public:

Tetrad2_Share()  {}

Tetrad2_Share(DATATYPE a, DATATYPE b, DATATYPE c) 
{
    mv = a;
    l0 = b;
    l1 = c;
}

Tetrad2_Share public_val(DATATYPE a)
{
    return Tetrad2_Share(a,SET_ALL_ZERO(),SET_ALL_ZERO());
}

Tetrad2_Share Not() const
{
   return Tetrad2_Share(NOT(mv),l0,l1);
}

template <typename func_add>
Tetrad2_Share Add(Tetrad2_Share b, func_add ADD) const
{
    return Tetrad2_Share(ADD(mv,b.mv),ADD(l0,b.l0),ADD(l1,b.l1));
}



template <typename func_add, typename func_sub, typename func_mul>
    Tetrad2_Share prepare_mult(Tetrad2_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
Tetrad2_Share c;
DATATYPE y2ab = ADD( ADD(MULT(l0,b.l1),MULT(l1,b.l0)), MULT(l0,l0));
DATATYPE u2 = getRandomVal(P_023);

//q:
c.mv = SET_ALL_ZERO();
c.l0 = receive_from_live(P_3);
store_compare_view(P_0, c.l0);
c.l1 = SET_ALL_ZERO();  //lambda3

DATATYPE s = getRandomVal(P_123);

DATATYPE y1 = SUB( XOR(y2ab,u2), ADD(MULT(l0,b.mv),MULT(mv,b.l0)) );
DATATYPE y3 = SUB(SET_ALL_ZERO(), ADD(MULT(l1,b.mv),MULT(mv,b.l1)));
send_to_live(P_1, y1);
DATATYPE z_r = ADD( ADD(y1, y3), MULT(mv,b.mv));

//Trick to store values neede later
c.storage = c.l0;
c.l0 = y1;
c.l1 = s;
c.mv = z_r;
return c;
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    DATATYPE y2 = receive_from_live(P_1);
    DATATYPE v = ADD(ADD(l0,l1),y2); // y1 + y2 + s for verification
    store_compare_view(P_012, v);
    mv = ADD(mv, y2);

    //p:
    DATATYPE pl1 = SET_ALL_ZERO(); // known by all
    DATATYPE pl2 = SET_ALL_ZERO(); // known by all
    DATATYPE pl3 = getRandomVal(P_123); //hide from P_0
    DATATYPE pmv = ADD(mv,pl3);                                   
    send_to_live(P_0, pmv);



    //o = p + q
    mv = pmv;
    l0 = storage; //lambda2
    l1 = pl3; //lambda3
}


void prepare_reveal_to_all()
{
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
                              
DATATYPE lambda1 = receive_from_live(P_3);
store_compare_view(P_0, lambda1); //get help from P_0 to veriy
DATATYPE result = SUB(mv, lambda1);
result = SUB(result, l0);
result = SUB(result, l1);
return result;
}



template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    mv = get_input_live();
    l0 = getRandomVal(P_023); //l2
    l1 = getRandomVal(P_123); //l3
    DATATYPE l2 = SET_ALL_ZERO();
    mv = ADD( ADD(mv, l2), ADD(l0,l1));
    send_to_live(P_0, mv);
    send_to_live(P_1, mv);
}
else if constexpr(id == P_0)
{
    l0 = getRandomVal(P_023);
    l1 = SET_ALL_ZERO();
}
else if constexpr(id == P_1)
{
    l0 = SET_ALL_ZERO();
    l1 = getRandomVal(P_123);
}
else if constexpr(id == P_3)
{
    l0 = getRandomVal(P_023);
    l1 = getRandomVal(P_123);
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id != PSELF)
{
    mv = receive_from_live(id);

        if(id != P_0)
                store_compare_view(P_0,mv);
        if(id != P_1)
                store_compare_view(P_1,mv);


    
}
}




static void send()
{
    send_live();
}

static void receive()
{
    receive_live();
}

static void communicate()
{
/* #if PRE == 0 */
    communicate_live();
/* #endif */
}

};
