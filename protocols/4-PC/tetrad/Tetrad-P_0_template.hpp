#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE Tetrad0_Share
template <typename Datatype>
class Tetrad0_Share
{
private:
    DATATYPE mv;
    DATATYPE l0;
    DATATYPE l1;
    DATATYPE storage; // used for storing results needed later
public:

Tetrad0_Share()  {}

Tetrad0_Share(DATATYPE a, DATATYPE b, DATATYPE c) 
{
    mv = a;
    l0 = b;
    l1 = c;
}

Tetrad0_Share public_val(DATATYPE a)
{
    return Tetrad0_Share(a,SET_ALL_ZERO(),SET_ALL_ZERO());
}

Tetrad0_Share Not() const
{
   return Tetrad0_Share(NOT(mv),l0,l1);
}

template <typename func_add>
Tetrad0_Share Add(Tetrad0_Share b, func_add ADD) const
{
    return Tetrad0_Share(ADD(mv,b.mv),ADD(l0,b.l0),ADD(l1,b.l1));
}



template <typename func_add, typename func_sub, typename func_mul>
    Tetrad0_Share prepare_mult(Tetrad0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Tetrad0_Share c;
DATATYPE y3ab = ADD( ADD(MULT(l0,b.l1),MULT(l1,b.l0)), MULT(l0,l0));
DATATYPE u1 = getRandomVal(P_013);
DATATYPE u2 = getRandomVal(P_023);
DATATYPE r = SUB(y3ab, ADD(u1,u2));
Tetrad0_Share q;

//q:
c.mv = SET_ALL_ZERO();
c.l1 = getRandomVal(P_013); //lambda2
c.l0 = SUB(SET_ALL_ZERO(),ADD(r,c.l1));  //lambda1
store_compare_view(P_2, c.l0); // verify if P_2 gets correct value from P_3

DATATYPE v = ADD(u1,u2);

v = SUB( v, ADD( MULT( ADD(l0,l1), b.mv) , MULT ( ADD(b.l0,b.l1), mv)));
c.mv = v; //Trick, can be set to zero later on
return c;
}


template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    DATATYPE w = receive_from_live(P_3);
    mv = ADD(mv,w);
    store_compare_view(P_012,mv);
    mv = SET_ALL_ZERO(); //restore actual value of c.mv

    Tetrad0_Share p; 
    p.l0 = SET_ALL_ZERO(); //lambda1
    p.l1 = SET_ALL_ZERO(); //lambda2
    p.mv = receive_from_live(P_2);
    store_compare_view(P_1,p.mv);
    

    //o = p + q
    mv = ADD(mv,p.mv);
    l0 = ADD(l0,p.l0);
    l1 = ADD(l1,p.l1);
}



void prepare_reveal_to_all()
{
#if PRE == 0
    send_to_live(P_3, mv);
#endif
}    



template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
    DATATYPE l3 = receive_from_live(P_3);
    DATATYPE result = SUB(mv, l3);
    result = SUB(result, l0);
    result = SUB(result, l1);
    store_compare_view(P_1, l3); //verify own value

    store_compare_view(P_1, l1);  // verify others
    store_compare_view(P_2, l0); 
    return result;
}

template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    mv = get_input_live();
    l0 = getRandomVal(P_013); //l1
    l1 = getRandomVal(P_023);
    DATATYPE l3 = SET_ALL_ZERO();
    mv = ADD( ADD(mv, l3), ADD(l0,l1));
    send_to_live(P_1, mv);
    send_to_live(P_2, mv);
}
else if constexpr(id == P_1)
{
    l0 = getRandomVal(P_013);
    l1 = SET_ALL_ZERO();
}
else if constexpr(id == P_2)
{
    l0 = SET_ALL_ZERO();
    l1 = getRandomVal(P_023);
}
else if constexpr(id == P_3)
{
    l0 = getRandomVal(P_013);
    l1 = getRandomVal(P_023);
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id != PSELF)
{
    mv = receive_from_live(id);

        if(id != P_1)
                store_compare_view(P_1,mv);
        if(id != P_2)
                store_compare_view(P_2,mv);
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
