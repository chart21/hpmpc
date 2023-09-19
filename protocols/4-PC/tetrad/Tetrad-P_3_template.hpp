#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Tetrad3_Share
{
private:
    DATATYPE l1;
    DATATYPE l2;
    DATATYPE l3;
    DATATYPE storage; // used for storing results needed later
public:

Tetrad3_Share()  {}

Tetrad3_Share(DATATYPE a, DATATYPE b, DATATYPE c) 
{
    l1 = a;
    l2 = b;
    l3 = c;
}

Tetrad3_Share public_val(DATATYPE a)
{
    return Tetrad3_Share(SET_ALL_ZERO(),SET_ALL_ZERO(),SET_ALL_ZERO());
}

Tetrad3_Share Not() const
{
    return *this;
}

template <typename func_add>
Tetrad3_Share Add(Tetrad3_Share b, func_add ADD) const
{
    return Tetrad3_Share(ADD(l1,b.l1),ADD(l2,b.l2),ADD(l3,b.l3));
}



template <typename func_add, typename func_sub, typename func_mul>
    Tetrad3_Share prepare_mult(Tetrad3_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Tetrad3_Share c;
DATATYPE y1ab = ADD( ADD(AND(l1,b.l3),AND(l3,b.l1)), AND(l3,l3));
DATATYPE y2ab = ADD( ADD(AND(l2,b.l3),AND(l3,b.l2)), AND(l2,l2));
DATATYPE y3ab = ADD( ADD(AND(l1,b.l2),AND(l2,b.l1)), AND(l1,l1));
DATATYPE u1 = getRandomVal(P_013);
DATATYPE u2 = getRandomVal(P_023);
DATATYPE r = SUB(y3ab, ADD(u1,u2));
Tetrad3_Share q;

DATATYPE s = getRandomVal(P_123);
DATATYPE w = ADD(s, ADD(y1ab,y2ab));
send_to_live(P_0, w);
//q:
c.l2 = getRandomVal(P_013); //lambda2
c.l1 = SUB(SET_ALL_ZERO(), ADD(r,c.l2));  //lambda1 
c.l3 = SET_ALL_ZERO(); //lambda3
send_to_live(P_2, c.l2);
return c;
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    Tetrad3_Share p; 
    p.l1 = SET_ALL_ZERO(); //lambda1
    p.l2 = SET_ALL_ZERO(); //lambda2
    p.l3 = getRandomVal(P_123); //lambda3

    //o = p + q
    l1 = ADD(l1,p.l1);
    l2 = ADD(l2,p.l2);
    l3 = ADD(l3,p.l3);
}



void prepare_reveal_to_all()
{
    send_to_live(P_2, l1);
    send_to_live(P_1, l2);
    send_to_live(P_0, l3);
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{

//receive lambda3 from P_3
DATATYPE mv = receive_from_live(P_0);
DATATYPE result = SUB(mv, l3);
result = SUB(result, l1);
result = SUB(result, l2);
store_compare_view(P_1, mv); //verify own value

return result;
}



template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
    DATATYPE mv = get_input_live();
    l1 = getRandomVal(P_013); //l1
    l2 = getRandomVal(P_023);
    l3 = getRandomVal(P_123);
    mv = ADD( ADD(mv, l3), ADD(l1,l2));
    send_to_live(P_0, mv);
    send_to_live(P_1, mv);
    send_to_live(P_2, mv);
}
else if constexpr(id == P_1)
{
    l1 = getRandomVal(P_013);
    l2 = SET_ALL_ZERO();
    l3 = getRandomVal(P_123);
}
else if constexpr(id == P_2)
{
    l1 = SET_ALL_ZERO();
    l2 = getRandomVal(P_023);
    l3 = getRandomVal(P_123);
}
else if constexpr(id == P_0)
{
    l1 = getRandomVal(P_013);
    l2 = getRandomVal(P_023); 
    l3 = SET_ALL_ZERO();
}
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
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
