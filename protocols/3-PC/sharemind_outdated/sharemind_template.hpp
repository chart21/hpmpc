#pragma once
#include "../../generic_share.hpp"

template <typename Datatype>
class Sharemind_Share
{
private:
Datatype val;
Datatype helper;
public:
Sharemind_Share(){}
Sharemind_Share(Datatype v, Datatype helper): val(v), helper(helper){}
Sharemind_Share(Datatype v): val(v){};

template <typename func_add>
Sharemind_Share Add( Sharemind_Share b, func_add ADD) const
{
    return Sharemind_Share(ADD(val,b.val),ADD(helper,b.helper));
}


Sharemind_Share public_val(Datatype a)
{
#if PARTY == 0
    return Sharemind_Share(a,SET_ALL_ZERO());
#else
    return Sharemind_Share(SET_ALL_ZERO(),SET_ALL_ZERO());
#endif
}

Sharemind_Share Not() const
{
    return Sharemind_Share(NOT(val),helper);
}


    template <typename func_add, typename func_sub>
Datatype reshare(Datatype a, func_add ADD, func_sub SUB) const
{
Datatype u[3];
u[pprev] = getRandomVal(pprev);
u[pnext] = getRandomVal(pnext);
u[2] = SUB(u[pprev],u[pnext]);
u[2] = ADD(a,u[2]);
return u[2];
}

template <typename func_add, typename func_sub, typename func_mul>
    Sharemind_Share prepare_mult(Sharemind_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype u = reshare(val, ADD, SUB);
Datatype v = reshare(b.val, ADD, SUB);
send_to_live(pnext, u);
send_to_live(pprev, v);
Sharemind_Share c;
c.val = MULT(u,v);
c.helper = v;
return c;
}

    template <typename func_add, typename func_sub, typename func_mul>
void complete_mult(func_add ADD, func_sub SUB, func_mul MULT)
{

Datatype u_p = receive_from_live(pprev);
Datatype v_n = receive_from_live(pnext);
Datatype v_i = helper;
val = ADD (val,   ADD ( MULT(u_p,v_i) , MULT(u_p,v_n) ));
}


void prepare_reveal_to_all()
{
    for(int t = 0; t < num_players-1; t++) 
        send_to_live(t, val);
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
Datatype result = val;
for(int t = 0; t < num_players-1; t++) 
    result = ADD(result,receive_from_live(t));
return result;
}


template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
        val = SUB( get_input_live(), ADD(getRandomVal(PPREV),getRandomVal(PNEXT)) );
    else
        val = getRandomVal(id);
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
    communicate_live();
}

};
