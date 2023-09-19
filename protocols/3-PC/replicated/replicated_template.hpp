#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Replicated_Share{
private:
Datatype x;
Datatype a;
    public:
Replicated_Share()  {}
Replicated_Share(Datatype x) { this->x = x; }
Replicated_Share(Datatype x, Datatype a) { this->x = x; this->a = a; }

Replicated_Share share_SRNG(Datatype a)
{
Replicated_Share s[3];
s[pprev].x = getRandomVal(pprev);
s[pnext].x = getRandomVal(pnext);
s[2].x =XOR(s[pprev].x,s[pnext].x);

s[pprev].a = XOR(s[pnext].x,a); //xi + x(i-1) + a
s[pnext].a = XOR(s[2].x,a); //xi + x(i-1) + a
s[2].a = XOR(s[pprev].x,a); //xi + x(i-1) + a

send_to_live(pprev, s[pprev].a);
send_to_live(pnext, s[pnext].a);

return s[2];
}






template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
    {
        *this = share_SRNG(get_input_live());
    }
    else
        x = getRandomVal(id);
}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id != PSELF)
    a = receive_from_live(id);
}

template <typename func_add>
Replicated_Share Add( Replicated_Share b, func_add ADD) const
{
    return Replicated_Share(ADD(x,b.x),ADD(a,b.a));
}


Replicated_Share public_val(Datatype a)
{
    return Replicated_Share(SET_ALL_ZERO(),a);
}

Replicated_Share Not() const
{
    return Replicated_Share(x,NOT(a));
}


void reshare(Datatype a, Datatype u[])
{
u[pprev] = getRandomVal(pprev);
u[pnext] = getRandomVal(pnext);
u[2] = XOR(u[pprev],u[pnext]);
u[2] = XOR(a,u[2]);
}

template <typename func_add, typename func_sub, typename func_mul>
    Replicated_Share prepare_mult(Replicated_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Replicated_Share c;
Datatype corr = XOR( getRandomVal(pprev), getRandomVal(pnext) );
Datatype r =  XOR( XOR(  AND(x,b.x), AND(a,b.a) ) , corr);  
c.a = r; //used to access value in complete and 
send_to_live(pnext, r);
return c;
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
x = XOR(a, receive_from_live(pprev));
}

void prepare_reveal_to_all()
{
    send_to_live(pnext, x);
}    


/* void prepare_reveal_to(Datatype a, int id) */
/* { */
/*     if(PSELF != id) */
/*     { */
/*         send_to_live(id, a); */
/* } */
/* } */

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
    Datatype result;
    result = XOR(a, receive_from_live(pprev));
    return result;
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
