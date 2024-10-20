#pragma once
#include "../../beaver_tiples.hpp"
template <typename Datatype>
class ABY2_ONLINE_Share{
using BT = triple<Datatype>;
private:
Datatype m;
Datatype l;
    public:
ABY2_ONLINE_Share()  {}
ABY2_ONLINE_Share(Datatype x) { this->m = x; }
ABY2_ONLINE_Share(Datatype x, Datatype l) { this->m = x; this->l = l; }



template <typename func_mul>
ABY2_ONLINE_Share mult_public(const Datatype b, func_mul MULT) const
{
    return ABY2_ONLINE_Share(MULT(m,b),MULT(l,b));
}


// P_i shares mx - lxi, P_j sets lxj to 0
template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
    {
        l = getRandomVal(PSELF);
        m = ADD(val,l);
        send_to_live(PNEXT, m);
    }
    else
    {
        l = SET_ALL_ZERO();
    }
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id != PSELF)
        m = receive_from_live(id);
}

template <typename func_add>
ABY2_ONLINE_Share Add( ABY2_ONLINE_Share b, func_add ADD) const
{
    return ABY2_ONLINE_Share(ADD(m,b.m),ADD(l,b.l));
}

void prepare_reveal_to_all() const
{
    send_to_live(PNEXT, l);
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
    return SUB(m,ADD(receive_from_live(PNEXT),l));
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_mult(ABY2_ONLINE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype lxly = retreive_shares();
ABY2_ONLINE_Share c;
c.l = getRandomVal(PSELF);
#if PARTY == 0
c.m = ADD(msg, MULT(m,b.m));
#else
c.m = SET_ALL_ZERO();
#endif
c.m = SUB(c.m, ADD(ADD(MULT(m,b.l),MULT(l,b.m),SUB(lxly,c.l)))); // mx my - (mx[ly] + my[lx] + [lxly] - [lz])
return c;
}
    
    template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    Datatype msg = receive_from_live(PPREV);
    m = ADD(m,msg);
}



ABY2_ONLINE_Share public_val(Datatype a)
{
    return ABY2_ONLINE_Share(a,SET_ALL_ZERO());
}

ABY2_ONLINE_Share Not() const
{
    return ABY2_ONLINE_Share(NOT(m),l);
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

