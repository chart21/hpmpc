#pragma once
#include "../../beaver_triples.hpp"
template <typename Datatype>
class Add_Share{
using BT = triple<Datatype>;
private:
Datatype v;
    public:
Add_Share()  {}
Add_Share(Datatype x) { this->v = x; }



template <typename func_mul>
Add_Share mult_public(const Datatype b, func_mul MULT) const
{
    return Add_Share(MULT(v,b));
}


// P_i shares mx - lxi, P_j sets lxj to 0
template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
    {
        v = val;
    }
    else
    {
        v = SET_ALL_ZERO();
    }
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
}

template <typename func_add>
Add_Share Add( Add_Share b, func_add ADD) const
{
    return Add_Share(ADD(v,b.v));
}

void prepare_reveal_to_all() const
{
    send_to_live(PNEXT, v);
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
    return ADD(v,receive_from_live(PNEXT));
}

template <typename func_add, typename func_sub, typename func_mul>
    Add_Share prepare_mult(Add_Share y, func_add ADD, func_sub SUB, func_mul MULT) const
{
BT t;
if constexpr(std::is_same_v<func_add, FUNC_XOR>)
{
    t = retrieveBooleanTriple<Datatype>();
}
else
{
    t = retrieveArithmeticTriple<Datatype>();
}
Datatype xpa = ADD(v,t.a);
Datatype ypb = ADD(y.v,t.b);
send_to_live(PNEXT, xpa);
send_to_live(PNEXT, ypb);
store_output_share(y.v);
store_output_share(t.a);
return Add_Share(ADD(SUB( MULT(xpa,y.v), MULT(ypb,t.a)), t.c));
}
    
    template <typename func_add, typename func_sub, typename func_mul>
void complete_mult(func_add ADD, func_sub SUB, func_mul MULT)
{
    Datatype xpa = receive_from_live(PNEXT);
    Datatype ypb = receive_from_live(PNEXT);
    Datatype yv = retrieve_output_share();
    Datatype ta = retrieve_output_share();
    v = ADD(v, SUB(MULT(xpa,yv), MULT(ypb,ta)));
}



Add_Share public_val(Datatype a)
{
#if PARTY == 0
    return Add_Share(a);
#else
    return Add_Share(SET_ALL_ZERO());
#endif
}

Add_Share Not() const
{
#if PARTY == 0
    return Add_Share(NOT(v));
#else
    return Add_Share(v);
#endif
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

