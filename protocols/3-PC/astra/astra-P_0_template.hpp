#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class ASTRA0_Share
{
private:
Datatype v;

public:

ASTRA0_Share()  {}
ASTRA0_Share(Datatype a) { v = a; }

ASTRA0_Share public_val(DATATYPE a)
{
    return ASTRA0_Share(a);
}

ASTRA0_Share Not() const
{
   return *this;
}

template <typename func_add>
ASTRA0_Share Add( ASTRA0_Share b, func_add ADD) const
{
   return ASTRA0_Share(ADD(v,b.v));
}



template <typename func_add, typename func_sub, typename func_mul>
    ASTRA0_Share prepare_mult(ASTRA0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
ASTRA0_Share c;
DATATYPE yxy = MULT(v,b.v);
c.v = ADD(getRandomVal(P_1),getRandomVal(P_2)); //yz
DATATYPE yxy2 = SUB(yxy,getRandomVal(P_1)); //yxy,2
#if PRE == 0
send_to_live(P_2, yxy2);
#else
pre_send_to_live(P_2, yxy2);
#endif
return c;
}

template <typename func_add, typename func_sub, typename func_mul>
ASTRA0_Share prepare_dot(const ASTRA0_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
ASTRA0_Share c;
c.v = MULT(v,b.v);
return c;
}


template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
Datatype yxy = v;
v = ADD(getRandomVal(P_1),getRandomVal(P_2)); //yz
DATATYPE yxy2 = SUB(yxy,getRandomVal(P_1)); //yxy,2
#if PRE == 0
send_to_live(P_2, yxy2);
#else
pre_send_to_live(P_2, yxy2);
#endif
}
template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}


void prepare_reveal_to_all()
{
#if PRE == 0
    send_to_live(P_1, v);
    send_to_live(P_2, v);
#else
    pre_send_to_live(P_1, v);
    pre_send_to_live(P_2, v);
#endif
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(func_add ADD, func_sub SUB)
{
#if PRE == 0 
return SUB(receive_from_live(P_2),v);
#elif PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share(v);
return SET_ALL_ZERO();
#endif
}




template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P_0)
{
#if OPT_SHARE == 1
    v = get_input_live(); 
    DATATYPE lx1 = getRandomVal(P_1);
#if PRE == 0
    send_to_live(P_2, ADD(v,lx1));
#else
    pre_send_to_live(P_2, ADD(v,lx1));
#endif

#else
    DATATYPE lv1 = getRandomVal(P_1); 
    DATATYPE lv2 = getRandomVal(P_2);
    v = ADD(lv1,lv2);// lv
    DATATYPE mv = ADD(v,get_input_live());
#if PRE == 0
    send_to_live(P_1, mv);
    send_to_live(P_2, mv);
#else
    pre_send_to_live(P_1, mv);
    pre_send_to_live(P_2, mv);
#endif
#endif
}
else if constexpr(id == P_1){
    v = getRandomVal(P_1);


}
else if constexpr(id == P_2)// id ==2
{
        v = getRandomVal(P_2);

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
#if PRE == 0
    communicate_live();
#endif
}

};
