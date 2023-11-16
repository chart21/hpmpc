#pragma once
#include "../generic_share.hpp"
template <typename Datatype>
class TTP_init
{
public:
TTP_init() {}
TTP_init(Datatype a) {}

Datatype get_p1()
{
    return SET_ALL_ZERO();
}

    template <typename func_mul, typename func_trunc>
TTP_init mult_public_fixed(const DATATYPE b, func_mul MULT, func_trunc TRUNC) const
{
   return TTP_init();
}



TTP_init public_val(Datatype a)
{
    return TTP_init();
}

TTP_init Not() const
{
    return TTP_init();
}

template <typename func_add>
TTP_init Add(TTP_init b, func_add ADD) const
{
    return TTP_init();
}
    
    template <typename func_add, typename func_sub, typename func_mul>
TTP_init prepare_dot(const TTP_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return TTP_init();
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
}
    
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}

    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}



template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_mult(TTP_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return TTP_init();
}
template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}


void prepare_reveal_to_all() const
{
#if PARTY == 2 && PROTOCOL != 13
        for(int t = 0; t < num_players-1; t++) 
        {
            send_to_(t);
        }//add to send buffer
#endif 
}    

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
#if PARTY != 2 && PROTOCOL != 13
    receive_from_(P_2);
#endif
return SET_ALL_ZERO();
}

template <int id,typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE val, func_add ADD, func_sub SUB)
{
#if PROTOCOL != 13 && PARTY != 2
if constexpr(id == PSELF)
{
        send_to_(P_2);
}
#endif
}



template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    prepare_receive_from<id>(SET_ALL_ZERO(), ADD, SUB);
}

    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
#if PARTY == 2 && PROTOCOL != 13
if constexpr(id == PSELF)
    return;
receive_from_(id);
#endif
}

static void send()
{
#if PROTOCOL != 13
send_();
#endif
}
static void receive()
{
#if PROTOCOL != 13
    receive_();
#endif
}
static void communicate()
{
#if PROTOCOL != 13
communicate_();
#endif
}

static void finalize(std::string* ips)
{
#if PROTOCOL != 13
    finalize_(ips);
#endif
}

static void prepare_A2B_S1(int k, TTP_init in[], TTP_init out[])
{
}


static void prepare_A2B_S2(int k, TTP_init in[], TTP_init out[])
{
}

static void complete_A2B_S1(int k, TTP_init out[])
{

}
static void complete_A2B_S2(int k, TTP_init out[])
{

}

void prepare_bit_injection_S1(TTP_init out[])
{
}

void prepare_bit_injection_S2(TTP_init out[])
{
}

static void complete_bit_injection_S1(TTP_init out[])
{
    
}

static void complete_bit_injection_S2(TTP_init out[])
{


}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_dot3(TTP_init b, TTP_init c, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_init();
}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_mult3(TTP_init b, TTP_init c, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_init();
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_dot4(TTP_init b, TTP_init c, TTP_init d, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_init();
}

template <typename func_add, typename func_sub, typename func_mul>
    TTP_init prepare_mult4(TTP_init b, TTP_init c, TTP_init d, func_add ADD, func_sub SUB, func_mul MULT) const
{
return TTP_init();
}


template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){}


TTP_init relu() const
{
    return TTP_init();
}

};
