#pragma once
#include "../generic_share.hpp"
template <typename Datatype>
class TTP_init
{
public:
TTP_init() {}




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


void prepare_reveal_to_all()
{
#if PARTY == 2
        for(int t = 0; t < num_players-1; t++) 
        {
            send_to_(t);
        }//add to send buffer
#endif 
}    

template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
#if PARTY != 2
    receive_from_(P2);
#endif
return SET_ALL_ZERO();
}



template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF && PARTY != 2)
{
        send_to_(P2);
}
}

    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
{
    return;
}
#if PARTY == 2
receive_from_(id);
#endif
}

static void send()
{
send_();
}
static void receive()
{
    receive_();
}
static void communicate()
{
communicate_();
}

static void finalize(std::string* ips)
{
    finalize_(ips);
}

static void prepare_A2B_S1(TTP_init in[], TTP_init out[])
{
}


static void prepare_A2B_S2(TTP_init in[], TTP_init out[])
{
}

static void complete_A2B_S1(TTP_init out[])
{

}
static void complete_A2B_S2(TTP_init out[])
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


};
