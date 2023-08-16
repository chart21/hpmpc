#pragma once
#include "../sharemind/sharemind_base.hpp"
template <typename Datatype>
class OECL1_init
{
public:
OECL1_init() {}

OECL1_init public_val(Datatype a)
{
    return OECL1_init();
}

OECL1_init Not() const
{
    return OECL1_init();
}

template <typename func_add>
OECL1_init Add(OECL1_init b, func_add ADD) const
{
   return OECL1_init();
}



template <typename func_add, typename func_sub, typename func_mul>
    OECL1_init prepare_mult(OECL1_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
send_to_(P2);
return OECL1_init();

//return u[player_id] * v[player_id];
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    receive_from_(P2);
}


void prepare_reveal_to_all()
{
return;
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
    pre_receive_from_(P0);
#else
    receive_from_(P0);
#endif
    return SET_ALL_ZERO();
}

template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P1)
{
        send_to_(P2);

}
}


    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P2)
{
    receive_from_(P2);
}
#if OPT_SHARE == 0
else if constexpr(id == P0)
{
    #if PRE == 1 && SHARE_PREP == 1
    pre_receive_from_(P0);
    #else
    receive_from_(P0);
    #endif
}
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

static void finalize(std::string* ips, receiver_args* ra, sender_args* sa)
{
    finalize_(ips, ra, sa);
}

static void prepare_A2B_S1(OECL1_init in[], OECL1_init out[])
{
}


static void prepare_A2B_S2(OECL1_init in[], OECL1_init out[])
{
}

static void complete_A2B_S1(OECL1_init out[])
{
}
static void complete_A2B_S2(OECL1_init out[])
{

}

void prepare_bit_injection_S1( OECL1_init out[])
{
}

void prepare_bit_injection_S2( OECL1_init out[])
{
}

static void complete_bit_injection_S1(OECL1_init out[])
{
    
}

static void complete_bit_injection_S2(OECL1_init out[])
{


}


};
