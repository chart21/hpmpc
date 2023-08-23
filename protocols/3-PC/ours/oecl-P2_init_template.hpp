#pragma once
#include "../sharemind/sharemind_base.hpp"
template <typename Datatype>
class OECL2_init
{
public:
OECL2_init() {}



OECL2_init public_val(Datatype a)
{
    return a;
}

OECL2_init Not() const
{
    return OECL2_init();
}

template <typename func_add>
OECL2_init Add(OECL2_init b, func_add ADD) const
{
   return OECL2_init();
}
       template <typename func_add, typename func_sub, typename func_mul>
void prepare_dot_add(OECL2_init a, OECL2_init b , OECL2_init &c, func_add ADD, func_sub SUB, func_mul MULT)
{
} 
    template <typename func_add, typename func_sub, typename func_mul>
OECL2_init prepare_dot(const OECL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OECL2_init();
}

template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
#if PRE == 1
    pre_receive_from_(P0);
#else
    receive_from_(P0);
#endif
    send_to_(P1);
}




template <typename func_add, typename func_sub, typename func_mul>
    OECL2_init prepare_mult(OECL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PRE == 1
pre_receive_from_(P0);
#else
receive_from_(P0);
#endif

send_to_(P1);
//return u[player_id] * v[player_id];
return OECL2_init();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    receive_from_(P1);
}

void prepare_reveal_to_all()
{
send_to_(P0);
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
if constexpr(id == P2)
{
        send_to_(P1);

}
}


    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == P1)
{
    receive_from_(P1);
}
else if constexpr(id == P0)
{
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
    pre_receive_from_(P0);
#else
    receive_from_(P0);
#endif
} 
/* if(id == player_id) */
/*     return; */
/* int offset = {id > player_id ? 1 : 0}; */
/* int player = id - offset; */
/* for(int i = 0; i < l; i++) */
/*     receiving_args[player].elements_to_rec[receiving_args[player].rec_rounds -1] += 1; */
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

static void prepare_A2B_S1(OECL2_init in[], OECL2_init out[])
{
}


static void prepare_A2B_S2(OECL2_init in[], OECL2_init out[])
{
}

static void complete_A2B_S1(OECL2_init out[])
{

}
static void complete_A2B_S2(OECL2_init out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        #if PRE == 1
        pre_receive_from_(P0);
        #else
        receive_from_(P0);
        #endif
    }
}

void prepare_bit_injection_S1( OECL2_init out[])
{
}

void prepare_bit_injection_S2( OECL2_init out[])
{
}

static void complete_bit_injection_S1(OECL2_init out[])
{
    
}

static void complete_bit_injection_S2(OECL2_init out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        #if PRE == 1
        pre_receive_from_(P0);
        #else
        receive_from_(P0);
        #endif
    }

}



};
