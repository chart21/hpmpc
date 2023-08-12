#pragma once
#include "oecl_base.hpp"
#include <cstdint>
template <typename Datatype>
class OECL0_init 
{
public:
static constexpr int VALS_PER_SHARE = 1;
OECL0_init() {}



OECL0_init public_val(Datatype a)
{
    return OECL0_init();
}

OECL0_init Not() const
{
    return OECL0_init();
}

template <typename func_add>
OECL0_init Add(OECL0_init b, func_add ADD) const
{
   return OECL0_init();
}

template <typename func_add, typename func_sub, typename func_mul>
    OECL0_init prepare_mult(OECL0_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PRE == 1
    pre_send_to_(P2);
#else
send_to_(P2);
#endif
return OECL0_init();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB){}




void prepare_reveal_to_all()
    {
    for(int t = 0; t < 2; t++) 
    {
        #if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
    pre_send_to_(t);
#else
    send_to_(t);
#endif

    }//add to send buffer
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
    {
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
receive_from_(P2);
#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share_();
#endif
return SET_ALL_ZERO();
}


template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
    {

/* return; */
/* old: */

if constexpr(id == P0)
{
#if OPT_SHARE == 1
        #if PRE == 1 && SHARE_PREP == 1
            pre_send_to_(P2);
        #else
            send_to_(P2);
        #endif

#else
        #if PRE == 1
        pre_send_to_(P1);
        pre_send_to_(P2);
        #else
        send_to_(P1);
        send_to_(P2);
        #endif

#endif
}
else
{}
} 

    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
    {
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

static void prepare_A2B_S1(OECL0_init in[], OECL0_init out[])
{
}


static void prepare_A2B_S2(OECL0_init in[], OECL0_init out[])
{
}

static void complete_A2B_S1(OECL0_init out[])
{

}
static void complete_A2B_S2(OECL0_init out[])
{

}


};
