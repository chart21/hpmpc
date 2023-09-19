#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL3_init
{
public:
OEC_MAL3_init() {}



OEC_MAL3_init public_val(Datatype a)
{
    return OEC_MAL3_init();
}

OEC_MAL3_init Not() const
{
    return OEC_MAL3_init();
}

template <typename func_add>
OEC_MAL3_init Add(OEC_MAL3_init b, func_add ADD) const
{
   return OEC_MAL3_init();
}
    
template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL3_init prepare_dot(const OEC_MAL3_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OEC_MAL3_init();
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_init prepare_mult(OEC_MAL3_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PROTOCOL == 12 || PROTOCOL == 8
#if PRE == 1
    pre_send_to_(P_2);
#else
    send_to_(P_2);
#endif
#else
store_compare_view_init(P_2);
#endif
#if PROTOCOL == 10 || PROTOCOL == 12 || PROTOCOL == 8
#if PRE == 1
    pre_send_to_(P_0);
#else
    send_to_(P_0);
#endif
#elif PROTOCOL == 11
store_compare_view_init(P_0);
#endif
return OEC_MAL3_init();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
}


void prepare_reveal_to_all()
{
    #if PROTOCOL == 8
    for(int t = 0; t < 3; t++) 
    {
        #if PRE == 1 
    pre_send_to_(t);
    #else
    send_to_(t);
    #endif

    }
 #else
#if PRE == 1
    pre_send_to_(P_0);
#else
    send_to_(P_0);
#endif
#endif
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
receive_from_(P_0);
#if PROTOCOL == 8
store_compare_view_init(P_1);
#else
store_compare_view_init(P_123);
store_compare_view_init(P_0123);
#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share_();
store_output_share_();
#endif
Datatype dummy;
return dummy;
}


template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
/* return; */
/* old: */

if constexpr(id == PSELF)
{
        #if PRE == 1
        pre_send_to_(P_0);
        pre_send_to_(P_1);
        pre_send_to_(P_2);
        #else
        send_to_(P_0);
        send_to_(P_1);
        send_to_(P_2);
        #endif
}
}

    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
}





static void send()
{
send_();
}

// P_0 only has 1 receive round
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


};
