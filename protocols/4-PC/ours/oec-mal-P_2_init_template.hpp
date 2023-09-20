#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL2_init
{
public:
OEC_MAL2_init() {}



OEC_MAL2_init public_val(Datatype a)
{
    return OEC_MAL2_init();
}

OEC_MAL2_init Not() const
{
    return OEC_MAL2_init();
}

template <typename func_add>
OEC_MAL2_init Add(OEC_MAL2_init b, func_add ADD) const
{
   return OEC_MAL2_init();
}
    
template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL2_init prepare_dot(const OEC_MAL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OEC_MAL2_init();
}

    template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
#if PROTOCOL == 12 || PROTOCOL == 8
store_compare_view_init(P_0);
#if PRE == 1
pre_receive_from_(P_3);
#else
receive_from_(P_3);
#endif
#else
store_compare_view_init(P_3);
#if PRE == 1
pre_receive_from_(P_0);
#else
receive_from_(P_0);
#endif
#endif
send_to_(P_1);
#if PROTOCOL == 10 || PROTOCOL == 12
send_to_(P_0);
#endif
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL2_init prepare_mult(OEC_MAL2_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PROTOCOL == 12 || PROTOCOL == 8
store_compare_view_init(P_0);
#if PRE == 1
pre_receive_from_(P_3);
#else
receive_from_(P_3);
#endif
#else
store_compare_view_init(P_3);
#if PRE == 1
pre_receive_from_(P_0);
#else
receive_from_(P_0);
#endif
#endif
send_to_(P_1);
#if PROTOCOL == 10 || PROTOCOL == 12
send_to_(P_0);
#endif

//return u[player_id] * v[player_id];
return OEC_MAL2_init();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    receive_from_(P_1);
#if PROTOCOL == 11
send_to_(P_0); // let P_0 obtain ab
send_to_(P_0); // let P_0 verify m_2 XOR m_3
#endif

#if PROTOCOL == 10 || PROTOCOL == 12 || PROTOCOL == 8
store_compare_view_init(P_012);
#endif

#if PROTOCOL == 8
 send_to_(P_0); // let P_0 obtain ab
#endif
}

void prepare_reveal_to_all()
{
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
receive_from_(P_0);
#if PROTOCOL == 8
    #if PRE == 1 
    pre_receive_from_(P_3);
#else
    receive_from_(P_3);
#endif

    store_compare_view_init(P_0);
#else
    store_compare_view_init(P_123);
    store_compare_view_init(P_0123);
#endif
Datatype dummy;
return dummy;
}



template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id == PSELF)
{
        send_to_(P_0);
        send_to_(P_1);
}
}

    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id != PSELF)
    {
#if PRE == 1
            if constexpr(id == P_3)
                pre_receive_from_(P_3);
            else
                receive_from_(id);
#else
            receive_from_(id);
#endif
        if constexpr(id != P_0)
                store_compare_view_init(P_0);
        if constexpr(id != P_1)
                store_compare_view_init(P_1);
    }
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


};
