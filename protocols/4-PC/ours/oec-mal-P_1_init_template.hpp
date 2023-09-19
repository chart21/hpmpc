#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL1_init
{
public:
OEC_MAL1_init() {}



OEC_MAL1_init public_val(Datatype a)
{
    return OEC_MAL1_init();
}

OEC_MAL1_init Not() const
{
    return OEC_MAL1_init();
}

template <typename func_add>
OEC_MAL1_init Add(OEC_MAL1_init b, func_add ADD) const
{
   return OEC_MAL1_init();
}
    
template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL1_init prepare_dot(const OEC_MAL1_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OEC_MAL1_init();
}


template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_init prepare_mult(OEC_MAL1_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
send_to_(P_2);
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view_init(P_0); // compare a1b1 + r123_2 with P_0
#endif
return OEC_MAL1_init();
//return u[player_id] * v[player_id];
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    receive_from_(P_2);
#if PROTOCOL == 10 || PROTOCOL == 12 || PROTOCOL == 8
store_compare_view_init(P_012);
#endif
#if PROTOCOL == 11 || PROTOCOL == 8
store_compare_view_init(P_0);
#endif
#if PROTOCOL == 11
store_compare_view_init(P_0);
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
    store_compare_view_init(P_3);
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
        send_to_(P_2);

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
        if constexpr(id != P_2)
                store_compare_view_init(P_2);
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
