#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL0_init
{
public:
OEC_MAL0_init() {}



OEC_MAL0_init public_val(Datatype a)
{
    return OEC_MAL0_init();
}

OEC_MAL0_init Not() const
{
    return OEC_MAL0_init();
}

template <typename func_add>
OEC_MAL0_init Add(OEC_MAL0_init b, func_add ADD) const
{
   return OEC_MAL0_init();
}
    
template <typename func_add, typename func_sub, typename func_mul>
OEC_MAL0_init prepare_dot(const OEC_MAL0_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OEC_MAL0_init();
}

template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
#if PROTOCOL == 12 || PROTOCOL == 8
    store_compare_view_init(P_2);
#else
#if PRE == 1
    pre_send_to_(P_2);
#else
    send_to_(P_2);
#endif
#endif
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL0_init prepare_mult(OEC_MAL0_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
#if PROTOCOL == 12 || PROTOCOL == 8
    store_compare_view_init(P_2);
#else
#if PRE == 1
    pre_send_to_(P_2);
#else
    send_to_(P_2);
#endif
#endif
return OEC_MAL0_init();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
#if PROTOCOL == 10 || PROTOCOL == 12 || PROTOCOL == 8
#if PRE == 1
    pre_receive_from_(P_3);
#else
receive_from_(P_3);
#endif

receive_from_(P_2);

store_compare_view_init(P_1);
/* store_compare_view_init(P_1); */
store_compare_view_init(P_012);
# elif PROTOCOL == 11
receive_from_(P_2);
receive_from_(P_2); // receive ab from P_2
store_compare_view_init(P_1);
store_compare_view_init(P_1);
store_compare_view_init(P_3);
#endif
}

void prepare_reveal_to_all()
{
send_to_(P_1);
send_to_(P_2);

send_to_(P_3);
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
#if PRE == 1
    pre_receive_from_(P_3);
    /* send_to_(P_3); */
#else
receive_from_(P_3);
#endif
#if PROTOCOL == 8
    store_compare_view_init(P_1);
    store_compare_view_init(P_1);
    store_compare_view_init(P_2);
#else
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
        send_to_(P_1);
        send_to_(P_2);
}
}

    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id != PSELF)
    {
#if PRE == 1
            if(id == P_3)
                pre_receive_from_(P_3);
            else
                receive_from_(id);
#else
            receive_from_(id);
#endif
        if constexpr(id != P_1)
                store_compare_view_init(P_1);
        if constexpr(id != P_2)
                store_compare_view_init(P_2);
        
    }
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
