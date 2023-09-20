#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Replicated_init{
    public:
Replicated_init()  {}




template <int id,typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
    {
        send_to_(pnext);
        send_to_(pprev);
    }
}
    

    template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
if constexpr(id != PSELF)
    receive_from_(id);
}
Replicated_init public_val(Datatype a)
{
    return Replicated_init();
}

Replicated_init Not() const
{
    return Replicated_init();
}

template <typename func_add>
Replicated_init Add(Replicated_init b, func_add ADD) const
{
   return Replicated_init();
}
    
    template <typename func_add, typename func_sub, typename func_mul>
Replicated_init prepare_dot(const Replicated_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return Replicated_init();
}

template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
send_to_(pnext);
}



template <typename func_add, typename func_sub, typename func_mul>
    Replicated_init prepare_mult(Replicated_init b, func_add ADD, func_sub SUB, func_mul MULT) const
{
send_to_(pnext);
return Replicated_init();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
receive_from_(pprev);
}


void prepare_reveal_to_all()
{
    send_to_(pnext);
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
receive_from_(pprev);
Datatype result;
return result;
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

};
