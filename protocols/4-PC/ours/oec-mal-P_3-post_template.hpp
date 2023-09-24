#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OECL_MAL3_POST_Share
{
private:
    Datatype r0;
    Datatype r1;
public:
    
OECL_MAL3_POST_Share() {}
OECL_MAL3_POST_Share(Datatype r0, Datatype r1) : r0(r0), r1(r1) {}
OECL_MAL3_POST_Share(Datatype r0) : r0(r0) {}


    

OECL_MAL3_POST_Share public_val(Datatype a)
{
    return OECL_MAL3_POST_Share();
}

OECL_MAL3_POST_Share Not() const
{
    return OECL_MAL3_POST_Share();
}

template <typename func_add>
OECL_MAL3_POST_Share Add(OECL_MAL3_POST_Share b, func_add ADD) const
{
    return OECL_MAL3_POST_Share();
}



template <typename func_add, typename func_sub, typename func_mul>
    OECL_MAL3_POST_Share prepare_mult(OECL_MAL3_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OECL_MAL3_POST_Share();
}

template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
}


void prepare_reveal_to_all()
{
}    

template <typename func_add, typename func_sub, typename func_mul>
OECL_MAL3_POST_Share prepare_dot(const OECL_MAL3_POST_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
    return OECL_MAL3_POST_Share();
}

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
}
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}

    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
}



template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB)
{
#if PROTOCOL == 8
Datatype mv = receive_from_live(P_0);
store_compare_view(P_1, mv); //verify own value
Datatype result = SUB(mv, retrieve_output_share());
result = SUB(result, retrieve_output_share());
result = SUB(result, retrieve_output_share());
#else
Datatype result = SUB(receive_from_live(P_0),retrieve_output_share());
store_compare_view(P_123, retrieve_output_share());
store_compare_view(P_0123, result);
#endif
return result;
}



template <int id, typename func_add, typename func_sub>
void prepare_receive_from(func_add ADD, func_sub SUB)
{}

template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{}




static void send()
{
    send_live();
}

static void receive()
{
    receive_live();
}

static void communicate()
{
    communicate_live();
}

};


