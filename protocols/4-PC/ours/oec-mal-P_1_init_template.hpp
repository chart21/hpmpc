#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL1_init
{
public:
OEC_MAL1_init() {}



static OEC_MAL1_init public_val(Datatype a)
{
    return OEC_MAL1_init();
}

template <typename func_mul>
OEC_MAL1_init mult_public(const Datatype b, func_mul MULT) const
{
    return OEC_MAL1_init();
}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void prepare_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OEC_MAL1_init& r_mk2, OEC_MAL1_init& r_msb, OEC_MAL1_init& c, OEC_MAL1_init& c_prime){
    store_compare_view_init(P_0);
    store_compare_view_init(P_0);
}

template <typename func_add, typename func_sub, typename func_xor, typename func_and, typename func_trunc>
void complete_trunc_2k_inputs(func_add ADD, func_sub SUB, func_xor XOR, func_and AND, func_trunc trunc, OEC_MAL1_init& r_mk2, OEC_MAL1_init& r_msb, OEC_MAL1_init& c, OEC_MAL1_init& c_prime){
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
OEC_MAL1_init prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC) const
{
    store_compare_view_init(P_0);
    return OEC_MAL1_init();
} 

    template <typename func_add, typename func_sub>
void complete_public_mult_fixed( func_add ADD, func_sub SUB)
{
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

    template <typename func_add, typename func_sub>
void mask_and_send_dot( func_add ADD, func_sub SUB)
{
send_to_(P_2);
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view_init(P_0); // compare a1b1 + r123_2 with P_0
#endif
}
    template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
send_to_(P_2); 
}
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
receive_from_(P_2); // v^1,2 = m^1 + m^2
#if PROTOCOL == 11
store_compare_view_init(P_0); // compare m1 + m2 + r123 with P_0
#else
store_compare_view_init(P_012); // v^1,2 + r_1,2,3
#endif
store_compare_view_init(P_0);
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

void prepare_reveal_to_all() const
{
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
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
        send_to_(P_2);

}
}
    template <int id,typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    prepare_receive_from<id>(ADD, SUB);
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

static void prepare_A2B_S1(int m, int k,OEC_MAL1_init in[], OEC_MAL1_init out[])
{
    for (int j = m; j < k; j++)
    {
    store_compare_view_init(P_0);
    }

}


static void prepare_A2B_S2(int m, int k, OEC_MAL1_init in[], OEC_MAL1_init out[])
{
}

static void complete_A2B_S1(int k, OEC_MAL1_init out[])
{
}

static void complete_A2B_S2(int k, OEC_MAL1_init out[])
{

}

void prepare_bit2a( OEC_MAL1_init out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        send_to_(P_2);
        store_compare_view_init(P_0);
    }
}

void complete_bit2a()
{
        receive_from_(P_2);
        store_compare_view_init(P_012);
}

void prepare_opt_bit_injection(OEC_MAL1_init a[], OEC_MAL1_init out[])
{
    for(int i = 0; i < BITLENGTH; i++)
    {
        send_to_(P_2);
        store_compare_view_init(P_0);
    }
}

void complete_opt_bit_injection()
{
        receive_from_(P_2);
        store_compare_view_init(P_012);
}


void prepare_bit_injection_S1(OEC_MAL1_init out[])
{
    for (int j = 0; j < BITLENGTH; j++)
    {
    store_compare_view_init(P_0);
    }
}

void prepare_bit_injection_S2(OEC_MAL1_init out[])
{
}

static void complete_bit_injection_S1(OEC_MAL1_init out[])
{
    
}

static void complete_bit_injection_S2(OEC_MAL1_init out[])
{
}
template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_init prepare_dot3(OEC_MAL1_init b, OEC_MAL1_init c, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL1_init d;
return d;
}
template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_init prepare_mult3(OEC_MAL1_init b, OEC_MAL1_init c, func_add ADD, func_sub SUB, func_mul MULT) const
{
store_compare_view_init(P_0);
send_to_(P_2);
OEC_MAL1_init d;
return d;
}

template <typename func_add, typename func_sub>
void complete_mult3(func_add ADD, func_sub SUB){
receive_from_(P_2);
store_compare_view_init(P_012); //compare d_0 s
}


template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_init prepare_dot4(OEC_MAL1_init b, OEC_MAL1_init c, OEC_MAL1_init d, func_add ADD, func_sub SUB, func_mul MULT) const
{
OEC_MAL1_init e;
return e;
}

template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL1_init prepare_mult4(OEC_MAL1_init b, OEC_MAL1_init c, OEC_MAL1_init d, func_add ADD, func_sub SUB, func_mul MULT) const
{
store_compare_view_init(P_0);
send_to_(P_2);
OEC_MAL1_init e;
return e;
}

template <typename func_add, typename func_sub>
void complete_mult4(func_add ADD, func_sub SUB){
receive_from_(P_2);
store_compare_view_init(P_012); 
}



};
