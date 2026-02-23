#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL3_NO_CV_init
{
  public:
    OEC_MAL3_NO_CV_init() {}

    static OEC_MAL3_NO_CV_init public_val(Datatype a) { return OEC_MAL3_NO_CV_init(); }

    OEC_MAL3_NO_CV_init Not() const { return OEC_MAL3_NO_CV_init(); }

    template <typename func_add>
    OEC_MAL3_NO_CV_init Add(OEC_MAL3_NO_CV_init b, func_add ADD) const
    {
        return OEC_MAL3_NO_CV_init();
    }
    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL3_NO_CV_init prepare_mult(OEC_MAL3_NO_CV_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {

#if PRE == 1
        pre_send_to_(P_2);
#else
        send_to_(P_2);
#endif
        return OEC_MAL3_NO_CV_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        pre_send_to_(P_0);
#else
        send_to_(P_0);
#endif
    }

    void prepare_reveal_to_all() const
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

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        receive_from_(P_1);
        receive_from_(P_2);
#if PRE == 1 && HAS_POST_PROTOCOL == 1
        store_output_share_();
#endif
        Datatype dummy;
        return dummy;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
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
    
    template <typename func_add, typename func_sub>
    void complete_mult2(func_add ADD, func_sub SUB)
    {
    }
    
    template <int id, typename func_add, typename func_sub>
    void complete_receive_from2(func_add ADD, func_sub SUB)
    {
    }
    
    template <typename func_mul>
    OEC_MAL3_NO_CV_init mult_public(const Datatype b, func_mul MULT) const
    {
        return OEC_MAL3_NO_CV_init();
    }

    static void send() { send_(); }

    // P_0 only has 1 receive round
    static void receive() { receive_(); }

    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

};