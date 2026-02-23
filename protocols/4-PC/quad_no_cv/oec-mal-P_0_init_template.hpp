#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class OEC_MAL0_NO_CV_init
{
  public:
    OEC_MAL0_NO_CV_init() {}

    static OEC_MAL0_NO_CV_init public_val(Datatype a) { return OEC_MAL0_NO_CV_init(); }

    OEC_MAL0_NO_CV_init Not() const { return OEC_MAL0_NO_CV_init(); }

    template <typename func_add>
    OEC_MAL0_NO_CV_init Add(OEC_MAL0_NO_CV_init b, func_add ADD) const
    {
        return OEC_MAL0_NO_CV_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    OEC_MAL0_NO_CV_init prepare_mult(OEC_MAL0_NO_CV_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        send_to_(P_2);
        return OEC_MAL0_NO_CV_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PRE == 1
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
#endif

        receive_from_(P_2);
        receive_from_(P_1);
    }
    
    template <typename func_add, typename func_sub>
    void complete_mult2(func_add ADD, func_sub SUB)
    {
        receive_from_(P_1);
        receive_from_(P_2);
    }


    void prepare_reveal_to_all() const
    {
        send_to_(P_1);
        send_to_(P_2);
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
#if PRE == 1
        pre_receive_from_(P_3);
#else
        receive_from_(P_3);
#endif
        receive_from_(P_2);
        Datatype dummy;
        return dummy;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(func_add ADD, func_sub SUB)
    {

        if constexpr (id == PSELF)
        {
            send_to_(P_1);
            send_to_(P_2);
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {
#if PRE == 1
            if (id == P_3)
                pre_receive_from_(P_3);
            else
                receive_from_(id);
#else
            receive_from_(id);
#endif
            if constexpr (id != P_1)
                receive_from_(P_1);
            if constexpr (id != P_2)
                receive_from_(P_2);
        }
    }

    static void send() { send_(); }

    // P_0 only has 1 receive round
    static void receive() { receive_(); }

    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }

};
