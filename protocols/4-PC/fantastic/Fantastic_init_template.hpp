#pragma once
#include "../../generic_share.hpp"
template <typename Datatype>
class Fantastic_Four_init
{
  public:
    Fantastic_Four_init() {}

    Fantastic_Four_init public_val(Datatype a) { return Fantastic_Four_init(); }

    Fantastic_Four_init Not() const { return Fantastic_Four_init(); }

    template <typename func_add>
    Fantastic_Four_init Add(Fantastic_Four_init b, func_add ADD) const
    {
        return Fantastic_Four_init();
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Fantastic_Four_init prepare_dot(const Fantastic_Four_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        return Fantastic_Four_init();
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {
#if PARTY == 0

        send_to_(P_2);
        send_to_(P_1);
        /* store_compare_view_init(P_3); */

#elif PARTY == 1

        send_to_(P_3);
        send_to_(P_0);
        /* store_compare_view_init(P_0); */

#elif PARTY == 2

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        send_to_(P_0);
        /* store_compare_view_init(P_1); */
        /* store_compare_view_init(P_1); */

#elif PARTY == 3

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        send_to_(P_1);
        /* store_compare_view_init(P_2); */
        /* store_compare_view_init(P_0); */

#endif
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Fantastic_Four_init prepare_mult(Fantastic_Four_init b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
#if PARTY == 0

        send_to_(P_2);
        send_to_(P_1);
        /* store_compare_view_init(P_3); */

#elif PARTY == 1

        send_to_(P_3);
        send_to_(P_0);
        /* store_compare_view_init(P_0); */

#elif PARTY == 2

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        send_to_(P_0);
        /* store_compare_view_init(P_1); */
        /* store_compare_view_init(P_1); */

#elif PARTY == 3

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        send_to_(P_1);
        /* store_compare_view_init(P_2); */
        /* store_compare_view_init(P_0); */

#endif
        return Fantastic_Four_init();
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PARTY == 0

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + r013
        receive_from_(P_1);
        store_compare_view_init(P_3);
        store_compare_view_init(P_3);

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
        receive_from_(P_2);
        store_compare_view_init(P_1);

#elif PARTY == 1

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        receive_from_(P_3);
        store_compare_view_init(P_0);
        store_compare_view_init(P_2);

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
        receive_from_(P_0);
        store_compare_view_init(P_2);

        // receive second term from P_0

#elif PARTY == 2

        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
        receive_from_(P_0);
        store_compare_view_init(P_1);
        store_compare_view_init(P_1);
        store_compare_view_init(P_3);
        // receive rest from P_0, verify with P_3

#elif PARTY == 3

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 - r123) + r013
        receive_from_(P_1);
        store_compare_view_init(P_0);
        store_compare_view_init(P_0);
        store_compare_view_init(P_2);
#endif
    }

    void prepare_reveal_to_all() const { send_to_(PNEXT); }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        receive_from_(PPREV);
        store_compare_view_init(P_0123);
        Datatype dummy;
        return dummy;
    }

    template <typename func_mul>
    Fantastic_Four_init mult_public(const Datatype b, func_mul MULT) const
    {
        return Fantastic_Four_init();
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
#if PARTY == 0
            send_to_(P_1);
            send_to_(P_2);
#elif PARTY == 1
            send_to_(P_0);
            send_to_(P_2);
#elif PARTY == 2
            send_to_(P_0);
            send_to_(P_1);
#else  // PARTY == 3
            send_to_(P_0);
            send_to_(P_1);
#endif
        }
    }

    template <int id, typename func_add, typename func_sub>
    void complete_receive_from(func_add ADD, func_sub SUB)
    {
        if constexpr (id != PSELF)
        {
#if PARTY == 0
            if constexpr (id == P_1)
            {
                receive_from_(P_1);
                store_compare_view_init(P_2);
            }
            else if constexpr (id == P_2)
            {
                receive_from_(P_2);
                store_compare_view_init(P_1);
            }
            else  // id == P_3
            {
                receive_from_(P_3);
                store_compare_view_init(P_1);
            }
#elif PARTY == 1
            if constexpr (id == P_0)
            {
                receive_from_(P_0);
                store_compare_view_init(P_2);
            }
            else if constexpr (id == P_2)
            {
                receive_from_(P_2);
                store_compare_view_init(P_0);
            }
            else  // id == P_3
            {
                receive_from_(P_3);
                store_compare_view_init(P_0);
            }
#elif PARTY == 2
            if constexpr (id == P_0)
            {
                receive_from_(P_0);
                store_compare_view_init(P_1);
            }
            else if constexpr (id == P_1)
            {
                receive_from_(P_1);
                store_compare_view_init(P_0);
            }
#endif
        }
    }

    static void send() { send_(); }
    static void receive() { receive_(); }
    static void communicate() { communicate_(); }

    static void finalize(std::string* ips) { finalize_(ips); }

    static void finalize(std::string* ips, receiver_args* ra, sender_args* sa) { finalize_(ips, ra, sa); }
};
