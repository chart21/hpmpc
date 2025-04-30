#pragma once
#include "../../generic_share.hpp"
#define PRE_SHARE Fantastic_Four_Share
template <typename Datatype>
class Fantastic_Four_Share
{
  private:
    Datatype v0;
    Datatype v1;
    Datatype v2;
    Datatype verify_store0;  // used for saving messages for verification
    Datatype verify_store1;  // used for saving messages for verification
    Datatype verify_store2;  // used for saving messages for verification
  public:
    Fantastic_Four_Share() {}
    Fantastic_Four_Share(Datatype v0, Datatype v1, Datatype v2) : v0(v0), v1(v1), v2(v2) {}

    template <typename func_mul>
    Fantastic_Four_Share mult_public(const Datatype b, func_mul MULT) const
    {
        return Fantastic_Four_Share(MULT(v0, b), MULT(v1, b), MULT(v2, b));
    }

    Fantastic_Four_Share public_val(Datatype a)
    {
#if PARTY == 0
        return Fantastic_Four_Share(SET_ALL_ZERO(), SET_ALL_ZERO(), SET_ALL_ZERO());
#else
        return Fantastic_Four_Share(a, SET_ALL_ZERO(), SET_ALL_ZERO());  // a + a + a + 0 + 0 ... = a (Valid for XOR)
#endif
    }

    Fantastic_Four_Share Not() const
    {
#if PARTY < 3
        return Fantastic_Four_Share(v0, v1, NOT(v2));
#else
        return *this;
#endif
        /* return Fantastic_Four_Share(NOT(a.v0),NOT(a.v1),NOT(a.v2)); */
    }

    template <typename func_add>
    Fantastic_Four_Share Add(Fantastic_Four_Share b, func_add ADD) const
    {
        return Fantastic_Four_Share(ADD(v0, b.v0), ADD(v1, b.v1), ADD(v2, b.v2));
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Fantastic_Four_Share prepare_mult(Fantastic_Four_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Datatype cross_term1 = ADD(MULT(v0, b.v1), MULT(v1, b.v0));
        Datatype cross_term2 = ADD(MULT(v0, b.v2), MULT(v2, b.v0));
        Datatype cross_term3 = ADD(MULT(v1, b.v2), MULT(v2, b.v1));
        Fantastic_Four_Share c;

        /* c.v1 = XOR (XOR(cross_term1,cross_term3), AND(a.v1, b.v1)); */
        /* c.v2 = XOR (XOR(cross_term2,cross_term3), AND(a.v2, b.v2)); */

#if PARTY == 0

        Datatype r012 = getRandomVal(P_012);
        Datatype r013 = getRandomVal(P_013);
        Datatype r023 = getRandomVal(P_023);
        Datatype r023_2 = getRandomVal(P_023);

        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2

        Datatype send_Term2 = SUB(cross_term1, r013);  // sent by P_0 to P_2, verified by P_3
        send_to_live(P_2, send_Term2);
        c.v0 = ADD(ADD(r023, r023_2), ADD(MULT(v0, b.v0), send_Term2));

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + r013
        Datatype verifyTerm3 = SUB(cross_term3, r012);  // sent by P_1 to P_3, verified by P_0
        c.verify_store0 = verifyTerm3;
        /* store_compare_view(P_3,verifyTerm3); */
        c.v1 = ADD(r013, ADD(MULT(v1, b.v1), verifyTerm3));
        // receive next of c.v1 from P_1

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023_2) + r012
        Datatype send_Term1 = SUB(cross_term2, r023_2);  // sent by P_0 to P_1, verified by P_2
        send_to_live(P_1, send_Term1);
        c.v2 = ADD(send_Term1, ADD(MULT(v2, b.v2), r012));
// receive next of c.v2 from P_2
#elif PARTY == 1

        Datatype r012 = getRandomVal(P_012);
        Datatype r013 = getRandomVal(P_013);
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        c.v0 = ADD(MULT(v0, b.v0), ADD(r123, r123_2));
        // receive next term by P_3

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 + r123_2  ) + r013
        Datatype send_Term3 = SUB(cross_term3, r012);    // sent by P_1 to P_3, verified by P_0
        Datatype send_Term0 = SUB(cross_term1, r123_2);  // sent by P_1 to P_0, verified by P_3
        send_to_live(P_3, send_Term3);
        send_to_live(P_0, send_Term0);

        c.v1 = ADD(r013, ADD(MULT(v1, b.v1), ADD(send_Term3, send_Term0)));

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
        Datatype verifyTerm0 = SUB(cross_term2, r123);  // sent by P_2 to P_0, verified by P_1
        c.verify_store0 = verifyTerm0;
        /* store_compare_view(P_0,verifyTerm0); */
        c.v2 = ADD(verifyTerm0, ADD(MULT(v2, b.v2), r012));
        // receive second term from P_0

#elif PARTY == 2

        Datatype r012 = getRandomVal(P_012);
        Datatype r023 = getRandomVal(P_023);
        Datatype r023_2 = getRandomVal(P_023);
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        Datatype verifyTerm1 = SUB(cross_term1, r023);  // sent by P_3 to P_1, verified by P_2
        c.verify_store0 = verifyTerm1;
        /* store_compare_view(P_1,verifyTerm1); */
        c.v0 = ADD(ADD(r123, r123_2), ADD(MULT(v0, b.v0), verifyTerm1));

        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
        c.v1 = ADD(ADD(r023, r023_2), MULT(v1, b.v1));
        // receive rest from P_0, verify with P_3

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023_2) + r012
        Datatype send_Term0 = SUB(cross_term2, r123);  // sent by P_2 to P_0, verified by P_1
        send_to_live(P_0, send_Term0);
        Datatype verifyTerm1_2 = SUB(cross_term3, r023_2);  // sent by P_0 to P_1, verified by P_2
        c.verify_store1 = verifyTerm1_2;
        /* store_compare_view(P_1,verifyTerm1_2); */
        c.v2 = ADD(r012, ADD(MULT(v2, b.v2), ADD(send_Term0, verifyTerm1_2)));

#elif PARTY == 3

        Datatype r013 = getRandomVal(P_013);
        Datatype r023 = getRandomVal(P_023);
        Datatype r023_2 = getRandomVal(P_023);
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        Datatype send_Term1 = SUB(cross_term1, r023);  // sent by P_3 to P_1, verified by P_2
        send_to_live(P_1, send_Term1);
        c.v0 = ADD(ADD(r123, r123_2), ADD(MULT(v0, b.v0), send_Term1));

        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
        Datatype verifyTerm2 = SUB(cross_term3, r013);  // sent by P_0 to P_2, verified by P_3
        c.verify_store0 = verifyTerm2;
        /* store_compare_view(P_2,verifyTerm2); */
        c.v1 = ADD(ADD(r023, r023_2), ADD(MULT(v1, b.v1), verifyTerm2));

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 - r123_2) + r013
        Datatype verifyTerm0 = SUB(cross_term2, r123_2);  // sent by P_1 to P_0, verified by P_3
        c.verify_store1 = verifyTerm0;
        /* store_compare_view(P_0,verifyTerm0); */
        c.v2 = ADD(r013, ADD(MULT(v2, b.v2), verifyTerm0));
        // receive rest from P_1

#endif
        return c;
    }

    template <typename func_add, typename func_sub, typename func_mul>
    Fantastic_Four_Share prepare_dot(const Fantastic_Four_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
    {
        Fantastic_Four_Share c;
        c.verify_store0 = ADD(MULT(v0, b.v1), MULT(v1, b.v0));
        c.verify_store1 = ADD(MULT(v0, b.v2), MULT(v2, b.v0));
        c.verify_store2 = ADD(MULT(v1, b.v2), MULT(v2, b.v1));
        c.v0 = MULT(v0, b.v0);
        c.v1 = MULT(v1, b.v1);
        c.v2 = MULT(v2, b.v2);
        return c;
    }

    template <typename func_add, typename func_sub>
    void mask_and_send_dot(func_add ADD, func_sub SUB)
    {

        /* c.v1 = XOR (XOR(cross_term1,cross_term3), AND(a.v1, b.v1)); */
        /* c.v2 = XOR (XOR(cross_term2,cross_term3), AND(a.v2, b.v2)); */

#if PARTY == 0

        Datatype r012 = getRandomVal(P_012);
        Datatype r013 = getRandomVal(P_013);
        Datatype r023 = getRandomVal(P_023);
        Datatype r023_2 = getRandomVal(P_023);

        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2

        Datatype send_Term2 = SUB(verify_store0, r013);  // sent by P_0 to P_2, verified by P_3
        send_to_live(P_2, send_Term2);
        v0 = ADD(ADD(r023, r023_2), ADD(v0, send_Term2));

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + r013
        Datatype verifyTerm3 = SUB(verify_store2, r012);  // sent by P_1 to P_3, verified by P_0
        verify_store0 = verifyTerm3;
        /* store_compare_view(P_3,verifyTerm3); */
        v1 = ADD(r013, ADD(v1, verifyTerm3));
        // receive next of c.v1 from P_1

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023_2) + r012
        Datatype send_Term1 = SUB(verify_store1, r023_2);  // sent by P_0 to P_1, verified by P_2
        send_to_live(P_1, send_Term1);
        v2 = ADD(send_Term1, ADD(v2, r012));
// receive next of c.v2 from P_2
#elif PARTY == 1

        Datatype r012 = getRandomVal(P_012);
        Datatype r013 = getRandomVal(P_013);
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        v0 = ADD(v0, ADD(r123, r123_2));
        // receive next term by P_3

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 + r123_2  ) + r013
        Datatype send_Term3 = SUB(verify_store2, r012);    // sent by P_1 to P_3, verified by P_0
        Datatype send_Term0 = SUB(verify_store0, r123_2);  // sent by P_1 to P_0, verified by P_3
        send_to_live(P_3, send_Term3);
        send_to_live(P_0, send_Term0);

        v1 = ADD(r013, ADD(v1, ADD(send_Term3, send_Term0)));

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
        Datatype verifyTerm0 = SUB(verify_store1, r123);  // sent by P_2 to P_0, verified by P_1
        verify_store0 = verifyTerm0;
        /* store_compare_view(P_0,verifyTerm0); */
        v2 = ADD(verifyTerm0, ADD(v2, r012));
        // receive second term from P_0

#elif PARTY == 2

        Datatype r012 = getRandomVal(P_012);
        Datatype r023 = getRandomVal(P_023);
        Datatype r023_2 = getRandomVal(P_023);
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        Datatype verifyTerm1 = SUB(verify_store0, r023);  // sent by P_3 to P_1, verified by P_2
        verify_store0 = verifyTerm1;
        /* store_compare_view(P_1,verifyTerm1); */
        v0 = ADD(ADD(r123, r123_2), ADD(v0, verifyTerm1));

        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
        v1 = ADD(ADD(r023, r023_2), v1);
        // receive rest from P_0, verify with P_3

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023_2) + r012
        Datatype send_Term0 = SUB(verify_store1, r123);  // sent by P_2 to P_0, verified by P_1
        send_to_live(P_0, send_Term0);
        Datatype verifyTerm1_2 = SUB(verify_store2, r023_2);  // sent by P_0 to P_1, verified by P_2
        verify_store1 = verifyTerm1_2;
        /* store_compare_view(P_1,verifyTerm1_2); */
        v2 = ADD(r012, ADD(v2, ADD(send_Term0, verifyTerm1_2)));

#elif PARTY == 3

        Datatype r013 = getRandomVal(P_013);
        Datatype r023 = getRandomVal(P_023);
        Datatype r023_2 = getRandomVal(P_023);
        Datatype r123 = getRandomVal(P_123);
        Datatype r123_2 = getRandomVal(P_123);

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        Datatype send_Term1 = SUB(verify_store0, r023);  // sent by P_3 to P_1, verified by P_2
        send_to_live(P_1, send_Term1);
        v0 = ADD(ADD(r123, r123_2), ADD(v0, send_Term1));

        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
        Datatype verifyTerm2 = SUB(verify_store2, r013);  // sent by P_0 to P_2, verified by P_3
        verify_store0 = verifyTerm2;
        /* store_compare_view(P_2,verifyTerm2); */
        v1 = ADD(ADD(r023, r023_2), ADD(v1, verifyTerm2));

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 - r123_2) + r013
        Datatype verifyTerm0 = SUB(verify_store1, r123_2);  // sent by P_1 to P_0, verified by P_3
        verify_store1 = verifyTerm0;
        /* store_compare_view(P_0,verifyTerm0); */
        v2 = ADD(r013, ADD(v2, verifyTerm0));
        // receive rest from P_1

#endif
    }

    template <typename func_add, typename func_sub>
    void complete_mult(func_add ADD, func_sub SUB)
    {
#if PARTY == 0

        store_compare_view(P_3, verify_store0);
        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + r013
        Datatype receive_term1 = receive_from_live(P_1);
        store_compare_view(P_3, receive_term1);
        v1 = ADD(v1, receive_term1);

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
        Datatype receive_term2 = receive_from_live(P_2);
        store_compare_view(P_1, receive_term2);
        v2 = ADD(v2, receive_term2);

#elif PARTY == 1

        // c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
        store_compare_view(P_0, verify_store0);

        Datatype receive_term3 = receive_from_live(P_3);
        store_compare_view(P_2, receive_term3);
        v0 = ADD(v0, receive_term3);

        // c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
        Datatype receive_term0 = receive_from_live(P_0);
        store_compare_view(P_2, receive_term0);
        v2 = ADD(v2, receive_term0);

        // receive second term from P_0

#elif PARTY == 2

        store_compare_view(P_1, verify_store0);
        store_compare_view(P_1, verify_store1);
        // c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
        Datatype receive_term0 = receive_from_live(P_0);
        store_compare_view(P_3, receive_term0);
        v1 = ADD(v1, receive_term0);
        // receive rest from P_0, verify with P_3

#elif PARTY == 3

        // c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 - r123) + r013
        Datatype receive_term1 = receive_from_live(P_1);
        store_compare_view(P_0, receive_term1);
        v2 = ADD(v2, receive_term1);
        store_compare_view(P_0, verify_store1);
        store_compare_view(P_2, verify_store0);
#endif
    }

    void prepare_reveal_to_all() const
    {
#if PARTY == 0
        send_to_live(P_1, v0);
#elif PARTY == 1
        send_to_live(P_2, v1);
#elif PARTY == 2
        send_to_live(P_3, v2);
#else
        send_to_live(P_0, v0);
#endif
    }

    template <typename func_add, typename func_sub>
    Datatype complete_Reveal(func_add ADD, func_sub SUB) const
    {
        Datatype result = ADD(ADD(v1, v2), ADD(v0, receive_from_live(PPREV)));
        store_compare_view(P_0123, result);
        return result;
    }

    template <int id, typename func_add, typename func_sub>
    void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
    {
        if constexpr (id == PSELF)
        {
#if PARTY == 0
            v0 = getRandomVal(P_023);
            v1 = getRandomVal(P_013);

            v2 = val;
            v2 = SUB(v2, (ADD(v0, v1)));

            send_to_live(P_1, v2);
            send_to_live(P_2, v2);
#elif PARTY == 1
            v0 = getRandomVal(P_123);
            v1 = getRandomVal(P_013);
            v2 = SUB(val, (ADD(v0, v1)));
            send_to_live(P_0, v2);
            send_to_live(P_2, v2);
#elif PARTY == 2
            v0 = getRandomVal(P_123);
            v1 = getRandomVal(P_023);
            v2 = SUB(val, (ADD(v0, v1)));
            send_to_live(P_0, v2);
            send_to_live(P_1, v2);
#else  // PARTY == 3
            v0 = getRandomVal(P_123);
            v1 = getRandomVal(P_023);
            v2 = SUB(val, (ADD(v0, v1)));
            send_to_live(P_0, v2);
            send_to_live(P_1, v2);
#endif
        }
        else
        {
            if constexpr (id == P_0)
            {
#if PARTY == 1

                v0 = SET_ALL_ZERO();
                v1 = getRandomVal(P_013);
                // receive
#elif PARTY == 2
                v0 = SET_ALL_ZERO();
                v1 = getRandomVal(P_023);
                // receive
#else  // PARTY == 3
                v0 = SET_ALL_ZERO();
                v1 = getRandomVal(P_023);
                v2 = getRandomVal(P_013);
#endif
            }
            else if constexpr (id == P_1)
            {
#if PARTY == 0
                v0 = SET_ALL_ZERO();
                v1 = getRandomVal(P_013);
                // receive
#elif PARTY == 2
                v0 = getRandomVal(P_123);
                v1 = SET_ALL_ZERO();
                // receive
#else  // PARTY == 3
                v0 = getRandomVal(P_123);
                v1 = SET_ALL_ZERO();
                v2 = getRandomVal(P_013);
#endif
            }
            else if constexpr (id == P_2)
            {
#if PARTY == 0
                v0 = getRandomVal(P_023);
                v1 = SET_ALL_ZERO();
                // receive
#elif PARTY == 1
                v0 = getRandomVal(P_123);
                v1 = SET_ALL_ZERO();
                // receive
#else  // PARTY == 3
                v0 = getRandomVal(P_123);
                v1 = getRandomVal(P_023);
                v2 = SET_ALL_ZERO();
#endif
            }
            else  // id == P_3
            {
#if PARTY == 0
                v0 = getRandomVal(P_023);
                v2 = SET_ALL_ZERO();
                // receive
#elif PARTY == 1
                v0 = getRandomVal(P_123);
                // receive
                v2 = SET_ALL_ZERO();
#else  // PARTY == 2
                v0 = getRandomVal(P_123);
                v1 = getRandomVal(P_023);
                v2 = SET_ALL_ZERO();
#endif
            }
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
                v2 = receive_from_live(P_1);
                store_compare_view(P_2, v2);
            }
            else if constexpr (id == P_2)
            {
                v2 = receive_from_live(P_2);
                store_compare_view(P_1, v2);
            }
            else  // id == P_3
            {
                v1 = receive_from_live(P_3);
                store_compare_view(P_1, v1);
            }
#elif PARTY == 1
            if constexpr (id == P_0)
            {
                v2 = receive_from_live(P_0);
                store_compare_view(P_2, v2);
            }
            else if constexpr (id == P_2)
            {
                v2 = receive_from_live(P_2);
                store_compare_view(P_0, v2);
            }
            else  // id == P_3
            {
                v1 = receive_from_live(P_3);
                store_compare_view(P_0, v1);
            }
#elif PARTY == 2
            if constexpr (id == P_0)
            {
                v2 = receive_from_live(P_0);
                store_compare_view(P_1, v2);
            }
            else if constexpr (id == P_1)
            {
                v2 = receive_from_live(P_1);
                store_compare_view(P_0, v2);
            }
#endif
        }
    }

    static void send() { send_live(); }

    static void receive() { receive_live(); }

    static void communicate()
    {
        /* #if PRE == 0 */
        communicate_live();
        /* #endif */
    }
};
