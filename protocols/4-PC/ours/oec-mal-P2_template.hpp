#pragma once
#include "oec-mal_base.hpp"
class OEC_MAL2
{
bool optimized_sharing;
public:
OEC_MAL2(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

OEC_MAL_Share public_val(DATATYPE a)
{
    return OEC_MAL_Share(a,SET_ALL_ZERO());
}

OEC_MAL_Share Not(OEC_MAL_Share a)
{
    return OEC_MAL_Share(NOT(a.v),a.r);
}

// Receive sharing of ~XOR(a,b) locally
OEC_MAL_Share Xor(OEC_MAL_Share a, OEC_MAL_Share b)
{
   return OEC_MAL_Share(XOR(a.v,b.v),XOR(a.r,b.r));
}


#if FUNCTION_IDENTIFIER < 5
/* void prepare_and(OEC_MAL_Share a, OEC_MAL_Share b, OEC_MAL_Share &c) */
/* { */
/*    c.r = XOR(getRandomVal(P023), getRandomVal(P123)); */
/*    /1* DATATYPE r234 = getRandomVal(P123); *1/ */
/*    DATATYPE r234 = */
/*        getRandomVal(P123); // Probably sufficient to only generate with P3 -> */
/*                            // Probably not because of verification */
/* /1* c.r = getRandomVal(P3); *1/ */
/* #if PROTOCOL == 12 */
/* #if PRE == 1 */
/*    DATATYPE o1 = pre_receive_from_live(P3); */
/* #else */
/*    DATATYPE o1 = receive_from_live(P3); */
/* #endif */
/*    store_compare_view(P0, o1); */
/* #else */
/*    DATATYPE o1 = receive_from_live(P0); */
/*    store_compare_view(P3, o1); */
/* #endif */
/*    c.m = XOR( c.r, XOR(XOR(o1, AND(a.v, b.r)), AND(b.v, a.r))); */
/*    send_to_live(P1, c.m); */


/*    /1* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); *1/ */
/*    c.v = AND(a.v, b.v); */

/* #if PROTOCOL == 10 || PROTOCOL == 12 */
/*    /1* DATATYPE m3_prime = XOR(XOR(r234, c.r), c.v); *1/ */
/*    /1* DATATYPE m3_prime = XOR(XOR(r234, c.r), AND(XOR(a.v, a.r), XOR(b.v, b.r))); *1/ */
/*    send_to_live(P0, XOR(r234 ,XOR(c.v,c.r))); */
/* #endif */
/*    c.v = XOR(c.v, c.m); */
/*    c.m = XOR(c.m, r234); */
/* } */

/* void complete_and(OEC_MAL_Share &c) */
/* { */

/* DATATYPE m2 = receive_from_live(P1); */
/* c.v = XOR(c.v, m2); */
/* /1* c.m = XOR(c.m, m2); *1/ */
/* /1* DATATYPE cm = XOR(c.m, m2); *1/ */
/* #if PROTOCOL == 11 */
/* send_to_live(P0, XOR(c.m, m2)); // let P0 verify m_2 XOR m_3 */
/* send_to_live(P0, XOR(c.v,c.r)); // let P0 obtain ab, Problem for arithmetic circuits: P0 wants ab+c3, P2 has ab+c1 -> P2 needs to add c_3, P1 needs to substract c_1 on receiving */
/* #endif */

/* #if PROTOCOL == 10 || PROTOCOL == 12 */
/* store_compare_view(P012, XOR(c.m, m2)); */
/* #endif */
/* /1* store_compare_view(P0, c.v); *1/ */
/* } */
void prepare_and(OEC_MAL_Share a, OEC_MAL_Share b, OEC_MAL_Share &c)
{
    c.r = getRandomVal(P023);
   /* c.r = ADD(getRandomVal(P023), getRandomVal(P123)); */
   /* DATATYPE r234 = getRandomVal(P123); */
   /* DATATYPE r234_1 = */
   /*     getRandomVal(P123); // Probably sufficient to only generate with P3 -> */
   DATATYPE r234_2 =
       getRandomVal(P123_2); // Probably sufficient to only generate with P3 ->
                           // Probably not because of verification
/* c.r = getRandomVal(P3); */
#if PROTOCOL == 12
#if PRE == 1
   DATATYPE o1 = pre_receive_from_live(P3);
#else
   DATATYPE o1 = receive_from_live(P3);
#endif
   store_compare_view(P0, o1);
#else
   DATATYPE o1 = receive_from_live(P0);
   store_compare_view(P3, o1);
#endif
   c.v = XOR(XOR(AND(a.v, b.r),o1), AND(b.v, a.r));
   send_to_live(P1, c.v);


   /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */
   DATATYPE a1b1 = AND(a.v, b.v);

#if PROTOCOL == 10 || PROTOCOL == 12
   /* DATATYPE m3_prime = XOR(XOR(r234, c.r), c.v); */
   /* DATATYPE m3_prime = XOR(XOR(r234, c.r), AND(XOR(a.v, a.r), XOR(b.v, b.r))); */
   /* send_to_live(P0, ADD(r234 ,ADD(c.v,c.r))); */
   send_to_live(P0, XOR(a1b1,r234_2));
#endif
#if PROTOCOL == 11
c.m = XOR(c.v, r234_2); // store m_2 + m_3 + r_234_2 to send P0 later
#endif
   c.v = XOR(a1b1, c.v);
   /* c.m = XOR(c.m, r234); */
}

void complete_and(OEC_MAL_Share &c)
{
DATATYPE m2 = receive_from_live(P1);
c.v = XOR(c.v, m2);
/* c.m = XOR(c.m, m2); */
/* DATATYPE cm = XOR(c.m, m2); */
#if PROTOCOL == 11
send_to_live(P0, XOR(c.m, m2)); // let P0 verify m_2 XOR m_3, obtain m_2 + m_3 + r_234_2
send_to_live(P0, XOR(c.v,getRandomVal(P123))); // let P0 obtain ab + c1 + r234_1
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P012, XOR(getRandomVal(P123), c.v)); // compare ab + c1 + r234_1
#endif
/* store_compare_view(P0, c.v); */
}
#endif

#if FUNCTION_IDENTIFIER > 4
void prepare_mult(OEC_MAL_Share a, OEC_MAL_Share b, OEC_MAL_Share &c)
{
    //1* get Random val
    c.r = getRandomVal(P023);
   /* c.r = ADD(getRandomVal(P023), getRandomVal(P123)); */
   /* DATATYPE r234 = getRandomVal(P123); */
   /* DATATYPE r234_1 = */
   /*     getRandomVal(P123); // Probably sufficient to only generate with P3 -> */
   DATATYPE r234_2 =
       getRandomVal(P123_2); // Probably sufficient to only generate with P3 ->
                           // Probably not because of verification
/* c.r = getRandomVal(P3); */
#if PROTOCOL == 12
#if PRE == 1
   DATATYPE o1 = pre_receive_from_live(P3);
#else
   DATATYPE o1 = receive_from_live(P3);
#endif
   store_compare_view(P0, o1);
#else
   DATATYPE o1 = receive_from_live(P0);
   store_compare_view(P3, o1);
#endif
   c.v = ADD(SUB(MULT(a.v, b.r),o1), MULT(b.v, a.r));
   send_to_live(P1, c.v);


   /* c.v = AND(XOR(a.v, a.r), XOR(b.v, b.r)); */
   DATATYPE a1b1 = MULT(a.v, b.v);

#if PROTOCOL == 10 || PROTOCOL == 12
   /* DATATYPE m3_prime = XOR(XOR(r234, c.r), c.v); */
   /* DATATYPE m3_prime = XOR(XOR(r234, c.r), AND(XOR(a.v, a.r), XOR(b.v, b.r))); */
   /* send_to_live(P0, ADD(r234 ,ADD(c.v,c.r))); */
   send_to_live(P0, ADD(a1b1,r234_2));
#endif
#if PROTOCOL == 11
c.m = ADD(c.v, r234_2); // store m_2 + m_3 + r_234_2 to send P0 later
#endif
   c.v = SUB(a1b1, c.v);
   /* c.m = ADD(c.m, r234); */

}

void complete_mult(OEC_MAL_Share &c)
{

DATATYPE m2 = receive_from_live(P1);
c.v = SUB(c.v, m2);
/* c.m = XOR(c.m, m2); */
/* DATATYPE cm = XOR(c.m, m2); */
#if PROTOCOL == 11
send_to_live(P0, ADD(c.m, m2)); // let P0 verify m_2 XOR m_3, obtain m_2 + m_3 + r_234_2
send_to_live(P0, ADD(c.v,getRandomVal(P123))); // let P0 obtain ab + c1 + r234_1
#endif

#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view(P012, ADD(getRandomVal(P123), c.v)); // compare ab + c1 + r234_1
#endif
/* store_compare_view(P0, c.v); */
}

#endif

void prepare_reveal_to_all(OEC_MAL_Share a)
{
}


DATATYPE complete_Reveal(OEC_MAL_Share a)
{
#if PRE == 1
DATATYPE result = XOR(a.v, pre_receive_from_live(P3));
#else
DATATYPE result = XOR(a.v, receive_from_live(P3));
#endif
store_compare_view(P0123, result);
return result;
}

OEC_MAL_Share* alloc_Share(int l)
{
    return new OEC_MAL_Share[l];
}


void prepare_receive_from(OEC_MAL_Share a[], int id, int l)
{
if(id == PSELF)
{
for(int i = 0; i < l; i++)
{
    DATATYPE x_1 = getRandomVal(P023); // held by P1,P3 (this),P4
    DATATYPE x_2 = getRandomVal(P123); // held by P2,P3 (this),P4
    DATATYPE x_3 = XOR(x_1,x_2);
    a[i].r = x_3;
    a[i].v = get_input_live();
    /* a[i].p1 = getRandomVal(0); *1/ */
    send_to_live(P0, XOR(a[i].v,x_3));
    send_to_live(P1, XOR(a[i].v,x_1));
    a[i].v = XOR(a[i].v,x_1);
}
}
else if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
        a[i].r = getRandomVal(P023);
    }
}
else if(id == P1)
{
    for(int i = 0; i < l; i++)
    {
        a[i].r = getRandomVal(P123);
    }
}
else if(id == P3)
{
    for(int i = 0; i < l; i++)
    {
    DATATYPE r023 = getRandomVal(P023);
    DATATYPE r123 = getRandomVal(P123);

        a[i].r = XOR(r023,r123);
    }
}

}

void complete_receive_from(OEC_MAL_Share a[], int id, int l)
{
if(id != PSELF)
{
    for(int i = 0; i < l; i++)
    {
        a[i].v = receive_from_live(id);
    }
        if(id != P0)
            for(int i = 0; i < l; i++)
                store_compare_view(P0,XOR(a[i].v,a[i].r));
        if(id != P1)
            for(int i = 0; i < l; i++)
                store_compare_view(P1,a[i].v);
}

}


void send()
{
    send_live();
}

void receive()
{
    receive_live();
}

void communicate()
{
    communicate_live();
}

};
