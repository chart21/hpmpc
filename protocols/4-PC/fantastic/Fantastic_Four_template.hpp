#pragma once
#include "fantastic_base.hpp"
#define PRE_SHARE Fantastic_Share
class Fantastic_Four
{
bool optimized_sharing;
public:
Fantastic_Four(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

void prepare_reveal_to_all(Fantastic_Share a)
{
#if PARTY == 0
send_to_live(P_1, a.v0);
#elif PARTY == 1
send_to_live(P_2, a.v1);
#elif PARTY == 2
send_to_live(P_3, a.v2);
#else
send_to_live(P_0, a.v0);
#endif
}    


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(Fantastic_Share a, Fantastic_Share b, Fantastic_Share &c, func_add ADD, func_sub SUB, func_mul MULT)
{
DATATYPE cross_term1 = ADD( MULT(a.v0,b.v1), MULT(a.v1,b.v0));
DATATYPE cross_term2 = ADD( MULT(a.v0,b.v2), MULT(a.v2,b.v0));
DATATYPE cross_term3 = ADD( MULT(a.v1,b.v2), MULT(a.v2,b.v1));


/* c.v1 = XOR (XOR(cross_term1,cross_term3), AND(a.v1, b.v1)); */
/* c.v2 = XOR (XOR(cross_term2,cross_term3), AND(a.v2, b.v2)); */

#if PARTY == 0

DATATYPE r012 = getRandomVal(P_012);
DATATYPE r013 = getRandomVal(P_013);
DATATYPE r023 = getRandomVal(P_023);
DATATYPE r023_2 = getRandomVal(P_023);


//c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2

DATATYPE send_Term2 = SUB(cross_term1, r013); // sent by P_0 to P_2, verified by P_3
send_to_live(P_2, send_Term2);
c.v0 = ADD( ADD(r023, r023_2) ,ADD( MULT(a.v0, b.v0), send_Term2)); 

//c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + r013
DATATYPE verifyTerm3 = SUB(cross_term3, r012); // sent by P_1 to P_3, verified by P_0
c.verify_store0 = verifyTerm3;
/* store_compare_view(P_3,verifyTerm3); */
c.v1 = ADD( r013 ,ADD( MULT(a.v1, b.v1), verifyTerm3));
//receive next of c.v1 from P_1

//c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023_2) + r012
DATATYPE send_Term1 = SUB(cross_term2, r023_2); // sent by P_0 to P_1, verified by P_2
send_to_live(P_1, send_Term1);
c.v2 = ADD(send_Term1 ,ADD( MULT(a.v2, b.v2), r012));
//receive next of c.v2 from P_2
#elif PARTY == 1

DATATYPE r012 = getRandomVal(P_012);
DATATYPE r013 = getRandomVal(P_013);
DATATYPE r123 = getRandomVal(P_123);
DATATYPE r123_2 = getRandomVal(P_123);


// c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
c.v0 = ADD (MULT(a.v0, b.v0), ADD(r123, r123_2));
//receive next term by P_3

//c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 + r123_2  ) + r013
DATATYPE send_Term3 = SUB(cross_term3, r012); // sent by P_1 to P_3, verified by P_0
DATATYPE send_Term0 = SUB(cross_term1, r123_2); // sent by P_1 to P_0, verified by P_3
send_to_live(P_3, send_Term3);
send_to_live(P_0, send_Term0);

c.v1 = ADD(r013 ,ADD( MULT(a.v1, b.v1), ADD(send_Term3, send_Term0)));


//c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
DATATYPE verifyTerm0 = SUB(cross_term2, r123); // sent by P_2 to P_0, verified by P_1
c.verify_store0 = verifyTerm0;
/* store_compare_view(P_0,verifyTerm0); */
c.v2 = ADD(verifyTerm0,  ADD (MULT(a.v2, b.v2), r012));
// receive second term from P_0

#elif PARTY == 2

DATATYPE r012 = getRandomVal(P_012);
DATATYPE r023 = getRandomVal(P_023);
DATATYPE r023_2 = getRandomVal(P_023);
DATATYPE r123 = getRandomVal(P_123);
DATATYPE r123_2 = getRandomVal(P_123);


// c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
DATATYPE verifyTerm1 = SUB(cross_term1, r023); // sent by P_3 to P_1, verified by P_2
c.verify_store0 = verifyTerm1;
/* store_compare_view(P_1,verifyTerm1); */
c.v0 = ADD( ADD(r123, r123_2) ,ADD( MULT(a.v0, b.v0), verifyTerm1));

//c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
c.v1 = ADD (ADD(r023,r023_2), MULT(a.v1, b.v1));
//receive rest from P_0, verify with P_3

//c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023_2) + r012
DATATYPE send_Term0 = SUB(cross_term2, r123); // sent by P_2 to P_0, verified by P_1
send_to_live(P_0, send_Term0);
DATATYPE verifyTerm1_2 = SUB(cross_term3, r023_2); // sent by P_0 to P_1, verified by P_2
c.verify_store1 = verifyTerm1_2;
/* store_compare_view(P_1,verifyTerm1_2); */
c.v2 = ADD(r012, ADD( MULT(a.v2, b.v2), ADD(send_Term0, verifyTerm1_2)));


#elif PARTY == 3

DATATYPE r013 = getRandomVal(P_013);
DATATYPE r023 = getRandomVal(P_023);
DATATYPE r023_2 = getRandomVal(P_023);
DATATYPE r123 = getRandomVal(P_123);
DATATYPE r123_2 = getRandomVal(P_123);

//c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
DATATYPE send_Term1 = SUB(cross_term1, r023); // sent by P_3 to P_1, verified by P_2
send_to_live(P_1, send_Term1);
c.v0 = ADD( ADD(r123, r123_2) ,ADD( MULT(a.v0, b.v0), send_Term1));

//c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
DATATYPE verifyTerm2 = SUB(cross_term3, r013); // sent by P_0 to P_2, verified by P_3
c.verify_store0 = verifyTerm2;
/* store_compare_view(P_2,verifyTerm2); */
c.v1 = ADD (ADD(r023, r023_2), ADD( MULT(a.v1, b.v1), verifyTerm2));

//c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 - r123_2) + r013
DATATYPE verifyTerm0 = SUB(cross_term2, r123_2); // sent by P_1 to P_0, verified by P_3
c.verify_store1 = verifyTerm0;
/* store_compare_view(P_0,verifyTerm0); */
c.v2 = ADD(r013 ,ADD( MULT(a.v2, b.v2), verifyTerm0));
//receive rest from P_1

#endif
}



template <typename func_add, typename func_sub>
void complete_mult(Fantastic_Share &c, func_add ADD, func_sub SUB)
{
#if PARTY == 0

store_compare_view(P_3,c.verify_store0);
//c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + r013
DATATYPE receive_term1 = receive_from_live(P_1);
store_compare_view(P_3,receive_term1);
c.v1 = ADD(c.v1, receive_term1);


//c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
DATATYPE receive_term2 = receive_from_live(P_2);
store_compare_view(P_1,receive_term2);
c.v2 = ADD(c.v2, receive_term2);


#elif PARTY == 1

// c0 = a0 b0 + (a0 b1 + a1 b0 - r023) + r123 + r123_2
store_compare_view(P_0,c.verify_store0);

DATATYPE receive_term3 = receive_from_live(P_3);
store_compare_view(P_2,receive_term3);
c.v0 = ADD(c.v0, receive_term3);

//c3 = a3 b3 + a3 b2 + (a3 b0 + a0 b3 - r123) + (a1b3 + a3b1 - r023) + r012
DATATYPE receive_term0 = receive_from_live(P_0);
store_compare_view(P_2,receive_term0);
c.v2 = ADD(c.v2, receive_term0);

// receive second term from P_0


#elif PARTY == 2

store_compare_view(P_1,c.verify_store0);
store_compare_view(P_1,c.verify_store1);
//c1 = a1 b1 + a1 b2 + (a1 b2 + a2 b1 - r013) + r023 + r023_2
DATATYPE receive_term0 = receive_from_live(P_0);
store_compare_view(P_3,receive_term0);
c.v1 = ADD(c.v1, receive_term0);
//receive rest from P_0, verify with P_3



#elif PARTY == 3


//c2 = a2 b2 + a2 b1 + (a2 b3 + a3 b2 - r012) + (a0 b2 + a2 b0 - r123) + r013
DATATYPE receive_term1 = receive_from_live(P_1);
store_compare_view(P_0,receive_term1);
c.v2 = ADD(c.v2, receive_term1);
store_compare_view(P_0,c.verify_store1);
store_compare_view(P_2,c.verify_store0);
#endif
}

template <typename func_add>
Fantastic_Share Add(Fantastic_Share a, Fantastic_Share b, func_add ADD)
{
    return Fantastic_Share(ADD(a.v0,b.v0),ADD(a.v1,b.v1),ADD(a.v2,b.v2));
}

template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(Fantastic_Share a, func_add ADD, func_sub SUB)
{
DATATYPE result = ADD( ADD(a.v1,a.v2) ,ADD(a.v0, receive_from_live(PPREV)));
store_compare_view(P_0123, result);
return result;
}


Fantastic_Share* alloc_Share(int l)
{
    return new Fantastic_Share[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(Fantastic_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == PSELF)
{
#if PARTY == 0
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_023);
    a[i].v1 = getRandomVal(P_013);
    
    
    a[i].v2 = get_input_live();
    a[i].v2 = SUB(a[i].v2 , (ADD(a[i].v0, a[i].v1)));

    send_to_live(P_1, a[i].v2);
    send_to_live(P_2, a[i].v2);
    }
#elif PARTY == 1
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = getRandomVal(P_013);
    a[i].v2 = SUB(get_input_live() , (ADD(a[i].v0, a[i].v1)));
    send_to_live(P_0, a[i].v2);
    send_to_live(P_2, a[i].v2);
    }
#elif PARTY == 2
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = getRandomVal(P_023);
    a[i].v2 = SUB(get_input_live() , (ADD(a[i].v0, a[i].v1)));
    send_to_live(P_0, a[i].v2);
    send_to_live(P_1, a[i].v2);
    }
#else // PARTY == 3
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = getRandomVal(P_023);
    a[i].v2 = SUB(get_input_live() , (ADD(a[i].v0, a[i].v1)));
    send_to_live(P_0, a[i].v2);
    send_to_live(P_1, a[i].v2);
    }
#endif
}
else{
if(id == P_0)
{
 #if PARTY == 1
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = SET_ALL_ZERO();
    a[i].v1 = getRandomVal(P_013);
    // receive
    }
#elif PARTY == 2
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = SET_ALL_ZERO();
    a[i].v1 = getRandomVal(P_023);
    // receive
    }
#else // PARTY == 3
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = SET_ALL_ZERO();
    a[i].v1 = getRandomVal(P_023);
    a[i].v2 = getRandomVal(P_013);
    }
#endif
}
else if(id == P_1)
{
 #if PARTY == 0
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = SET_ALL_ZERO();
    a[i].v1 = getRandomVal(P_013);
    // receive
    }
#elif PARTY == 2
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = SET_ALL_ZERO();
    // receive
    }
#else // PARTY == 3
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = SET_ALL_ZERO();
    a[i].v2 = getRandomVal(P_013);
    }
#endif
}
else if(id == P_2)
{
 #if PARTY == 0
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_023);
    a[i].v1 = SET_ALL_ZERO();
    // receive
    }
#elif PARTY == 1
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = SET_ALL_ZERO();
    // receive
    }
#else // PARTY == 3
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = getRandomVal(P_023);
    a[i].v2 = SET_ALL_ZERO();
    }
#endif
}
else // id == P_3
{
 #if PARTY == 0
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_023);
    a[i].v2 = SET_ALL_ZERO();
    // receive
    }
#elif PARTY == 1
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    //receive
    a[i].v2 = SET_ALL_ZERO();
    }
#else // PARTY == 2
    for(int i = 0; i < l; i++)
    {
    a[i].v0 = getRandomVal(P_123);
    a[i].v1 = getRandomVal(P_023);
    a[i].v2 = SET_ALL_ZERO();
    }
#endif
}
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(Fantastic_Share a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id != PSELF)
{
#if PARTY == 0
    if(id == P_1)
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v2 = receive_from_live(P_1);
    store_compare_view(P_2,a[i].v2);
    }
    }
    else if(id == P_2)
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v2 = receive_from_live(P_2);
    store_compare_view(P_1,a[i].v2);
    }
    }
    else // id == P_3
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v1 = receive_from_live(P_3);
    store_compare_view(P_1,a[i].v1);
    }
    }
#elif PARTY == 1
    if(id == P_0)
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v2 = receive_from_live(P_0);
    store_compare_view(P_2,a[i].v2);
    }
    }
    else if(id == P_2)
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v2 = receive_from_live(P_2);
    store_compare_view(P_0,a[i].v2);
    }
    }
    else // id == P_3
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v1 = receive_from_live(P_3);
    store_compare_view(P_0,a[i].v1);
    }
    }
#elif PARTY == 2
    if(id == P_0)
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v2 = receive_from_live(P_0);
    store_compare_view(P_1,a[i].v2);
    }
    }
    else if(id == P_1)
    {
    for(int i = 0; i < l; i++)
    {
    a[i].v2 = receive_from_live(P_1);
    store_compare_view(P_0,a[i].v2);
    }
    } 
#endif
}
}


Fantastic_Share public_val(DATATYPE a)
{
    #if PARTY == 0
    return Fantastic_Share(SET_ALL_ZERO(),SET_ALL_ZERO(),SET_ALL_ZERO());
    #else
    return Fantastic_Share(a,SET_ALL_ZERO() ,SET_ALL_ZERO()); // a + a + a + 0 + 0 ... = a (Valid for XOR)
    #endif
}

Fantastic_Share Not(Fantastic_Share a)
{
#if PARTY < 3
    return Fantastic_Share(a.v0,a.v1,NOT(a.v2));
#else
    return a;
#endif
   /* return Fantastic_Share(NOT(a.v0),NOT(a.v1),NOT(a.v2)); */
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
/* #if PRE == 0 */
    communicate_live();
/* #endif */
}

};
