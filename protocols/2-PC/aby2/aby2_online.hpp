#pragma once
#include "../../beaver_triples.hpp"
template <typename Datatype>
class ABY2_ONLINE_Share{
using BT = triple<Datatype>;
private:
Datatype m;
Datatype l;
    public:
ABY2_ONLINE_Share()  {}
ABY2_ONLINE_Share(Datatype x) { this->m = x; }
ABY2_ONLINE_Share(Datatype x, Datatype l) { this->m = x; this->l = l; }



template <typename func_mul>
ABY2_ONLINE_Share mult_public(const Datatype b, func_mul MULT) const
{
    return ABY2_ONLINE_Share(MULT(m,b),MULT(l,b));
}

template <typename func_mul, typename func_add, typename func_sub, typename func_trunc>
ABY2_ONLINE_Share prepare_mult_public_fixed(const Datatype b, func_mul MULT, func_add ADD, func_sub SUB, func_trunc TRUNC, int fractional_bits = FRACTIONAL) const
{
ABY2_ONLINE_Share c;
#if PARTY == 0
c.m = TRUNC(MULT(SUB(m,l),b),fractional_bits); // Share Trunc(mv1 * b)
#else
c.m = SUB(SET_ALL_ZERO(), TRUNC(MULT(l,b),fractional_bits)); // Share Trunc -(lv1 * b)
#endif
c.l = getRandomVal(PSELF);
c.m = ADD(c.m,c.l);
send_to_live(PNEXT, c.m);
return c;
}
    
    template <typename func_add, typename func_sub>
void complete_public_mult_fixed(func_add ADD, func_sub SUB)
{
    Datatype msg = receive_from_live(PNEXT);
    m = ADD(m,msg); //recv Trunc(mv1 * b) - TRunc(lv1 * b)
}



// P_i shares mx - lxi, P_j sets lxj to 0
template <int id, typename func_add, typename func_sub>
void prepare_receive_from(Datatype val, func_add ADD, func_sub SUB)
{
    if constexpr(id == PSELF)
    {
        l = getRandomVal(PSELF);
        m = ADD(val,l);
        send_to_live(PNEXT, m);
    }
    else
    {
        l = SET_ALL_ZERO();
    }
}


template <int id, typename func_add, typename func_sub>
void complete_receive_from(func_add ADD, func_sub SUB)
{
    if constexpr(id != PSELF)
        m = receive_from_live(id);
}

template <typename func_add>
ABY2_ONLINE_Share Add( ABY2_ONLINE_Share b, func_add ADD) const
{
    return ABY2_ONLINE_Share(ADD(m,b.m),ADD(l,b.l));
}

void prepare_reveal_to_all() const
{
}    


template <typename func_add, typename func_sub>
Datatype complete_Reveal(func_add ADD, func_sub SUB) const
{
    return SUB(m,ADD(retrieve_output_share(),l));
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_mult(ABY2_ONLINE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype lxly;
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
   lxly = retrieve_output_share_bool();
else
   lxly = retrieve_output_share_arithmetic();
ABY2_ONLINE_Share c;
c.l = getRandomVal(PSELF);
#if PARTY == 0
c.m = MULT(m,b.m);
#else
c.m = SET_ALL_ZERO();
#endif
c.m = SUB(c.m, SUB(ADD(MULT(m,b.l),MULT(l,b.m)),ADD(lxly,c.l))); // mx my - (mx[ly] + my[lx] - [lxly] - [lz])
send_to_live(PNEXT, c.m);
return c;
}

template <typename func_add, typename func_sub, typename func_mul>
    ABY2_ONLINE_Share prepare_dot(ABY2_ONLINE_Share b, func_add ADD, func_sub SUB, func_mul MULT) const
{
Datatype lxly;
if constexpr(std::is_same_v<func_add(), FUNC_XOR>)
   lxly = retrieve_output_share_bool();
else
   lxly = retrieve_output_share_arithmetic();
ABY2_ONLINE_Share c;
/* c.l = getRandomVal(PSELF); */
#if PARTY == 0
c.m = MULT(m,b.m);
#else
c.m = SET_ALL_ZERO();
#endif
c.m = SUB(c.m, SUB(ADD(MULT(m,b.l),MULT(l,b.m)),lxly)); // mx my - (mx[ly] + my[lx] - [lxly])
return c;
}

/* template <typename func_add, typename func_sub, typename func_mul, typename func_trunc> */
/*     ABY2_ONLINE_Share prepare_dot_with_trunc(ABY2_ONLINE_Share b, func_add ADD, func_sub SUB, func_mul MULT, func_trunc TRUNC) const */
/* { */
/* Datatype lxly; */
/* if constexpr(std::is_same_v<func_add(), FUNC_XOR>) */
/*    lxly = retrieve_output_share_bool(); */
/* else */
/*    lxly = retrieve_output_share_arithmetic(); */
/* ABY2_ONLINE_Share c; */
/* c.l = getRandomVal(PSELF); */
/* #if PARTY == 0 */
/* c.m = MULT(m,b.m); */
/* #else */
/* c.m = SET_ALL_ZERO(); */
/* #endif */
/* c.m = SUB(c.m, SUB(ADD(MULT(m,b.l),MULT(l,b.m)),lxly)); // mx my - (mx[ly] + my[lx] - [lxly]) */
/* c.m = ADD(TRUNC(c.m),c.l); // [c] = [c]^t + [lz] */
/* return c; */
/* } */

template <typename func_add, typename func_sub>
void mask_and_send_dot(func_add ADD, func_sub SUB)
{
    l = getRandomVal(PSELF);
    m = ADD(m,l);
    send_to_live(PNEXT, m);
}
    
template <typename func_add, typename func_sub, typename func_trunc>
void mask_and_send_dot_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
    l = getRandomVal(PSELF);
#if PARTY == 0
    /* m = ADD(TRUNC(m),l); */
    m = ADD(OP_SUB(SET_ALL_ZERO(),TRUNC(OP_SUB(SET_ALL_ZERO(),m))),l); // whyever this is necessary ...
#else
    m = ADD(TRUNC(m),l);
#endif
    send_to_live(PNEXT, m);
}
    
    
    template <typename func_add, typename func_sub>
void complete_mult(func_add ADD, func_sub SUB)
{
    Datatype msg = receive_from_live(PNEXT);
    m = ADD(m,msg);
}
    template <typename func_add, typename func_sub, typename func_trunc>
void complete_mult_with_trunc(func_add ADD, func_sub SUB, func_trunc TRUNC)
{
    complete_mult(ADD,SUB);
}

static void prepare_A2B_S1(int m, int k, ABY2_ONLINE_Share in[], ABY2_ONLINE_Share out[])
{
#if PARTY == 0
    Datatype temp_p1[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp_p1[i] = OP_SUB(in[i].m,in[i].l) ; // set first share to mv - lv1
    }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp_p1, temp2);
    orthogonalize_boolean(temp2, temp_p1);
    
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
        out[i-m].m = FUNC_XOR(temp_p1[i],out[i-m].l);
        send_to_live(PNEXT, out[i-m].m);
    }
#endif
}

static void prepare_A2B_S2(int m, int k, ABY2_ONLINE_Share in[], ABY2_ONLINE_Share out[])
{
#if PARTY == 1
    Datatype temp_p1[BITLENGTH];
    for(int i = 0; i < BITLENGTH; i++)
    {
        temp_p1[i] = OP_SUB(SET_ALL_ZERO(),in[i].l) ; // set second share to -lv2
    }
    alignas(sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_arithmetic(temp_p1, temp2);
    orthogonalize_boolean(temp2, temp_p1);
    
    for(int i = m; i < k; i++)
    {
        out[i-m].l = getRandomVal(PSELF);
        out[i-m].m = FUNC_XOR(temp_p1[i],out[i-m].l);
        send_to_live(PNEXT, out[i-m].m);
    }
#endif
}

static void complete_A2B_S1(int k, ABY2_ONLINE_Share out[])
{
#if PARTY == 1
    for(int i = 0; i < k; i++)
    {
        out[i].m = receive_from_live(PNEXT);
        out[i].l = SET_ALL_ZERO();
    }
#endif
}

static void complete_A2B_S2(int k, ABY2_ONLINE_Share out[])
{
#if PARTY == 0
    for(int i = 0; i < k; i++)
    {
        out[i].m = receive_from_live(PNEXT);
        out[i].l = SET_ALL_ZERO();
    }
#endif
}

void prepare_bit2a(ABY2_ONLINE_Share out[])
{
    Datatype b0[BITLENGTH]{0};
    b0[BITLENGTH - 1] = m; //convert b0 to an arithemtic value
    alignas (sizeof(Datatype)) UINT_TYPE temp2[DATTYPE];
    unorthogonalize_boolean(b0, temp2);
    orthogonalize_arithmetic(temp2, b0);
    Datatype lb[BITLENGTH]{0};
    lb[BITLENGTH - 1] = l;
    unorthogonalize_boolean(lb, temp2);
    orthogonalize_arithmetic(temp2, lb);
    for(int i = 0; i < BITLENGTH; i++)
    {
        Datatype lxly = retrieve_output_share_arithmetic();
#if PARTY == 0
        out[i].m = b0[i];
#else
        out[i].m = SET_ALL_ZERO();
#endif
        out[i].m = OP_SUB(out[i].m, OP_MULT( OP_SUB(OP_ADD(b0[i], b0[i]),PROMOTE(1)), OP_ADD(lb[i], lxly))); // mb -  (2mb - 1) * (lvi + lxly)
        out[i].l = getRandomVal(PSELF); 
        out[i].m = OP_ADD(out[i].m, out[i].l);
        send_to_live(PNEXT, out[i].m);
    }
}

void complete_bit2a()
{
        m = OP_ADD(m, receive_from_live(PNEXT));
}


static ABY2_ONLINE_Share public_val(Datatype a)
{
    return ABY2_ONLINE_Share(a,SET_ALL_ZERO());
}

ABY2_ONLINE_Share Not() const
{
    return ABY2_ONLINE_Share(NOT(m),l);
}


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

