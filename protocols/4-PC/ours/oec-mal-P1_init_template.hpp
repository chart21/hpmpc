#pragma once
#include "../../generic_share.hpp"
class OEC_MAL1_init
{
bool optimized_sharing;
public:
OEC_MAL1_init(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}


XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return a;
}

// Receive sharing of ~XOR(a,b) locally
DATATYPE Xor(DATATYPE a, DATATYPE b)
{
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(DATATYPE a, DATATYPE b, DATATYPE &c)
{
send_to_(P2);
#if PROTOCOL == 10 || PROTOCOL == 12
store_compare_view_init(P0); // compare a1b1 + r123_2 with P0
#endif

//return u[player_id] * v[player_id];
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(DATATYPE &c)
{
    receive_from_(P2);
#if PROTOCOL == 10 || PROTOCOL == 12 || PROTOCOL == 8
store_compare_view_init(P012);
#endif
#if PROTOCOL == 11 || PROTOCOL == 8
store_compare_view_init(P0);
#endif
#if PROTOCOL == 11
store_compare_view_init(P0);
#endif
}
#if FUNCTION_IDENTIFIER > 4
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c)
{
    prepare_and(a,b,c);
}

void complete_mult(DATATYPE &c)
{
    complete_and(c);
}
#endif

void prepare_reveal_to_all(DATATYPE a)
{
return;
}    



DATATYPE complete_Reveal(DATATYPE a)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
#if PRE == 1 
    pre_receive_from_(P3);
#else
    receive_from_(P3);
#endif
#if PROTOCOL == 8
store_compare_view_init(P0);
store_compare_view_init(P3);

store_compare_view_init(P0);
#else
    store_compare_view_init(P0123);
#endif
    return a;
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


void prepare_receive_from(DATATYPE a[], int id, int l)
{
if(id == PSELF)
{
    for(int i = 0; i < l; i++)
    {
        send_to_(P0);
        send_to_(P2);
    }

}
}


void complete_receive_from(DATATYPE a[], int id, int l)
{
    if(id != PSELF)
    {
        for(int i = 0; i < l; i++)
            receive_from_(id);
        if(id != P0)
            for(int i = 0; i < l; i++)
                store_compare_view_init(P0);
        if(id != P2)
            for(int i = 0; i < l; i++)
                store_compare_view_init(P2);
    }
}
/* if(id == player_id) */
/*     return; */
/* int offset = {id > player_id ? 1 : 0}; */
/* int player = id - offset; */
/* for(int i = 0; i < l; i++) */
/*     receiving_args[player].elements_to_rec[receiving_args[player].rec_rounds -1] += 1; */


void send()
{
send_();
}
void receive()
{
    receive_();
}
void communicate()
{
communicate_();
}

void finalize(std::string* ips)
{
    finalize_(ips);
}

void finalize(std::string* ips, receiver_args* ra, sender_args* sa)
{
    finalize_(ips, ra, sa);
}


};
