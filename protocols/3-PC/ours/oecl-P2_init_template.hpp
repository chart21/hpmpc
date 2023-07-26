#pragma once
#include "../sharemind/sharemind_base.hpp"
class OECL2_init
{
bool optimized_sharing;
public:
OECL2_init(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}



XOR_Share public_val(DATATYPE a)
{
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return a;
}

template <typename func_add>
DATATYPE Add(DATATYPE a, DATATYPE b, func_add ADD)
{
   return a;
}



template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c, func_add ADD, func_sub SUB, func_mul MUL)
{
#if PRE == 1
pre_receive_from_(P0);
#else
receive_from_(P0);
#endif

send_to_(P1);
//return u[player_id] * v[player_id];
}

template <typename func_add, typename func_sub>
void complete_mult(DATATYPE &c, func_add ADD, func_sub SUB)
{
    receive_from_(P1);
}


void prepare_reveal_to_all(DATATYPE a)
{
send_to_(P0);
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
    pre_receive_from_(P0);
#else
    receive_from_(P0);
#endif
return a;
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P2)
{
    for(int i = 0; i < l; i++)
    {
        send_to_(P1);
    }

}
}


template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P1)
{
for(int i = 0; i < l; i++)
    receive_from_(P1);
}
else if(id == P0)
{
for(int i = 0; i < l; i++)
#if (SHARE_PREP == 1 || OPT_SHARE == 0) && PRE == 1
    pre_receive_from_(P0);
#else
    receive_from_(P0);
#endif
} 
/* if(id == player_id) */
/*     return; */
/* int offset = {id > player_id ? 1 : 0}; */
/* int player = id - offset; */
/* for(int i = 0; i < l; i++) */
/*     receiving_args[player].elements_to_rec[receiving_args[player].rec_rounds -1] += 1; */
}




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
