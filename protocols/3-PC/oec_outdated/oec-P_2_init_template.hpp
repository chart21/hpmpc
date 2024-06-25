#pragma once
#include "../sharemind/sharemind_base.hpp"
class OEC2_init
{
public:
bool optimized_sharing;
public:
OEC2_init(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}



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
pre_receive_from_(P_0);
pre_receive_from_(P_0);
#else
receive_from_(P_0);
receive_from_(P_0);
#endif
send_to_(P_1);
//return u[player_id] * v[player_id];
}

template <typename func_add, typename func_sub>
void complete_mult(DATATYPE &c, func_add ADD, func_sub SUB)
{
receive_from_(P_1);
}

void prepare_reveal_to_all(DATATYPE a)
{
send_to_(P_0);
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) // OPT_SHARE is input dependent, can only be sent in prepocessing phase if allowed
    pre_receive_from_(P_0);
#else
    receive_from_(P_0);
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
if(id == P_2)
{
    for(int i = 0; i < l; i++)
    {
        send_to_(P_1);
    }

}
}

template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P_1)
{
for(int i = 0; i < l; i++)
    receive_from_(P_1);
}
else if(id == P_0 && optimized_sharing == false)
{
for(int i = 0; i < l; i++)
#if SHARE_PREP == 1 && PRE == 1
    pre_receive_from_(P_0);
#else
    receive_from_(P_0);
#endif
}
/* if(id == player_id) */
/*     return; */
/* for(int i = 0; i < l; i++) */
/*     receiving_args[id].elements_to_rec[receiving_args[id].rec_rounds -1] += 1; */
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
