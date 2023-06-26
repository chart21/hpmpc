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

// Receive sharing of ~XOR(a,b) locally
DATATYPE Xor(DATATYPE a, DATATYPE b)
{
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(DATATYPE a, DATATYPE b, DATATYPE &c)
{
#if PRE == 1
pre_receive_from_(P0);
pre_receive_from_(P0);
#else
receive_from_(P0);
receive_from_(P0);
#endif
send_to_(P1);
//return u[player_id] * v[player_id];
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(DATATYPE &c)
{
receive_from_(P1);
}

void prepare_reveal_to_all(DATATYPE a)
{
send_to_(P0);
}    



DATATYPE complete_Reveal(DATATYPE a)
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


void prepare_receive_from(DATATYPE a[], int id, int l)
{
if(id == P2)
{
    for(int i = 0; i < l; i++)
    {
        send_to_(P1);
    }

}
}


void complete_receive_from(DATATYPE a[], int id, int l)
{
if(id == P1)
{
for(int i = 0; i < l; i++)
    receive_from_(P1);
}
else if(id == P0 && optimized_sharing == false)
{
for(int i = 0; i < l; i++)
#if SHARE_PREP == 1 && PRE == 1
    pre_receive_from_(P0);
#else
    receive_from_(P0);
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
