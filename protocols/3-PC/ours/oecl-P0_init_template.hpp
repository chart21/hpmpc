#pragma once
#include "../../generic_share.hpp"
class OECL0_init
{
bool optimized_sharing;
public:
OECL0_init(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}


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
    pre_send_to_(P2);
#else
send_to_(P2);
#endif
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(DATATYPE &c)
{
}

void prepare_reveal_to_all(DATATYPE a)
{
    for(int t = 0; t < 2; t++) 
    {
        #if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
    pre_send_to_(t);
#else
    send_to_(t);
#endif

    }//add to send buffer
}    



DATATYPE complete_Reveal(DATATYPE a)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
receive_from_(P2);
return a;
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


void prepare_receive_from(DATATYPE a[], int id, int l)
{
/* return; */
/* old: */

if(id == P0)
{
#if OPT_SHARE == 1
{
    for(int i = 0; i < l; i++)
    {
        #if PRE == 1 && SHARE_PREP == 1
            pre_send_to_(P2);
        #else
            send_to_(P2);
        #endif
    }

}
#else
{
    for(int i = 0; i < l; i++)
    {
        #if PRE == 1
        pre_send_to_(P1);
        pre_send_to_(P2);
#else
        send_to_(P1);
        send_to_(P2);
#endif
    }

}
#endif
}
}

void complete_receive_from(DATATYPE a[], int id, int l)
{
    return;
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
