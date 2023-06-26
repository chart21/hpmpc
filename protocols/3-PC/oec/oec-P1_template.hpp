#pragma once
#include "../sharemind/sharemind_base.hpp"
#define SHARE DATATYPE
class OEC1
{
bool optimized_sharing;
public:
OEC1(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

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
   return XOR(a,b);
}



//prepare AND -> send real value a&b to other P
void prepare_and(DATATYPE a, DATATYPE b, DATATYPE &c)
{
DATATYPE rl = getRandomVal(P0);
DATATYPE rr = getRandomVal(P0);
DATATYPE rx = getRandomVal(P0);
c = XOR(rx , XOR(AND(a,rl), AND(b,rr)));
send_to_live(P2,c);
}

// NAND both real Values to receive sharing of ~ (a&b) 
void complete_and(DATATYPE &c)
{
c = XOR(c, receive_from_live(P2));
}

void prepare_reveal_to_all(DATATYPE a)
{
return;
}    



DATATYPE complete_Reveal(DATATYPE a)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1) 
    a = XOR(a, pre_receive_from_live(P0));
#else 
    a = XOR(a,receive_from_live(P0));
#endif
return a;
}


XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


void prepare_receive_from(DATATYPE a[], int id, int l)
{
if(id == P1)
{
for(int i = 0; i < l; i++)
{
    a[i] = get_input_live();
    a[i] = XOR(a[i],getRandomVal(P0));
    send_to_live(P2,a[i]);
}
}
}

void complete_receive_from(DATATYPE a[], int id, int l)
{
if(id == P0)
{
    #if OPT_SHARE == 1
        for(int i = 0; i < l; i++)
            a[i] = SET_ALL_ZERO();
    #else
        for(int i = 0; i < l; i++)
        {
        #if PRE == 1 && SHARE_PREP == 1
        a[i] = pre_receive_from_live(P0);
#else
        a[i] = receive_from_live(P0);
        #endif
        }
    #endif
}

else if(id == P2)
{
for(int i = 0; i < l; i++)
{
    a[i] = receive_from_live(P2);
}


}

/* int offset = {id > player_id ? 1 : 0}; */
/* int player = id - offset; */
/* for(int i = 0; i < l; i++) */
/* { */
/* a[i] = receiving_args[player].received_elements[rounds-1][share_buffer[player]]; */
/* share_buffer[player] +=1; */
/* } */
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
