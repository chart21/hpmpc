#pragma once
#include "../sharemind/sharemind_base.hpp"
#define PRE_SHARE DATATYPE
#define SHARE DATATYPE
class OEC0
{
bool optimized_sharing;
public:
OEC0(bool optimized_sharing) {this->optimized_sharing = optimized_sharing;}

XOR_Share public_val(DATATYPE a)
{
    a = SET_ALL_ZERO();
    return a;
}

DATATYPE Not(DATATYPE a)
{
   return NOT(a);
}

template <typename func_add>
DATATYPE Add(DATATYPE a, DATATYPE b, func_add ADD)
{
   return ADD(a,b);
}


template <typename func_add, typename func_sub, typename func_mul>
void prepare_mult(DATATYPE a, DATATYPE b, DATATYPE &c, func_add ADD, func_sub SUB, func_mul MUL)
{
DATATYPE rl = getRandomVal(P1);
DATATYPE rr = getRandomVal(P1);

DATATYPE rx = getRandomVal(P1);
DATATYPE ry = getRandomVal(P2);

DATATYPE o1 = ADD(a,rr);
DATATYPE o2 = ADD(b,rl);
#if PRE == 1
    pre_send_to_live(P2,o1);
    pre_send_to_live(P2,o2);
#else
    send_to_live(P2,o1);
    send_to_live(P2,o2);
#endif
/* a = AND(a,rl); */
/* b = XOR(AND(b,rr),AND(rl,rr)); */
/* a = XOR(a,b); */
/* b = XOR(rx,ry); */
/* return XOR(a,b); */

c = XOR( XOR (rx, ry), XOR ( AND(a,rl), XOR(AND(b,rr),AND(rl,rr))   ) );

}

template <typename func_add, typename func_sub>
void complete_mult(DATATYPE &c, func_add ADD, func_sub SUB)
{
}


void prepare_reveal_to_all(DATATYPE a)
{
#if PRE == 1 && (OPT_SHARE == 0 || SHARE_PREP == 1)
    pre_send_to_live(P1,a);
    pre_send_to_live(P2,a);
#else
    send_to_live(P1,a);
    send_to_live(P2,a);
#endif
}    


template <typename func_add, typename func_sub>
DATATYPE complete_Reveal(DATATYPE a, func_add ADD, func_sub SUB)
{
#if PRE == 0
a = XOR(a, receive_from_live(P2));
rb+=1;
#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
store_output_share(a);
#endif


return a;
}



void communicate()
{
#if PRE == 0
communicate_live();
#else
    // sb = 0; //TODO replace with individual index for each player soon
#endif
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


#if OPT_SHARE == 0
void share_unoptimized(DATATYPE a[], int id, int l)
{

if(id == P0)
{
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P0 keeps it inputs, the other parties hold share=0
    a[i] = get_input_live();
        /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    DATATYPE r = getRandomVal(P0); //should be an SRNG shared by P0,P1,P2 to save communication
    a[i] = XOR(r,a[i]);
    send_to_live(P1,a[i]);
    send_to_live(P2,a[i]);
    a[i] = r;
    }

}
}
#endif

template <typename func_add, typename func_sub>
void prepare_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
if(id == P0)
{
    #if OPT_SHARE == 0
    share_unoptimized(a, id, l);
    #else
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P0 keeps it inputs, the other parties hold share=0
    a[i] = get_input_live();
        /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    /* DATATYPE r = getRandomVal(2); //should be an SRNG shared by P0,P1,P2 to save communication */
    /* a[i] = XOR(r,a[i]); */
    /* sending_args[0].sent_elements[sending_rounds][sb] = SET_ALL_ZERO(); */
    /* sending_args[1].sent_elements[sending_rounds][sb] = SET_ALL_ZERO(); */
    /* sb += 1; */
    /* a[i] = r; */
    }
#endif 
}
else{
for(int i = 0; i < l; i++)
    {
    a[i] = getRandomVal(id);
    }
}
}

template <typename func_add, typename func_sub>
void complete_receive_from(DATATYPE a[], int id, int l, func_add ADD, func_sub SUB)
{
    return;
}



void finalize(std::string* ips)
{
for(int t=0;t<(num_players-1);t++) {
    int offset = 0;
    if(t >= player_id)
        offset = 1; // player should not receive from itself
    receiving_args[t].player_count = num_players;
    receiving_args[t].received_elements = new DATATYPE*[receiving_args[t].rec_rounds]; //every thread gets its own pointer array for receiving elements
    
    /* receiving_args[t].elements_to_rec = new int[total_comm]; */
    /* for (int i = 1 - use_srng_for_inputs; i < total_comm -1; i++) { */
    /* receiving_args[t].elements_to_rec[i] = elements_per_round[i]; */
    /* } */
    /* receiving_args[t].elements_to_rec[0] = 0; // input sharing with SRNG */ 
    /* if(use_srng_for_inputs == 0) */
    /*     receiving_args[t].elements_to_rec[0] = input_length[t+offset]; //input shares to receive from that player */
    /* receiving_args[t].elements_to_rec[total_comm-1] = reveal_length[player_id]; //number of revealed values to receive from other players */
    receiving_args[t].player_id = player_id;
    receiving_args[t].connected_to = t+offset;
    receiving_args[t].ip = ips[t];
    receiving_args[t].hostname = (char*)"hostname";
    receiving_args[t].port = (int) base_port + player_id * (num_players-1) + t; //e.g. P0 receives on base port from P1, P2 on base port + num_players from P0 6000,6002
    /* std::cout << "In main: creating thread " << t << "\n"; */
    //init_srng(t, (t+offset) + player_id); 
}
for(int t=0;t<(num_players-1);t++) {
    int offset = 0;
    if(t >= player_id)
        offset = 1; // player should not send to itself
    sending_args[t].sent_elements = new DATATYPE*[sending_args[t].send_rounds];
    /* sending_args[t].elements_to_send[0] = 0; //input sharing with SRNGs */ 
    sending_args[t].player_id = player_id;
    sending_args[t].player_count = num_players;
    sending_args[t].connected_to = t+offset;
    sending_args[t].port = (int) base_port + (t+offset) * (num_players -1) + player_id - 1 + offset; //e.g. P0 sends on base port + num_players  for P1, P2 on base port + num_players for P0 (6001,6000)
    /* std::cout << "In main: creating thread " << t << "\n"; */
    sending_args[t].sent_elements[0] = NEW(DATATYPE[sending_args[t].elements_to_send[0]]); // Allocate memory for first round
       
}
rounds = 0;
sending_rounds = 0;
rb = 0;
sb = 0;
}


void send()
{
    send_live();
}

void receive()
{
    receive_live();
}


};
