#pragma once
#include "../sharemind/sharemind_base.hpp"
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

// Receive sharing of ~XOR(a,b) locally
DATATYPE Xor(DATATYPE a, DATATYPE b)
{
   return XOR(a,b);
}



//prepare AND -> send real value a&b to other P
void prepare_and(DATATYPE &a, DATATYPE &b)
{
DATATYPE rl = getRandomVal(0);
DATATYPE rr = getRandomVal(0);

DATATYPE rx = getRandomVal(0);
DATATYPE ry = getRandomVal(1);

DATATYPE o1 = XOR(a,rr);
DATATYPE o2 = XOR(b,rl);
sending_args[1].sent_elements[sending_rounds][sb] = o1; //P_0 should only have a link to P_2
sending_args[1].sent_elements[sending_rounds][sb+1] = o2; //P_0 should only have a link to P_2
sb+=2;
a = AND(a,rl);
b = XOR(AND(b,rr),AND(rl,rr));
a = XOR(a,b);
b = XOR(rx,ry);


}

DATATYPE complete_and(DATATYPE a, DATATYPE b)
{
return XOR(a,b);
}

void prepare_reveal_to_all(DATATYPE a)
{
    sending_args[0].sent_elements[sending_rounds][sb] = a;
    sending_args[1].sent_elements[sending_rounds][sb] = a;
    sb += 1;
}    



DATATYPE complete_Reveal(DATATYPE a)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
/* a = XOR(a,receiving_args[1].received_elements[rounds-1][rb]); */ 
/* rb+=1; */
return a;
}

void send()
{
sb = 0;      
    for(int t = 0; t < num_players-1; t++)
        sending_args[t].sent_elements[sending_rounds + 1] = NEW(DATATYPE[sending_args[t].elements_to_send[sending_rounds + 1]]); // Allocate memory for all sending buffers for next round
    pthread_mutex_lock(&mtx_send_next); 
     sending_rounds +=1;
      pthread_cond_broadcast(&cond_send_next); //signal all threads that sending buffer contains next data
      /* printf("boradcasted round %i \n", sending_rounds); */
      pthread_mutex_unlock(&mtx_send_next); 
}

void receive(){
        rounds+=1;  
        // receive_data
      //wait until all sockets have finished received their last data
      pthread_mutex_lock(&mtx_receive_next);
      
/* std::chrono::high_resolution_clock::time_point c1 = */
/*         std::chrono::high_resolution_clock::now(); */
      while(rounds > receiving_rounds) //wait until all threads received their data
          pthread_cond_wait(&cond_receive_next, &mtx_receive_next);
      
/* double time = std::chrono::duration_cast<std::chrono::microseconds>( */
/*                      std::chrono::high_resolution_clock::now() - c1) */
/*                      .count(); */
      /* printf("finished waiting for receive in round %i \n", rounds - 1); */
      pthread_mutex_unlock(&mtx_receive_next);
/* printf("Time spent waiting for data chrono: %fs \n", time / 1000000); */

rb = 0;
}

void communicate()
{
    /* send(); */
    /* receive(); */
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}



void share_unoptimized(DATATYPE a[], int id, int l)
{

if(id == 0)
{
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P_0 keeps it inputs, the other parties hold share=0
    a[i] = player_input[share_buffer[2]];
    share_buffer[2] += 1;
    /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    DATATYPE r = getRandomVal(2); //should be an SRNG shared by P_0,P_1,P_2 to save communication
    a[i] = XOR(r,a[i]);
    sending_args[0].sent_elements[sending_rounds][sb] = a[i];
    sending_args[1].sent_elements[sending_rounds][sb] = a[i];
    sb += 1;
    a[i] = r;
    }

}
}

void prepare_receive_from(DATATYPE a[], int id, int l)
{
if(id == 0)
{
    if(optimized_sharing == false)
    {
        share_unoptimized(a, id, l);
    }
    else
    {
    for(int i = 0; i < l; i++)
    {
    //special sharing technique, P_0 keeps it inputs, the other parties hold share=0
    a[i] = player_input[share_buffer[2]];
    share_buffer[2] += 1;
    /* sending_args[0].sent_elements[sending_rounds][sb] = 0; */
    /* sending_args[1].sent_elements[sending_rounds][sb] = 0; */
    /* sb += 1; */
    /* DATATYPE r = getRandomVal(2); //should be an SRNG shared by P_0,P_1,P_2 to save communication */
    /* a[i] = XOR(r,a[i]); */
    /* sending_args[0].sent_elements[sending_rounds][sb] = SET_ALL_ZERO(); */
    /* sending_args[1].sent_elements[sending_rounds][sb] = SET_ALL_ZERO(); */
    /* sb += 1; */
    /* a[i] = r; */
    }
    

}
}
else{
for(int i = 0; i < l; i++)
    {
    a[i] = getRandomVal(id - 1);
    }
}
}

void complete_receive_from(DATATYPE a[], int id, int l)
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
    receiving_args[t].port = (int) base_port + player_id * (num_players-1) + t; //e.g. P_0 receives on base port from P_1, P_2 on base port + num_players from P_0 6000,6002
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
    sending_args[t].port = (int) base_port + (t+offset) * (num_players -1) + player_id - 1 + offset; //e.g. P_0 sends on base port + num_players  for P_1, P_2 on base port + num_players for P_0 (6001,6000)
    /* std::cout << "In main: creating thread " << t << "\n"; */
    sending_args[t].sent_elements[0] = NEW(DATATYPE[sending_args[t].elements_to_send[0]]); // Allocate memory for first round
       
}
rounds = 0;
sending_rounds = 0;
rb = 0;
sb = 0;
}
};
