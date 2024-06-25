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
   return a;
}



//prepare AND -> send real value a&b to other P
void prepare_and(DATATYPE &a, DATATYPE &b)
{
DATATYPE rl = getRandomVal(0);
DATATYPE rr = getRandomVal(0);
DATATYPE rx = getRandomVal(0);
a = XOR(rx , XOR(AND(a,rl), AND(b,rr)));
sending_args[1].sent_elements[sending_rounds][sb] = a; 
sb+=1;

}

// NAND both real Values to receive sharing of ~ (a&b) 
DATATYPE complete_and(DATATYPE a, DATATYPE b)
{
b = receiving_args[1].received_elements[rounds-1][rb];
rb+=1;
return XOR(a, b); 
}

void prepare_reveal_to_all(DATATYPE a)
{
return;
}    



DATATYPE complete_Reveal(DATATYPE a)
{
/* for(int t = 0; t < num_players-1; t++) */ 
/*     receiving_args[t].elements_to_rec[rounds-1]+=1; */
a = XOR(a,receiving_args[0].received_elements[rounds-1][rb]); 
rb+=1;
return a;
}

void send()
{
for (int t = 0; t < 2; t++)
{
    sending_args[t].total_rounds += 1;
    sending_args[t].send_rounds += 1;
}
}

void receive()
{
for (int t = 0; t < 2; t++)
{
    receiving_args[t].total_rounds += 1;
    receiving_args[t].rec_rounds +=1;
}
}


void communicate()
{
    send_and_receive();
}

XOR_Share* alloc_Share(int l)
{
    return new DATATYPE[l];
}


void prepare_receive_from(DATATYPE a[], int id, int l)
{
if(id == 1)
{
for(int i = 0; i < l; i++)
{
    a[i] = player_input[share_buffer[2]];
    share_buffer[2] += 1;
    a[i] = XOR(a[i],getRandomVal(0));
    sending_args[1].sent_elements[sending_rounds][sb] = a[i];
    sb+=1;
}
}
}

void complete_receive_from(DATATYPE a[], int id, int l)
{
if(id == player_id)
    return;
else if(id == 0)
{
    if(optimized_sharing == true)
    {
        for(int i = 0; i < l; i++)
            a[i] = SET_ALL_ZERO();
    }
    else{
        for(int i = 0; i < l; i++)
        {
            a[i] = receiving_args[0].received_elements[rounds-1][share_buffer[0]];
            share_buffer[0] +=1;
        }
    }
}

else if(id == 2)
{
for(int i = 0; i < l; i++)
{
a[i] = receiving_args[1].received_elements[rounds-1][share_buffer[1]];
share_buffer[1] +=1;
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

void send_and_receive()
{
    send();
    receive();
}

};
