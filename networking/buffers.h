#pragma once
#include <chrono>
#include "sockethelper.h"
int player_id;
sender_args sending_args[num_players];
receiver_args receiving_args[num_players];

#if PRE == 1
sender_args sending_args_pre[num_players];
receiver_args receiving_args_pre[num_players];
#endif

int rounds;
int rb;
int sb;
int send_count[num_players] = {0};
int share_buffer[num_players] = {0}; //TODO: move to protocol layer
int send_count_pre[num_players] = {0};
int share_buffer_pre[num_players] = {0}; //TODO: move to protocol layer
int reveal_buffer[num_players] = {0};
int total_comm;
int* elements_per_round;
int input_length[num_players] = {0};
int reveal_length[num_players] = {0};
DATATYPE* player_input;

#if num_players == 4
    #define player_multiplier 2
#else
    #define player_multiplier 1
#endif
#if MAL == 1
DATATYPE* verify_buffer[num_players*player_multiplier]; // Verify buffer for each player
uint64_t verify_buffer_index[num_players*player_multiplier] = {0};

alignas(sizeof(DATATYPE)) uint32_t hash_val[num_players*player_multiplier][8]; // Hash value for each player
uint64_t elements_to_compare[num_players*player_multiplier] = {0};
#endif
//DATATYPE srng[num_players -1] = {0};
//DATATYPE* input_seed;
#if PRE == 1 && HAS_POST_PROTOCOL == 1 // Store preprocessed-output to get the correct results during post-processing
DATATYPE* preprocessed_outputs;
uint64_t preprocessed_outputs_index = 0;
#endif
int num_generated[num_players*player_multiplier] = {0};
int pnext;
int pprev;
int pmiddle;

int use_srng_for_inputs = 1;

int current_phase = 0; //Keeping track of current pahse
