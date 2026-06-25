#pragma once
#include "../include/pch.h"
#include "sockethelper.h"
int player_id;
sender_args sending_args[num_players];
receiver_args receiving_args[num_players];

#if PRE == 1
sender_args sending_args_pre[num_players];
receiver_args receiving_args_pre[num_players];
#endif

uint64_t total_send[num_players - 1] = {0};
uint64_t total_recv[num_players - 1] = {0};
#if PRE == 1
uint64_t total_send_pre[num_players - 1] = {0};
uint64_t total_recv_pre[num_players - 1] = {0};
#endif

#if FUSE_RELU_AVG == 1
int curr_denom = 1;
#endif

int rounds;
int rb;
int sb;
int send_count[num_players] = {0};
int share_buffer[num_players] = {0};  // TODO: move to protocol layer
int send_count_pre[num_players] = {0};
int share_buffer_pre[num_players] = {0};  // TODO: move to protocol layer
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
DATATYPE* verify_buffer[num_players * player_multiplier];  // Verify buffer for each player
uint64_t verify_buffer_index[num_players * player_multiplier] = {0};

#if DATTTYPE > 32
alignas(sizeof(DATATYPE)) uint32_t hash_val[num_players * player_multiplier][8];  // Hash value for each player
#else
uint32_t hash_val[num_players * player_multiplier][8];  // Hash value for each player
#endif
uint64_t elements_to_compare[num_players * player_multiplier] = {0};
#endif
#if (PRE == 1 && HAS_POST_PROTOCOL == 1) || \
    BEAVER == 1  // Store preprocessed-output to get the correct results during post-processing

#if BEAVER == 1 && PRE == 1
DATATYPE** preprocessed_outputs_bool = nullptr;
DATATYPE** preprocessed_outputs_arithmetic = nullptr;
uint64_t* preprocessed_outputs_bool_index = nullptr;
uint64_t* preprocessed_outputs_bool_input_index = nullptr;
uint64_t* preprocessed_outputs_arithmetic_input_index = nullptr;
uint64_t* preprocessed_outputs_arithmetic_index = nullptr;
#endif
DATATYPE* preprocessed_outputs = nullptr;
uint64_t preprocessed_outputs_input_index = 0;
uint64_t preprocessed_outputs_index = 0;
uint64_t total_preprocessed_outputs = 0;
#if A2B_ONLINE_OPT == 1
// Dedicated output-share buffer for the A2B MSB adder. The adder is deferred in PRE past the
// boolean-addition step (so its S2 operand can use the boolean-addition result c instead of bits(-l_i)),
// then run in a batch. Its zero_add output-shares live here (NOT the shared default buffer) so the
// batched fill order matches the online's forward-pass read order. Only used when A2B_ONLINE_OPT==1.
DATATYPE* preprocessed_outputs_a2b = nullptr;
uint64_t preprocessed_outputs_a2b_input_index = 0;   // PRE write cursor (batch)
uint64_t preprocessed_outputs_a2b_index = 0;         // online read cursor
uint64_t total_preprocessed_outputs_a2b = 0;
bool g_a2b_adder_active = false;                       // route zero_add output-shares to the dedicated buffer
#include <functional>
#include <vector>
std::vector<std::function<void()>> g_deferred_a2b_circuits;  // boolean circuits deferred past the boolean addition
uint64_t g_a2b_c_consume_index = 0;   // cursor into boolean_addition_triple_c used to set S2.l=c in the batch
uint64_t g_a2b_zero_add_count = 0;    // # of adder zero_adds (counted in the batch) -> dedicated buffer size

// dedicated-buffer accessors mirroring retrieve_output_share / store_output_share
inline DATATYPE retrieve_output_share_a2b()
{
    preprocessed_outputs_a2b_index += 1;
    return preprocessed_outputs_a2b[preprocessed_outputs_a2b_index - 1];
}
inline void store_output_share_a2b(DATATYPE val)
{
    preprocessed_outputs_a2b[preprocessed_outputs_a2b_input_index] = val;
    preprocessed_outputs_a2b_input_index += 1;
}
#endif
uint64_t send_in_last_round[num_players - 1] = {0};
#endif
uint64_t num_generated[num_players * player_multiplier] = {0};

int use_srng_for_inputs = 1;

int current_phase = 0;   // Keeping track of current pahse
int process_offset = 0;  // offsets the starting input for each process, base port must be multiple of 1000 to work

#if TRUNC_DELAYED == 1
bool delayed = false;  // For delayed truncation
bool isReLU = false;   // For ReLU truncation
#endif

#if TRUNC_APPROACH > 0
bool all_positive = false;  // for slack-based optiimzation
#endif

