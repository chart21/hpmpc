#pragma once
#include "../core/include/pch.h"
#include "../core/networking/buffers.h"
#include "../core/utils/print.hpp"

void send_()
{
    for (int t = 0; t < num_players - 1; t++)
    {
        /* if(sending_args[t].elements_to_send[sending_args[t].send_rounds] > 0) */
        /* { */
        sending_args[t].total_rounds += 1;
        sending_args[t].send_rounds += 1;
        sending_args[t].elements_to_send.push_back(0);
        sending_args[t].elements_to_send.push_back(0);
        /* } */
    }
}

void receive_()
{
    for (int t = 0; t < num_players - 1; t++)
    {
        /* if(receiving_args[t].elements_to_rec[receiving_args[t].rec_rounds] > 0) */
        /* { */
        receiving_args[t].total_rounds += 1;
        receiving_args[t].rec_rounds += 1;
        receiving_args[t].elements_to_rec.push_back(0);
        /* } */
    }
    sockets_received.push_back(0);
}

#if PRE == 1
void pre_send_to_(int player_index)
{
    sending_args_pre[player_index].elements_to_send[0] += 1;
    total_send_pre[player_index] += 1;
    /* sending_args_pre[player_index].elements_to_send[sending_args_pre[player_index].send_rounds] += 1; */
}

void pre_receive_from_(int player_index)
{
    /* receiving_args_pre[player_index].elements_to_rec[receiving_args_pre[player_index].rec_rounds -1] += 1; */
    receiving_args_pre[player_index].elements_to_rec[0] += 1;
    total_recv_pre[player_index] += 1;
}
#endif

void send_to_(int player_index);
void receive_from_(int player_index);

void send_to_(int player_index)
{
#if SEND_BUFFER > 0
    if (sending_args[player_index].elements_to_send[sending_args[player_index].send_rounds] == SEND_BUFFER)
    {
        send_();
    }
#endif
    sending_args[player_index].elements_to_send[sending_args[player_index].send_rounds] += 1;
    total_send[player_index] += 1;
    #if WAIT_AFTER_MESSAGES_IF_AHEAD >= 0
    if((SYNC_PARTY_RECV2 == player_index || SYNC_PARTY_RECV == player_index) && total_send[player_index] % WAIT_AFTER_MESSAGES_IF_AHEAD == 0)
    {
        receive_(); // receive sync message
        receive_from_(player_index); // expect sync message
    }
    #endif
}

void receive_from_(int player_index)
{
#if RECV_BUFFER > 0
    if (receiving_args[player_index].elements_to_rec[receiving_args[player_index].rec_rounds - 1] == RECV_BUFFER)
    {
        receive_();
    }
#endif
    receiving_args[player_index].elements_to_rec[receiving_args[player_index].rec_rounds - 1] += 1;
    total_recv[player_index] += 1;
#if WAIT_AFTER_MESSAGES_IF_AHEAD >= 0
    if(SYNC_PARTY_SEND == player_index && (total_recv[player_index]-1) % WAIT_AFTER_MESSAGES_IF_AHEAD == 0)
    {
        send_to_(player_index); // send sync message
        send_(); // send sync message
    }
#endif
}

void communicate_()
{
    send_();
    receive_();
}

#if MAL == 1

void store_compare_view_init(int player_id)
{
    elements_to_compare[player_id] += 1;
}

void compare_views_init()
{
#if DATTYPE >= 256
    int hash_chunks_to_send = 1;
#else
    int hash_chunks_to_send = (sizeof(uint32_t) * 8) / sizeof(DATATYPE);
#endif
    for (int player_id = 0; player_id < num_players * player_multiplier; player_id++)
    {
        if (elements_to_compare[player_id] > 0)
        {
            // exchange 1 sha256 hash. Do to DATATYPE constraints it may need to be split up to multiple chunks
            for (int i = 0; i < hash_chunks_to_send; i++)
            {
                if (player_id == 3)  // P_0123
                {
                    if (P_0 != PSELF)
                        send_to_(P_0);
                    if (P_1 != PSELF)
                        send_to_(P_1);
                    if (P_2 != PSELF)
                        send_to_(P_2);
                    if (P_3 != PSELF)
                        send_to_(P_3);
                }
                else if (player_id == 4)  // P_012
                {
                    if (P_0 != PSELF)
                        send_to_(P_0);
                    if (P_1 != PSELF)
                        send_to_(P_1);
                    if (P_2 != PSELF)
                        send_to_(P_2);
                }
                else if (player_id == 5)  // P_013
                {
                    if (P_0 != PSELF)
                        send_to_(P_0);
                    if (P_1 != PSELF)
                        send_to_(P_1);
                    if (P_3 != PSELF)
                        send_to_(P_3);
                }
                else if (player_id == 6)  // P_023
                {
                    if (P_0 != PSELF)
                        send_to_(P_0);
                    if (P_2 != PSELF)
                        send_to_(P_2);
                    if (P_3 != PSELF)
                        send_to_(P_3);
                }
                else if (player_id == 7)  // P_123
                {
                    if (P_1 != PSELF)
                        send_to_(P_1);
                    if (P_2 != PSELF)
                        send_to_(P_2);
                    if (P_3 != PSELF)
                        send_to_(P_3);
                }
                else
                    send_to_(player_id);
            }
        }
    }

    communicate_();  // TODO: check compatability with Preprocessing

    for (int player_id = 0; player_id < num_players * player_multiplier; player_id++)
    {
        if (elements_to_compare[player_id] > 0)
        {
            // exchange 1 sha256 hash. Do to DATATYPE constraints it may need to be split up to multiple chunks
            for (int i = 0; i < hash_chunks_to_send; i++)
            {
                if (player_id == 3)  // P_0123
                {
                    if (P_0 != PSELF)
                        receive_from_(P_0);
                    if (P_1 != PSELF)
                        receive_from_(P_1);
                    if (P_2 != PSELF)
                        receive_from_(P_2);
                    if (P_3 != PSELF)
                        receive_from_(P_3);
                }
                else if (player_id == 4)  // P_012
                {
                    if (P_0 != PSELF)
                        receive_from_(P_0);
                    if (P_1 != PSELF)
                        receive_from_(P_1);
                    if (P_2 != PSELF)
                        receive_from_(P_2);
                }
                else if (player_id == 5)  // P_013
                {
                    if (P_0 != PSELF)
                        receive_from_(P_0);
                    if (P_1 != PSELF)
                        receive_from_(P_1);
                    if (P_3 != PSELF)
                        receive_from_(P_3);
                }
                else if (player_id == 6)  // P_023
                {
                    if (P_0 != PSELF)
                        receive_from_(P_0);
                    if (P_2 != PSELF)
                        receive_from_(P_2);
                    if (P_3 != PSELF)
                        receive_from_(P_3);
                }
                else if (player_id == 7)  // P_123
                {
                    if (P_1 != PSELF)
                        receive_from_(P_1);
                    if (P_2 != PSELF)
                        receive_from_(P_2);
                    if (P_3 != PSELF)
                        receive_from_(P_3);
                }
                else
                    receive_from_(player_id);
            }
        }
    }
}

#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
void store_output_share_()
{
    preprocessed_outputs_index += 1;
}
#endif

void finalize_(std::string* ips)
{
    for (int t = 0; t < (num_players - 1); t++)
    {
        int offset = 0;
        if (t >= player_id)
            offset = 1;  // player should not receive from itself
        receiving_args[t].player_count = num_players;
        receiving_args[t].received_elements =
            new DATATYPE*[receiving_args[t].rec_rounds];  // every thread gets its own pointer array for receiving
                                                          // elements

        receiving_args[t].player_id = player_id;
        receiving_args[t].connected_to = t + offset;
        receiving_args[t].ip = ips[t];
        receiving_args[t].hostname = (char*)"hostname";
        receiving_args[t].port =
            (int)base_port + player_id * (num_players - 1) +
            t;  // e.g. P_0 receives on base port from P_1, P_2 on base port + num_players from P_0 6000,6002
        // init_srng(t, (t+offset) + player_id);
    }
    for (int t = 0; t < (num_players - 1); t++)
    {
        int offset = 0;
        if (t >= player_id)
            offset = 1;  // player should not send to itself
        sending_args[t].sent_elements = new DATATYPE*[sending_args[t].send_rounds];
        /* sending_args[t].elements_to_send[0] = 0; //input sharing with SRNGs */
        sending_args[t].player_id = player_id;
        sending_args[t].player_count = num_players;
        sending_args[t].connected_to = t + offset;
        sending_args[t].port = (int)base_port + (t + offset) * (num_players - 1) + player_id - 1 +
                               offset;  // e.g. P_0 sends on base port + num_players  for P_1, P_2 on base port +
                                        // num_players for P_0 (6001,6000)
        sending_args[t].sent_elements[0] =
            NEW(DATATYPE[sending_args[t].elements_to_send[0]]);  // Allocate memory for first round
    }

#if MAL == 1 && PRE == 0
    for (int t = 0; t < (num_players * player_multiplier); t++)
    {
#if VERIFY_BUFFER > 0
        verify_buffer[t] = new DATATYPE[VERIFY_BUFFER];
#else
        verify_buffer[t] = new DATATYPE[elements_to_compare[t]];
#endif
    }
#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
    preprocessed_outputs_index = 0;  // reset index for post phase
#endif
    rounds = 0;
    sending_rounds = 0;
    rb = 0;
    sb = 0;
    current_phase = PHASE_LIVE;
    print_communication();
#if WAIT_AFTER_MESSAGES_IF_AHEAD >= 0
    for (int t = 0; t < num_players - 1; t++)
    {
        total_recv[t] = 0;
        total_send[t] = 0;
    }
#endif

}

void init()
{
    for (int t = 0; t < (num_players - 1); t++)
    {
        sending_args[t].elements_to_send[0] = 0;
        receiving_args[t].elements_to_rec[0] = 0;
    }
}

void finalize_(std::string* ips, receiver_args* ra, sender_args* sa)
{
    for (int t = 0; t < (num_players - 1); t++)
    {
        int offset = 0;
        if (t >= player_id)
            offset = 1;  // player should not receive from itself
        ra[t].player_count = num_players;

        ra[t].received_elements =
            new DATATYPE*[ra[t].rec_rounds];  // every thread gets its own pointer array for receiving elements

        ra[t].player_id = player_id;
        ra[t].connected_to = t + offset;
        ra[t].ip = ips[t];
        ra[t].hostname = (char*)"hostname";
        ra[t].port = (int)base_port + (t + offset) * (num_players - 1) + player_id - 1 +
                     offset;  // e.g. P_0 sends on base port + num_players  for P_1, P_2 on base port + num_players for
                              // P_0 (6001,6000)
    }
    for (int t = 0; t < (num_players - 1); t++)
    {
        int offset = 0;
        if (t >= player_id)
            offset = 1;  // player should not send to itself
#if PRE == 1             // only in actual preprocessing phase
        sa[t].send_rounds = 1;
#endif

        sa[t].sent_elements = new DATATYPE*[sa[t].send_rounds];
        /* sending_args[t].elements_to_send[0] = 0; //input sharing with SRNGs */
        sa[t].player_id = player_id;
        sa[t].player_count = num_players;
        sa[t].connected_to = t + offset;
        sa[t].port = (int)base_port + player_id * (num_players - 1) +
                     t;  // e.g. P_0 receives on base port from P_1, P_2 on base port + num_players from P_0 6000,6002
        sa[t].sent_elements[0] = NEW(DATATYPE[sa[t].elements_to_send[0]]);  // Allocate memory for first round
        share_buffer[t] = 0;
    }
#if MAL == 1 && PRE == 1
    for (int t = 0; t < (num_players * player_multiplier); t++)
    {
#if VERIFY_BUFFER > 0
        verify_buffer[t] = new DATATYPE[VERIFY_BUFFER];
#else
        verify_buffer[t] = new DATATYPE[elements_to_compare[t]];
#endif
    }
#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
    preprocessed_outputs = new DATATYPE[preprocessed_outputs_index];
    preprocessed_outputs_index = 0;
#endif

    rounds = 0;
    sending_rounds = 0;
    rb = 0;
    sb = 0;
    current_phase = PHASE_PRE;
    /* print_communication(); */
}
