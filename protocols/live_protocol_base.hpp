#pragma once
#include "../core/include/pch.h"
#include "../core/networking/buffers.h"

#if MAL == 1
#ifdef __SHA__
#include "../core/crypto/sha/SHA_256_x86.h"
#elif ARM == 1
#include "../core/cryptosha/SHA_256_arm.h"
#else
#include "../core/crypto/sha/SHA_256.h"
#endif
#endif

void send_live()
{
    sb = 0;
    for (int t = 0; t < (num_players - 1); t++)
    {
        send_count[t] = 0;
    }
    // different in PRE
    for (int t = 0; t < (num_players - 1); t++)
        if (sending_rounds < sending_args[t].send_rounds - 1)  // don't allocate memory for the last+1 round
            sending_args[t].sent_elements[sending_rounds + 1] =
                NEW(DATATYPE[sending_args[t].elements_to_send[sending_rounds + 1]]);  // Allocate memory for all sending
                                                                                      // buffers for next round
    pthread_mutex_lock(&mtx_send_next);
    sending_rounds += 1;
    pthread_cond_broadcast(&cond_send_next);  // signal all threads that sending buffer contains next data
    /* printf("boradcasted round %i \n", sending_rounds); */
    pthread_mutex_unlock(&mtx_send_next);
}

void receive_live()
{
    for (int t = 0; t < (num_players - 1); t++)
        share_buffer[t] = 0;

    rounds += 1;
    // receive_data
    // wait until all sockets have finished received their last data
    pthread_mutex_lock(&mtx_receive_next);

    /* std::chrono::high_resolution_clock::time_point c1 = */
    /*         std::chrono::high_resolution_clock::now(); */
    while (rounds > receiving_rounds)  // wait until all threads received their data
        pthread_cond_wait(&cond_receive_next, &mtx_receive_next);

    /* double time = std::chrono::duration_cast<std::chrono::microseconds>( */
    /*                      std::chrono::high_resolution_clock::now() - c1) */
    /*                      .count(); */
    /* printf("finished waiting for receive in round %i \n", rounds - 1); */
    pthread_mutex_unlock(&mtx_receive_next);
    for (int t = 0; t < (num_players - 1); t++)
    {
        if (rounds > 2)
            if (receiving_args[t].rec_rounds > 1)
                /* if(receiving_args[t].received_elements[rounds - 2] != NULL) */
                delete[] receiving_args[t].received_elements[rounds - 2];  // delete memory of last round
        /* printf("Time spent waiting for data chrono: %fs \n", time / 1000000); */
    }
    rb = 0;
}

void communicate_live()
{
    /* #if PRE == 0 */
    send_live();
    receive_live();
    /* #endif */
}

void send_to_live(int player_id);
DATATYPE receive_from_live(int player_id);

void send_to_live(int player_id, DATATYPE a)
{
/* sending_args[player_id].sent_elements[sending_args[player_id].send_rounds][send_count[player_id]] = a; */
#if SEND_BUFFER > 0
    if (send_count[player_id] == SEND_BUFFER)
    {
        send_live();
    }
#endif
    sending_args[player_id].sent_elements[sending_rounds][send_count[player_id]] = a;
    send_count[player_id] += 1;
#if WAIT_AFTER_MESSAGES_IF_AHEAD >= 0
    total_send[player_id] += 1;
    if ((SYNC_PARTY_RECV == player_id || SYNC_PARTY_RECV2 == player_id) && (total_send[player_id]) % WAIT_AFTER_MESSAGES_IF_AHEAD == 0)
    {
        receive_live();  // expect sync message
        receive_from_live(player_id);  // expect sync message 
    }
#endif
}

DATATYPE receive_from_live(int player_id)
{
#if RECV_BUFFER > 0
    if (share_buffer[player_id] == RECV_BUFFER)
    {
        receive_live();
    }
#endif
    share_buffer[player_id] += 1;
#if WAIT_AFTER_MESSAGES_IF_AHEAD >= 0
    total_recv[player_id] += 1;
if (SYNC_PARTY_SEND == player_id && (total_recv[player_id] -1) % WAIT_AFTER_MESSAGES_IF_AHEAD == 0)
{
    send_to_live(player_id, SET_ALL_ZERO()); // send sync message
    send_live();  // send sync message
}
#endif
    return receiving_args[player_id].received_elements[rounds - 1][share_buffer[player_id] - 1];
    /* return
     * receiving_args[player_id].received_elements[receiving_args[player_id].rec_rounds-1][share_buffer[player_id] -1];
     */
}

/* void send_to_live(int player_id1, int player_id2, DATATYPE a) */
/* { */
/*     send_to_live(player_id1, a); */
/*     send_to_live(player_id2, a); */
/* } */

#if PRE == 1
void pre_send_to_live(int player_id, DATATYPE a)
{
    sending_args_pre[player_id].sent_elements[0][send_count_pre[player_id]] = a;
    /* sending_args_pre[player_id].sent_elements[sending_args_pre[player_id].send_rounds][send_count[player_id]] = a; */
    send_count_pre[player_id] += 1;
}

DATATYPE pre_receive_from_live(int player_id)
{
    share_buffer_pre[player_id] += 1;
    return receiving_args_pre[player_id].received_elements[0][share_buffer_pre[player_id] - 1];
    /* return
     * receiving_args_pre[player_id].received_elements[receiving_args_pre[player_id].rec_rounds][share_buffer[player_id]
     * -1]; */
}

DATATYPE fetch_from_pre(int player_id)
{
    return sending_args_pre[player_id].sent_elements[0][sending_args_pre[player_id].fetch_counter++];
}
#endif

DATATYPE get_input_live()
{
    share_buffer[num_players - 1] += 1;
    return player_input[share_buffer[num_players - 1] - 1];
}

#if MAL == 1

void perform_compare_view(int player_id)
{
    if (verify_buffer_index[player_id] == 0)
        return;
#ifdef __SHA__
    sha256_process_x86(
        hash_val[player_id], (uint8_t*)verify_buffer[player_id], sizeof(DATATYPE) * verify_buffer_index[player_id]);
#elif ARM == 1
    sha256_process_arm(
        hash_val[player_id], (uint8_t*)verify_buffer[player_id], sizeof(DATATYPE) * verify_buffer_index[player_id]);
#else
    sha256_process(
        hash_val[player_id], (uint8_t*)verify_buffer[player_id], sizeof(DATATYPE) * verify_buffer_index[player_id]);
#endif
    verify_buffer_index[player_id] = 0;
}

/* void store_compare_view(int player_id, DATATYPE element_to_compare) */
/* { */
/* #if VERIFY_BUFFER > 0 */
/* if(verify_buffer_index[player_id] == VERIFY_BUFFER) */
/* { */
/*     perform_compare_view(player_id); */
/* } */
/* #endif */
/* verify_buffer[player_id][verify_buffer_index[player_id]] = element_to_compare; */
/* verify_buffer_index[player_id] +=1; */
/* } */

void store_compare_view(int player_id, DATATYPE element_to_compare)
{
#if VERIFY_BUFFER > 0
#if VERIFY_BUFFER == 1
#ifdef __SHA__
    sha256_process_x86(hash_val[player_id], (uint8_t*)&element_to_compare, sizeof(DATATYPE));
#else
    sha256_process(hash_val[player_id], (uint8_t*)&element_to_compare, sizeof(DATATYPE));
#endif
    return;
#endif
    if (verify_buffer_index[player_id] == VERIFY_BUFFER)
        perform_compare_view(player_id);
#endif
    verify_buffer[player_id][verify_buffer_index[player_id]] = element_to_compare;
    verify_buffer_index[player_id] += 1;
}

void compare_views()
{
#if DATTYPE >= 256
    int hash_chunks_to_send = 1;  // fill up to size of DATATYPE to avoid memory
                                  // leaks
#else
    int hash_chunks_to_send = (sizeof(uint32_t) * 8) / sizeof(DATATYPE);
#endif
    DATATYPE val_to_send[num_players * player_multiplier][hash_chunks_to_send];
    DATATYPE val_recieved[num_players * player_multiplier][hash_chunks_to_send];
    for (int player_id = 0; player_id < num_players * player_multiplier; player_id++)
    {
        if (elements_to_compare[player_id] > 0)
        {
            perform_compare_view(player_id);
            // exchange 1 sha256 hash. Do to DATATYPE constraints it may need to be
            // split up to multiple chunks
            int index_slice = 0;
            for (int i = 0; i < hash_chunks_to_send; i++)
            {
#if DATTYPE < 64
                uint8_t* addr_to_send = ((uint8_t*)(hash_val[player_id])) + index_slice;
                index_slice += sizeof(DATATYPE);  // hash is stored in 4 byte chunks -> need
                                                  // smaller slices for small DATATYPE
#elif DATTYPE >= 256
                uint32_t values_to_send[8 * sizeof(DATATYPE) / (sizeof(uint32_t) * 8)]{0};
                for (int j = 0; j < 8; j++)
                    values_to_send[j] = hash_val[player_id][j];
                uint32_t* addr_to_send = values_to_send;
#else
                uint32_t* addr_to_send = hash_val[player_id] + index_slice;
                index_slice += sizeof(DATATYPE) / 4;  // hash is stored in 4 byte chunks -> move
                                                      // up index by multiplier
#endif
                val_to_send[player_id][i] = *((DATATYPE*)addr_to_send);
                if (player_id == 3)  // P_0123
                {
                    if (P_0 != PSELF)
                        send_to_live(P_0, val_to_send[player_id][i]);
                    if (P_1 != PSELF)
                        send_to_live(P_1, val_to_send[player_id][i]);
                    if (P_2 != PSELF)
                        send_to_live(P_2, val_to_send[player_id][i]);
                    if (P_3 != PSELF)
                        send_to_live(P_3, val_to_send[player_id][i]);
                }
                else if (player_id == 4)  // P_012
                {
                    if (P_0 != PSELF)
                        send_to_live(P_0, val_to_send[player_id][i]);
                    if (P_1 != PSELF)
                        send_to_live(P_1, val_to_send[player_id][i]);
                    if (P_2 != PSELF)
                        send_to_live(P_2, val_to_send[player_id][i]);
                }
                else if (player_id == 5)  // P_013
                {
                    if (P_0 != PSELF)
                        send_to_live(P_0, val_to_send[player_id][i]);
                    if (P_1 != PSELF)
                        send_to_live(P_1, val_to_send[player_id][i]);
                    if (P_3 != PSELF)
                        send_to_live(P_3, val_to_send[player_id][i]);
                }
                else if (player_id == 6)  // P_023
                {
                    if (P_0 != PSELF)
                        send_to_live(P_0, val_to_send[player_id][i]);
                    if (P_2 != PSELF)
                        send_to_live(P_2, val_to_send[player_id][i]);
                    if (P_3 != PSELF)
                        send_to_live(P_3, val_to_send[player_id][i]);
                }
                else if (player_id == 7)  // P_123
                {
                    if (P_1 != PSELF)
                        send_to_live(P_1, val_to_send[player_id][i]);
                    if (P_2 != PSELF)
                        send_to_live(P_2, val_to_send[player_id][i]);
                    if (P_3 != PSELF)
                        send_to_live(P_3, val_to_send[player_id][i]);
                }
                else
                    send_to_live(player_id, val_to_send[player_id][i]);
            }
        }
    }
    communicate_live();
    for (int player_id = 0; player_id < num_players * player_multiplier; player_id++)
    {

        if (elements_to_compare[player_id] > 0)
        {
            bool verified = true;
            for (int i = 0; i < hash_chunks_to_send; i++)
            {
                DATATYPE val_rec[3];
                int counter = 0;
                if (player_id > num_players - 2)
                {
                    if (player_id == 3)  // P_0123
                    {
                        if (P_0 != PSELF)
                            val_rec[counter++] = receive_from_live(P_0);
                        if (P_1 != PSELF)
                            val_rec[counter++] = receive_from_live(P_1);
                        if (P_2 != PSELF)
                            val_rec[counter++] = receive_from_live(P_2);
                        if (P_3 != PSELF)
                            val_rec[counter++] = receive_from_live(P_3);
                        val_rec[0] = XOR(XOR(val_rec[0], val_rec[1]),
                                         val_rec[2]);  // trick to ensure equality of all three values for check later
                                                       // (since comparing vectorized variables is tedious)
                    }
                    else if (player_id == 4)  // P_012
                    {
                        if (P_0 != PSELF)
                            val_rec[counter++] = receive_from_live(P_0);
                        if (P_1 != PSELF)
                            val_rec[counter++] = receive_from_live(P_1);
                        if (P_2 != PSELF)
                            val_rec[counter++] = receive_from_live(P_2);
                        val_rec[0] =
                            AND(val_rec[0],
                                val_rec[1]);  // trick to ensure equality of both values with certain probability
                    }
                    else if (player_id == 5)  // P_013
                    {
                        if (P_0 != PSELF)
                            val_rec[counter++] = receive_from_live(P_0);
                        if (P_1 != PSELF)
                            val_rec[counter++] = receive_from_live(P_1);
                        if (P_3 != PSELF)
                            val_rec[counter++] = receive_from_live(P_3);
                        val_rec[0] =
                            AND(val_rec[0],
                                val_rec[1]);  // trick to ensure equality of both values with certain probability
                    }
                    else if (player_id == 6)  // P_023
                    {
                        if (P_0 != PSELF)
                            val_rec[counter++] = receive_from_live(P_0);
                        if (P_2 != PSELF)
                            val_rec[counter++] = receive_from_live(P_2);
                        if (P_3 != PSELF)
                            val_rec[counter++] = receive_from_live(P_3);
                        val_rec[0] =
                            AND(val_rec[0],
                                val_rec[1]);  // trick to ensure equality of both values with certain probability
                    }
                    else if (player_id == 7)  // P_123
                    {
                        if (P_1 != PSELF)
                            val_rec[counter++] = receive_from_live(P_1);
                        if (P_2 != PSELF)
                            val_rec[counter++] = receive_from_live(P_2);
                        if (P_3 != PSELF)
                            val_rec[counter++] = receive_from_live(P_3);
                        val_rec[0] =
                            AND(val_rec[0],
                                val_rec[1]);  // trick to ensure equality of both values with certain probability
                    }
                    val_recieved[player_id][i] = val_rec[0];
                }
                else
                    val_recieved[player_id][i] = receive_from_live(player_id);
            }
            for (unsigned long j = 0; j < sizeof(DATATYPE); j++)
            {
                if (*(((uint8_t*)&(val_recieved[player_id])) + j) != *(((uint8_t*)&(val_to_send[player_id])) + j))
                {
                    verified = false;
                }
            }
            if (!verified)
            {
                printf("P%i: Compareviews failed! \n", PARTY);
                /* exit(0); */
            }
            /* else */
            /* printf("Compareviews of Player %i sucessfull! \n", */
            /*         PARTY); */
        }
    }
}
#endif

#if PRE == 1 && HAS_POST_PROTOCOL == 1
void store_output_share(DATATYPE val)
{
    preprocessed_outputs[preprocessed_outputs_index] = val;
    preprocessed_outputs_index += 1;
}

DATATYPE retrieve_output_share()
{
    preprocessed_outputs_index += 1;
    return preprocessed_outputs[preprocessed_outputs_index - 1];
}
#endif
