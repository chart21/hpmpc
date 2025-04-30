#pragma once
#include "../include/pch.h"

#ifndef BOOL_COMPRESS
#define DTYPE DATATYPE
#else
#define DTYPE uint8_t
#endif
#include "socket.hpp"
#include "sockethelper.h"

void client_signal_connection_established(int player_count)
{
    pthread_mutex_lock(&mtx_connection_established);
    num_successful_connections += 1;
    if (num_successful_connections == 2 * (player_count - 1))
    {
        pthread_cond_signal(&cond_successful_connection);  // signal main thread that all threads have connected
    }
    pthread_mutex_unlock(&mtx_connection_established);

    pthread_mutex_lock(&mtx_start_communicating);
    while (num_successful_connections != -1)
    {  // wait for start signal from main thread
        pthread_cond_wait(&cond_start_signal, &mtx_start_communicating);
    }
    pthread_mutex_unlock(&mtx_start_communicating);
}

void signal_all_data_received_in_round(int& rounds, int player_count)
{
    pthread_mutex_lock(&mtx_data_received);
    sockets_received[rounds] += 1;
    if (sockets_received[rounds] == player_count - 1)
    {
        pthread_mutex_lock(&mtx_receive_next);    // Mutex probably neccessary if one thread is alrady one round further
        receiving_rounds += 1;                    // increase global receiving_rounds
        pthread_cond_signal(&cond_receive_next);  // signal main thread that receiving is finished
        pthread_mutex_unlock(&mtx_receive_next);
    }
    pthread_mutex_unlock(&mtx_data_received);
    rounds += 1;
}

void* receiver(void* threadParameters)
{
    Socket client;
    client.Connect(((receiver_args*)threadParameters)->ip, ((receiver_args*)threadParameters)->port);
#if PRINT == 1
    printf("P%i: Receiving Socket connected to Player %i \n",
           ((receiver_args*)threadParameters)->player_id,
           ((receiver_args*)threadParameters)->connected_to);
#endif
    client_signal_connection_established(((receiver_args*)threadParameters)->player_count);

    int rounds = 0;
    while (rounds < ((receiver_args*)threadParameters)->rec_rounds)
    {
        // Allocate new memory for received data, check correctness
        ((receiver_args*)threadParameters)->received_elements[rounds] =
            NEW(DATATYPE[((receiver_args*)threadParameters)->elements_to_rec[rounds]]);

        if (((receiver_args*)threadParameters)->elements_to_rec[rounds] > 0)  // should data be received in this round?
        {
#ifndef BOOL_COMPRESS
            int64_t elements_to_rec = ((receiver_args*)threadParameters)->elements_to_rec[rounds];
            elements_to_rec = elements_to_rec * sizeof(DATATYPE);
#endif
#ifdef BOOL_COMPRESS
            int64_t elements_to_rec = (((receiver_args*)threadParameters)->elements_to_rec[rounds] + 7) / 8;
            uint8_t* rec_buffer = new (std::align_val_t(sizeof(uint64_t))) uint8_t[elements_to_rec];
            unpack(rec_buffer,
                   ((receiver_args*)threadParameters)->received_elements[rounds],
                   ((receiver_args*)threadParameters)->elements_to_rec[rounds]);
            delete[] rec_buffer;
#endif
            client.Receive_all(((char*)((receiver_args*)threadParameters)->received_elements[rounds]),
                               &elements_to_rec);
#if PRINT == 1
            printf("P%i: Received %li bytes from player %i in round %i out of %i \n",
                   PARTY,
                   elements_to_rec,
                   ((receiver_args*)threadParameters)->connected_to,
                   rounds + 1,
                   ((receiver_args*)threadParameters)->rec_rounds);
#endif
        }
        // If all sockets received, signal main_thread
        signal_all_data_received_in_round(rounds, ((receiver_args*)threadParameters)->player_count);
    }
    /* client.Close_Context(); */
    pthread_exit(NULL);
}
