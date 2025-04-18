#pragma once
#include "../arch/DATATYPE.h"
#include "../include/pch.h"

pthread_mutex_t mtx_connection_established;
pthread_mutex_t mtx_start_communicating;
pthread_cond_t cond_successful_connection;
pthread_cond_t cond_start_signal;
int num_successful_connections = 0;

// int sending_rounds = 0;
int receiving_rounds = 0;
pthread_mutex_t mtx_receive_next;
pthread_cond_t cond_receive_next;

std::vector<int> sockets_received;
pthread_mutex_t mtx_data_received;
pthread_cond_t cond_data_received;

int sending_rounds = 0;
pthread_mutex_t mtx_send_next;
pthread_cond_t cond_send_next;

int sockets_sent = 0;
pthread_mutex_t mtx_data_sent;
pthread_cond_t cond_data_sent;

typedef struct receiver_arguments
{
    int player_count;
    int player_id;
    int connected_to;
    DATATYPE** received_elements;
    int inputs_size;  // depricated
    std::string ip;
    int port;
    char* hostname;
    int rec_rounds;
    std::vector<int64_t> elements_to_rec;
    int total_rounds;  // depricated
    // char *data;
    // char *length
} receiver_args;

typedef struct sender_arguments
{
    DATATYPE** sent_elements;
    int inputs_size;  // depricated
    int port;
    int player_id;
    int player_count;
    int connected_to;
    int send_rounds;
    std::vector<int64_t> elements_to_send;
    int total_rounds;  // depricated
#if PRE == 1
    uint64_t fetch_counter;  // use to fetch sent in pre-processing round
#endif

    // char *data;
} sender_args;

void client_signal_main()
{

    pthread_mutex_lock(&mtx_connection_established);
    num_successful_connections += 1;
    if (num_successful_connections == 2 * num_players - 1)
    {
        pthread_cond_signal(&cond_successful_connection);  // signal main thread that all threads have connected
        printf("client %i \n", num_successful_connections);
    }
    pthread_mutex_unlock(&mtx_connection_established);

    pthread_mutex_lock(&mtx_start_communicating);
    while (num_successful_connections != -1)
    {  // wait for start signal from main thread
        pthread_cond_wait(&cond_start_signal, &mtx_start_communicating);
    }
    pthread_mutex_unlock(&mtx_start_communicating);
}
