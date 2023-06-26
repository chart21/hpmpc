#pragma once
#include <cstdint>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>

#include "sockethelper.h"
#include "socket.hpp"

#ifndef BOOL_COMPRESS
#define DTYPE DATATYPE
#else
#define DTYPE uint8_t
#endif


void server_signal_connection_established(int player_count)
{
            pthread_mutex_lock(&mtx_connection_established);
        num_successful_connections += 1; 
    if(num_successful_connections == 2 * (player_count -1)) {
        pthread_cond_signal(&cond_successful_connection); //signal main thread that all threads have connected
        /* printf("server %i \n",num_successful_connections); */
    }
    pthread_mutex_unlock(&mtx_connection_established);
    /* printf("Player: Unlocked conn \n"); */

    pthread_mutex_lock(&mtx_start_communicating); 

    while (num_successful_connections != -1) { // wait for start signal from main thread
        /* printf("Player: Unlocking conn and waiting for signal \n"); */ 
        pthread_cond_wait(&cond_start_signal, &mtx_start_communicating);
    }
        /* printf("Player: Done waiting, unlocking \n"); */
        pthread_mutex_unlock(&mtx_start_communicating);

}


void *sender(void* threadParameters)
{
    Socket server;
    server.Bind(((sender_args*) threadParameters)->port);
    server.Listen(2);
    Socket client = server.Accept();
    #if PRINT == 1
    printf("Player %i: Connected to Player %i\n", PARTY, ((sender_args*) threadParameters)->connected_to);
    #endif
    server_signal_connection_established(((sender_args*) threadParameters)->player_count);
    // Send data to the client
        int rounds = 0;

        while (rounds < ((sender_args*) threadParameters)->send_rounds) //continue until all data is sent
        {
            pthread_mutex_lock(&mtx_send_next); 
            while(rounds >= sending_rounds)
                    pthread_cond_wait(&cond_send_next, &mtx_send_next); //make sure that there is new data to send, singaled by main
            pthread_mutex_unlock(&mtx_send_next);
            //char buf[1024] = { 0 };
            //recv(new_fd, buf, 32, MSG_WAITALL);
            /* printf("sending round %i out of %i \n", rounds, ((sender_args*) threadParameters)->send_rounds); */
            if(((sender_args*) threadParameters)->elements_to_send[rounds] > 0)
            {
#ifndef BOOL_COMPRESS
            int64_t elements_to_send =  ((sender_args*) threadParameters)->elements_to_send[rounds];
            elements_to_send = elements_to_send * sizeof(DATATYPE);
#else
            int64_t elements_to_send =  (((sender_args*) threadParameters)->elements_to_send[rounds] + 7)/8;
            uint8_t* send_buf = new (std::align_val_t(sizeof(uint64_t))) uint8_t[elements_to_send];
pack(((sender_args*) threadParameters)->sent_elements[rounds],send_buf,((sender_args*) threadParameters)->elements_to_send[rounds]);
#endif

client.Send_all( ((char*) ((sender_args*) threadParameters)->sent_elements[rounds]), &elements_to_send);
#if PRINT == 1
printf("Player %i :Sent %li bytes to player %i in round %i out of %i \n", PARTY, elements_to_send , ((sender_args*) threadParameters)->connected_to, rounds + 1, ((sender_args*) threadParameters)->send_rounds);
#endif
            }
                //Delete sent data
           //free(((thargs_p*) threadParameters)->sent_elements[rounds]);

            rounds += 1;
            /* printf("Next sending round \n"); */
        

        }





#if PRINT == 1
printf("Closing connection to Player %i\n", ((sender_args*) threadParameters)->connected_to);
#endif
/* client.Close_Context(); */
/* server.Close_Context(); */
pthread_exit( NULL );

}

