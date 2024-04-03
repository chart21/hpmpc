#pragma once
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <cstring>
#include <pthread.h>
#include <random>
#include <bitset>
#include <new>
#include <memory>
#include "arch/DATATYPE.h"
#include "protocols/init_protocol_base.hpp"
#include "protocols/live_protocol_base.hpp"

#if FUNCTION_IDENTIFIER == 0
#include "programs/search_init.hpp"
#elif FUNCTION_IDENTIFIER == 11
#include "programs/share_conversion_init.hpp"
#elif FUNCTION_IDENTIFIER >= 1 && FUNCTION_IDENTIFIER <= 6
#include "programs/mult_init.hpp"
#elif FUNCTION_IDENTIFIER >= 7 && FUNCTION_IDENTIFIER <= 9
#include "programs/debug_init.hpp"
#elif FUNCTION_IDENTIFIER > 11 && FUNCTION_IDENTIFIER <= 39
#include "programs/mat_mul_init.hpp"
#elif FUNCTION_IDENTIFIER >= 40 && FUNCTION_IDENTIFIER <= 65
#include "programs/functions/sevare.hpp"
#elif FUNCTION_IDENTIFIER < 400
#include "programs/functions/NN.hpp"
#elif FUNCTION_IDENTIFIER >= 500 && FUNCTION_IDENTIFIER <= 510
#include "programs/functions/mpspdz.hpp"
#else 
#include "programs/functions/cryptgpu_piranha_bench.hpp"
#endif

#include "utils/xorshift.h"

#include "config.h"
#include "protocols/Protocols.h"

#include "utils/randomizer.h"
#include "utils/timing.hpp"
#include "utils/print.hpp"
#include "networking/client.hpp"
#include "networking/server.hpp"

#include "networking/sockethelper.h"
#include "networking/buffers.h"
#if LIVE == 1 && INIT == 0 && NO_INI == 0
#include "protocols/CircuitInfo.hpp"
#endif

struct timespec i1, p1, p2, l1, l2;

int modulo(int x,int N){
    return (x % N + N) %N;
}



void init_srng(uint64_t link_id, uint64_t link_seed)
{
    #if RANDOM_ALGORITHM == 0
    UINT_TYPE gen_seeds[DATTYPE];
   for (int i = 0; i < DATTYPE; i++) {
      gen_seeds[i] = link_seed * (i+1); // replace with independant seeds in the future
   }
/* int incr = (DATTYPE -1) / 64 + 1; */
/* for (int i = 0; i < 64; i+=incr) { */
/* orthogonalize(gen_seeds+i, srng[link_id]+i); */
orthogonalize_boolean(gen_seeds, srng[link_id]);

#elif RANDOM_ALGORITHM == 1
    UINT_TYPE gen_keys[11][DATTYPE*128/BITLENGTH];
    for (int i = 0; i < 11; i++) {
        for (int j = 0; j < DATTYPE*128/BITLENGTH; j++) {
            gen_keys[i][j] = link_seed * ((i+1)*j); // replace with independant seeds in the future
        }
    }
    for (int i = 0; i < 11; i++) {
            for(int j = 0; j < DATTYPE*128/BITLENGTH; j++)
                orthogonalize_boolean(gen_keys[i]+j, key[link_id][i]+j);
            }
    
#elif RANDOM_ALGORITHM == 2
srand(link_seed);
alignas(AES_DATTYPE/8) uint64_t counter[AES_DATTYPE/64];
#if USE_SSL_AES == 1
    for (int j = 0; j < 64; j++)
        aes_counter[link_id][j] = rand() % 256 | rand() % 256; //generate random 8-bit number, to stay consistent with other approaches, 2 times rand is used
#else
    for (int j = 0; j < AES_DATTYPE/64; j++)
        counter[j] = ((uint64_t) rand() << 32) | rand(); //generate random 64-bit number
        
#if defined(__AVX512F__ ) && defined(__VAES__)
    aes_counter[link_id] = _mm512_set_epi64(counter[7], counter[6], counter[5], counter[4], counter[3], counter[2], counter[1], counter[0]);
#elif defined(__AVX2__) && defined(__VAES__)
    aes_counter[link_id] = _mm256_set_epi64x(counter[3], counter[2], counter[1], counter[0]);
#elif defined(__AES__)
    aes_counter[link_id] = _mm_set_epi64x(counter[1], counter[0]);
#endif
#endif


alignas(AES_DATTYPE/8) uint8_t seed[AES_DATTYPE/8];
        for(int i = 0; i < AES_DATTYPE/8; i++)
            {
                /* seed[link_id][i] = rand() % 256; */
                seed[i] = rand() % 256; //
                /* seed[j][i] = 10; */
            }
#if USE_SSL_AES == 1
    key_schedule[link_id] = EVP_CIPHER_CTX_new();
    if (!key_schedule[link_id])
        handleErrors();
#endif
    aes_load_enc(seed, key_schedule[link_id]);


#endif
        init_buffers(link_id);
#if MAL == 1

    // Ensure all players have the same initial state
    /* initial state */
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    for(int i = 0; i < num_players-1; i++)
    {
        for(int j = 0; j < 8; j++)
            hash_val[i][j] = state[j];
    }

#endif

}

void init_muetexes()
{
pthread_mutex_init(&mtx_connection_established, NULL);
pthread_mutex_init(&mtx_start_communicating, NULL);
pthread_mutex_init(&mtx_send_next, NULL);
pthread_cond_init(&cond_successful_connection, NULL);
pthread_cond_init(&cond_start_signal, NULL);
pthread_cond_init(&cond_send_next, NULL);

}

void init_circuit(std::string ips[])
{

/* clock_t time_init_start = clock (); */
/* std::chrono::high_resolution_clock::time_point t_init = std::chrono::high_resolution_clock::now(); */
#if PRINT == 1
print("Initializing circuit ...\n");
#endif
/* //replace with vector soon !! */
sockets_received.push_back(0);
for(int t=0;t<(num_players-1);t++) { // ???
    #if LIVE == 1 
    /* receiving_args[t].elements_to_rec = std::vector<int>(); */ 
    /* sending_args[t].elements_to_send = std::vector<int>(); */
    receiving_args[t].elements_to_rec.push_back(0);
    sending_args[t].elements_to_send.push_back(0);
#endif
    #if PRE == 1
    /* receiving_args_pre[t].elements_to_rec = std::vector<int>(); */ 
    /* sending_args_pre[t].elements_to_send = std::vector<int>(); */
    receiving_args_pre[t].elements_to_rec.push_back(0);
    sending_args_pre[t].elements_to_send.push_back(0);
    receiving_args_pre[t].rec_rounds = 1;
    sending_args_pre[t].send_rounds = 1;
    #endif
    }
    #if INIT == 1 && NO_INI == 0
    /* auto p_init = PROTOCOL_INIT<DATATYPE>(); */
    auto garbage = new RESULTTYPE;
    FUNCTION<PROTOCOL_INIT<DATATYPE>>(garbage);
    #if MAL==1
        compare_views_init();
        /* p_init.communicate(); */
    #endif

    #if PRE == 1
        PROTOCOL_INIT<DATATYPE>::finalize(ips,receiving_args_pre,sending_args_pre);
    #else
        PROTOCOL_INIT<DATATYPE>::finalize(ips); //TODO change to new version
    #endif
    /* #if LIVE == 0 */
    /*     export_Details_to_file(); */
    /* #endif */
#endif
#if LIVE == 1 && INIT == 0 && NO_INI == 0
    init_from_file();
    finalize(ips);
#endif


}


#if PRE == 1
void preprocess_circuit(std::string ips[])
{
pthread_t sending_Threads_pre[num_players-1];
pthread_t receiving_threads_pre[num_players-1];
int ret_pre;

    for(int t=0;t<(num_players-1);t++) {
        ret_pre = pthread_create(&receiving_threads_pre[t], NULL, receiver, &receiving_args_pre[t]);
        if (ret_pre){
            print("ERROR; return code from pthread_create() is %d\n", ret_pre);
            exit(-1);
            }
    }

    /// Creating sending threads
    for(int t=0;t<(num_players-1);t++) {
        ret_pre = pthread_create(&sending_Threads_pre[t], NULL, sender, &sending_args_pre[t]);
        if (ret_pre){
            print("ERROR; return code from pthread_create() is %d\n", ret_pre);
            exit(-1);
            }
    }



    // waiting until all threads connected
    //#endif

    pthread_mutex_lock(&mtx_connection_established);
    while (num_successful_connections < 2 * (num_players -1)) {
    pthread_cond_wait(&cond_successful_connection, &mtx_connection_established);
    }
    num_successful_connections = -1; 
    pthread_cond_broadcast(&cond_start_signal); //signal all threads to start receiving
    pthread_mutex_unlock(&mtx_connection_established);
    print("All parties connected sucessfully, starting protocol and timer! \n");


#if PRINT == 1
print("Preprocessing phase ...\n");
#endif
clock_t time_pre_function_start = clock ();
clock_gettime(CLOCK_REALTIME, &p1);
std::chrono::high_resolution_clock::time_point p =
            std::chrono::high_resolution_clock::now();

#if PROTOCOL_PRE == -1
            // receive only
    #else
        /* auto p_pre = PROTOCOL_PRE(); */
        auto garbage_PRE = new RESULTTYPE;
        FUNCTION<PROTOCOL_PRE<DATATYPE>>(garbage_PRE);
    #endif
    // manual send

    sb = 0;      
    pthread_mutex_lock(&mtx_send_next); 
     sending_rounds +=1;
      pthread_cond_broadcast(&cond_send_next); //signal all threads that sending buffer contains next data
      pthread_mutex_unlock(&mtx_send_next); 

    
    // manual receive

    rounds+=1;  
        // receive_data
      //wait until all sockets have finished received their last data
    pthread_mutex_lock(&mtx_receive_next);
      
    while(rounds > receiving_rounds) //wait until all threads received their data
          pthread_cond_wait(&cond_receive_next, &mtx_receive_next);
      
    pthread_mutex_unlock(&mtx_receive_next);

    rb = 0;
    
    // Join threads to avoid address rebind
    for(int t=0;t<(num_players-1);t++) {
    pthread_join(receiving_threads_pre[t],NULL);
    pthread_join(sending_Threads_pre[t],NULL);
    }

    double time_pre = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now() - p)
                         .count();
    /* searchComm__<Sharemind,DATATYPE>(protocol,*found); */
    clock_gettime(CLOCK_REALTIME, &p2);
    double accum_pre = ( p2.tv_sec - p1.tv_sec )
    + (double)( p2.tv_nsec - p1.tv_nsec ) / (double) 1000000000L;
    clock_t time_pre_function_finished = clock ();
   

    print("Time measured to perform preprocessing clock: %fs \n", double((time_pre_function_finished - time_pre_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform preprocessing getTime: %fs \n", accum_pre);
    print("Time measured to perform preprocessing chrono: %fs \n", time_pre / 1000000);

#if LIVE == 1
    // reset all variables
num_successful_connections = 0;
std::fill(sockets_received.begin(), sockets_received.end(), 0);
share_buffer[0] = 0;
share_buffer[1] = 0;
send_count[0] = 0;
send_count[1] = 0;
rb = 0;
sb = 0;
rounds = 0;
sending_rounds = 0;
receiving_rounds = 0;
    #if INIT == 0 && NO_INI == 0
         init_from_file();
        finalize(ips);
    #else
        auto p_init = PROTOCOL_INIT<DATATYPE>();
        p_init.finalize(ips);
        /* p_init.finalize(ips,receiving_args,sending_args); */
    #endif
#endif

}
#endif


#if LIVE == 1
void live_circuit()
{
pthread_t sending_Threads[num_players-1];
pthread_t receiving_threads[num_players-1];
int ret;

//TODO check, recently commented
    for(int t=0;t<(num_players-1);t++) {
        ret = pthread_create(&receiving_threads[t], NULL, receiver, &receiving_args[t]);
        if (ret){
            print("ERROR; return code from pthread_create() is %d\n", ret);
            exit(-1);
            }
    }

    /// Creating sending threads



    for(int t=0;t<(num_players-1);t++) {
        ret = pthread_create(&sending_Threads[t], NULL, sender, &sending_args[t]);
        if (ret){
            print("ERROR; return code from pthread_create() is %d\n", ret);
            exit(-1);
            }
    }



    // waiting until all threads connected

    /* printf("m: locking conn \n"); */
    pthread_mutex_lock(&mtx_connection_established);
    /* printf("m: locked conn \n"); */
    while (num_successful_connections < 2 * (num_players -1)) {
    /* printf("m: unlocking conn and waiting \n"); */
    pthread_cond_wait(&cond_successful_connection, &mtx_connection_established);
    }
    /* printf("m: done waiting, modifying conn \n"); */
    num_successful_connections = -1; 
    pthread_cond_broadcast(&cond_start_signal); //signal all threads to start receiving
    pthread_mutex_unlock(&mtx_connection_established);
    /* printf("m: unlocked conn \n"); */
    print("All parties connected sucessfully, starting protocol and timer! \n");
    clock_gettime(CLOCK_REALTIME, &l1);
    /* clock_gettime(CLOCK_REALTIME, &i3); */




    /// Processing Inputs ///
    /* Sharemind protocol = Sharemind(); */
    clock_t time_function_start = clock ();
    std::chrono::high_resolution_clock::time_point c1 =
            std::chrono::high_resolution_clock::now();
    
    /* auto p_live = PROTOCOL_LIVE(); */
    auto result = new RESULTTYPE;
    FUNCTION<PROTOCOL_LIVE<DATATYPE>>(result);
    #if MAL==1
        compare_views();
        /* p_live.communicate(); */
    #endif
    
    
    
    for(int t=0;t<(num_players-1);t++) {
        pthread_join(receiving_threads[t],NULL);
        pthread_join(sending_Threads[t],NULL);
        /* sending_args[t].elements_to_send.clear(); */

    }

    double time = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now() - c1)
                         .count();
    /* searchComm__<Sharemind,DATATYPE>(protocol,*found); */
    clock_gettime(CLOCK_REALTIME, &l2);
    double accum = ( l2.tv_sec - l1.tv_sec )
    + (double)( l2.tv_nsec - l1.tv_nsec ) / (double) 1000000000L;
    #if PRINT == 1
    print_result(result); //different for other functions
    #endif
    clock_t time_function_finished = clock ();

    double init_time = ( l1.tv_sec - i1.tv_sec )
    + (double)( l1.tv_nsec - i1.tv_nsec ) / (double) 1000000000L;
    #if PRE == 1
    double accum_pre = ( p2.tv_sec - p1.tv_sec )
    + (double)( p2.tv_nsec - p1.tv_nsec ) / (double) 1000000000L;
    init_time = init_time - accum_pre;
    #endif
    print("Time measured to initialize program: %fs \n", init_time);
    /* printf("Time measured to read and receive inputs: %fs \n", double((time_data_received - time_application_start)) / CLOCKS_PER_SEC); */
    print("Time measured to perform computation clock: %fs \n", double((time_function_finished - time_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform computation getTime: %fs \n", accum);
    print("Time measured to perform computation chrono: %fs \n", time / 1000000);
    // Join threads to ensure closing of sockets
    print_layer_stats();

}
#endif

#if PROTOCOL != 13
void executeProgram(int argc, char *argv[], int process_id, int process_num)
{
clock_gettime(CLOCK_REALTIME, &i1);

player_id = PARTY;

#if num_players == 3
/* init_srng(0,0); */
/* init_srng(1,0); */
/* init_srng(2,0); */
init_srng(PPREV, modulo((player_id - 1),  num_players)*101011 + 5000);
init_srng(PNEXT,player_id*101011 + 5000);
init_srng(num_players-1, player_id * 291932 +6000); // used for sharing inputs
#elif num_players == 4
init_srng(0,0);
init_srng(1,0);
init_srng(2,0);
init_srng(3,0);
init_srng(4,0);
init_srng(5,0);
init_srng(6,0);
init_srng(7,0);
#endif


/// Connecting to other Players
std::string ips[num_players-1];

//char* hostnames[num_players-1];
for(int i=0; i < num_players -1; i++)
{
    if(i < argc - 1 )
        ips[i] = std::string(argv[i+1]);
    else
    {
        ips[i] = "127.0.0.1";
    }
}


init_muetexes();


generateElements();

init_circuit(ips);

#if PRE == 1
    preprocess_circuit(ips);
#endif

#if PRE == 1 && LIVE == 0
double dummy_time = 0.00;
    print("Time measured to initialize program: %fs \n", dummy_time);
    print("Time measured to perform computation clock: %fs \n", dummy_time);
    print("Time measured to perform computation getTime: %fs \n", dummy_time);
    print("Time measured to perform computation chrono: %fs \n", dummy_time);
#endif

#if LIVE == 1
    live_circuit();
#endif

}


#else
void simulate_live()
{
    current_phase = PHASE_LIVE;
    clock_t time_function_start = clock ();
    clock_gettime(CLOCK_REALTIME, &l1);
    std::chrono::high_resolution_clock::time_point c1 =
            std::chrono::high_resolution_clock::now();
    
    auto result = new RESULTTYPE;
    FUNCTION<PROTOCOL_LIVE<DATATYPE>>(result);

    double time = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::high_resolution_clock::now() - c1)
                         .count();
    /* searchComm__<Sharemind,DATATYPE>(protocol,*found); */
    clock_gettime(CLOCK_REALTIME, &l2);
    double accum = ( l2.tv_sec - l1.tv_sec )
    + (double)( l2.tv_nsec - l1.tv_nsec ) / (double) 1000000000L;
    #if PRINT == 1
    print_result(result); //different for other functions
    #endif
    clock_t time_function_finished = clock ();

    print("Time measured to perform computation clock: %fs \n", double((time_function_finished - time_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform computation getTime: %fs \n", accum);
    print("Time measured to perform computation chrono: %fs \n", time / 1000000);
    // Join threads to ensure closing of sockets

}

void executeProgram(int argc, char *argv[], int process_id, int process_num)
{
    current_phase = PHASE_LIVE;
    init_srng(0,0);
    generateElements();
    simulate_live();
}
#endif
