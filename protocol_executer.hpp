#pragma once
#include "core/init.hpp"
#include "protocols/Protocols.h"

#if FUNCTION_IDENTIFIER < 8
#include "programs/benchmarks/bench_basic_primitives.hpp"
#elif FUNCTION_IDENTIFIER < 13
#include "programs/benchmarks/bench_rounds.hpp"
#elif FUNCTION_IDENTIFIER < 24
#include "programs/benchmarks/bench_statistics.hpp"
#elif FUNCTION_IDENTIFIER < 33
#include "programs/benchmarks/bench_use_cases.hpp"
#elif FUNCTION_IDENTIFIER < 48
#include "programs/benchmarks/bench_nn.hpp"
#elif FUNCTION_IDENTIFIER < 54
#include "programs/benchmarks/bench_conv_alt.hpp"
#elif FUNCTION_IDENTIFIER == 54
#include "programs/tests/test_basic_primitives.hpp"
#elif FUNCTION_IDENTIFIER == 55
#include "programs/tests/test_fixed_point_arithmetic.hpp"
#elif FUNCTION_IDENTIFIER == 56
#include "programs/tests/test_truncation.hpp"
#elif FUNCTION_IDENTIFIER == 57
#include "programs/tests/test_mat_mul.hpp"
#elif FUNCTION_IDENTIFIER == 58
#include "programs/tests/test_multi_input.hpp"
#elif FUNCTION_IDENTIFIER == 59
#include "programs/tests/test_comparisons.hpp"
#elif FUNCTION_IDENTIFIER == 60
#include "programs/tests/test_all.hpp"
#elif FUNCTION_IDENTIFIER == 61
#include "programs/tutorials/basic_tutorial.hpp"
#elif FUNCTION_IDENTIFIER == 62
#include "programs/tutorials/fixed_point_tutorial.hpp"
#elif FUNCTION_IDENTIFIER == 63
#include "programs/tutorials/mixed_circuits_tutorial.hpp"
#elif FUNCTION_IDENTIFIER == 64
#include "programs/tutorials/matrix_operations_tutorial.hpp"
#elif FUNCTION_IDENTIFIER == 65
#include "programs/tutorials/YourFirstProgram.hpp"
#elif FUNCTION_IDENTIFIER < 400
#include "programs/NN.hpp"
#elif FUNCTION_IDENTIFIER >= 500 && FUNCTION_IDENTIFIER <= 534
#include "programs/functions/mpspdz.hpp"
#endif

void init_circuit(std::string ips[])
{

#if PRINT == 1
    print("Initializing circuit ...\n");
#endif
    sockets_received.push_back(0);
    for (int t = 0; t < (num_players - 1); t++)
    {
#if LIVE == 1
        receiving_args[t].elements_to_rec.push_back(0);
        sending_args[t].elements_to_send.push_back(0);
#endif
#if PRE == 1
        receiving_args_pre[t].elements_to_rec.push_back(0);
        sending_args_pre[t].elements_to_send.push_back(0);
        receiving_args_pre[t].rec_rounds = 1;
        sending_args_pre[t].send_rounds = 1;
#endif
    }
#if INIT == 1 && NO_INI == 0
    auto garbage = new RESULTTYPE;
    FUNCTION<PROTOCOL_INIT<DATATYPE>>(garbage);
#if MAL == 1
    compare_views_init();
#endif

#if PRE == 1
    PROTOCOL_INIT<DATATYPE>::finalize(ips, receiving_args_pre, sending_args_pre);
#else
    PROTOCOL_INIT<DATATYPE>::finalize(ips);  // TODO change to new version
#endif
#endif
#if LIVE == 1 && INIT == 0 && NO_INI == 0
    init_from_file();
    finalize(ips);
#endif

#if TRUNC_DELAYED == 1
    delayed = false;
#endif
}

#if PRE == 1
void preprocess_circuit(std::string ips[])
{
    pthread_t sending_Threads_pre[num_players - 1];
    pthread_t receiving_threads_pre[num_players - 1];
    int ret_pre;

    for (int t = 0; t < (num_players - 1); t++)
    {
        ret_pre = pthread_create(&receiving_threads_pre[t], NULL, receiver, &receiving_args_pre[t]);
        if (ret_pre)
        {
            print("ERROR; return code from pthread_create() is %d\n", ret_pre);
            exit(-1);
        }
    }

    /// Creating sending threads
    for (int t = 0; t < (num_players - 1); t++)
    {
        ret_pre = pthread_create(&sending_Threads_pre[t], NULL, sender, &sending_args_pre[t]);
        if (ret_pre)
        {
            print("ERROR; return code from pthread_create() is %d\n", ret_pre);
            exit(-1);
        }
    }

    // waiting until all threads connected
    // #endif

    pthread_mutex_lock(&mtx_connection_established);
    while (num_successful_connections < 2 * (num_players - 1))
    {
        pthread_cond_wait(&cond_successful_connection, &mtx_connection_established);
    }
    num_successful_connections = -1;
    pthread_cond_broadcast(&cond_start_signal);  // signal all threads to start receiving
    pthread_mutex_unlock(&mtx_connection_established);
    print("All parties connected sucessfully, starting protocol and timer! \n");

#if PRINT == 1
    print("Preprocessing phase ...\n");
#endif
    clock_t time_pre_function_start = clock();
    clock_gettime(CLOCK_REALTIME, &p1);
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();

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
    sending_rounds += 1;
    pthread_cond_broadcast(&cond_send_next);  // signal all threads that sending
                                              // buffer contains next data
    pthread_mutex_unlock(&mtx_send_next);

    // manual receive

    rounds += 1;
    // receive_data
    // wait until all sockets have finished received their last data
    pthread_mutex_lock(&mtx_receive_next);

    while (rounds > receiving_rounds)  // wait until all threads received their
                                       // data
        pthread_cond_wait(&cond_receive_next, &mtx_receive_next);

    pthread_mutex_unlock(&mtx_receive_next);

    rb = 0;

    // Join threads to avoid address rebind
    for (int t = 0; t < (num_players - 1); t++)
    {
        pthread_join(receiving_threads_pre[t], NULL);
        pthread_join(sending_Threads_pre[t], NULL);
    }

    double time_pre =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - p).count();
    /* searchComm__<Sharemind,DATATYPE>(protocol,*found); */
    clock_gettime(CLOCK_REALTIME, &p2);
    double accum_pre = (p2.tv_sec - p1.tv_sec) + (double)(p2.tv_nsec - p1.tv_nsec) / (double)1000000000L;
    clock_t time_pre_function_finished = clock();

    print("Time measured to perform preprocessing clock: %fs \n",
          double((time_pre_function_finished - time_pre_function_start)) / CLOCKS_PER_SEC);
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
#endif
#endif
#if TRUNC_DELAYED == 1
    delayed = false;
#endif
}
#endif

#if LIVE == 1
void live_circuit()
{
    pthread_t sending_Threads[num_players - 1];
    pthread_t receiving_threads[num_players - 1];
    int ret;

    // TODO check, recently commented
    for (int t = 0; t < (num_players - 1); t++)
    {
        ret = pthread_create(&receiving_threads[t], NULL, receiver, &receiving_args[t]);
        if (ret)
        {
            print("ERROR; return code from pthread_create() is %d\n", ret);
            exit(-1);
        }
    }

    /// Creating sending threads

    for (int t = 0; t < (num_players - 1); t++)
    {
        ret = pthread_create(&sending_Threads[t], NULL, sender, &sending_args[t]);
        if (ret)
        {
            print("ERROR; return code from pthread_create() is %d\n", ret);
            exit(-1);
        }
    }

    // waiting until all threads connected
    /* printf("m: locking conn \n"); */
    print("Initialized circuit, waiting for all parties to connect ... \n");
    pthread_mutex_lock(&mtx_connection_established);
    /* printf("m: locked conn \n"); */
    while (num_successful_connections < 2 * (num_players - 1))
    {
        /* printf("m: unlocking conn and waiting \n"); */
        pthread_cond_wait(&cond_successful_connection, &mtx_connection_established);
    }
    /* printf("m: done waiting, modifying conn \n"); */
    num_successful_connections = -1;
    pthread_cond_broadcast(&cond_start_signal);  // signal all threads to start receiving
    pthread_mutex_unlock(&mtx_connection_established);
    /* printf("m: unlocked conn \n"); */
    print("All parties connected sucessfully, starting protocol and timer! \n");
    clock_gettime(CLOCK_REALTIME, &l1);
    /* clock_gettime(CLOCK_REALTIME, &i3); */

    /// Processing Inputs ///
    /* Sharemind protocol = Sharemind(); */
    clock_t time_function_start = clock();
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();

    /* auto p_live = PROTOCOL_LIVE(); */
    auto result = new RESULTTYPE;
    FUNCTION<PROTOCOL_LIVE<DATATYPE>>(result);
#if MAL == 1
    compare_views();
    /* p_live.communicate(); */
#endif

    for (int t = 0; t < (num_players - 1); t++)
    {
        pthread_join(receiving_threads[t], NULL);
        pthread_join(sending_Threads[t], NULL);
        /* sending_args[t].elements_to_send.clear(); */
    }

    double time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - c1).count();
    /* searchComm__<Sharemind,DATATYPE>(protocol,*found); */
    clock_gettime(CLOCK_REALTIME, &l2);
    double accum = (l2.tv_sec - l1.tv_sec) + (double)(l2.tv_nsec - l1.tv_nsec) / (double)1000000000L;
#if PRINT == 1
    print_result(result);  // different for other functions
#endif
    clock_t time_function_finished = clock();

    double init_time = (l1.tv_sec - i1.tv_sec) + (double)(l1.tv_nsec - i1.tv_nsec) / (double)1000000000L;
#if PRE == 1
    double accum_pre = (p2.tv_sec - p1.tv_sec) + (double)(p2.tv_nsec - p1.tv_nsec) / (double)1000000000L;
    init_time = init_time - accum_pre;
#endif
    print("Time measured to initialize program: %fs \n", init_time);
    print("Time measured to perform computation clock: %fs \n",
          double((time_function_finished - time_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform computation getTime: %fs \n", accum);
    print("Time measured to perform computation chrono: %fs \n", time / 1000000);
    // Join threads to ensure closing of sockets
#if FUNCTION_IDENTIFIER >= 70
    print_layer_stats();
#endif
}
#endif

#if PROTOCOL != 13
void executeProgram(int argc, char* argv[], int process_id, int process_num)
{
    clock_gettime(CLOCK_REALTIME, &i1);

    player_id = PARTY;
#if num_players == 2
    init_srng(PPREV, SRNG_SEED);
    init_srng(PSELF, PARTY + 1 + SRNG_SEED);
#elif num_players == 3
/* init_srng(PPREV,SRNG_SEED); */
/* init_srng(PNEXT,SRNG_SEED); */
/* init_srng(PSELF,PARTY+1+SRNG_SEED); */
#if PROTOCOL == 6 || PROTOCOL == 7  // TTP
    init_srng(0, SRNG_SEED);
    init_srng(1, SRNG_SEED);
    init_srng(2, SRNG_SEED);
#else
    init_srng(PPREV, modulo((player_id - 1), num_players) * 101011 + 5000 + SRNG_SEED);
    init_srng(PNEXT, player_id * 101011 + 5000 + SRNG_SEED);
    init_srng(num_players - 1,
              player_id * 291932 + 6000 + SRNG_SEED);  // used for sharing inputs
#endif
#elif num_players == 4
    init_srng(0, SRNG_SEED);
    init_srng(1, SRNG_SEED);
    init_srng(2, SRNG_SEED);
    init_srng(3, SRNG_SEED);
    init_srng(4, SRNG_SEED);
    init_srng(5, SRNG_SEED);
    init_srng(6, SRNG_SEED);
    init_srng(7, SRNG_SEED);
#endif

    /// Connecting to other Players
    std::string ips[num_players - 1];

    // char* hostnames[num_players-1];
    for (int i = 0; i < num_players - 1; i++)
    {
        if (i < argc - 1)
            ips[i] = std::string(argv[i + 1]);
        else
        {
            ips[i] = "127.0.0.1";
        }
    }

    init_muetexes();

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
    clock_t time_function_start = clock();
    clock_gettime(CLOCK_REALTIME, &l1);
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();

    auto result = new RESULTTYPE;
    FUNCTION<PROTOCOL_LIVE<DATATYPE>>(result);

    double time =
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - c1).count();
    /* searchComm__<Sharemind,DATATYPE>(protocol,*found); */
    clock_gettime(CLOCK_REALTIME, &l2);
    double accum = (l2.tv_sec - l1.tv_sec) + (double)(l2.tv_nsec - l1.tv_nsec) / (double)1000000000L;
#if PRINT == 1
    print_result(result);  // different for other functions
#endif
    clock_t time_function_finished = clock();

    print("Time measured to perform computation clock: %fs \n",
          double((time_function_finished - time_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform computation getTime: %fs \n", accum);
    print("Time measured to perform computation chrono: %fs \n", time / 1000000);
    // Join threads to ensure closing of sockets
}

void executeProgram(int argc, char* argv[], int process_id, int process_num)
{
    current_phase = PHASE_LIVE;
    init_srng(0, 0);
    simulate_live();
}
#endif
