#pragma once
#include "config.h"
#include "core/init.hpp"
#include "core/networking/buffers.h"
#include "protocols/Protocols.h"
#include "protocols/live_protocol_base.hpp"
#include <sys/ucontext.h>
#if BEAVER == 1
#include "protocols/beaver_triples.hpp"
#endif
#if CV_FIX == 1 && PROTOCOL == 12
#include "programs/functions/check_eqs.hpp"
#endif

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
#elif FUNCTION_IDENTIFIER == 53
#include "programs/tests/test_conv_pool.hpp"
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
        receiving_args_pre[t].rec_rounds = 0;
        sending_args_pre[t].send_rounds = 0;
// #if PROTOCOL != 4
//         receiving_args_pre[t].total_rounds += 1;
//         receiving_args_pre[t].rec_rounds += 1;
// #endif
#endif
    }
#if PRE == 1 && BEAVER == 1
    num_arithmetic_triples.push_back(0);  // temporary
    num_arithmetic_triples.push_back(0);
    num_boolean_triples.push_back(0);
    num_boolean_triples.push_back(0);
    num_ab2_boolean_triples.push_back(0);
    num_ab2_boolean_triples.push_back(0);
    num_ab2_arithmetic_triples.push_back(0);
    num_ab2_arithmetic_triples.push_back(0);
    num_boolean_addition_triples = 0;
    // boolean_triple_index.push_back(0);
    // boolean_triple_index.push_back(0);
    // arithmetic_triple_index.push_back(0);
    // arithmetic_triple_index.push_back(0);
    preprocessed_outputs_arithmetic_index = new uint64_t[2]{0};
    preprocessed_outputs_bool_index = new uint64_t[2]{0};
    preprocessed_outputs_arithmetic_input_index = new uint64_t[2]{0};
    preprocessed_outputs_bool_input_index = new uint64_t[2]{0};
#endif

#if INIT == 1 && NO_INI == 0
    RESULTTYPE garbage;
    FUNCTION<PROTOCOL_INIT<DATATYPE>>(&garbage);
    /* delete garbage; */
#if CV_FIX == 1 && PROTOCOL == 12
    check_eqs_quad_init();
#endif
#if PRE == 1 && BEAVER == 1
    PROTOCOL_INIT<DATATYPE>::complete_preprocessing();
#elif PRE == 1
    communicate_pre_();
#endif

#if MAL == 1 && (CV_FIX == 0 || PROTOCOL != 12)
    compare_views_init();
#endif

#if PRE == 1 && SKIP_PRE == 0
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

#if BEAVER == 1
void beaver(std::string ips[])
{
    for (auto i : num_arithmetic_triples)
        total_arithmetic_triples_num += i;
    for (auto i : num_boolean_triples)
        total_boolean_triples_num += i;
    for (auto i : num_ab2_arithmetic_triples)
        total_ab2_arithmetic_triples_num += i;
    for (auto i : num_ab2_boolean_triples)
        total_ab2_boolean_triples_num += i;
#if LX_TRIPLES == 1
    init_beaverAB(0);
    init_beaverAB2(0);
    init_ConvAB();
    init_BatchNorm2DAB();
    init_FullyConnectedAB();
#if A2B_ONLINE_OPT == 1
    init_booleanAdditionBeaverAB();
#endif
#if BIT_INJECTION_PREPROCESSING_OPT == 1
    init_multiplexerBeaverAB();
    init_cotBeaverAB();
#endif
#else
    init_beaver();
#endif
    print_num_triples();
#if SKIP_PRE == 1
    print("SKIP_PRE set to 1, skipping preprocessing phase and Beaver triples generation ... \n");
    return;
#elif LX_TRIPLES == 1
    return;
#else

#if FAKE_TRIPLES == 1
    print("Fake Triples set to 1, generating fake triples ... \n");
#else
    print("Generating Beaver Triples ... \n");
#endif
    clock_t time_beaver_function_start = clock();
    clock_gettime(CLOCK_REALTIME, &p1);
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();

    generate_beaver_triples(ips, base_port, process_offset);

    clock_gettime(CLOCK_REALTIME, &p2);
    double accum_beaver = (p2.tv_sec - p1.tv_sec) + (double)(p2.tv_nsec - p1.tv_nsec) / (double)1000000000L;
    clock_t time_beaver_function_finished = clock();
    print("Time measured to perform beaver triple generation clock: %fs \n",
          double((time_beaver_function_finished - time_beaver_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform beaver triple generation getTime: %fs \n", accum_beaver);
    print("Time measured to perform beaver triple generation chrono: %fs \n",
          double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - p)
                     .count()) /
              1000000);
#endif
}
#endif

// #if BEAVER == 1 && LX_TRIPLES == 1
// void generate_triples_from_lx_ly()
// {
// #if FAKE_TRIPLES == 1
//     print("Fake Triples set to 1, generating fake triples ... \n");
// #else
//     print("Generating Beaver Triples ... \n");
// #endif
//     clock_t time_beaver_function_start = clock();
//     clock_gettime(CLOCK_REALTIME, &p1);
//     std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();

//     generate_beaver_triples(ips, base_port, process_offset);

//     clock_gettime(CLOCK_REALTIME, &p2);
//     double accum_beaver = (p2.tv_sec - p1.tv_sec) + (double)(p2.tv_nsec - p1.tv_nsec) / (double)1000000000L;
//     clock_t time_beaver_function_finished = clock();
//     print("Time measured to perform beaver triple generation clock: %fs \n",
//           double((time_beaver_function_finished - time_beaver_function_start)) / CLOCKS_PER_SEC);
//     print("Time measured to perform beaver triple generation getTime: %fs \n", accum_beaver);
//     print("Time measured to perform beaver triple generation chrono: %fs \n",
//           double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - p)
//                      .count()) /
//               1000000);

// }
// #endif
#if STORE_PREPROCESSING == 1
void store_preprocessed_data() 
{
    #if ROT_PREPROCESSING_OPT == 1
    float total_ouputs_MB = (total_num_boolean_output_triples[0] + total_num_arithmetic_output_triples[0] +
                             total_num_boolean_output_triples[1] + total_num_arithmetic_output_triples[1] +
                             total_boolean_triples_num * 3 + num_random_multiplications * 2 +
                             num_beaver_3_tuples * 7 + num_beaver_4_tuples *15 +
                             total_preprocessed_outputs)/1000000.0 * sizeof(DATATYPE);
    #else
    float total_ouputs_MB = (total_num_boolean_output_triples[0] + total_num_arithmetic_output_triples[0] +
                             total_num_boolean_output_triples[1] + total_num_arithmetic_output_triples[1] +
                             total_preprocessed_outputs)/1000000.0 * sizeof(DATATYPE);
    #endif

    print("Storing %f MB of preprocessing data to file \n", total_ouputs_MB);
    
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();
    #if ROT_PREPROCESSING_OPT == 1
    store_preprocessed_data(preprocessed_outputs_bool[0], total_num_boolean_output_triples[0],
                            preprocessed_outputs_arithmetic[0], total_num_arithmetic_output_triples[0],
                            preprocessed_outputs_bool[1], total_num_boolean_output_triples[1],
                            preprocessed_outputs_arithmetic[1], total_num_arithmetic_output_triples[1],
                            boolean_triple_a, boolean_triple_b, boolean_triple_c, total_boolean_triples_num,
                            random_multiplication_a, random_multiplication_b, num_random_multiplications,
                            beaver_3_tuples.a, beaver_3_tuples.b, beaver_3_tuples.c,
                            beaver_3_tuples.ab, beaver_3_tuples.ac, beaver_3_tuples.bc,
                            beaver_3_tuples.abc, num_beaver_3_tuples,
                            beaver_4_tuples.a, beaver_4_tuples.b, beaver_4_tuples.c, beaver_4_tuples.d,
                            beaver_4_tuples.ab, beaver_4_tuples.ac, beaver_4_tuples.ad,
                            beaver_4_tuples.bc, beaver_4_tuples.bd, beaver_4_tuples.cd,
                            beaver_4_tuples.abc, beaver_4_tuples.abd, beaver_4_tuples.acd, beaver_4_tuples.bcd,
                            beaver_4_tuples.abcd, num_beaver_4_tuples,
                            preprocessed_outputs, total_preprocessed_outputs);
    #else
    store_preprocessed_data(preprocessed_outputs_bool[0], total_num_boolean_output_triples[0],
                            preprocessed_outputs_arithmetic[0], total_num_arithmetic_output_triples[0],
                            preprocessed_outputs_bool[1], total_num_boolean_output_triples[1],
                            preprocessed_outputs_arithmetic[1], total_num_arithmetic_output_triples[1],
                            preprocessed_outputs, total_preprocessed_outputs);
    #endif
    delete[] preprocessed_outputs_bool[0];
    delete[] preprocessed_outputs_bool[1];
    delete[] preprocessed_outputs_arithmetic[0];
    delete[] preprocessed_outputs_arithmetic[1];
    delete[] preprocessed_outputs;
    delete[] preprocessed_outputs_bool;
    delete[] preprocessed_outputs_arithmetic;
    #if ROT_PREPROCESSING_OPT == 1
    delete[] boolean_triple_a;
    delete[] boolean_triple_b;
    delete[] boolean_triple_c;
    delete[] random_multiplication_a;
    delete[] random_multiplication_b;
    deinit_beaver_3_tuples();
    deinit_beaver_4_tuples();
    #endif
    print("Time measured to store preprocessing data chrono: %fs \n",
          double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - p)
                     .count()) /
              1000000);
}



#endif
#if LOAD_PREPROCESSING == 1
void load_preprocessed_data()
{
    #if ROT_PREPROCESSING_OPT == 1
    float total_ouputs_MB = (total_num_boolean_output_triples[0] + total_num_arithmetic_output_triples[0] +
                             total_num_boolean_output_triples[1] + total_num_arithmetic_output_triples[1] +
                             total_boolean_triples_num * 3 + num_random_multiplications * 2 +
                             num_beaver_3_tuples * 7 + num_beaver_4_tuples * 15 +
                             total_preprocessed_outputs)/1000000.0 * sizeof(DATATYPE);
    #else
    float total_ouputs_MB = (total_num_boolean_output_triples[0] + total_num_arithmetic_output_triples[0] +
                             total_num_boolean_output_triples[1] + total_num_arithmetic_output_triples[1] +
                             total_preprocessed_outputs)/1000000.0 * sizeof(DATATYPE);
    #endif
    print("Loading %f MB of preprocessing data from file \n", total_ouputs_MB);
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();
    preprocessed_outputs_bool = new DATATYPE*[2];
    preprocessed_outputs_arithmetic = new DATATYPE*[2];
    preprocessed_outputs_bool[0] = new DATATYPE[total_num_boolean_output_triples[0]];
    preprocessed_outputs_bool[1] = new DATATYPE[total_num_boolean_output_triples[1]];
    preprocessed_outputs_arithmetic[0] = new DATATYPE[total_num_arithmetic_output_triples[0]];
    preprocessed_outputs_arithmetic[1] = new DATATYPE[total_num_arithmetic_output_triples[1]];
    preprocessed_outputs = new DATATYPE[total_preprocessed_outputs];

    #if ROT_PREPROCESSING_OPT == 1
    boolean_triple_a = new DATATYPE[total_boolean_triples_num];
    boolean_triple_b = new DATATYPE[total_boolean_triples_num];
    boolean_triple_c = new DATATYPE[total_boolean_triples_num];
    random_multiplication_a = new DATATYPE[num_random_multiplications];
    random_multiplication_b = new DATATYPE[num_random_multiplications];
    init_beaver_3_tuples();
    init_beaver_4_tuples();
    load_preprocessed_data(preprocessed_outputs_bool[0], total_num_boolean_output_triples[0],
                            preprocessed_outputs_arithmetic[0], total_num_arithmetic_output_triples[0],
                            preprocessed_outputs_bool[1], total_num_boolean_output_triples[1],
                            preprocessed_outputs_arithmetic[1], total_num_arithmetic_output_triples[1],
                            boolean_triple_a, boolean_triple_b, boolean_triple_c, total_boolean_triples_num,
                            random_multiplication_a, random_multiplication_b, num_random_multiplications,
                            beaver_3_tuples.a, beaver_3_tuples.b, beaver_3_tuples.c,
                            beaver_3_tuples.ab, beaver_3_tuples.ac, beaver_3_tuples.bc,
                            beaver_3_tuples.abc, num_beaver_3_tuples,
                            beaver_4_tuples.a, beaver_4_tuples.b, beaver_4_tuples.c, beaver_4_tuples.d,
                            beaver_4_tuples.ab, beaver_4_tuples.ac, beaver_4_tuples.ad,
                            beaver_4_tuples.bc, beaver_4_tuples.bd, beaver_4_tuples.cd,
                            beaver_4_tuples.abc, beaver_4_tuples.abd, beaver_4_tuples.acd, beaver_4_tuples.bcd,
                            beaver_4_tuples.abcd, num_beaver_4_tuples,
                            preprocessed_outputs, total_preprocessed_outputs);
    #else
    load_preprocessed_data(preprocessed_outputs_bool[0], total_num_boolean_output_triples[0],
                            preprocessed_outputs_arithmetic[0], total_num_arithmetic_output_triples[0],
                            preprocessed_outputs_bool[1], total_num_boolean_output_triples[1],
                            preprocessed_outputs_arithmetic[1], total_num_arithmetic_output_triples[1],
                            preprocessed_outputs, total_preprocessed_outputs);
    #endif
    print("Time measured to load preprocessing data chrono: %fs \n",
          double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - p)
                     .count()) /
              1000000);
}
#endif


#if PRE == 1 && SKIP_PRE == 0
void preprocess_circuit(std::string ips[])
{
#if PROTOCOL == 4
#if BEAVER == 1
    curr_beaver_3_triple_index = 0;
    curr_beaver_4_triple_index = 0;
    curr_boolean_triple_index = 0;
    curr_arithmetic_triple_index = 0;
    curr_arithmetic_ab2_triple_index = 0;
    curr_boolean_ab2_triple_index = 0;
    curr_random_multiplication_index = 0;
#endif
#if ROT_PREPROCESSING_OPT == 1 
        clock_t time_pre_function_start = clock();
        clock_gettime(CLOCK_REALTIME, &p1);
        std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();
        init_beaverC_boolean(0);
        generate_beaver_triples(
                ips, base_port, process_offset, 0, num_boolean_triples[0], "LXLY");
        init_beaverAB2C_boolean(0);
        generate_beaver_triples(
                ips, base_port, process_offset, 0, num_ab2_boolean_triples[0], "LXLY2");
#if BEAVER_N_TUPLES == 1
        init_beaver_3_tuples();
        init_beaver_4_tuples();
        generate_beaver_triples(
                ips, base_port, process_offset, 0, 0, "BEAVER_N_TUPLES");
#endif
#if RESHARE_OPT == 1
init_random_multiplications();
generate_beaver_triples(
                ips, base_port, process_offset, 0, 0, "RANDOM_MULTIPLICATION");
#endif
#if A2B_CONV_BAKE_ACTIVE
        // Commit ia/lz and load the boolean-addition inputs, then run the boolean addition EARLY (here,
        // not in complete_preprocessing) so [c] is available to prepare_A2B_S2 in BOTH the PRE and LIVE
        // passes - the msb adder's beaver triples (built in PRE from s2.l = [c]) must match LIVE.
        init_a2b_bake<DATATYPE>(num_boolean_addition_triples, std::minus<DATATYPE>());
        if (num_boolean_addition_triples > 0)
        {
            init_booleanAdditionBeaverC();
            generate_beaver_triples(
                    ips, base_port, process_offset, num_boolean_addition_triples, 0, "BOOLEANADDITION");
            a2b_bake_store_c(num_boolean_addition_triples);
        }
        g_a2b_layer_base = 0;  // conv-mask layer base starts at 0 for the PRE pass
        g_a2b_c_cursor = 0;  // A2B-S2 [c] cursor starts at 0 for the PRE pass
#endif
#endif
#if CHEETAH_DISCONNECT == 0
    CheetahDisconnect(ips[0], base_port + process_offset);
#endif
#endif
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

#if ROT_PREPROCESSING_OPT == 0
    clock_t time_pre_function_start = clock();
    clock_gettime(CLOCK_REALTIME, &p1);
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();
#endif
    

#if PROTOCOL_PRE == -1
    // receive only
#else
    RESULTTYPE garbage_PRE;
    FUNCTION<PROTOCOL_PRE<DATATYPE>>(&garbage_PRE);
    #if CV_FIX == 1 && PROTOCOL == 12
        check_eqs_quad_pre();
    #endif
#endif
    // manual send

    /* sb = 0; */
    /* pthread_mutex_lock(&mtx_send_next); */
    /* sending_rounds += 1; */
    /* pthread_cond_broadcast(&cond_send_next); // signal all threads that sending */
    /*                                          // buffer contains next data */
    /* pthread_mutex_unlock(&mtx_send_next); */

    /* // manual receive */

    /* rounds += 1; */
    /* // receive_data */
    /* // wait until all sockets have finished received their last data */
    /* pthread_mutex_lock(&mtx_receive_next); */

    /* while (rounds > receiving_rounds) // wait until all threads received their */
    /*                                   // data */
    /*   pthread_cond_wait(&cond_receive_next, &mtx_receive_next); */

    /* pthread_mutex_unlock(&mtx_receive_next); */

    /* rb = 0; */

#if BEAVER == 1
    PROTOCOL_PRE<DATATYPE>::complete_preprocessing(
        ips, base_port, process_offset);
    Iface::printTripleStats(CHEETAH_PARTY, process_offset);
    Iface::resetTripleStats();
#else
    communicate_pre();
#endif

    // Join threads to avoid address rebind
    for (int t = 0; t < (num_players - 1); t++)
    {
        pthread_join(receiving_threads_pre[t], NULL);
        pthread_join(sending_Threads_pre[t], NULL);
    }

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
    curr_boolean_triple_index = 0;
#if INIT == 0 && NO_INI == 0
    init_from_file();
    finalize(ips);
#else
    auto p_init = PROTOCOL_INIT<DATATYPE>();
    p_init.finalize(ips);
#endif
#endif

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

#if BEAVER == 1
    curr_beaver_3_triple_index = 0;
    curr_beaver_4_triple_index = 0;
    curr_boolean_triple_index = 0;
    curr_arithmetic_triple_index = 0;
    curr_arithmetic_ab2_triple_index = 0;
    curr_boolean_ab2_triple_index = 0;
    curr_random_multiplication_index = 0;
#endif
#if A2B_CONV_BAKE_ACTIVE
    g_a2b_layer_base = 0;  // conv-mask layer base restarts for the LIVE pass (same committed lz)
    g_a2b_c_cursor = 0;  // A2B-S2 [c] cursor restarts for the LIVE pass (same committed [c] as PRE)
#endif

    /// Processing Inputs ///
    /* Sharemind protocol = Sharemind(); */
    clock_t time_function_start = clock();
    std::chrono::high_resolution_clock::time_point c1 = std::chrono::high_resolution_clock::now();

    /* auto p_live = PROTOCOL_LIVE(); */
    RESULTTYPE result;
    FUNCTION<PROTOCOL_LIVE<DATATYPE>>(&result);
#if MAL == 1
#if CV_FIX == 1 && PROTOCOL == 12
    check_eqs_quad_live();
#else
    compare_views();
#endif
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

#if BEAVER == 1 

#if LX_TRIPLES == 1
#else
    deinit_beaver();
#endif
#endif

#if FUNCTION_IDENTIFIER >= 70
    print_layer_stats();
#endif
    /* delete result; //alternatively do something with the result */
}
#endif

#if PROTOCOL != 13
void executeProgram(int argc, char* argv[], int process_id, int process_num)
{
    clock_gettime(CLOCK_REALTIME, &i1);
    player_id = PARTY;
    init_srngs();
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

#if BEAVER == 1
    beaver(ips);
#endif

#if PRE == 1 && SKIP_PRE == 0
    preprocess_circuit(ips);

#endif


#if STORE_PREPROCESSING == 1 && SKIP_PRE == 0
store_preprocessed_data();
#endif
#if LOAD_PREPROCESSING == 1 
load_preprocessed_data();
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

    RESULTTYPE result;
    FUNCTION<PROTOCOL_LIVE<DATATYPE>>(&result);

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
