#pragma once

// === Basic Settings ===

#ifndef PROTOCOL
#define PROTOCOL 5
#endif

// Use a preprocessing phase? Only supported by Protocols 3,5,8,11,12
#ifndef PRE
#define PRE 0
#endif

// Party ID (starting from 0)
#ifndef PARTY
#define PARTY 2
#endif

// Bitlength of integers
#ifndef BITLENGTH
#define BITLENGTH 32
#endif

// Fractional bits to use for fixed point arithmetic
#ifndef FRACTIONAL
#define FRACTIONAL 5
#endif

// 70+ Neural network architectures (LeNet, AlexNet, VGG, ResNet, etc.) on different dataset sizes (MNIST, CIFAR-10,
// Imagenet). Refer to programs/functions/NN.hpp
#ifndef FUNCTION_IDENTIFIER
#define FUNCTION_IDENTIFIER 15
#endif

// Number of inputs (mostly used by Benchmarking functions or Neural Networks)
#ifndef NUM_INPUTS
#define NUM_INPUTS 1
#endif

// === Concurrency Settings ===

// Register size to use for SIMD parallelization (Bitslicing/vectorization). Supported:
// 1,8,16,32,64,128(SSE),256(AVX-2),512(AVX-512) Info: MULT64 is supported by DATTYPE 64 and 512. MULT32 is supported
// for DATTYPE 32 and all DATATYPEs >= 128
#ifndef DATTYPE
#define DATTYPE 32
#endif

// Number of parallel processes to use
#ifndef PROCESS_NUM
#define PROCESS_NUM 1
#endif

// === Hardware Acceleration Settings ===

// 0 = xorshift, 1 = AES_BS, 2 = VAES/AES-NI. 0 is not secure.
#ifndef RANDOM_ALGORITHM
#define RANDOM_ALGORITHM 2
#endif

#ifndef USE_SSL_AES  // USE SSl's AES implementation instead
#define USE_SSL_AES 0
#endif

#ifndef ARM
#define ARM 0  // 1 if ARM processor, 0 otherwise. Can speed up Sha hashing.
#endif

// USE CUDA for matrix multiplication?
#ifndef USE_CUDA_GEMM
#define USE_CUDA_GEMM 0
#endif

// === Tweaks ===

// How many gates should be buffered until sending them to the receiving party? 0 means the data of an entire
// communication round is buffered
#ifndef SEND_BUFFER
#define SEND_BUFFER 10000
#endif

// How many receiving messages should be buffered until the main thread is signaled that data is ready? 0 means that all
// data of a communication round needs to be ready before the main thread is signaled.
#ifndef RECV_BUFFER
#define RECV_BUFFER 10000
#endif

// How many messages should be buffered until a combined hash is performed? 0 means all hashes are calculated at the
// very end of the protocol.
#ifndef VERIFY_BUFFER
#define VERIFY_BUFFER 512 / DATTYPE
#endif

#if PRE == 0
    #ifndef WAIT_AFTER_MESSAGES_IF_AHEAD //Experimental, P0/P3 will not get correct output
    #define WAIT_AFTER_MESSAGES_IF_AHEAD -1  // In case of interleaved computation, the preprocessing party will at most be x messages ahead to save RAM
#endif
#endif
// === Network Settings ===

// Use SSL encrypted communication?
#ifndef USE_SSL
#define USE_SSL 0
#endif

// === Neural Network Settings ===

#ifndef MODELOWNER
#define MODELOWNER \
    -1  // Who holds the model parameters? (-1: Dummy model parameters, P_0/P_1/P_2/P_3: Read locally from
        // P_0/P_1/P_2/P_3 followed by secret sharing). Important: Use "P_0" not "0"!
#endif

#ifndef DATAOWNER
#define DATAOWNER \
    -1  // Who holds the data? (-1: Dummy dataset, P_0/P_1/P_2/P_3: Read locally from P_0/P_1/P_2/P_3 followed by secret
        // sharing). Important: Use "P_0" not "0"!
#endif

#ifndef TRUNC_THEN_MULT
#define TRUNC_THEN_MULT 0  // 0: Truncate after multiplication, 1: Truncate before multiplication
#endif

#ifndef TRUNC_APPROACH
#define TRUNC_APPROACH \
    0  // 0: Probabilistic truncation, 1: Reduced Slack Truncation, 2: Exact Truncation //3: Optimized exact truncation
#endif

#ifndef TRUNC_DELAYED
#define TRUNC_DELAYED 0  // Delay CONV truncation until next ReLU
#endif

#ifndef AVG_OPT
#define AVG_OPT \
    1  // Optimize average pooling truncation failure by reducing fractional bits, Currently only works with
       // trunc_approach 0,1,2,4 (not 3)
#endif

#ifndef AVG_OPT_THRESHOLD
#define AVG_OPT_THRESHOLD 0  // Threshold of tolerated precision decrease for one bit of FRACTIONAL reduction
#endif

#ifndef MSB0_OPT
#define MSB0_OPT 1  // Exploit that the MSB of many layers is 0 when uing truncation
#endif

#ifndef AVG1_OPT  // OPtimze avaerage pooling for denominator = 1
#define AVG1_OPT 1
#endif

#ifndef COMPUTE_ARGMAX
#define COMPUTE_ARGMAX 0  // 0: skip final argmax during inference, 1: Compute final argmax during inference
#endif

#ifndef PUBLIC_WEIGHTS
#define PUBLIC_WEIGHTS 0  // 0: weights are secretly shared, 1: weights are public
#endif

#ifndef COMPRESS
#define COMPRESS 0
#endif

// Reduced Bitlength that might be used for RELU, etc
#if COMPRESS == 0
#ifndef REDUCED_BITLENGTH_k
#define REDUCED_BITLENGTH_k BITLENGTH
#endif

#ifndef REDUCED_BITLENGTH_m
#define REDUCED_BITLENGTH_m 0
#endif

#else
#ifndef REDUCED_BITLENGTH_k
#define REDUCED_BITLENGTH_k 20
#endif

#ifndef REDUCED_BITLENGTH_m
#define REDUCED_BITLENGTH_m 12
#endif

#endif

#ifndef IS_TRAINING
#define IS_TRAINING 0  // Training or inference phase? Training is not supported yet.
#endif

// === Debugging Settings ===

// Print additional info?
#ifndef PRINT
#define PRINT 0
#endif

#ifndef PRINT_TIMINGS
#define PRINT_TIMINGS 1
#endif

#ifndef PRINT_IMPORTANT
#define PRINT_IMPORTANT 1
#endif

// === Other Settings ===

#ifndef FLOATTYPE
#if MODELOWNER == -1 || DATAOWNER == -1
#define FLOATTYPE float
#else
#define FLOATTYPE double  // might be useful for float/fixed conversion
#endif
#endif

#ifndef SRNG_SEED
#define SRNG_SEED 0  // Seed for the random number generator.
#endif

// Starting port for required port range of the sockets, must be multiple of 1000 for some applications
#ifndef BASE_PORT
#define BASE_PORT 10000
#endif

#ifndef SPLIT_ROLES_OFFSET
#define SPLIT_ROLES_OFFSET 0
#endif

int base_port = BASE_PORT;  // temporary solution

// Timeout in seconds when connecting to a socket
#ifndef CONNECTION_TIMEOUT
#define CONNECTION_TIMEOUT 500
#endif

// Timeout in milliseconds before attempting to connect again to a socket
#ifndef CONNECTION_RETRY
#define CONNECTION_RETRY 5
#endif

#ifndef FUSE_DOT
#define FUSE_DOT 1  // Fuse multiple dot products into one
#endif

#ifndef INTERLEAVE_COMM
#define INTERLEAVE_COMM 1  // Interleave communication
#endif

// === Legacy Settings ===

// Allow sharing of inputs in offline phase
#ifndef SHARE_PREP
#define SHARE_PREP 1
#endif

// Use optimized secret sharing? Often utilizes SRNG instead of secret sharing with communication
#ifndef OPT_SHARE
#define OPT_SHARE 1
#endif

// Use the initialization phase or import initialization data from a file?
#ifndef NO_INI
#define NO_INI 0
#endif

// Use the initialization phase or import initialization data from a file?
#ifndef INIT
#define INIT 1
#endif

// Use the online phase?
#ifndef LIVE
#define LIVE 1
#endif

// Use random inputs or inputs from a file?
#ifndef INPUT
#define INPUT 'r'
#endif

// === Internal Settings, do not change ===

#if PROTOCOL == 4
#define HAS_POST_PROTOCOL 1
#elif (PROTOCOL == 3 || PROTOCOL == 5) && PARTY == 0
#define HAS_POST_PROTOCOL 1
#elif (PROTOCOL == 8 || PROTOCOL == 11 || PROTOCOL == 12) && PARTY == 3
#define HAS_POST_PROTOCOL 1
#endif

#if PROTOCOL == 4
#ifndef num_players
#define num_players 2
#endif
#elif PROTOCOL < 7
#ifndef num_players
#define num_players 3
#endif
#else
#ifndef num_players
#define num_players 4
#endif
#endif

#if PROTOCOL > 7
#ifndef MAL
#define MAL 1
#endif
#endif

#if num_players == 2
#define PNEXT 0
#define PPREV 0
#define PSELF 1
#if PARTY == 0
#define P_0 1
#define P_1 0
#elif PARTY == 1
#define P_0 0
#define P_1 1
#endif
#elif num_players == 3
#define PSELF 2
#if PARTY == 0
#define P_0 2
#define P_1 0
#define P_2 1
#define PPREV 1
#define PNEXT 0
#elif PARTY == 1
#define P_0 0
#define P_1 2
#define P_2 1
#define PPREV 0
#define PNEXT 1
#elif PARTY == 2
#define P_0 0
#define P_1 1
#define P_2 2
#define PPREV 1
#define PNEXT 0
#endif
#elif num_players == 4
#define PSELF 3
#define P_0123 3
#define P_012 4
#define P_013 5
#define P_023 6
#define P_123 7
#define P_123_2 3  // Trick for Protocols 10-12

#if PARTY == 0
#define P_0 3
#define P_1 0
#define P_2 1
#define P_3 2
#define PPREV 2
#define PNEXT 0
#define PMIDDLE 1
#elif PARTY == 1
#define P_0 0
#define P_1 3
#define P_2 1
#define P_3 2
#define PPREV 0
#define PNEXT 1
#define PMIDDLE 2
#elif PARTY == 2
#define P_0 0
#define P_1 1
#define P_2 3
#define P_3 2
#define PPREV 1
#define PNEXT 2
#define PMIDDLE 0
#elif PARTY == 3
#define P_0 0
#define P_1 1
#define P_2 2
#define P_3 3
#define PPREV 2
#define PNEXT 0
#define PMIDDLE 1
#endif
#endif

#if WAIT_AFTER_MESSAGES_IF_AHEAD >= 0

#if PROTOCOL == 5
#if PARTY == 0
#define SYNC_PARTY_RECV P_2
#elif PARTY == 2
#define SYNC_PARTY_SEND P_0
#endif
#endif

#if PROTOCOL == 12
#if PARTY == 3
#define SYNC_PARTY_RECV P_2
#define SYNC_PARTY_RECV2 P_0
#elif PARTY == 0 || PARTY == 2
#define SYNC_PARTY_SEND P_3
#endif
#endif

#ifndef SYNC_PARTY_RECV
#define SYNC_PARTY_RECV -1
#endif
#ifndef SYNC_PARTY_SEND
#define SYNC_PARTY_SEND -1
#endif
#ifndef SYNC_PARTY_RECV2
#define SYNC_PARTY_RECV2 -1
#endif

#endif

#if FUNCTION_IDENTIFIER > 5
#ifndef MULTI_INPUT
#define MULTI_INPUT 1
#endif
#else
#ifndef MULTI_INPUT
#define MULTI_INPUT 0
#endif
#endif
/* #endif */

#if FUNCTION_IDENTIFIER < 65
#if FUNCTION_IDENTIFIER == 42 || FUNCTION_IDENTIFIER == 46 || FUNCTION_IDENTIFIER == 49 || \
    FUNCTION_IDENTIFIER == 52 || FUNCTION_IDENTIFIER == 59 || FUNCTION_IDENTIFIER == 62  // RCA
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 1
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 0
#endif
#elif FUNCTION_IDENTIFIER == 43 || FUNCTION_IDENTIFIER == 47 || FUNCTION_IDENTIFIER == 50 || \
    FUNCTION_IDENTIFIER == 53 || FUNCTION_IDENTIFIER == 60 || FUNCTION_IDENTIFIER == 63  // PPA
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 0
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 0
#endif
#elif FUNCTION_IDENTIFIER == 44 || FUNCTION_IDENTIFIER == 48 || FUNCTION_IDENTIFIER == 51 || \
    FUNCTION_IDENTIFIER == 54 || FUNCTION_IDENTIFIER == 61 || FUNCTION_IDENTIFIER == 64  // PPA 4-Way
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 0
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 1
#endif
#endif
#elif FUNCTION_IDENTIFIER < 400
#if FUNCTION_IDENTIFIER < 100  // RCA
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED \
    1  // 1 if bandwidth optimized (e.g. Ripple Carry Adder), 0 if Latency optimized (e.g. Multi-input AND gates,
       // Parallel Prefix Adder)
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED \
    0  // 1 if online optimized (e.g. MULTI_INPUT AND gates), 0 if optimized for total communication (e.g. no
       // MULTI_INPUT AND gates)
#endif
#elif FUNCTION_IDENTIFIER < 200  // PPA
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 0
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 0
#endif
#elif FUNCTION_IDENTIFIER < 300
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 0  // PPA 4-Way
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 1
#endif
#endif

#elif FUNCTION_IDENTIFIER >= 500 || FUNCTION_IDENTIFIER <= 530  // MP-SPDZ
#if FUNCTION_IDENTIFIER == 507 || FUNCTION_IDENTIFIER == 513 || FUNCTION_IDENTIFIER == 516 || \
    FUNCTION_IDENTIFIER == 521 || FUNCTION_IDENTIFIER == 526 || FUNCTION_IDENTIFIER == 529 || \
    FUNCTION_IDENTIFIER == 532
#define BANDWIDTH_OPTIMIZED 1
#define ONLINE_OPTIMIZED 0
#elif FUNCTION_IDENTIFIER == 508 || FUNCTION_IDENTIFIER == 514 || FUNCTION_IDENTIFIER == 517 || \
    FUNCTION_IDENTIFIER == 522 || FUNCTION_IDENTIFIER == 527 || FUNCTION_IDENTIFIER == 530 ||   \
    FUNCTION_IDENTIFIER == 533  // PPA
#define BANDWIDTH_OPTIMIZED 0
#define ONLINE_OPTIMIZED 0
#elif FUNCTION_IDENTIFIER == 509 || FUNCTION_IDENTIFIER == 515 || FUNCTION_IDENTIFIER == 518 || \
    FUNCTION_IDENTIFIER == 523 || FUNCTION_IDENTIFIER == 528 || FUNCTION_IDENTIFIER == 531 ||   \
    FUNCTION_IDENTIFIER == 534  // PPA 4-Way
#define BANDWIDTH_OPTIMIZED 0
#define ONLINE_OPTIMIZED 1
#endif

#else
#if FUNCTION_IDENTIFIER == 404 || FUNCTION_IDENTIFIER == 407 || FUNCTION_IDENTIFIER == 412  // RCA
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 1
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 0
#endif
#elif FUNCTION_IDENTIFIER == 405 || FUNCTION_IDENTIFIER == 408 || FUNCTION_IDENTIFIER == 413  // PPA
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 0
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 0
#endif
#elif FUNCTION_IDENTIFIER == 406 || FUNCTION_IDENTIFIER == 409 || FUNCTION_IDENTIFIER == 414  // PPA 4-Way
#ifndef BANDWIDTH_OPTIMIZED
#define BANDWIDTH_OPTIMIZED 0
#endif

#ifndef ONLINE_OPTIMIZED
#define ONLINE_OPTIMIZED 1
#endif
#endif
#endif

#ifndef SIMULATE_MPC_FUNCTIONS
#define SIMULATE_MPC_FUNCTIONS 1
#endif

#if BITLENGTH == 64
#ifndef INT_TYPE
#define INT_TYPE int64_t
#endif

#ifndef UINT_TYPE
#define UINT_TYPE uint64_t
#endif

#ifndef LOG2_BITLENGTH
#define LOG2_BITLENGTH 6
#endif

#ifndef LOG4_BITLENGTH
#define LOG4_BITLENGTH 3
#endif
#elif BITLENGTH == 32
#ifndef INT_TYPE
#define INT_TYPE int32_t
#endif

#ifndef UINT_TYPE
#define UINT_TYPE uint32_t
#endif

#ifndef LOG2_BITLENGTH
#define LOG2_BITLENGTH 5
#endif

#ifndef LOG4_BITLENGTH
#define LOG4_BITLENGTH 3
#endif
#elif BITLENGTH == 16
#ifndef INT_TYPE
#define INT_TYPE int16_t
#endif

#ifndef UINT_TYPE
#define UINT_TYPE uint16_t
#endif

#ifndef LOG2_BITLENGTH
#define LOG2_BITLENGTH 4
#endif

#ifndef LOG4_BITLENGTH
#define LOG4_BITLENGTH 2
#endif
#elif BITLENGTH == 8
#ifndef INT_TYPE
#define INT_TYPE int8_t
#endif

#ifndef UINT_TYPE
#define UINT_TYPE uint8_t
#endif

#ifndef LOG2_BITLENGTH
#define LOG2_BITLENGTH 3
#endif

#ifndef LOG4_BITLENGTH
#define LOG4_BITLENGTH 2
#endif
#endif

#ifndef JIT_VEC
#define JIT_VEC \
    1  // 0: vectorize and share inputs from the beginning, 1: vectorize and share inputs just in time, load a batch of
       // images, then vectorize
#endif

#ifndef BASETYPE
#define BASETYPE 0  // 0: Additive_Share, 1: sint
#endif

#if JIT_VEC == 0
#ifndef BASE_DIV
#define BASE_DIV 1
#endif
#else
#if BASETYPE == 0
#ifndef BASE_DIV
#define BASE_DIV DATTYPE / BITLENGTH
#endif
#else
#ifndef BASE_DIV
#define BASE_DIV DATTYPE
#endif
#endif
#endif

#ifndef PHASE_LIVE
#define PHASE_LIVE 1
#endif

#ifndef PHASE_INIT
#define PHASE_INIT 0
#endif

#ifndef PHASE_PRE
#define PHASE_PRE 2
#endif
