#pragma once

#define MODELOWNER -1 //Who holds the model parameters? (-1: Dummy model parameters, P_0/P_1/P_2/P_3: Read locally from P_0/P_1/P_2/P_3 followed by secret sharing)
#define DATAOWNER -1 //Who holds the data? (-1: Dummy dataset, P_0/P_1/P_2/P_3: Read locally from P_0/P_1/P_2/P_3 followed by secret sharing)

#define PROTOCOL 5

// Party ID (starting from 0)
#define PARTY 2

//0: Search 
//1-6: Multiplicatios: 1,2,3: 1-bit,32-bit,64-bit with 1 communication round, 4,5,6: 1-bit,32-bit,64-bit with 1000 communication rounds
//7-9: Debug: 7: 1-bit, 8: 32-bit, 9: 64-bit
//13,14: Dot product, 16,17 RELU, 20,21 Conv Forward (*10), Conv Backwards (*10), 22 MatMul (*10), 23,24 Forward Backwards (Different Sizes), 25,26 Forward Backwards (Different Sizes), 27 Mat Mul Eigen, 28 max/min/argmax/argmin, 29 mult3, 30 mult 4, 31-34 dot2/dot3/dot4/dotmixed, 
// 40-65 Various benchmarks (Elementary operations such as mult, div. Statistical operations such as avg, max. Set Intersection, AES, Private Auction, Logistic Regression, etc. Refer to programs/functions/sevare.hpp
// 70+ Neural network architectures (LeNet, AlexNet, VGG, ResNet, etc.) on different dataset sizes (MNIST, CIFAR-10, Imagenet). Refer to programs/functions/NN.hpp
#define FUNCTION_IDENTIFIER 70

// Registersize to use for SIMD parallelization (Bitslicing/vectorization). Supported: 1,8,16,32,64,128(SSE),256(AVX-2),512(AVX-512)
//Info: MULT64 is supported by DATTYPE 64 and 512. MULT32 is supported for DATTYPE 32 and all DATATYPEs >= 128
#define DATTYPE 32

// Use a preprocessing phase? Currently only supported by Protocols 4,5,12
#define PRE 0

// Number of inputs (depends on the problem)
#define NUM_INPUTS 10

// Number of parallel processes to use
#define PROCESS_NUM 1

// 0 = xorshift, 1 = AES_BS, 2 = VAES/AES-NI. 0 is not secure.
#define RANDOM_ALGORITHM 2
#define USE_SSL_AES 0

// Use SSL encrypted communication?
#define USE_SSL 0

// How many gates should be buffered until sending them to the receiving party? 0 means the data of an entire communication round is buffered
#define SEND_BUFFER 10000

// How many reciving messages should be buffered until the main thread is signaled that data is ready? 0 means that all data of a communication round needs to be ready before the main thread is signaled.
#define RECV_BUFFER 10000

// How many messages should be buffered until a combined hash is performed? 0 means all hashes are calculated at the very end of the protocol.
#define VERIFY_BUFFER 16
// Print additional info?
#define PRINT 0
#define PRINT_TIMINGS 0

#define FRACTIONAL 5 // fractional bits for fixed point numbers

// Starting port for required port range of the sockets, must be multiple of 1000 for some applications
#define BASE_PORT 10000
#define SPLIT_ROLES_OFFSET 0
int base_port = BASE_PORT; // temporary solution

// Timeout in seconds when connecting to a socket
#define CONNECTION_TIMEOUT 500

// Timeout in millisecond before attempting to connect again to a socket
#define CONNECTION_RETRY 5

#define ARM 0 // 1 if ARM processor, 0 otherwise. Can speed up Sha hashing.

// Allow sharing of inputs in offline phase
#define SHARE_PREP 1

// Compress binary data into chars before sending them over the netowrk? Only relevant for DATTYPE = 1
#define COMPRESS 1

// Use optimized secret sharing? Often utilizes SRNG instead of secret sharing with communication
#define OPT_SHARE 1

// Use the initialization phase or import initiliazation data from a file?
#define NO_INI 0

// Use the initialization phase or import initiliazation data from a file?
#define INIT 1

// Use the online phase?
#define LIVE 1

// Use random inputs or inputs from a file? TODO: File inputs to be implemented
#define INPUT 'r'

// Bitlength of integers (currently not used)
#define BITLENGTH 32
// Reduced Bitlength that might be used for RELU, etc

#if COMPRESS == 0
#define REDUCED_BITLENGTH_k 32
#define REDUCED_BITLENGTH_m 0
#else
#define REDUCED_BITLENGTH_k 20
#define REDUCED_BITLENGTH_m 12
#endif

#if BANDWIDTH_OPTIMIZED == 0 || ONLINE_OPTIMIZED == 1 //if BANDWIDTH_OPTIMIZED and not ONLINE_OPTIMIZED we don't need MULTI_INPUT_AND gates
#define MULTI_INPUT 1 // activate multi input Multiplication gates?
#else
#define MULTI_INPUT 0
#endif

#define SIMULATE_QUANT 0 // Simulate 8-bit quantization

#if FUNCTION_IDENTIFIER > 65
#if FUNCTION_IDENTIFIER < 100  //RCA
#define BANDWIDTH_OPTIMIZED 1 // 1 if bandwidth optimized (e.g. Ripple Carry Adder), 0 if Latency optimized (e.g. Multi-input AND gates, Parallel Prefix Adder)
#define ONLINE_OPTIMIZED 0 // 1 if online optimized (e.g. MULTI_INPUT AND gates), 0 if optimized for total communication (e.g. no MULTI_INPUT AND gates)
#elif FUNCTION_IDENTIFIER < 200 // PPA4
#define BANDWIDTH_OPTIMIZED 0 
#define ONLINE_OPTIMIZED 1 
#elif FUNCTION_IDENTIFIER < 300 
#define BANDWIDTH_OPTIMIZED 0 // PPA2
#define ONLINE_OPTIMIZED 0 
#endif
#endif

#if PROTOCOL < 7
#define num_players 3
#else
#define num_players 4
#endif
#if PROTOCOL > 7
#define MAL 1
#endif

#define SIMULATE_MPC_FUNCTIONS 1

#if BITLENGTH == 64
    #define INT_TYPE int64_t
    #define UINT_TYPE uint64_t
    #define LOG2_BITLENGTH 6
    #define LOG4_BITLENGTH 3
#elif BITLENGTH == 32
    #define INT_TYPE int32_t
    #define UINT_TYPE uint32_t
    #define LOG2_BITLENGTH 5
    #define LOG4_BITLENGTH 3
#elif BITLENGTH == 16
    #define INT_TYPE int16_t
    #define UINT_TYPE uint16_t
    #define LOG2_BITLENGTH 4
    #define LOG4_BITLENGTH 2
#elif BITLENGTH == 8
    #define INT_TYPE int8_t
    #define UINT_TYPE uint8_t
    #define LOG2_BITLENGTH 3
    #define LOG4_BITLENGTH 2
#endif


#define TRUNC_THEN_MULT 0 // 0 = mult then trunc, 1 = trunc then mult
#define TRUNC_APPROACH 0 // 0: cut, 1: interactive
#define TRUNC_DELAYED 0 // 0: truncate after each fixed point multiplication, 1: truncate after next ReLU (might produce errors in some networks)
#define COMPUTE_ARGMAX 0 // 0: skip final argmax during inference, 1: Compute final argmax during inference
#define PUBLIC_WEIGHTS 0 // 0: weights are secretly shared, 1: weights are public


#define JIT_VEC 1 // 0: vectorize and share inputs from the beginning, 1: vectorize and share inputs just in time, load a batch of images, then vectorize
#define BASETYPE 0 // 0: Additive_Share, 1: sint

#if JIT_VEC == 0
    #define BASE_DIV 1
#else
    #if BASETYPE == 0
        #define BASE_DIV DATTYPE/BITLENGTH
    #else
        #define BASE_DIV DATTYPE
    #endif
#endif

#define IS_TRAINING 0


#define PHASE_LIVE 1
#define PHASE_INIT 0
#define PHASE_PRE 2
