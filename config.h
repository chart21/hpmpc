#pragma once

// 1: Sharemind (no support for dot products), 2: Replicated, 3: Astra, 4: ODUP (depricated), 5: OURS (3-PC), 6: TTP (3-PC), 7: TTP (4-PC), 8: Tetrad, 9: FantasticFour, 10: Ours: Base (4-PC), 11: Ours: Het (4-PC), 12: Ours: Off/On (4-PC), 13: Simulator
#define PROTOCOL 13

// Party ID (starting from 0)
#define PARTY 1

//0: Search 
//1-6: Multiplicatios: 1,2,3: 1-bit,32-bit,64-bit with 1 communication round, 4,5,6: 1-bit,32-bit,64-bit with 1000 communication rounds
//7-9: Debug: 7: 1-bit, 8: 32-bit, 9: 64-bit
//13,14: Dot product, 16,17 RELU, 20,21 Conv Forward (*10), Conv Backwards (*10), 22 MatMul (*10), 23,24 Forward Backwards (Different Sizes), 25,26 Forward Backwards (Different Sizes), 27 Mat Mul Eigen, 28 max/min/argmax/argmin, 29 mult3, 30 mult 4, 31-34 dot2/dot3/dot4/dotmixed, 
//Info: MULT64 is supported by DATTYPE 64 and 512. MULT32 is supported for DATTYPE 32 and all DATATYPEs >= 128
#define FUNCTION_IDENTIFIER 28

// Registersize to use for SIMD parallelization (Bitslicing/vectorization). Supported: 0,8,32,64,128(SSE),256(AVX-2),512(AVX-512)
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
#define SEND_BUFFER 1000

// How many reciving messages should be buffered until the main thread is signaled that data is ready? 0 means that all data of a communication round needs to be ready before the main thread is signaled.
#define RECV_BUFFER 1000

// How many messages should be buffered until a combined hash is performed? 0 means all hashes are calculated at the very end of the protocol.
#define VERIFY_BUFFER 0
// Print additional info?
#define PRINT 0

#define FRACTIONAL 0 // fractional bits for fixed point numbers

// Starting port for required port range of the sockets
#define BASE_PORT 11000
int base_port = BASE_PORT; // temporary solution

// Timeout in seconds when connecting to a socket
#define CONNECTION_TIMEOUT 30 

// Timeout in millisecond before attempting to connect again to a socket
#define CONNECTION_RETRY 5

#define ARM 0 // 1 if ARM processor, 0 otherwise. Can speed up Sha hashing.

// Allow sharing of inputs in offline phase
#define SHARE_PREP 1

// Compress binary data into chars before sending them over the netowrk? Only relevant for DATTYPE = 1
#define COMPRESS 0

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
#define REDUCED_BITLENGTH 32

#define MULTI_INPUT 1 // activate multi input Multiplication gates?

#if PROTOCOL < 7
#define num_players 3
#else
#define num_players 4
#endif
#if PROTOCOL > 7
#define MAL 1
#endif



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

