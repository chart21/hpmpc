#pragma once

// 1: Sharemind, 2: Replicated, 3: Astra, 4: ODUP, 5: OURS (3-PC), 6: TTP (3-PC), 7: TTP (4-PC), 8: Tetrad, 9: FantasticFour, 10: Ours: Base (4-PC), 11: Ours: Het (4-PC), 12: Ours: Off/On (4-PC)
#define PROTOCOL 6

// Party ID (starting from 0)
#define PARTY 2

//0: Search 1: XORNOTAND, 2: AND 1 comm round 3: AND 1000 comm rounds  4: Debug 5: MULT32 1 comm round 6: MULT64 1 comm round 7: Debug 9: Mult_32 1000 comm rounds 10: Mult64 1000 comm rounds. Currently, Protocols 9-12 support MULT. MULT64 is supported by DATATYPE 64 and 512. MULT32 is supported for DATATYPE 32 and all DATATYPEs >= 128
#define FUNCTION_IDENTIFIER 1

// Registersize to use for SIMD parallelization (Bitslicing/vectorization). Supported: 0,8,32,64,128(SSE),256(AVX-2),512(AVX-512)
#define DATTYPE 128

// Use a preprocessing phase? Currently only supported by Protocols 4,5,12
#define PRE 0

// Number of inputs (depends on the problem)
#define NUM_INPUTS 2

// Number of parallel processes to use
#define PROCESS_NUM 1

// 0 = xorshift, 1 = AES_BS, 2 = VAES/AES-NI. 0 is not secure.
#define RANDOM_ALGORITHM 2 

// Use SSL encrypted communication?
#define USE_SSL 0

// How many gates should be buffered until sending them to the receiving party? 0 means the data of an entire communication round is buffered
#define SEND_BUFFER 0

// How many reciving messages should be buffered until the main thread is signaled that data is ready? 0 means that all data of a communication round needs to be ready before the main thread is signaled.
#define RECV_BUFFER 0

// How many messages should be buffered until a combined hash is performed? 0 means all hashes are calculated at the very end of the protocol.
#define VERIFY_BUFFER 0
// Print additional info?
#define PRINT 1

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
#define BITLENGTH 64


#if PROTOCOL < 7
#define num_players 3
#else
#define num_players 4
#endif
#if PROTOCOL > 7
#define MAL 1
#endif


//temporary solution
#if (PROTOCOL == 4 || PROTOCOL == 5) && PARTY == 0
#define HAS_POST_PROTOCOL 1
#elif PROTOCOL == 12 && PARTY == 3
#define HAS_POST_PROTOCOL 1
#endif

#if BITLENGTH == 64
    #define INT_TYPE int64_t
    #define UINT_TYPE uint64_t
    #define LOG2_BITLENGTH 6
#elif BITLENGTH == 32
    #define INT_TYPE int32_t
    #define UINT_TYPE uint32_t
    #define LOG2_BITLENGTH 5
#elif BITLENGTH == 16
    #define INT_TYPE int16_t
    #define UINT_TYPE uint16_t
    #define LOG2_BITLENGTH 4
#elif BITLENGTH == 8
    #define INT_TYPE int8_t
    #define UINT_TYPE uint8_t
    #define LOG2_BITLENGTH 3
#endif

