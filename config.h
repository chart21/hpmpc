#pragma once
#define PROTOCOL 12
#if PROTOCOL < 7
#define num_players 3
#else
#define num_players 4
#endif
#if PROTOCOL > 7
#define MAL 1
#endif

//0: search 1: XORNOTAND, 2: AND 1 comm round 3: AND 1000 comm rounds  4: debug 5: MULT_32 6: MULT64 7: Debug 9: Mult_32 1000 comm rounds 10: Mult64 1000 comm rounds
#define FUNCTION_IDENTIFIER 2


#define NEW_WAY 1

#define DATTYPE 128 // Registersize to use for SIMD parallelization (Bitslicing/vectorization)

#if FUNCTION_IDENTIFIER == 5 || FUNCTION_IDENTIFIER == 7 || FUNCTION_IDENTIFIER == 9
    #define MULT(a,b) MUL_SIGNED(a,b,32) 
    #define ADD(a,b) ADD_SIGNED(a,b,32)
    #define SUB(a,b) SUB_SIGNED(a,b,32)
    
#elif FUNCTION_IDENTIFIER == 6 || FUNCTION_IDENTIFIER == 10
    #define MULT(a,b) MUL_SIGNED(a,b,64)
    #define ADD(a,b) ADD_SIGNED(a,b,64)
    #define SUB(a,b) SUB_SIGNED(a,b,64)
#elif FUNCTION_IDENTIFIER == 8
    #if DATTYPE == 128
        #define MULT32 _mm_mullo_epi32
        #define ADD32 _mm_add_epi32
        #define SUB32 _mm_sub_epi32
    #elif DATTYPE == 256
        #define MULT32 _mm256_mullo_epi32
        #define ADD32 _mm256_add_epi32
        #define SUB32 _mm256_sub_epi32
    #elif DATTYPE == 512
        #define MULT32 _mm512_mullo_epi32
        #define ADD32 _mm512_add_epi32
        #define SUB32 _mm512_sub_epi32
    #endif
#endif

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

// Use the offline phase?
#define PRE 0

// Allow sharing of inputs in offline phase
#define SHARE_PREP 1

// Party ID (starting from 0)
#define PARTY 3

// Use random inputs or inputs from a file? TODO: File inputs to be implemented
#define INPUT 'r'

// Number of inputs (depends on the problem)
#define NUM_INPUTS 10000

// Bitlength of integers
#define BITLENGTH 64

// Number of players in the protocol

// Starting port for required port range of the sockets
#define BASE_PORT 11000
int base_port = BASE_PORT; // temporary solution

// Use SSL encrypted communication?
#define USE_SSL 0

// Number of parallel processes to use
#define PROCESS_NUM 1

// 0 = xorshift, 1 = AES_BS, 2 = AES_NI
#define RANDOM_ALGORITHM 2 

#define ARM 0 // 1 if ARM processor, 0 otherwise. Can speed up Sha hashing.

// Timeout in seconds when connecting to a socket
#define CONNECTION_TIMEOUT 30 

// Timeout in millisecond before attempting to connect again to a socket
#define CONNECTION_RETRY 5

// How many gates should be buffered until sending them to the receiving party? 0 means the data of an entire communication round is buffered
#define SEND_BUFFER 0

// How many reciving messages should be buffered until the main thread is signaled that data is ready? 0 means that all data of a communication round needs to be ready before the main thread is signaled.
#define RECV_BUFFER 0

// How many messages should be buffered until a combined hash is performed? 0 means all hashes are calculated at the very end of the protocol.
#define VERIFY_BUFFER 0
// Print additional info?
#define PRINT 0
