#pragma once
#include "../core/generate_beaver_tiples.hpp"
#include "../core/init.hpp"
#include "../config.h"  
#include "generic_share.hpp"
#include "../core/include/pch.h"

// std::vector<uint64_t> arithmetic_triple_index;
// std::vector<uint64_t> boolean_triple_index;
std::vector<uint64_t> num_arithmetic_triples;
std::vector<uint64_t> num_boolean_triples;
std::vector<uint64_t> triple_type_index;
std::vector<uint8_t*> triple_type;
uint64_t total_arithmetic_triples_num = 0;
uint64_t total_boolean_triples_num = 0;
uint64_t total_arithmetic_triples_index = 0;
uint64_t total_boolean_triples_index = 0;
/* uint64_t boolean_triple_index = 0; */
/* uint64_t num_arithmetic_triples = 0; */
/* uint64_t num_boolean_triples = 0; */
/* uint64_t triple_type_index = 0; */
/* uint8_t* triple_type; */
uint64_t arithmetic_triple_index = 0;
uint64_t boolean_triple_index = 0;
DATATYPE* arithmetic_triple_a;
DATATYPE* arithmetic_triple_b;
DATATYPE* arithmetic_triple_c;
DATATYPE* boolean_triple_a;
DATATYPE* boolean_triple_b;
DATATYPE* boolean_triple_c;
bool deinit = false;
bool deinitAB = false;
bool deinitC = false;

template <typename Datatype>
struct triple
{
    Datatype a;
    Datatype b;
    Datatype c;  // c = a*b
};

template <typename Datatype>
triple<Datatype> retrieveArithmeticTriple()
{
    total_arithmetic_triples_index++;
    return triple<Datatype>{arithmetic_triple_a[total_arithmetic_triples_index - 1],
                            arithmetic_triple_b[total_arithmetic_triples_index - 1],
                            arithmetic_triple_c[total_arithmetic_triples_index - 1]};
}

template <typename Datatype>
triple<Datatype> retrieveBooleanTriple()
{
    total_boolean_triples_index++;
    return triple<Datatype>{boolean_triple_a[total_boolean_triples_index - 1],
                            boolean_triple_b[total_boolean_triples_index - 1],
                            boolean_triple_c[total_boolean_triples_index - 1]};
    /* return triple<Datatype>{boolean_triple_a[boolean_triple_index], boolean_triple_b[boolean_triple_index],
     * boolean_triple_c[boolean_triple_index++]}; */
}

    template <typename Datatype>
void storeArithmeticABTriple(const Datatype a, const Datatype b)
{
    arithmetic_triple_a[arithmetic_triple_index] = a;
#if AB2 != 1 || PARTY != 1
    arithmetic_triple_b[arithmetic_triple_index] = b; //B1 is not needed for the AB2 protocol
#endif
    arithmetic_triple_index++;
}

template <typename Datatype>
void storeBooleanABTriple(const Datatype a, const Datatype b)
{
    boolean_triple_a[boolean_triple_index] = a;
#if AB2 != 1 || PARTY != 1
    boolean_triple_b[boolean_triple_index] = b; //B1 is not needed for the AB2 protocol
#endif
    boolean_triple_index++;
}
    

template <typename Datatype>
Datatype retrieveBooleanLXLY()
{
    total_boolean_triples_index++;
    return boolean_triple_c[total_boolean_triples_index - 1];
}


template <typename Datatype>
Datatype retrieveArithmeticLXLY()
{
    total_arithmetic_triples_index++;
    return arithmetic_triple_c[total_arithmetic_triples_index - 1];
}



void init_beaver()
{
    /* arithmetic_triple_index = 0; */
    /* boolean_triple_index = 0; */
    arithmetic_triple_a = new DATATYPE[total_arithmetic_triples_num];
    arithmetic_triple_b = new DATATYPE[total_arithmetic_triples_num];
    arithmetic_triple_c = new DATATYPE[total_arithmetic_triples_num];
    boolean_triple_a = new DATATYPE[total_boolean_triples_num];
    boolean_triple_b = new DATATYPE[total_boolean_triples_num];
    boolean_triple_c = new DATATYPE[total_boolean_triples_num];
}
void deinit_beaver()
{
    if (!deinit)
    {
        delete[] arithmetic_triple_a;
        delete[] arithmetic_triple_b;
        delete[] arithmetic_triple_c;
        delete[] boolean_triple_a;
        delete[] boolean_triple_b;
        delete[] boolean_triple_c;
        deinit = true;
    }
}

void deinit_beaverAB()
{
    if (!deinitAB)
    {
        delete[] arithmetic_triple_c;
        delete[] boolean_triple_c;
        deinit = true;
    }
}

void deinit_beaverC()
{
    if (!deinitC)
    {
        delete[] arithmetic_triple_c;
        delete[] boolean_triple_c;
        deinit = true;
    }
}

struct timespec k1, k2;

void generate_beaver_triples(std::string ips[], int base_port, int process_offset, uint64_t num_arith_triples, uint64_t num_bool_triples)
{
    uint64_t l_num_arithmetic_triples = num_arith_triples * DATTYPE / BITLENGTH;
    uint64_t l_num_boolean_triples = num_bool_triples * DATTYPE;
#if FAKE_TRIPLES == 1
    print("Fake Triples set to 1, generating fake triples ... \n");
#else
    print("Generating Beaver Triples ... \n");
#endif
    clock_t time_beaver_function_start = clock();
    clock_gettime(CLOCK_REALTIME, &k1);
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();

#if num_players == 2
    generateArithmeticTriples(arithmetic_triple_a,
                              arithmetic_triple_b,
                              arithmetic_triple_c,
                              BITLENGTH,
                              l_num_arithmetic_triples,
                              ips[0],
                              base_port + process_offset);
    generateBooleanTriples(boolean_triple_a,
                           boolean_triple_b,
                           boolean_triple_c,
                           BITLENGTH,
                           l_num_boolean_triples,
                           ips[0],
                           base_port + process_offset);
#else
    std::cerr << "Beaver triples not implemented for more than 2 parties" << std::endl;
    exit(1);
#endif

    clock_gettime(CLOCK_REALTIME, &k2);
    double accum_beaver = (k2.tv_sec - k1.tv_sec) + (double)(k2.tv_nsec - k1.tv_nsec) / (double)1000000000L;
    clock_t time_beaver_function_finished = clock();
    print("Time measured to perform beaver triple generation clock: %fs \n",
          double((time_beaver_function_finished - time_beaver_function_start)) / CLOCKS_PER_SEC);
    print("Time measured to perform beaver triple generation getTime: %fs \n", accum_beaver);
    print("Time measured to perform beaver triple generation chrono: %fs \n",
          double(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - p)
                     .count()) /
              1000000);


}

void print_num_triples()
{
#if PRINT_IMPORTANT == 1
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Arithmetic Beaver Triples Required: " << total_arithmetic_triples_num * DATTYPE / BITLENGTH
              << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Boolean Beaver Triples Required: " << total_boolean_triples_num * DATTYPE / BITLENGTH << std::endl;
#endif
}
