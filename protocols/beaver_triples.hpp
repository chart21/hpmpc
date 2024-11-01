#pragma once
#include "generic_share.hpp"
#include "../core/generate_beaver_tiples.hpp"
#include "../core/init.hpp"
#include <cstdint>
#include <sys/types.h>

std::vector<uint64_t> arithmetic_triple_index;
std::vector<uint64_t> boolean_triple_index;
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
DATATYPE* arithmetic_triple_a;
DATATYPE* arithmetic_triple_b;
DATATYPE* arithmetic_triple_c;
DATATYPE* boolean_triple_a;
DATATYPE* boolean_triple_b;
DATATYPE* boolean_triple_c;
bool deinit = false;

template <typename Datatype>
struct triple
{
    Datatype a;
    Datatype b;
    Datatype c; //c = a*b
};

template <typename Datatype>
triple<Datatype> retrieveArithmeticTriple()
{
    total_arithmetic_triples_index++;
    return triple<Datatype>{arithmetic_triple_a[total_arithmetic_triples_index-1], arithmetic_triple_b[total_arithmetic_triples_index-1], arithmetic_triple_c[total_arithmetic_triples_index-1]};
}

template <typename Datatype>
triple<Datatype> retrieveBooleanTriple()
{
    total_boolean_triples_index++;
    return triple<Datatype>{boolean_triple_a[total_boolean_triples_index-1], boolean_triple_b[total_boolean_triples_index-1], boolean_triple_c[total_boolean_triples_index-1]};
    /* return triple<Datatype>{boolean_triple_a[boolean_triple_index], boolean_triple_b[boolean_triple_index], boolean_triple_c[boolean_triple_index++]}; */
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
    if(! deinit)
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
  
void generate_beaver_triples(std::string ips[], int base_port, int process_offset)
{
    uint64_t l_num_arithmetic_triples = total_arithmetic_triples_num*DATTYPE/BITLENGTH;
    uint64_t l_num_boolean_triples = total_boolean_triples_index*DATTYPE;
#if num_players == 2
    generateArithmeticTriples(arithmetic_triple_a, arithmetic_triple_b, arithmetic_triple_c, BITLENGTH, l_num_arithmetic_triples, ips[0], base_port+process_offset);
    generateBooleanTriples(boolean_triple_a, boolean_triple_b, boolean_triple_c, BITLENGTH, l_num_boolean_triples, ips[0], base_port+process_offset);
#else
    std::cerr << "Beaver triples not implemented for more than 2 parties" << std::endl;
    exit(1);
#endif
}


void print_num_triples()
{
#if PRINT_IMPORTANT == 1
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset <<  ": " << "Arithmetic Beaver Triples Required: " << total_arithmetic_triples_num*DATTYPE/BITLENGTH << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset <<  ": " << "Boolean Beaver Triples Required: " << total_boolean_triples_num*DATTYPE/BITLENGTH << std::endl;
#endif
}

