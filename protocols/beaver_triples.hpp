#pragma once
#include "../core/generate_beaver_tiples.hpp"
#include "../core/init.hpp"
#include "../config.h"  
#include "generic_share.hpp"
#include "../core/include/pch.h"

// std::vector<uint64_t> arithmetic_triple_index;
// std::vector<uint64_t> boolean_triple_index;
std::vector<uint64_t> num_arithmetic_triples;
std::vector<uint64_t> num_ab2_arithmetic_triples;
std::vector<uint64_t> num_boolean_triples;
uint64_t num_boolean_addition_triples;
uint64_t num_multiplexer_triples;
uint64_t num_cot_triples;
std::vector<uint64_t> num_ab2_boolean_triples;
std::vector<uint64_t> triple_type_index;
std::vector<uint8_t*> triple_type;
/* uint64_t boolean_triple_index = 0; */
/* uint64_t num_arithmetic_triples = 0; */
/* uint64_t num_boolean_triples = 0; */
/* uint64_t triple_type_index = 0; */
/* uint8_t* triple_type; */

std::vector<uint64_t> total_num_boolean_output_triples;
std::vector<uint64_t> total_num_arithmetic_output_triples;


uint64_t total_arithmetic_triples_num = 0;
uint64_t total_boolean_triples_num = 0;
uint64_t total_arithmetic_triples_index = 0;
uint64_t total_boolean_triples_index = 0;

uint64_t arithmetic_triple_index = 0;
uint64_t boolean_triple_index = 0;
uint64_t curr_arithmetic_triple_index = 0;
uint64_t curr_boolean_triple_index = 0;
DATATYPE* arithmetic_triple_a = nullptr;
DATATYPE* arithmetic_triple_b = nullptr;
DATATYPE* arithmetic_triple_c = nullptr;
DATATYPE* boolean_triple_a = nullptr;
DATATYPE* boolean_triple_b= nullptr;
DATATYPE* boolean_triple_c = nullptr;

uint64_t total_ab2_arithmetic_triples_num = 0;
uint64_t total_ab2_boolean_triples_num = 0;
uint64_t total_ab2_arithmetic_triples_index = 0;
uint64_t total_ab2_boolean_triples_index = 0;

uint64_t curr_arithmetic_ab2_triple_index = 0;
uint64_t curr_boolean_ab2_triple_index = 0;
uint64_t arithmetic_ab2_triple_index = 0;
uint64_t boolean_ab2_triple_index = 0;
DATATYPE* arithmetic_ab2_triple_a = nullptr;
DATATYPE* arithmetic_ab2_triple_b = nullptr;
DATATYPE* arithmetic_ab2_triple_c = nullptr;
DATATYPE* boolean_ab2_triple_a = nullptr;
DATATYPE* boolean_ab2_triple_b = nullptr;
DATATYPE* boolean_ab2_triple_c = nullptr;


uint64_t curr_boolean_addition_triple_index = 0;
uint64_t boolean_addition_triple_index = 0;
DATATYPE* boolean_addition_triple_a = nullptr;
DATATYPE* boolean_addition_triple_b= nullptr;
DATATYPE* boolean_addition_triple_c = nullptr;

uint64_t curr_multiplexer_triple_index = 0;
uint64_t arithmetic_multiplexer_triple_index = 0;
uint64_t boolean_multiplexer_triple_index = 0;
DATATYPE* multiplexer_triple_a = nullptr;
DATATYPE* multiplexer_triple_b= nullptr;
DATATYPE* multiplexer_triple_c = nullptr;

uint64_t curr_cot_triple_index = 0;
uint64_t arithmetic_cot_triple_index = 0;
uint64_t boolean_cot_triple_index = 0;
DATATYPE* cot_triple_a = nullptr;
DATATYPE* cot_triple_b= nullptr;
DATATYPE* cot_triple_c = nullptr;
        

DATATYPE** conv_triple_w = nullptr;
DATATYPE** conv_triple_x = nullptr;
DATATYPE* conv_triple_y = nullptr;
uint64_t curr_conv_triple_index = 0;
uint64_t num_conv_c_triples = 0;
std::vector<ConvolutionParameter> conv_triple_params;

DATATYPE** fc_triple_w = nullptr;
DATATYPE** fc_triple_x = nullptr;
DATATYPE* fc_triple_y = nullptr;
uint64_t curr_fc_triple_index = 0;
uint64_t num_fc_c_triples = 0;
std::vector<FullyConnectedParameter> fc_triple_params;

DATATYPE** bc2D_triple_w = nullptr;
DATATYPE** bc2D_triple_x = nullptr;
DATATYPE* bc2D_triple_y = nullptr;
uint64_t curr_bc2D_triple_index = 0;
uint64_t num_bc2D_c_triples = 0;
std::vector<BatchNorm2DParameter> bc2D_triple_params;


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
    arithmetic_triple_b[arithmetic_triple_index] = b;
    arithmetic_triple_index++;
}

template <typename Datatype>
void storeBooleanABTriple(const Datatype a, const Datatype b)
{
    boolean_triple_a[boolean_triple_index] = a;
    boolean_triple_b[boolean_triple_index] = b; //B1 is not needed for the AB2 protocol
    boolean_triple_index++;
}

    template <typename Datatype>
void storeArithmeticAB2Triple(const Datatype a, const Datatype b)
{
#if PARTY == 0
    arithmetic_ab2_triple_a[arithmetic_ab2_triple_index] = a; //P0 holds A0 in plain in AB2 setting
#endif
#if PARTY != 0
    arithmetic_ab2_triple_b[arithmetic_ab2_triple_index] = b; //B1 is not needed for the AB2 protocol
#endif
    arithmetic_ab2_triple_index++;
}

template <typename Datatype>
void storeBooleanAB2Triple(const Datatype a, const Datatype b)
{
#if PARTY == 0
    boolean_ab2_triple_a[boolean_ab2_triple_index] = a;
#endif
#if PARTY != 0
    boolean_ab2_triple_b[boolean_ab2_triple_index] = b; //B1 is not needed for the AB2 protocol
#endif
    boolean_ab2_triple_index++;
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


#if LX_TRIPLES == 1
void init_beaverAB(int rounds)
{
    arithmetic_triple_a = new DATATYPE[num_arithmetic_triples[rounds] ];
    arithmetic_triple_b = new DATATYPE[num_arithmetic_triples[rounds] ];
    boolean_triple_a = new DATATYPE[num_boolean_triples[rounds] ];
    boolean_triple_b = new DATATYPE[num_boolean_triples[rounds] ];
    // std::cout << "Initialized beaver AB for round " + std::to_string(rounds) + " with " + std::to_string(num_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverAB_arithmetic(int rounds)
{
    arithmetic_triple_a = new DATATYPE[num_arithmetic_triples[rounds] ];
    arithmetic_triple_b = new DATATYPE[num_arithmetic_triples[rounds] ];
}

void init_beaverAB_boolean(int rounds)
{
    boolean_triple_a = new DATATYPE[num_boolean_triples[rounds] ];
    boolean_triple_b = new DATATYPE[num_boolean_triples[rounds] ];
}


#if A2B_ONLINE_OPT == 1
void init_booleanAdditionBeaverAB()
{
    if(num_boolean_addition_triples == 0)
        return;
#if PARTY == 0
    boolean_addition_triple_a = new DATATYPE[num_boolean_addition_triples];
#else
    boolean_addition_triple_b = new DATATYPE[num_boolean_addition_triples];
#endif
}

void init_booleanAdditionBeaverC()
{
    boolean_addition_triple_c = new DATATYPE[num_boolean_addition_triples];
}

void deinit_booleanAdditionBeaverAB()
{
#if PARTY == 0
    delete[] boolean_addition_triple_a;
#else
    delete[] boolean_addition_triple_b;
#endif
}

void deinit_booleanAdditionBeaverC()
{
    delete[] boolean_addition_triple_c;
}
#endif

#if BIT_INJECTION_PREPROCESSING_OPT == 1

void init_multiplexerBeaverAB()
{
    multiplexer_triple_a = new DATATYPE[num_multiplexer_triples];
    multiplexer_triple_b = new DATATYPE[num_multiplexer_triples / BITLENGTH];
}

void init_multiplexerBeaverC()
{
    multiplexer_triple_c = new DATATYPE[num_multiplexer_triples];
}

void deinit_multiplexerBeaverAB()
{
    delete[] multiplexer_triple_a;
    delete[] multiplexer_triple_b;
}


void deinit_multiplexerBeaverC()
{
    delete[] multiplexer_triple_c;
}

void init_cotBeaverAB()
{
#if PARTY == 0
    cot_triple_a = new DATATYPE[num_cot_triples];
#else
    cot_triple_a = multiplexer_triple_b; //reuse lb share
#endif
}

void init_cotBeaverC()
{
    cot_triple_c = new DATATYPE[num_cot_triples];
}

void deinit_cotBeaverAB()
{
#if PARTY == 0
    delete[] cot_triple_a;
#else
    cot_triple_a = nullptr; // no need to delet since multiplexer is reused
#endif
}

void deinit_cotBeaverC()
{
    delete[] cot_triple_c;
}

#endif


void init_beaverC(int rounds)
{
    arithmetic_triple_c = new DATATYPE[num_arithmetic_triples[rounds] ];
    boolean_triple_c = new DATATYPE[num_boolean_triples[rounds] ];
    // std::cout << "Initialized beaver C for round " + std::to_string(rounds) + " with " + std::to_string(num_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverC_arithmetic(int rounds)
{
    arithmetic_triple_c = new DATATYPE[num_arithmetic_triples[rounds] ];
}

void init_beaverC_boolean(int rounds)
{
    boolean_triple_c = new DATATYPE[num_boolean_triples[rounds] ];
}


template <typename LayerParameter>
void deinit_LayerAB(DATATYPE** x, DATATYPE** w, std::vector<LayerParameter> p)
{
    for(int i = 0; i < p.size(); i++)
    {
#if PARTY == 0 || A_KNOWN == 0 // Party0 holds W in plain in AB2 setting
        delete[] w[i];
#endif
#if PARTY == 1 || A_KNOWN == 0 // Party 0 does not need X triples in AB2 setting
        delete[] x[i];
#endif
    }
    delete[] w;
    delete[] x;
}

void init_ConvAB()
{
    conv_triple_w = new DATATYPE*[conv_triple_params.size()]; 
    conv_triple_x = new DATATYPE*[conv_triple_params.size()];
}

void init_BatchNorm2DAB()
{
    bc2D_triple_w = new DATATYPE*[bc2D_triple_params.size()];
    bc2D_triple_x = new DATATYPE*[bc2D_triple_params.size()];
}

void init_FullyConnectedAB()
{
    fc_triple_w = new DATATYPE*[fc_triple_params.size()];
    fc_triple_x = new DATATYPE*[fc_triple_params.size()];
}

void init_ConvC()
{
    conv_triple_y = new DATATYPE[num_conv_c_triples];
}

void init_BatchNorm2DC()
{
    bc2D_triple_y = new DATATYPE[num_bc2D_c_triples];
}

void init_FullyConnectedC()
{
    fc_triple_y = new DATATYPE[num_fc_c_triples];
}

void deinit_ConvAB()
{
    deinit_LayerAB(conv_triple_x, conv_triple_w, conv_triple_params);
}

void deinit_ConvC()
{
    delete[] conv_triple_y;
}

void deinit_BatchNorm2DAB()
{
    deinit_LayerAB(bc2D_triple_x, bc2D_triple_w, bc2D_triple_params);
}

void deinit_BatchNorm2DC()
{
    delete[] bc2D_triple_y;
}

void deinit_FullyConnectedAB()
{
    deinit_LayerAB(fc_triple_x, fc_triple_w, fc_triple_params);
}

void deinit_FullyConnectedC()
{
    delete[] fc_triple_y;
}

void init_beaverAB2(int rounds)
{
#if PARTY == 0 // P0 holds a in plain in AB2 setting
    arithmetic_ab2_triple_a = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
    boolean_ab2_triple_a = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
#if PARTY == 1 // P0 doesn't need B1 for AB2
    arithmetic_ab2_triple_b = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
    boolean_ab2_triple_b = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
    // std::cout << "Initialized beaver AB2 for round " + std::to_string(rounds) + " with " + std::to_string(num_ab2_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_ab2_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverAB2_arithmetic(int rounds)
{
#if PARTY == 0 // P0 holds a in plain in AB2 setting
    arithmetic_ab2_triple_a = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
#endif
#if PARTY == 1 // P0 doesn't need B1 for AB2
    arithmetic_ab2_triple_b = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
#endif
}

void init_beaverAB2_boolean(int rounds)
{
#if PARTY == 0 // P0 holds a in plain in AB2 setting
    boolean_ab2_triple_a = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
#if PARTY == 1 // P0 doesn't need B1 for AB2
    boolean_ab2_triple_b = new DATATYPE[num_ab2_boolean_triples[rounds] ];
#endif
}

void init_beaverAB2C(int rounds)
{
    if(num_ab2_arithmetic_triples[rounds] > 0)
        arithmetic_ab2_triple_c = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
    if(num_ab2_boolean_triples[rounds] > 0)
        boolean_ab2_triple_c = new DATATYPE[num_ab2_boolean_triples[rounds] ];
    // std::cout << "Initialized beaver AB2 C for round " + std::to_string(rounds) + " with " + std::to_string(num_ab2_arithmetic_triples[rounds] * DATTYPE/BITLENGTH) + " arithmetic triples and " + std::to_string(num_ab2_boolean_triples[rounds] * DATTYPE) + " boolean triples.\n";
}

void init_beaverAB2C_arithmetic(int rounds)
{
    if(num_ab2_arithmetic_triples[rounds] > 0)
        arithmetic_ab2_triple_c = new DATATYPE[num_ab2_arithmetic_triples[rounds] ];
}

void init_beaverAB2C_boolean(int rounds)
{
    if(num_ab2_boolean_triples[rounds] > 0)
        boolean_ab2_triple_c = new DATATYPE[num_ab2_boolean_triples[rounds] ];
}
#else
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

    arithemtic_ab2_triple_a = new DATATYPE[total_ab2_arithmetic_triples_num];
    arithmetic_ab2_triple_b = new DATATYPE[total_ab2_arithmetic_triples_num];
    arithmetic_ab2_triple_c = new DATATYPE[total_ab2_arithmetic_triples_num];
    boolean_ab2_triple_a = new DATATYPE[total_ab2_boolean_triples_num];
    boolean_ab2_triple_b = new DATATYPE[total_ab2_boolean_triples_num];
    boolean_ab2_triple_c = new DATATYPE[total_ab2_boolean_triples_num];
}
#endif

void deinit_beaverAB2()
{
    // print("Deleting beaver AB2 arrays.");
#if PARTY == 0 
    delete[] arithmetic_ab2_triple_a;
    delete[] boolean_ab2_triple_a;
#elif PARTY == 1
    delete[] arithmetic_ab2_triple_b;
    delete[] boolean_ab2_triple_b;
#endif
}

void deinit_beaverAB2_arithmetic()
{
#if PARTY == 0 
    delete[] arithmetic_ab2_triple_a;
#elif PARTY == 1
    delete[] arithmetic_ab2_triple_b;
#endif
}

void deinit_beaverAB2_boolean()
{
#if PARTY == 0
    delete[] boolean_ab2_triple_a;
#elif PARTY == 1
    delete[] boolean_ab2_triple_b;
#endif
}

void deinit_beaverAB2C()
{
    // print("Deleting beaver AB2 C arrays.");
    if(arithmetic_ab2_triple_c != nullptr) 
    {
        delete[] arithmetic_ab2_triple_c;
        arithmetic_ab2_triple_c = nullptr;
    }
    if(boolean_ab2_triple_c != nullptr)
    {
        delete[] boolean_ab2_triple_c;
        boolean_ab2_triple_c = nullptr;
    }
}

void deinit_beaverAB2C_arithmetic()
{
    if(arithmetic_ab2_triple_c != nullptr) 
    {
        delete[] arithmetic_ab2_triple_c;
        arithmetic_ab2_triple_c = nullptr;
    }
}

void deinit_beaverAB2C_boolean()
{
    if(boolean_ab2_triple_c != nullptr)
    {
        delete[] boolean_ab2_triple_c;
        boolean_ab2_triple_c = nullptr;
    }
}


void deinit_beaverAB()
{
    // std::cout << "Deleting beaver AB arrays." << std::endl;
    delete[] arithmetic_triple_a;
    delete[] arithmetic_triple_b;
    delete[] boolean_triple_a;
    delete[] boolean_triple_b;
}

void deinit_beaverAB_arithmetic()
{
    delete[] arithmetic_triple_a;
    delete[] arithmetic_triple_b;
}

void deinit_beaverAB_boolean()
{
    delete[] boolean_triple_a;
    delete[] boolean_triple_b;
}

void deinit_beaverC()
{
    // std::cout << "Deleting beaver C arrays." << std::endl;
    if(arithmetic_triple_c != nullptr) 
    {
        delete[] arithmetic_triple_c;
        arithmetic_triple_c = nullptr;
    }
    if(boolean_triple_c != nullptr)
    {
        delete[] boolean_triple_c;
        boolean_triple_c = nullptr;
    }
}

void deinit_beaverC_arithmetic()
{
    if(arithmetic_triple_c != nullptr) 
    {
        delete[] arithmetic_triple_c;
        arithmetic_triple_c = nullptr;
    }
}

void deinit_beaverC_boolean()
{
    if(boolean_triple_c != nullptr)
    {
        delete[] boolean_triple_c;
        boolean_triple_c = nullptr;
    }
}

struct timespec k1, k2;

void generate_beaver_triples(std::string ips[], int base_port, int process_offset, uint64_t num_arith_triples, uint64_t num_bool_triples, std::string triple_type)
{
    uint64_t l_num_arithmetic_triples = num_arith_triples * DATTYPE / BITLENGTH;
    uint64_t l_num_boolean_triples = num_bool_triples * DATTYPE;
    uint64_t l_num_multiplexer_triples = num_multiplexer_triples * DATTYPE / BITLENGTH;
    uint64_t l_num_cot_triples = num_cot_triples * DATTYPE / BITLENGTH;
    uint64_t l_num_boolean_addition_triples = num_boolean_addition_triples * DATTYPE;
#if FAKE_TRIPLES == 1
    print("Fake Triples set to 1, generating fake triples ... \n");
#else
    // print("Generating ", triple_type.data(), "  Triples ... \n");
    print("Generating %s Triples ... \n", triple_type.c_str());
#endif
    clock_t time_beaver_function_start = clock();
    clock_gettime(CLOCK_REALTIME, &k1);
    std::chrono::high_resolution_clock::time_point p = std::chrono::high_resolution_clock::now();

#if num_players == 2
if(triple_type == "LXLY") {
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
} else if(triple_type == "LXLY2") {
    generateArithmeticAB2Triples(arithmetic_ab2_triple_a,
                                 arithmetic_ab2_triple_b,
                                 arithmetic_ab2_triple_c,
                                 BITLENGTH,
                                 l_num_arithmetic_triples,
                                 ips[0],
                                 base_port + process_offset);
    generateBooleanAB2Triples(boolean_ab2_triple_a,
                                boolean_ab2_triple_b,
                                boolean_ab2_triple_c,
                                BITLENGTH,
                                l_num_boolean_triples,
                                ips[0],
                                base_port + process_offset);

} 
else if (triple_type == "CONV") {
    generateConvTriples(conv_triple_w,
                        conv_triple_x,
                        conv_triple_y,
                        BITLENGTH,
                        conv_triple_params,
                        ips[0],
                        base_port + process_offset);
}
else if (triple_type == "FC") {
    generateFCTriples(fc_triple_w,
                     fc_triple_x,
                     fc_triple_y,
                     BITLENGTH,
                     fc_triple_params,
                     ips[0],
                     base_port + process_offset);
}
else if (triple_type == "BATCHNORM2D") {
    generateBatchNorm2DTriples(bc2D_triple_w,
                              bc2D_triple_x,
                              bc2D_triple_y,
                              BITLENGTH,
                              bc2D_triple_params,
                              ips[0],
                              base_port + process_offset);
}
#if A2B_ONLINE_OPT == 1
else if (triple_type == "BOOLEANADDITION") {
    generateBooleanAdditionTriples(boolean_addition_triple_a,
                                   boolean_addition_triple_b,
                                   boolean_addition_triple_c,
                                   BITLENGTH,
                                   l_num_boolean_addition_triples,
                                   ips[0],
                                   base_port + process_offset);
}
#endif
#if BIT_INJECTION_PREPROCESSING_OPT == 1
else if (triple_type == "MULTIPLEXER") {
    generateMultiplexerTriples(multiplexer_triple_a,
                               multiplexer_triple_b,
                               multiplexer_triple_c,
                               BITLENGTH,
                               l_num_multiplexer_triples,
                               ips[0],
                               base_port + process_offset);
}
else if (triple_type == "COT") {
    generateCOTTriples(cot_triple_a,
                       cot_triple_c,
                       BITLENGTH,
                       l_num_cot_triples,
                       ips[0],
                       base_port + process_offset);
}
#endif
else {
    std::cerr << "Unknown triple type: " << triple_type << std::endl;
    exit(1);
}
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
              << "Boolean Beaver Triples Required: " << total_boolean_triples_num * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Arithmetic AB2 Beaver Triples Required: " << total_ab2_arithmetic_triples_num * DATTYPE / BITLENGTH
              << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
                << "Boolean AB2 Beaver Triples Required: " << total_ab2_boolean_triples_num * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Boolean Addition Triples Required: " << num_boolean_addition_triples * DATTYPE << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
              << "Multiplexer Triples Required: " << num_multiplexer_triples * DATTYPE / BITLENGTH << std::endl;
    std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": " 
              << "COT Triples Required: " << num_cot_triples * DATTYPE / BITLENGTH << std::endl;
#if A_KNOWN == 0
    std::string triple_type_str = "AB";
#else
    std::string triple_type_str = "AB2";
#endif
    for(int i = 0; i < conv_triple_params.size(); i++)
    {
        std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
            << "Convolution " << triple_type_str << " Triples Required for Conv layer " << i << ": " 
                  << conv_triple_params[i].batchSize * (((conv_triple_params[i].out_h + 0) / 1) * (((conv_triple_params[i].out_w + 0) / 1)) * conv_triple_params[i].dout)
                  << std::endl;
    }
    for(int i = 0; i < fc_triple_params.size(); i++)
    {
        std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
                  << "Fully Connected " << triple_type_str << " Triples Required for FC layer " << i << ": " 
                  << fc_triple_params[i].out_feat * fc_triple_params[i].batchSize
                  << std::endl;
    }
    for(int i = 0; i < bc2D_triple_params.size(); i++)
    {
        std::cout << "P" << PARTY << ", PRE, PID" << process_offset << ": "
                  << "BatchNorm2D " << triple_type_str << " Triples Required for BN2D layer " << i << ": " 
                  << bc2D_triple_params[i].batchSize * bc2D_triple_params[i].ch * bc2D_triple_params[i].h * bc2D_triple_params[i].w
                  << std::endl;
    }
#endif
}
