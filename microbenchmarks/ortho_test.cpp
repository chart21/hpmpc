#include "../arch/DATATYPE.h"
#include "../crypto/aes/AES.h"
#include "../crypto/aes/AES_BS_SHORT.h"
#include "../utils/xorshift.h"
#include "../crypto/sha/SHA_256.h"
#include "../config.h"

#include <cstdint>
#include <iostream>
#include <string>

int main(int argc, char *argv[])
{

    /* alignas(16) UINT_TYPE in[2] = {0, 1}; */
    /* std::cout << std::to_string(in[0]) << std::to_string(in[1]) << std::endl; */
    /* DATATYPE intrin = _mm_set_epi64x (in[0], in[1]); */
    /* _mm_store_si128 ((__m128i*)&(in[0]), intrin); */
    /* std::cout << std::to_string(in[0]) << std::to_string(in[1]) << std::endl; */


    UINT_TYPE* data = NEW(UINT_TYPE[DATTYPE]);
    for (int i = 0; i < DATTYPE; i++) {
        data[i] = i;
    }
    

    

    DATATYPE* arithmetic_data = NEW(DATATYPE[BITLENGTH]);
    orthogonalize_arithmetic(data, arithmetic_data); 
    unorthogonalize_arithmetic(arithmetic_data, data);

    for (UINT_TYPE i = 0; i < DATTYPE; i++) {
        if (data[i] != i) {
            std::cout << "ARITHMETIC ERROR: " << i << " " << std::to_string(data[i]) << std::endl;
            /* return 1; */
        }
    }
    
   
    
    DATATYPE* bool_data = NEW(DATATYPE[BITLENGTH]);

    orthogonalize_boolean(data, bool_data);
    unorthogonalize_boolean(bool_data, data);
    for (UINT_TYPE i = 0; i < DATTYPE; i++) {
        if (data[i] != i) {
            std::cout << "BOOLEAN ERROR: " << i << " " << std::to_string(data[i]) << std::endl;
            /* return 1; */
        }
    }
    std::cout << "finished" << std::endl;
    return 0;
}
