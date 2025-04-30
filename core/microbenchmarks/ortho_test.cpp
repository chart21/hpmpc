#include "../../config.h"
#include "../arch/DATATYPE.h"
#include "../crypto/aes/AES.h"
#include "../crypto/aes/AES_BS_SHORT.h"
#include "../crypto/sha/SHA_256.h"
#include "../utils/xorshift.h"

int main(int argc, char* argv[])
{

    UINT_TYPE* data = NEW(UINT_TYPE[DATTYPE]);
    for (int i = 0; i < DATTYPE; i++)
    {
        data[i] = i;
    }

    DATATYPE* arithmetic_data = NEW(DATATYPE[BITLENGTH]);
    orthogonalize_arithmetic(data, arithmetic_data);
    unorthogonalize_arithmetic(arithmetic_data, data);

    for (UINT_TYPE i = 0; i < DATTYPE; i++)
    {
        if (data[i] != i)
        {
            std::cout << "ARITHMETIC ERROR: " << i << " " << std::to_string(data[i]) << std::endl;
            /* return 1; */
        }
    }

    DATATYPE* bool_data = NEW(DATATYPE[BITLENGTH]);

    orthogonalize_boolean(data, bool_data);
    unorthogonalize_boolean(bool_data, data);
    for (UINT_TYPE i = 0; i < DATTYPE; i++)
    {
        if (data[i] != i)
        {
            std::cout << "BOOLEAN ERROR: " << i << " " << std::to_string(data[i]) << std::endl;
            /* return 1; */
        }
    }

    // Variant 1, easy ortho

    auto data_array = NEW(UINT_TYPE[3][DATTYPE]);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < DATTYPE; j++)
        {
            data_array[i][j] = i * DATTYPE + j;
        }
    }

    auto arithmetic_data_array = NEW(DATATYPE[3][BITLENGTH]);
    for (int i = 0; i < 3; i++)
    {
        orthogonalize_arithmetic(data_array[i], arithmetic_data_array[i]);
        unorthogonalize_arithmetic(arithmetic_data_array[i], data_array[i]);
    }

    for (int i = 0; i < 3; i++)
    {
        for (UINT_TYPE j = 0; j < DATTYPE; j++)
        {
            if (data_array[i][j] != i * DATTYPE + j)
            {
                std::cout << "ARITHMETIC ERROR: " << i << " " << j << " " << std::to_string(data_array[i][j])
                          << std::endl;
                /* return 1; */
            }
        }
    }

    auto bool_data_array = NEW(DATATYPE[3][BITLENGTH]);
    for (int i = 0; i < 3; i++)
    {
        orthogonalize_boolean(data_array[i], bool_data_array[i]);
        unorthogonalize_boolean(bool_data_array[i], data_array[i]);
    }

    for (int i = 0; i < 3; i++)
    {
        for (UINT_TYPE j = 0; j < DATTYPE; j++)
        {
            if (data_array[i][j] != i * DATTYPE + j)
            {
                std::cout << "BOOLEAN ERROR: " << i << " " << j << " " << std::to_string(data_array[i][j]) << std::endl;
                /* return 1; */
            }
        }
    }

    // Variant 2, hard ortho
    auto data_array_2 = NEW(UINT_TYPE[DATTYPE][3]);

    for (int i = 0; i < 3; i++)
    {
        for (UINT_TYPE j = 0; j < DATTYPE; j++)
        {
            data_array_2[i][j] = i * DATTYPE + j;
        }
    }

    auto bool_data_array2 = NEW(DATATYPE[BITLENGTH][3]);
    for (int i = 0; i < 3; i++)
    {
        orthogonalize_boolean((UINT_TYPE*)(data_array_2) + i * DATTYPE, (DATATYPE*)(bool_data_array2) + i * BITLENGTH);

        unorthogonalize_boolean((DATATYPE*)(bool_data_array2) + i * BITLENGTH,
                                (UINT_TYPE*)(data_array_2) + i * DATTYPE);
    }

    for (int i = 0; i < 3; i++)
    {
        for (UINT_TYPE j = 0; j < DATTYPE; j++)
        {
            if (data_array[i][j] != i * DATTYPE + j)
            {
                std::cout << "BOOLEAN ERROR: " << i << " " << j << " " << std::to_string(data_array[i][j]) << std::endl;
            }
        }
    }

    // transpose data_array into data_array_2
    for (int i = 0; i < 3; i++)
    {
        for (UINT_TYPE j = 0; j < DATTYPE; j++)
        {
            data_array_2[j][i] = data_array[i][j];
        }
    }

    auto arithmetic_data_array2 = NEW(DATATYPE[BITLENGTH][3]);
    for (int i = 0; i < 3; i++)
    {
        orthogonalize_arithmetic((UINT_TYPE*)(data_array_2) + i * DATTYPE,
                                 (DATATYPE*)(arithmetic_data_array2) + i * BITLENGTH);
        unorthogonalize_arithmetic((DATATYPE*)(arithmetic_data_array2) + i * BITLENGTH,
                                   (UINT_TYPE*)(data_array_2) + i * DATTYPE);
    }

    // transpose data_array_2 back into data_array
    for (int i = 0; i < 3; i++)
    {
        for (UINT_TYPE j = 0; j < DATTYPE; j++)
        {
            data_array[i][j] = data_array_2[j][i];
        }
    }

    for (int i = 0; i < 3; i++)
    {
        for (UINT_TYPE j = 0; j < DATTYPE; j++)
        {
            if (data_array[i][j] != i * DATTYPE + j)
            {
                std::cout << "ARITHMETIC ERROR: " << i << " " << j << " " << std::to_string(data_array[i][j])
                          << std::endl;
                /* return 1; */
            }
        }
    }

    std::cout << "finished" << std::endl;
    return 0;
}
