#include <chrono>
#include <cstdint>
#include <iostream>
/* #define DATTYPE 128 */
#include "../../config.h"
#include "../arch/DATATYPE.h"
#include "../crypto/aes/AES.h"
#include "../crypto/aes/AES_BS_SHORT.h"
#include "../crypto/sha/SHA_256.h"
#include "../utils/xorshift.h"
#ifdef __SHA__
#include "../crypto/sha/SHA_256_x86.h"
#elif ARM == 1
#include "../crypto/sha/SHA_256_arm.h"
#endif

int array_size = 1000000000;
double factor = (double)1000000000 / array_size;

std::vector<double> process_benchmark(int process_id)
{
    UINT_TYPE* data = NEW(UINT_TYPE[1000000000]);
    DATATYPE* ortho_data = NEW(DATATYPE[1000000000 / (DATTYPE / BITLENGTH)]);

    auto finish21 = std::chrono::high_resolution_clock::now();
    int counter = 0;
    for (int i = 0; i < 1000000000 / (DATTYPE / BITLENGTH); i += DATTYPE)
    {
        orthogonalize_boolean(data + i, ortho_data + counter);
        counter += BITLENGTH;
    }
    auto finish22 = std::chrono::high_resolution_clock::now();
    counter = 0;
    for (int i = 0; i < 1000000000 / DATTYPE; i += 2)
    {
        ortho_data[i] = FUNC_XOR(ortho_data[i], ortho_data[i + 1]);
    }
    for (int i = 0; i < 1000000000 / (DATTYPE / BITLENGTH); i += DATTYPE)
    {
        unorthogonalize_boolean(ortho_data + counter, data + i);
        counter += BITLENGTH;
    }
    auto finish23 = std::chrono::high_resolution_clock::now();
    auto finish24 = std::chrono::high_resolution_clock::now();
    orthogonalize_arithmetic(data, ortho_data, 1000000000 / (DATTYPE / BITLENGTH));
    auto finish25 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000000 / (DATTYPE / BITLENGTH); i += 2)
    {
        ortho_data[i] = OP_ADD(ortho_data[i], ortho_data[i + 1]);
    }
    auto finish26 = std::chrono::high_resolution_clock::now();
    unorthogonalize_arithmetic(ortho_data, data, 1000000000 / (DATTYPE / BITLENGTH));
    auto finish27 = std::chrono::high_resolution_clock::now();

    std::cout << "Process " << process_id << " - First element: " << data[0] << std::endl;

    delete[] data;
    delete[] ortho_data;

    std::vector<double> results;
    results.push_back(
        BITLENGTH /
        (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish22 - finish21).count() * factor) / 1000));
    results.push_back(
        BITLENGTH /
        (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish23 - finish22).count() * factor) / 1000));
    results.push_back(
        BITLENGTH /
        (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish25 - finish24).count() * factor) / 1000));
    results.push_back(
        BITLENGTH /
        (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish27 - finish26).count() * factor) / 1000));

    return results;
}

int main()
{

#ifdef __AES__
    std::cout << "AES-NI is supported" << std::endl;
#endif

#ifdef __VAES__
    std::cout << "VAES is supported" << std::endl;
#endif

#ifdef __SSE4_1__
    std::cout << "SSE4.1 is supported" << std::endl;
#endif

#ifdef __SSE4_2__
    std::cout << "SSE4.2 is supported" << std::endl;
#endif

#ifdef __AVX__
    std::cout << "AVX is supported" << std::endl;
#endif

#ifdef __AVX2__
    std::cout << "AVX2 is supported" << std::endl;
#endif

#ifdef __AVX512F__
    std::cout << "AVX512F is supported" << std::endl;
#endif

#ifdef __AVX512BW__
    std::cout << "AVX512BW is supported" << std::endl;
#endif

#ifdef __SHA__
    std::cout << "SHA is supported" << std::endl;
#endif

#ifdef __PEXT__
    std::cout << "PEXT is supported" << std::endl;
#endif

    /* std::cout << aes128_self_test() << std::endl; */
    DATATYPE plain__[128];
    DATATYPE key__[11][128];
    DATATYPE cipher__[128];
    DATATYPE seed[64];

    DATATYPE m;
    DATATYPE k[11];

    uint8_t message[128000];
    /* memset(message, 0x00, sizeof(message)); */
    message[0] = 0x80;

    /* initial state */
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    std::chrono::system_clock::time_point start;
    std::chrono::system_clock::time_point finish;
    std::chrono::system_clock::time_point finish2;
    std::chrono::system_clock::time_point finish3;
    std::chrono::system_clock::time_point finish4;
    std::chrono::system_clock::time_point finish5;
    std::chrono::system_clock::time_point finish6;
    std::chrono::system_clock::time_point finish7;
    std::chrono::system_clock::time_point finish8;
    std::chrono::system_clock::time_point finish9;
    std::chrono::system_clock::time_point finish10;
    std::chrono::system_clock::time_point finish11;
    std::chrono::system_clock::time_point finish12;
    std::chrono::system_clock::time_point finish13;
    std::chrono::system_clock::time_point finish14;
    std::chrono::system_clock::time_point finish15;
    std::chrono::system_clock::time_point finish16;
    std::chrono::system_clock::time_point finish17;
    std::chrono::system_clock::time_point finish18;
    std::chrono::system_clock::time_point finish19;
    std::chrono::system_clock::time_point finish20;
    std::chrono::system_clock::time_point finish21;
    std::chrono::system_clock::time_point finish22;
    std::chrono::system_clock::time_point finish23;
    std::chrono::system_clock::time_point finish24;
    std::chrono::system_clock::time_point finish25;
    std::chrono::system_clock::time_point finish26;
    std::chrono::system_clock::time_point finish27;

    // Warmup
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            DO_ENC_BLOCK(m, k);
            k[0] = m;
            m = m + 1;
        }
    }
    for (int i = 0; i < 1000; i++)
    {
        AES__(plain__, key__, cipher__);
        plain__[0] = cipher__[0];
    }
    for (int i = 0; i < 1000; i++)
    {
        for (int j = 0; j < 2; j++)
            xor_shift(seed);
    }

    for (int i = 0; i < 1000; i++)
    {
        sha256_process(state, message, sizeof(message));
    }

#ifdef __SHA__
    for (int i = 0; i < 1000; i++)
    {
        sha256_process_x86(state, message, sizeof(message));
    }
#endif

    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000000; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            DO_ENC_BLOCK(m, k);
            k[0] = m;
            m = m + 1;
        }
    }
    finish = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++)
    {
        AES__(plain__, key__, cipher__);
        plain__[0] = cipher__[0];
    }
    finish2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000; i++)
    {
        for (int j = 0; j < 2; j++)
            xor_shift(seed);
    }

    finish3 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 1000; i++)
    {
        sha256_process(state, message, sizeof(message));
    }
    finish4 = std::chrono::high_resolution_clock::now();

#ifdef __SHA__
    for (int i = 0; i < 1000; i++)
    {
        sha256_process_x86(state, message, sizeof(message));
    }
    finish5 = std::chrono::high_resolution_clock::now();
    std::cout << "SHA256 (hardware module): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(finish5 - finish4).count() << std::endl;
    std::cout << "SHA256 (hardware module) Throughput: "
              << 1.024 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish5 - finish4).count() / 1000)
              << "GB/s" << std::endl;
#elif ARM == 1
    for (int i = 0; i < 1000; i++)
    {
        sha256_process_arm(state, message, sizeof(message));
    }
    finish5 = std::chrono::high_resolution_clock::now();
#endif
    std::cout << "AES_NI: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
              << std::endl;
    std::cout << "AES_NI Throughput: "
              << (0.128 * DATTYPE) /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count() / 1000)
              << "GBit/s" << std::endl;
    std::cout << "AES_BS: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish2 - finish).count()
              << std::endl;
    std::cout << "AES_BS Throughput: "
              << (0.128 * DATTYPE) /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish2 - finish).count() / 1000)
              << "GBit/s" << std::endl;
    std::cout << "XOR_Shift: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish3 - finish2).count()
              << std::endl;
    std::cout << "XOR_Shift Throughput: "
              << (0.128 * DATTYPE) /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish3 - finish2).count() / 1000)
              << "GBit/s" << std::endl;
    std::cout << "SHA256: " << std::chrono::duration_cast<std::chrono::milliseconds>(finish4 - finish3).count() * 1000
              << std::endl;
    std::cout << "SHA256 Throughput: "
              << 1.024 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish4 - finish3).count() / 1000)
              << "GBit/s" << std::endl;
    /* std::cout << "Tested with 16384MB for AES, 1024MB for SHA256" << std::endl; */

    /* auto a = new UINT_TYPE[1000000000]; */
    auto b = new UINT_TYPE[1000000000];
    auto c = new UINT_TYPE[1000000000];
    finish6 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < 1000000000; i++)
    {
        c[i] = c[i - 1] * b[i];
    }
    finish7 = std::chrono::high_resolution_clock::now();

    finish13 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < 1000000000; i++)
    {
        c[i] = c[i - 1] + b[i];
    }
    finish14 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < 1000000000; i++)
    {
        c[i] = c[i - 1] ^ b[i];
    }
    finish20 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < 1000000000; i++)
    {
        c[i] = c[i - 1] && b[i];
    }
    finish21 = std::chrono::high_resolution_clock::now();

    /* delete[] a; */
    delete[] b;
    delete[] c;
    std::cout << "64-bit Mult Throughput in Gbps: "
              << 64 / ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish7 - finish6).count() / 1000)
              << std::endl;
    std::cout << "64-bit Add Throughput in Gbps: "
              << 64 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish14 - finish13).count() / 1000)
              << std::endl;
    std::cout << "64-bit XOR Throughput in Gbps: "
              << 64 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish20 - finish14).count() / 1000)
              << std::endl;
    std::cout << "64-bit AND Throughput in Gbps: "
              << 64 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish21 - finish20).count() / 1000)
              << std::endl;

    /* auto g = new uint32_t[1000000000]; */
    auto h = new uint32_t[1000000000];
    auto j = new uint32_t[1000000000];
    finish10 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < 1000000000; i++)
        j[i] = j[i - 1] * h[i];
    finish11 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < 1000000000; i++)
        j[i] = j[i - 1] + h[i];
    finish12 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < 1000000000; i++)
    {
        j[i] = j[i - 1] ^ h[i];
    }
    finish22 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < 1000000000; i++)
    {
        j[i] = j[i - 1] && h[i];
    }
    finish23 = std::chrono::high_resolution_clock::now();

    /* delete[] g; */
    delete[] h;
    delete[] j;
    std::cout << "32-bit Add Throughput in Gbps: "
              << 32 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish12 - finish11).count() / 1000)
              << std::endl;

    std::cout << "32-bit Mult Throughput in Gbps: "
              << 32 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish11 - finish10).count() / 1000)
              << std::endl;
    std::cout << "32-bit XOR Throughput in Gbps: "
              << 32 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish22 - finish12).count() / 1000)
              << std::endl;
    std::cout << "32-bit AND Throughput in Gbps: "
              << 32 /
                     ((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish23 - finish22).count() / 1000)
              << std::endl;

    /* auto d = new DATATYPE[1000000000]; */

    auto e = new DATATYPE[array_size];
    auto f = new DATATYPE[array_size];

    // Warmup
    for (int i = 1; i < array_size / 1000; i++)
    {
        f[i] = AND(f[i - 1], e[i]);
        f[i] = XOR(f[i - 1], e[i]);
    }

    finish8 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < array_size; i++)
    {
        f[i] = AND(f[i - 1], e[i]);
    }
    finish9 = std::chrono::high_resolution_clock::now();

    finish15 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i < array_size; i++)
    {
        f[i] = XOR(f[i - 1], e[i]);
    }

    finish16 = std::chrono::high_resolution_clock::now();

    std::cout << "DATTYPE AND Throughput in Gbps: "
              << DATTYPE / (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish9 - finish8).count() *
                             factor) /
                            1000)
              << std::endl;
    std::cout << "DATTYPE XOR Throughput in Gbps: "
              << DATTYPE /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish16 - finish15).count() *
                       factor) /
                      1000)
              << std::endl;

    // Warmup
    for (int i = 1; i < array_size / 1000; i++)
    {
        f[i] = ADD_SIGNED(f[i - 1], e[i], 32);
        f[i] = MUL_SIGNED(f[i - 1], e[i], 32);
    }

    for (int i = 1; i < array_size; i++)
    {
        f[i] = ADD_SIGNED(f[i - 1], e[i], 32);
    }

    finish17 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < array_size; i++)
    {
        f[i] = MUL_SIGNED(f[i - 1], e[i], 32);
    }

    finish18 = std::chrono::high_resolution_clock::now();
    std::cout << "Signed 32-bit Add DATTYPE Throughput in Gbps:"
              << DATTYPE /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish17 - finish16).count() *
                       factor) /
                      1000)
              << std::endl;
    std::cout << "Signed 32-bit Mult DATTYPE Throughput in Gbps:"
              << DATTYPE /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish18 - finish17).count() *
                       factor) /
                      1000)
              << std::endl;

#if defined(__AVX512F__) & defined(__AVX512DQ__)

    finish18 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < array_size; i++)
    {
        f[i] = ADD_SIGNED(f[i - 1], e[i], 64);
    }

    finish19 = std::chrono::high_resolution_clock::now();

    for (int i = 1; i < array_size; i++)
    {
        f[i] = MUL_SIGNED(f[i - 1], e[i], 64);
    }

    finish20 = std::chrono::high_resolution_clock::now();

    std::cout << "Signed 64-bit Add DATTYPE Throughput in Gbps:"
              << DATTYPE /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish19 - finish18).count() *
                       factor) /
                      1000)
              << std::endl;
    std::cout << "Signed 64-bit Mult DATTYPE Throughput in Gbps:"
              << DATTYPE /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish20 - finish19).count() *
                       factor) /
                      1000)
              << std::endl;
#endif

    /* delete[] d; */
    delete[] e;
    delete[] f;

    UINT_TYPE* data = NEW(UINT_TYPE[1000000000]);
    /* int multiplier = DATTYPE / 64; */
    DATATYPE* ortho_data = NEW(DATATYPE[1000000000 / (DATTYPE / BITLENGTH)]);
    finish21 = std::chrono::high_resolution_clock::now();
    int counter = 0;
    for (int i = 0; i < 1000000000 / (DATTYPE / BITLENGTH); i += DATTYPE)
    {
        orthogonalize_boolean(data + i, ortho_data + counter);
        counter += BITLENGTH;
    }
    finish22 = std::chrono::high_resolution_clock::now();
    counter = 0;
    for (int i = 0; i < 1000000000 / DATTYPE; i += 2)
    {
        ortho_data[i] = FUNC_XOR(ortho_data[i], ortho_data[i + 1]);
    }

    for (int i = 0; i < 1000000000 / (DATTYPE / BITLENGTH); i += DATTYPE)
    {
        unorthogonalize_boolean(ortho_data + counter, data + i);
        counter += BITLENGTH;
    }
    finish23 = std::chrono::high_resolution_clock::now();

    finish24 = std::chrono::high_resolution_clock::now();
    orthogonalize_arithmetic(data, ortho_data, 1000000000 / (DATTYPE / BITLENGTH));
    finish25 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000000000 / (DATTYPE / BITLENGTH); i += 2)
    {
        ortho_data[i] = OP_ADD(ortho_data[i], ortho_data[i + 1]);
    }
    finish26 = std::chrono::high_resolution_clock::now();
    unorthogonalize_arithmetic(ortho_data, data, 1000000000 / (DATTYPE / BITLENGTH));
    finish27 = std::chrono::high_resolution_clock::now();

    std::cout << data[0] << std::endl;

    delete[] data;
    delete[] ortho_data;

    std::cout << "Orthogonalize Boolean Throughput in Gbps: "
              << BITLENGTH /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish22 - finish21).count() *
                       factor) /
                      1000)
              << std::endl;
    std::cout << "Unorthogonalize Boolean Throughput in Gbps: "
              << BITLENGTH /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish23 - finish22).count() *
                       factor) /
                      1000)
              << std::endl;
    std::cout << "Orthogonalize Arithmetic Throughput in Gbps: "
              << BITLENGTH /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish25 - finish24).count() *
                       factor) /
                      1000)
              << std::endl;
    std::cout << "Unorthogonalize Arithmetic Throughput in Gbps: "
              << BITLENGTH /
                     (((double)std::chrono::duration_cast<std::chrono::milliseconds>(finish27 - finish26).count() *
                       factor) /
                      1000)
              << std::endl;

    // Main code (replace the original benchmark code with this)
    std::vector<std::thread> threads;
    std::vector<std::vector<double>> all_results(PROCESS_NUM);

    for (int i = 0; i < PROCESS_NUM; ++i)
    {
        threads.emplace_back([i, &all_results]() { all_results[i] = process_benchmark(i); });
    }

    for (auto& thread : threads)
    {
        thread.join();
    }

    // Calculate average results
    std::vector<double> avg_results(4, 0.0);
    for (const auto& result : all_results)
    {
        for (int i = 0; i < 4; ++i)
        {
            avg_results[i] += result[i];
        }
    }
    for (auto& avg : avg_results)
    {
        avg /= PROCESS_NUM;
    }

    // Print average results
    std::cout << "Orthogonalize Boolean Throughput (MT) in Gbps: " << avg_results[0] * PROCESS_NUM << std::endl;
    std::cout << "Unorthogonalize Boolean Throughput (MT) in Gbps: " << avg_results[1] * PROCESS_NUM << std::endl;
    std::cout << "Orthogonalize Arithmetic Throughput (MT) in Gbps: " << avg_results[2] * PROCESS_NUM << std::endl;
    std::cout << "Unorthogonalize Arithmetic Throughput (MT) in Gbps: " << avg_results[3] * PROCESS_NUM << std::endl;

    std::cout << m[0] << std::endl;
    std::cout << plain__[0][0] << std::endl;
    std::cout << seed << std::endl;
    std::cout << state[0] << std::endl;
    std::cout << message[0] << std::endl;
    std::cout << c[999999] << std::endl;
    std::cout << j[999999] << std::endl;

    return 0;
}
