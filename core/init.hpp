#pragma once
#include "include/pch.h"
#include "networking/buffers.h"
#include "networking/client.hpp"
#include "networking/server.hpp"
#include "utils/print.hpp"
#include "utils/randomizer.h"

#if LIVE == 1 && INIT == 0 && NO_INI == 0
#include "core/utils/CircuitInfo.hpp"
#endif

struct timespec i1, p1, p2, l1, l2;

int modulo(int x, int N)
{
    return (x % N + N) % N;
}

void init_srng(uint64_t link_id, uint64_t link_seed)
{
#if RANDOM_ALGORITHM == 0
    UINT_TYPE gen_seeds[DATTYPE];
    for (int i = 0; i < DATTYPE; i++)
    {
        gen_seeds[i] = link_seed * (i + 1);  // replace with independant seeds in the future
    }
    orthogonalize_boolean(gen_seeds, srng[link_id]);

#elif RANDOM_ALGORITHM == 1
    UINT_TYPE gen_keys[11][DATTYPE * 128 / BITLENGTH];
    for (int i = 0; i < 11; i++)
    {
        for (int j = 0; j < DATTYPE * 128 / BITLENGTH; j++)
        {
            gen_keys[i][j] = link_seed * ((i + 1) * j);  // replace with independant seeds in the future
        }
    }
    for (int i = 0; i < 11; i++)
    {
        for (int j = 0; j < DATTYPE * 128 / BITLENGTH; j++)
            orthogonalize_boolean(gen_keys[i] + j, key[link_id][i] + j);
    }

#elif RANDOM_ALGORITHM == 2
    srand(link_seed);
    alignas(AES_DATTYPE / 8) uint64_t counter[AES_DATTYPE / 64];
#if USE_SSL_AES == 1
    for (int j = 0; j < 64; j++)
        aes_counter[link_id][j] =
            rand() % 256 |
            rand() %
                256;  // generate random 8-bit number, to stay consistent with other approaches, 2 times rand is used
#else
    for (int j = 0; j < AES_DATTYPE / 64; j++)
        counter[j] = ((uint64_t)rand() << 32) | rand();  // generate random 64-bit number

#if defined(__AVX512F__) && defined(__VAES__)
    aes_counter[link_id] = _mm512_set_epi64(
        counter[7], counter[6], counter[5], counter[4], counter[3], counter[2], counter[1], counter[0]);
#elif defined(__AVX2__) && defined(__VAES__)
    aes_counter[link_id] = _mm256_set_epi64x(counter[3], counter[2], counter[1], counter[0]);
#elif defined(__AES__)
    aes_counter[link_id] = _mm_set_epi64x(counter[1], counter[0]);
#endif
#endif

    alignas(AES_DATTYPE / 8) uint8_t seed[AES_DATTYPE / 8];
    for (int i = 0; i < AES_DATTYPE / 8; i++)
    {
        /* seed[link_id][i] = rand() % 256; */
        seed[i] = rand() % 256;  //
        /* seed[j][i] = 10; */
    }
#if USE_SSL_AES == 1
    key_schedule[link_id] = EVP_CIPHER_CTX_new();
    if (!key_schedule[link_id])
        handleErrors();
#endif
    aes_load_enc(seed, key_schedule[link_id]);

#endif
    init_buffers(link_id);
#if MAL == 1

    // Ensure all players have the same initial state
    /* initial state */
    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};

    for (int i = 0; i < num_players - 1; i++)
    {
        for (int j = 0; j < 8; j++)
            hash_val[i][j] = state[j];
    }

#endif
}

void init_muetexes()
{
    pthread_mutex_init(&mtx_connection_established, NULL);
    pthread_mutex_init(&mtx_start_communicating, NULL);
    pthread_mutex_init(&mtx_send_next, NULL);
    pthread_cond_init(&cond_successful_connection, NULL);
    pthread_cond_init(&cond_start_signal, NULL);
    pthread_cond_init(&cond_send_next, NULL);
}
