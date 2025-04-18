#pragma once
#include "../include/pch.h"
#include "../networking/buffers.h"
#if RANDOM_ALGORITHM == 0
#include "../utils/xorshift.h"
#elif RANDOM_ALGORITHM == 1
#include "../crypto/aes/AES_BS_SHORT.h"
#elif RANDOM_ALGORITHM == 2
#if USE_SSL_AES == 1
#include "../crypto/aes/AES_SSL.h"
EVP_CIPHER_CTX* key_schedule[num_players * player_multiplier];
alignas(DATTYPE / 8) unsigned char aes_counter[num_players * player_multiplier][64];
#else
#include "../crypto/aes/AES.h"
AES_TYPE key_schedule[num_players * player_multiplier][11];
AES_TYPE aes_counter[num_players * player_multiplier];
#endif
#if AES_DATTYPE >= DATTYPE
#define BUFFER_SIZE AES_DATTYPE / DATTYPE
#else
#define BUFFER_SIZE -DATTYPE / AES_DATTYPE
#endif
#endif

#if RANDOM_ALGORITHM == 0
DATATYPE srng[num_players * player_multiplier][64]{0};
#elif RANDOM_ALGORITHM == 1
DATATYPE counter[num_players * player_multiplier][128]{0};
DATATYPE cipher[num_players * player_multiplier][128]{0};
DATATYPE key[num_players * player_multiplier][11][128]{0};
#endif

void init_buffers(int link_id)
{
#if RANDOM_ALGORITHM == 0
    num_generated[link_id] = 64;
#elif RANDOM_ALGORITHM == 1
    num_generated[link_id] = 128;
#elif RANDOM_ALGORITHM == 2
    num_generated[link_id] = BUFFER_SIZE;
#endif
}

DATATYPE getRandomVal(int link_id)
{
    /* if(link_id == 5 || link_id == 6 ) */
    /* return SET_ALL_ZERO(); */
#if RANDOM_ALGORITHM == 0
    if (num_generated[link_id] > 63)
    {
        num_generated[link_id] = 0;
        xor_shift(srng[link_id]);
    }
    num_generated[link_id] += 1;
    return srng[link_id][num_generated[link_id] - 1];
#elif RANDOM_ALGORITHM == 1
    if (num_generated[link_id] > 127)
    {
        num_generated[link_id] = 0;
        AES__(counter[link_id], key[link_id], cipher[link_id]);
        for (int i = 0; i < 128; i++)
        {
            counter[link_id][i] += 1;
        }
    }
    num_generated[link_id] += 1;
    return cipher[link_id][num_generated[link_id] - 1];
#elif RANDOM_ALGORITHM == 2
#if BUFFER_SIZE > 1
    if (num_generated[link_id] >= BUFFER_SIZE)
    {
        AES_enc(aes_counter[link_id], key_schedule[link_id]);
        /* DO_ENC_BLOCK(aes_counter[link_id], key_schedule[link_id]); */
        num_generated[link_id] = 0;
    }
#if USE_SSL_AES == 1
    return ((DATATYPE*)aes_counter[link_id])[num_generated[link_id]++];
#else
    alignas(sizeof(AES_TYPE)) DATATYPE ret[BUFFER_SIZE];
    MM_AES_STORE((AES_TYPE*)ret, aes_counter[link_id]);
    return ret[num_generated[link_id]++];
#endif
#elif BUFFER_SIZE == 1
    /* DO_ENC_BLOCK(aes_counter[link_id], key_schedule[link_id]); */
    AES_enc(aes_counter[link_id], key_schedule[link_id]);
    return ((DATATYPE*)aes_counter)[link_id];
#else
    DATATYPE ret;
    for (int i = 0; i > BUFFER_SIZE; i--)
    {
        AES_enc(aes_counter[link_id], key_schedule[link_id]);
        MM_AES_STORE(((AES_TYPE*)(&ret)) - i, aes_counter[link_id]);
    }
    return ret;
#endif

#endif
}

DATATYPE getRandomVal(int link_id1, int link_id2)
{
    return getRandomVal(num_players * (link_id1 + 1) + link_id2);
}
