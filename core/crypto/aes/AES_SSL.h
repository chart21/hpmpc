#ifndef __AES_NI_H__
#define __AES_NI_H__
#include "openssl/aes.h"
#include <openssl/err.h>
#include <openssl/evp.h>
// macros

#define AES_DATTYPE 512
#define AES_TYPE uint8_t

#define DO_ENC_BLOCK(m, k)                        \
    do                                            \
    {                                             \
        int out_len;                              \
        EVP_EncryptUpdate(k, m, &out_len, m, 64); \
    } while (0)
#endif

void AES_enc(uint8_t m[], EVP_CIPHER_CTX* ctx)
{
    int out_len;
    EVP_EncryptUpdate(ctx, m, &out_len, m, 64);
}

// Function to handle errors
void handleErrors()
{
    ERR_print_errors_fp(stderr);
    abort();
}

// Initialize the AES ECB context with the given key
void initialize_aes_ecb_context(const uint8_t* key, EVP_CIPHER_CTX* ctx)
{
    // Create and initialize the context

    // Initialize the operation (1 for encryption, 0 for decryption)
    if (1 != EVP_EncryptInit_ex(ctx, EVP_aes_128_ecb(), NULL, (const unsigned char*)key, NULL))
    {
        handleErrors();
    }
}
static void aes_load_enc(uint8_t* enc_key, EVP_CIPHER_CTX* ctx)
{
    initialize_aes_ecb_context(enc_key, ctx);
}
