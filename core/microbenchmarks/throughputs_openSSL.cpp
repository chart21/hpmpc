#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <chrono>
#include <cstring>
#include <iostream>

void MeasureSHAPerformance(const EVP_MD* type, const char* type_name)
{
    const size_t dataSize = 1 << 30;  // 1 GiB
    unsigned char* data = new unsigned char[dataSize];
    memset(data, 0, dataSize);  // Fill with zeros for simplicity

    unsigned char hash[EVP_MAX_MD_SIZE];
    auto start = std::chrono::high_resolution_clock::now();

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ctx, type, nullptr);

    for (size_t i = 0; i < dataSize; i += EVP_MD_block_size(type))
    {
        EVP_DigestUpdate(ctx, data + i, EVP_MD_block_size(type));
    }

    unsigned int out_len;
    EVP_DigestFinal_ex(ctx, hash, &out_len);

    auto end = std::chrono::high_resolution_clock::now();

    EVP_MD_CTX_free(ctx);

    std::chrono::duration<double> elapsed = end - start;
    double gigabits = (dataSize * 8) / (1 << 30);
    double throughput = gigabits / elapsed.count();

    std::cout << type_name << " throughput: " << throughput << " Gbit/s" << std::endl;

    delete[] data;
}

void MeasureAESPerformance(const EVP_CIPHER* type, const char* type_name)
{
    unsigned char key[EVP_MAX_KEY_LENGTH], iv[EVP_MAX_IV_LENGTH];

    // Generate a random key and IV
    if (!RAND_bytes(key, sizeof(key)) || !RAND_bytes(iv, sizeof(iv)))
    {
        std::cerr << "Failed to generate key or IV" << std::endl;
        return;
    }

    const size_t dataSize = 1 << 30;  // 1 GiB
    unsigned char* data = new unsigned char[dataSize];
    unsigned char* enc_data = new unsigned char[dataSize];
    memset(data, 0, dataSize);  // Fill with zeros for simplicity

    auto start = std::chrono::high_resolution_clock::now();

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit_ex(ctx, type, nullptr, key, iv);

    for (size_t i = 0; i < dataSize; i += EVP_CIPHER_block_size(type))
    {
        int out_len;
        EVP_EncryptUpdate(ctx, enc_data + i, &out_len, data + i, EVP_CIPHER_block_size(type));
    }

    int out_len;
    EVP_EncryptFinal_ex(ctx, enc_data + dataSize, &out_len);

    auto end = std::chrono::high_resolution_clock::now();

    EVP_CIPHER_CTX_free(ctx);

    std::chrono::duration<double> elapsed = end - start;
    double gigabits = (dataSize * 8) / (1 << 30);
    double throughput = gigabits / elapsed.count();

    std::cout << type_name << " throughput: " << throughput << " Gbit/s" << std::endl;

    delete[] data;
    delete[] enc_data;
}
void MeasureAESGCMPerformance()
{
    unsigned char key[AES_BLOCK_SIZE], iv[AES_BLOCK_SIZE];

    // Generate a random key and IV
    if (!RAND_bytes(key, sizeof(key)) || !RAND_bytes(iv, sizeof(iv)))
    {
        std::cerr << "Failed to generate key or IV" << std::endl;
        return;
    }

    const size_t dataSize = 1 << 30;  // 1 GiB
    unsigned char* data = new unsigned char[dataSize];
    unsigned char* enc_data = new unsigned char[dataSize];
    memset(data, 0, dataSize);  // Fill with zeros for simplicity

    auto start = std::chrono::high_resolution_clock::now();

    EVP_CIPHER_CTX* ctx = EVP_CIPHER_CTX_new();
    EVP_EncryptInit(ctx, EVP_aes_256_gcm(), key, iv);

    for (size_t i = 0; i < dataSize; i += AES_BLOCK_SIZE)
    {
        int out_len;
        EVP_EncryptUpdate(ctx, enc_data + i, &out_len, data + i, AES_BLOCK_SIZE);
    }

    int out_len;
    EVP_EncryptFinal(ctx, enc_data + dataSize, &out_len);

    auto end = std::chrono::high_resolution_clock::now();

    EVP_CIPHER_CTX_free(ctx);

    std::chrono::duration<double> elapsed = end - start;
    double gigabits = (dataSize * 8) / (1 << 30);
    double throughput = gigabits / elapsed.count();

    std::cout << "AES-256-GCM throughput: " << throughput << " Gbit/s" << std::endl;

    delete[] data;
    delete[] enc_data;
}
int main()
{
    MeasureAESPerformance(EVP_aes_256_cbc(), "AES-256-CBC");
    MeasureAESPerformance(EVP_aes_128_cbc(), "AES-128-CBC");
    MeasureAESPerformance(EVP_aes_256_ecb(), "AES-256-ECB");
    MeasureAESPerformance(EVP_aes_128_ecb(), "AES-128-ECB");
    MeasureAESPerformance(EVP_aes_256_cfb(), "AES-256-CFB");
    MeasureAESPerformance(EVP_aes_128_cfb(), "AES-128-CFB");

    MeasureSHAPerformance(EVP_sha1(), "SHA-1");
    MeasureSHAPerformance(EVP_sha224(), "SHA-224");
    MeasureSHAPerformance(EVP_sha256(), "SHA-256");
    MeasureSHAPerformance(EVP_sha384(), "SHA-384");
    MeasureSHAPerformance(EVP_sha512(), "SHA-512");

    MeasureAESGCMPerformance();

    return 0;
}
