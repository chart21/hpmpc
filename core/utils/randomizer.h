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

#if RNG_AHEAD == 1 && RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0 && BUFFER_SIZE > 1
// The PSELF stream of getRandomVal, produced ahead by a background thread from a copy of the generator
// state (rng_ahead_start). Single consumer: the main thread (parallel circuit levels take their values
// from getRandomVals on the main thread).
namespace rng_ahead
{
constexpr size_t kCap = size_t(1) << 24;  // values (64 MB for 32-bit words)
inline DATATYPE* ring = nullptr;
inline std::atomic<uint64_t> head{0};  // produced
inline uint64_t tail = 0;              // consumed
inline std::atomic<bool> active{false}, stop{false};
inline std::atomic<uint64_t> tail_pub{0};  // consumed, as seen by the producer
inline uint64_t tail_seen() { return tail_pub.load(std::memory_order_acquire); }

inline void produce(AES_TYPE counter, uint64_t numgen)
{
    alignas(sizeof(AES_TYPE)) DATATYPE block[BUFFER_SIZE];
    uint64_t h = 0;
    MM_AES_STORE((AES_TYPE*)block, counter);
    while (!stop.load(std::memory_order_relaxed))
    {
        if (numgen >= BUFFER_SIZE)
        {
            AES_enc(counter, key_schedule[PSELF]);
            MM_AES_STORE((AES_TYPE*)block, counter);
            numgen = 0;
        }
        while (h + BUFFER_SIZE > tail_seen() + kCap)  // full: let the consumer catch up
        {
            if (stop.load(std::memory_order_relaxed))
                return;
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
        for (; numgen < BUFFER_SIZE; numgen++, h++)
            ring[h & (kCap - 1)] = block[numgen];
        head.store(h, std::memory_order_release);
    }
}
inline void wait_for(uint64_t n)
{
    while (head.load(std::memory_order_acquire) < n)
        __builtin_ia32_pause();
}
}  // namespace rng_ahead

// switch getRandomVal(PSELF) to the ring, continuing the stream from the current generator state
inline void rng_ahead_start()
{
    using namespace rng_ahead;
    if (active.load())
        return;
    ring = new DATATYPE[kCap];
    tail = 0;
    head.store(0);
    tail_pub.store(0);
    std::thread(produce, aes_counter[PSELF], num_generated[PSELF]).detach();  // runs until the process exits
    active.store(true);
}
#endif

DATATYPE getRandomVal(int link_id)
{
#if RNG_AHEAD == 1 && RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0 && BUFFER_SIZE > 1
    if (link_id == PSELF && rng_ahead::active.load(std::memory_order_relaxed)
#if ADDITIONAL_RELU_THREADS > 0
        && !tl_stream
#endif
    )
    {
#if ADDITIONAL_RELU_THREADS > 0
        rnd_calls_self++;
#endif
        using namespace rng_ahead;
        wait_for(tail + 1);
        const DATATYPE v = ring[tail & (kCap - 1)];
        tail_pub.store(++tail, std::memory_order_release);
        return v;
    }
#endif
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_stream)
    {
        if (link_id != PSELF)
            stream_cursor_misuse("getRandomVal of a shared link");
        return *tl_stream->rnd++;
    }
    rnd_calls_self += link_id == PSELF;
#endif
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

// n consecutive getRandomVal(link_id) values (the same stream), without the per-call block store
void getRandomVals(int link_id, DATATYPE* out, size_t n)
{
#if RNG_AHEAD == 1 && RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0 && BUFFER_SIZE > 1
    if (link_id == PSELF && rng_ahead::active.load(std::memory_order_relaxed))
    {
        using namespace rng_ahead;
#if ADDITIONAL_RELU_THREADS > 0
        if (tl_stream)
            stream_cursor_misuse("getRandomVals");
        rnd_calls_self += n;
#endif
        for (size_t i = 0; i < n;)
        {
            const size_t chunk = std::min<size_t>(n - i, kCap / 4);
            wait_for(tail + chunk);
            const size_t at = tail & (kCap - 1), first = std::min(chunk, kCap - at);
            std::memcpy(out + i, ring + at, first * sizeof(DATATYPE));
            std::memcpy(out + i + first, ring, (chunk - first) * sizeof(DATATYPE));
            tail += chunk;
            tail_pub.store(tail, std::memory_order_release);
            i += chunk;
        }
        return;
    }
#endif
#if RANDOM_ALGORITHM == 2 && USE_SSL_AES == 0 && BUFFER_SIZE > 1
#if ADDITIONAL_RELU_THREADS > 0
    if (tl_stream)
        stream_cursor_misuse("getRandomVals");
    rnd_calls_self += link_id == PSELF ? n : 0;
#endif
    alignas(sizeof(AES_TYPE)) DATATYPE block[BUFFER_SIZE];
    MM_AES_STORE((AES_TYPE*)block, aes_counter[link_id]);
    for (size_t i = 0; i < n;)
    {
        if (num_generated[link_id] >= BUFFER_SIZE)
        {
            AES_enc(aes_counter[link_id], key_schedule[link_id]);
            MM_AES_STORE((AES_TYPE*)block, aes_counter[link_id]);
            num_generated[link_id] = 0;
        }
        const size_t take = std::min<size_t>(n - i, BUFFER_SIZE - num_generated[link_id]);
        std::memcpy(out + i, block + num_generated[link_id], take * sizeof(DATATYPE));
        num_generated[link_id] += take;
        i += take;
    }
#else
    for (size_t i = 0; i < n; i++)
        out[i] = getRandomVal(link_id);
#endif
}

DATATYPE getRandomVal(int link_id1, int link_id2)
{
    return getRandomVal(num_players * (link_id1 + 1) + link_id2);
}
