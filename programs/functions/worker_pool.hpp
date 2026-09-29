#pragma once
#include "../../config.h"
#if ADDITIONAL_GEMM_THREADS > 0 || ADDITIONAL_RELU_THREADS > 0
#include <atomic>
#include <functional>
#include <immintrin.h>
#include <x86intrin.h>
#include <thread>
#include <vector>

// Pause iterations before an idle worker sleeps. Single process: ~6 ms, longer than an online round trip, so the
// workers are still spinning when the next circuit level arrives (a futex wake per level cost 17-20% of the ImageNet
// online phase). Several processes share the cores (multi-batch): ~100 us.
#ifndef GEMM_POOL_MWAITX
#define GEMM_POOL_MWAITX 1  // idle workers wait with MONITORX/MWAITX where the CPU has it (AMD), else pause
#endif
#ifndef GEMM_POOL_SPIN
#if PROCESS_NUM > 1
#define GEMM_POOL_SPIN (1 << 14)
#else
#define GEMM_POOL_SPIN (1 << 20)
#endif
#endif

static_assert(ADDITIONAL_GEMM_THREADS == 0 || ADDITIONAL_RELU_THREADS == 0 || ADDITIONAL_GEMM_THREADS == ADDITIONAL_RELU_THREADS,
              "the GEMM and the ReLU share one pool: its size is the GEMM partition");
#define WORKER_POOL_THREADS (ADDITIONAL_GEMM_THREADS > ADDITIONAL_RELU_THREADS ? ADDITIONAL_GEMM_THREADS : ADDITIONAL_RELU_THREADS)

// The ADDITIONAL_GEMM_THREADS workers of GEMM_threaded.hpp, kept for the whole run: a layer calls the
// GEMM once per image, and spawning the threads on every call cost more than the parallel accumulation
// saved (CIFAR-10 ResNet50, batch 10: online 0.63 -> 1.0 s)
class GemmPool
{
  public:
    static GemmPool& get()
    {
        static GemmPool pool(WORKER_POOL_THREADS);
        return pool;
    }

    // f(t) for t = 0 .. WORKER_POOL_THREADS, the last one on the calling thread; returns when all are done.
    // Lock-free hand-off: with a mutex and condition variables, the 31 workers woke up into the same mutex
    // on every call, and the futex contention took a quarter of the online CPU time (CIFAR-10 ResNet50).
    void run(const std::function<void(int)>& f)
    {
        job_ = &f;
        pending_.store(int(workers_.size()), std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        generation_.notify_all();
        f(int(workers_.size()));
        for (int spin = 0; pending_.load(std::memory_order_acquire) != 0; ++spin)
        {
            if (spin < kSpin)
                _mm_pause();
            else
                std::this_thread::yield();
        }
        job_ = nullptr;
    }

    ~GemmPool()
    {
        stop_.store(true, std::memory_order_relaxed);
        generation_.fetch_add(1, std::memory_order_release);
        generation_.notify_all();
        for (auto& w : workers_)
            w.join();
    }

  private:
    // pause iterations before a worker sleeps (GEMMs of a layer follow closely; 1 << 14 ~ 100 us on Zen 4)
    static constexpr int kSpin = GEMM_POOL_SPIN;

    explicit GemmPool(int n)
    {
        for (int t = 0; t < n; ++t)
            workers_.emplace_back([this, t] { work(t); });
    }

    // Wait for the next job. With MONITORX/MWAITX (AMD, -march=native on Zen) a worker parks on generation_'s cache
    // line until it is written (or a short timer expires), without taking issue slots from its SMT sibling, which a
    // pause loop does: with 24 spinning workers the Zen 3 hosts' serial parts of the online phase slowed down.
    // Either way the worker sleeps on a futex after the spin budget.
    void work(int t)
    {
        uint64_t seen = 0;
        for (;;)
        {
            uint64_t g;
#if defined(__MWAITX__) && GEMM_POOL_MWAITX == 1
            for (int spin = 0; (g = generation_.load(std::memory_order_acquire)) == seen; ++spin)
            {
                if (spin < (kSpin >> 8))
                {
                    _mm_monitorx((void*) &generation_, 0, 0);
                    if ((g = generation_.load(std::memory_order_acquire)) != seen)
                        break;
                    _mm_mwaitx(2, 0, 1u << 14);  // timer enabled: at most 2^14 TSC ticks per wait
                }
                else
                    generation_.wait(seen, std::memory_order_acquire);
            }
#else
            for (int spin = 0; (g = generation_.load(std::memory_order_acquire)) == seen; ++spin)
            {
                if (spin < kSpin)
                    _mm_pause();
                else
                    generation_.wait(seen, std::memory_order_acquire);
            }
#endif
            seen = g;
            if (stop_.load(std::memory_order_relaxed))
                return;
            (*job_)(t);
            pending_.fetch_sub(1, std::memory_order_release);
        }
    }

    std::vector<std::thread> workers_;
    const std::function<void(int)>* job_ = nullptr;
    alignas(64) std::atomic<uint64_t> generation_{0};  // own cache line: MWAITX wakes on any write to it
    alignas(64) std::atomic<int> pending_{0};
    std::atomic<bool> stop_{false};
};
#endif
