// Thread-parallel accumulation with thread-safe prepare_dot variants.
// Parallelizes the row dimension of the tiled GEMM across ADDITIONAL_GEMM_THREADS+1 threads.
// All threads join before the serial mask_and_send_dot phase.
// Only activated when m*f >= 4096 (enough multiply-accumulates to amortize spawn cost).
//
// This file is included inline inside prepare_GEMM_CPU, guarded by:
//   #if ADDITIONAL_GEMM_THREADS > 0 && FUSE_DOT == 1 && FUSE_CONV_BN_SIM == 0 &&
//       (PUBLIC_WEIGHTS == 1 || CONV_TRIPLES == 1)
// Variables in scope: A, B, C, m, p, f, TILE_SIZE

    {
        constexpr int TILE_SIZE_T = 128;
        const int total_threads = ADDITIONAL_GEMM_THREADS + 1;
        if ((static_cast<long long>(m) * f >= 4096) && (m >= total_threads))
        {
            std::thread workers[ADDITIONAL_GEMM_THREADS];
            auto accum_rows = [&](int row_start, int row_end) {
                for (int i = row_start; i < row_end; i += TILE_SIZE_T)
                {
                    int i_max = std::min(i + TILE_SIZE_T, row_end);
                    for (int j = 0; j < p; j += TILE_SIZE_T)
                    {
                        int j_max = std::min(j + TILE_SIZE_T, p);
                        for (int k = 0; k < f; k += TILE_SIZE_T)
                        {
                            int k_max = std::min(k + TILE_SIZE_T, f);
                            for (int ii = i; ii < i_max; ++ii)
                            {
                                const int iip = ii * p;
                                const int iif = ii * f;
                                for (int jj = j; jj < j_max; ++jj)
                                {
                                    const int jjf = jj * f;
                                    auto temp = T(0);
                                    for (int kk = k; kk < k_max; ++kk)
                                    {
#if PUBLIC_WEIGHTS == 0
#if A_KNOWN == 1
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
                                        temp += A[iif + kk].prepare_dot_ex_lxly_a_known_pre(B[jjf + kk]);
#else
                                        temp += A[iif + kk].prepare_dot_ex_lxly_a_known(B[jjf + kk]);
#endif
#else
                                        temp += A[iif + kk].prepare_dot_ex_lxly(B[jjf + kk]);
#endif
#else
                                        temp += B[jjf + kk].mult_public(A[iif + kk]);
#endif
                                    }
                                    C[iip + jj] += temp;
                                }
                            }
                        }
                    }
                }
            };

            const int rows_per_thread = m / total_threads;
            int row_start = 0;
            for (int t = 0; t < ADDITIONAL_GEMM_THREADS; ++t)
            {
                int row_end = row_start + rows_per_thread;
                workers[t] = std::thread(accum_rows, row_start, row_end);
                row_start = row_end;
            }
            accum_rows(row_start, m); // main thread handles remainder

            for (int t = 0; t < ADDITIONAL_GEMM_THREADS; ++t)
                workers[t].join();

            // mask_and_send_dot must be serial (accesses shared protocol send buffer)
            for (int i = 0; i < m * p; ++i)
            {
#if PUBLIC_WEIGHTS == 0
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
#if INTERLEAVE_COMM == 1
#if A2B_ROUND_OPT_SIM == 1
                C[i].mask_and_send_dot(); // Simulate second send for XOR Share
#endif
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
                C[i].mask_and_send_dot_a_known_pre_with_triple(i);
#else
#if A_KNOWN == 1
                C[i].mask_and_send_dot_without_trunc_with_triple(i);
#else
                C[i].mask_and_send_dot();
#endif
#endif
#else // INTERLEAVE_COMM == 0
#if A_KNOWN == 1
                C[i].mask_and_send_dot_without_trunc_with_triple();
#else
                C[i].mask_and_send_dot();
#endif
#endif
#else // no TRUNC_DELAYED, no TRUNC_APPROACH
#if INTERLEAVE_COMM == 1
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
                C[i].mask_and_send_dot_a_known_pre_with_triple(i);
#else
#if A_KNOWN == 1
                C[i].mask_and_send_dot_with_triple(i);
#else
                C[i].mask_and_send_dot();
#endif
#endif
#else // INTERLEAVE_COMM == 0
#if A_KNOWN == 1
#if MODELWEIGHTS_KNOWN_DURING_PREPROCESSING == 1
                C[i].mask_and_send_dot_a_known_pre_with_triple();
#else
                C[i].mask_and_send_dot_with_triple();
#endif
#else
                C[i].mask_and_send_dot();
#endif
#endif
#endif
#else // PUBLIC_WEIGHTS == 1
#if TRUNC_DELAYED == 1 || TRUNC_APPROACH > 0
                // no truncation needed
#else
                C[i] = C[i].prepare_mult_public_fixed(1);
#endif
#endif
            }
#if INTERLEAVE_COMM == 1 && PROTOCOL == 4 && CONV_TRIPLES == 1 && A_KNOWN == 1 && PUBLIC_WEIGHTS == 0
            if(current_phase == PHASE_LIVE)
                preprocessed_outputs_arithmetic_index[0] += m * p;
#endif
            return;
        }
    }
