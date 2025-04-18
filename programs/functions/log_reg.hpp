#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../protocols/Protocols.h"
#include "comparisons.hpp"

const int NUM_FEATURES = 5;
template <typename Share>
void compute_gradient(const Additive_Share<DATATYPE, Share> X_Shared[NUM_INPUTS][NUM_FEATURES],
                      const Additive_Share<DATATYPE, Share> y_Shared[NUM_INPUTS],
                      const Additive_Share<DATATYPE, Share> weights[NUM_FEATURES],
                      Additive_Share<DATATYPE, Share> gradient[NUM_FEATURES])
{
    // Compute gradient
    auto z = new Additive_Share<DATATYPE, Share>[NUM_INPUTS] { 0 };
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        for (int j = 0; j < NUM_FEATURES; j++)
        {
            z[i] += X_Shared[i][j].prepare_dot(weights[j]);
        }
        z[i].mask_and_send_dot();
    }
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        z[i].complete_mult();
    }
    auto d = new Additive_Share<DATATYPE, Share>[NUM_INPUTS * 2];
    /* auto d_1 = new Additive_Share<DATATYPE, Share>[NUM_INPUTS]; */
    const int zero_point_five = 1 << (FRACTIONAL - 1);
    for (int i = 0; i < NUM_INPUTS * 2; i += 2)
    {
        d[i] = z[i / 2] - zero_point_five;
        d[i + 1] = Additive_Share<DATATYPE, Share>(0) - d[i];
        /* d_1[i] = z[i] - zero_point_five; */
        /* d_0[i] = Additive_Share<DATATYPE, Share>(0) - d_1[i]; */
    }
    delete[] z;
    auto sigmoid = new Additive_Share<DATATYPE, Share>[NUM_INPUTS];
    // DRELU
    /* DRELU_range_inplace<0,BITLENGTH>(d, NUM_INPUTS*2); */
    auto dp = new Additive_Share<DATATYPE, Share>[NUM_INPUTS * 2];
    pack_additive<0, BITLENGTH>(d, dp, NUM_INPUTS * 2, LTZ<0, BITLENGTH, Share, DATATYPE>);  // LTZ
    delete[] d;

    /* DRELU_range_inplace<0,BITLENGTH>(d_0, NUM_INPUTS); */

    for (int i = 0; i < NUM_INPUTS; i++)
    {
        /* sigmoid[i] = Additive_Share<DATATYPE, Share>(1) - d_1[i].prepare_dot(z[i] + zero_point_five); */
        sigmoid[i] = Additive_Share<DATATYPE, Share>(1) - dp[i * 2 + 1].prepare_dot(z[i] + zero_point_five);
        sigmoid[i].mask_and_send_dot();
    }
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        sigmoid[i].complete_mult();
    }
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        /* sigmoid[i] = ( Additive_Share<DATATYPE, Share>(1) - d_0[i]).prepare_dot(sigmoid[i]); */
        sigmoid[i] = (Additive_Share<DATATYPE, Share>(1) - dp[i * 2]).prepare_dot(sigmoid[i]);
        sigmoid[i].mask_and_send_dot();
    }
    delete[] dp;
    Share::communicate();
    for (int i = 0; i < NUM_INPUTS; i++)
    {
        sigmoid[i].complete_mult();
    }

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for (int j = 0; j < NUM_INPUTS; j++)
        {
            gradient[i] += (sigmoid[j] - y_Shared[j]) * X_Shared[j][i];
        }
    }

    /* delete[] d_0; */
    /* delete[] d_1; */
    delete[] sigmoid;
}

template <typename Share>
void update_weights(const Additive_Share<DATATYPE, Share> weights[NUM_FEATURES],
                    const Additive_Share<DATATYPE, Share> gradient[NUM_FEATURES],
                    Additive_Share<DATATYPE, Share> updated_weights[NUM_FEATURES])
{
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        updated_weights[i] = weights[i] - gradient[i];
    }
}

template <typename Share>
void logistic_regression(const Additive_Share<DATATYPE, Share> X_Shared[NUM_INPUTS][NUM_FEATURES],
                         const Additive_Share<DATATYPE, Share> y_Shared[NUM_INPUTS],
                         Additive_Share<DATATYPE, Share> weights[NUM_FEATURES])
{
    Additive_Share<DATATYPE, Share> gradient[NUM_FEATURES];
    Additive_Share<DATATYPE, Share> updated_weights[NUM_FEATURES];
    for (int i = 0; i < 100; i++)
    {
        compute_gradient(X_Shared, y_Shared, weights, gradient);
        update_weights(weights, gradient, updated_weights);
        weights = updated_weights;
    }
}
