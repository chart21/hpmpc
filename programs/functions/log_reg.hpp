#include <stdint.h>
#include "../protocols/XOR_Share.hpp"
#include "../protocols/Additive_Share.hpp"
#include "../../protocols/Protocols.h"

const int NUM_FEATURES = 10;
template<typename Share>
void compute_gradient(const Additive_Share<DATATYPE, Share> X_Shared[NUM_INPUTS][NUM_FEATURES], const Additive_Share<DATATYPE, Share> y_Shared[NUM_INPUTS], const Additive_Share<DATATYPE, Share> weights[NUM_FEATURES], Additive_Share<DATATYPE, Share> gradient[NUM_FEATURES])
{
    // Compute gradient
    Additive_Share<DATATYPE, Share> gradient[NUM_FEATURES];
    auto z = new Additive_Share<DATATYPE, Share>[NUM_INPUTS]{0};
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        for(int j = 0; j < NUM_FEATURES; j++)
        {
            z[i] += X_Shared[i][j].prepare_dot(weights[j]);
        }
        z[i].mask_and_send_dot();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        z[i].complete_mult();
    }
    auto d_0 = new Additive_Share<DATATYPE, Share>[NUM_INPUTS];
    auto d_1 = new Additive_Share<DATATYPE, Share>[NUM_INPUTS];
    const int zero_point_five = 1 << (FRACTIONAL -1);
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        d_1[i] = z[i] - zero_point_five; 
        d_0[i] = Additive_Share<DATATYPE, Share>(0) - d_1[i];
    }
    auto sigmoid = new Additive_Share<DATATYPE, Share>[NUM_INPUTS];
//DRELU
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        sigmoid[i] = 1 - d_1[i].prepare_dot(z[i] + zero_point_five);
        sigmoid[i].mask_and_send_dot();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        sigmoid[i].complete_mult();
    }
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        sigmoid[i] = ( 1 - d_0[i]).prepare_dot(sigmoid[i]);
        sigmoid[i].mask_and_send_dot();
    }
    Share::communicate();
    for(int i = 0; i < NUM_INPUTS; i++)
    {
        sigmoid[i].complete_mult();
    }

    for (int i = 0; i < NUM_FEATURES; i++)
    {
        for(int j = 0; j < NUM_INPUTS; j++)
        {
            gradient[i] += (sigmoid[j] - y_Shared[j]) * X_Shared[j][i];
        }
    }

    delete[] z;
    delete[] d_0;
    delete[] d_1;
    delete[] sigmoid;
}

template<typename Share>
void update_weights(const Additive_Share<DATATYPE, Share> weights[NUM_FEATURES], const Additive_Share<DATATYPE, Share> gradient[NUM_FEATURES], Additive_Share<DATATYPE, Share> updated_weights[NUM_FEATURES])
{
    for (int i = 0; i < NUM_FEATURES; i++)
    {
        updated_weights[i] = weights[i] - gradient[i];
    }
}

template<typename Share>
void logistic_regression(const Additive_Share<DATATYPE, Share> X_Shared[NUM_INPUTS][NUM_FEATURES], const Additive_Share<DATATYPE, Share> y_Shared[NUM_INPUTS], Additive_Share<DATATYPE, Share> weights[NUM_FEATURES])
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
