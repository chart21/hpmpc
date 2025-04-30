#pragma once
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/XOR_Share.hpp"
#include "../../datatypes/k_bitset.hpp"
#include "../../datatypes/k_sint.hpp"
#include "../../protocols/Protocols.h"
#include "share_conversion.hpp"

template <int bm, int bk, typename Datatype, typename Share>
void max_min_msb_range(sint_t<Additive_Share<Datatype, Share>>* val,
                       XOR_Share<Datatype, Share>* msb,
                       const int og_len,
                       int m,
                       int batch_size,
                       bool want_max)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;

    int len = (m + 1) / 2;
    sint* max_val = new sint[batch_size * len];
    /* for(int b = 0; b < batch_size*len; b++) */
    /*     max_val[b] = UINT_TYPE(0); */
    /* sint* min_val = new sint[batch_size*len]{UINT_TYPE(0)}; */
    int offset = m % 2;  // if m is odd, offset is 1
    int counter = 0;
    for (int b = 0; b < batch_size; b++)
    {
        counter = 0;
        for (int j = 1; j < m; j += 2)
        {
            if (want_max)
                max_val[counter + b * len] = val[j + b * og_len] - val[j - 1 + b * og_len];
            else
                max_val[counter + b * len] = val[j - 1 + b * og_len] - val[j + b * og_len];
            counter++;
        }
        if (offset == 1)
            max_val[counter + b * len] = UINT_TYPE(0);  // last uneven element is always pairwise max, override later
    }

    get_msb_range<bm, bk>(max_val, msb, len * batch_size);

    delete[] max_val;

    // get arithmetic version of msb to update values
    auto max_idx = new sint[batch_size * len];
    bit2A_range(msb, batch_size * len, max_idx);
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < counter; i++)
        {
            /* #if PARTY ==2 */
            /*             std::cout << "max idx: " << max_idx[i].get_p1() << std::endl; */
            /* #endif */
            max_idx[i + b * len] =
                max_idx[i + b * len].prepare_mult((val[2 * i + b * og_len] - val[2 * i + 1 + b * og_len]));
        }
    }
    Share::communicate();
    for (int b = 0; b < batch_size; b++)
    {
        for (int i = 0; i < counter; i++)
        {
            max_idx[b * len + i].complete_mult_without_trunc();
            max_idx[b * len + i] = max_idx[b * len + i] + val[b * og_len + 2 * i + 1];
            val[b * og_len + i] = max_idx[b * len + i];
            /* #if PARTY == 2 */
            /* std::cout << "updated val: " << val[i].get_p1() << std::endl; */
            /* #endif */
        }
    }
    if (offset == 1)
    {
        for (int b = 0; b < batch_size; b++)
        {
            val[b * og_len + counter] = val[b * og_len + m - 1];  // last uneven element is always pairwise max
            msb[b * len + counter] = SET_ALL_ONE();
        }
    }
    delete[] max_idx;
}

template <int bm, int bk, typename Datatype, typename Share>
void max_min(const sint_t<Additive_Share<Datatype, Share>>* begin,
             int len,
             sint_t<Additive_Share<Datatype, Share>>* output,
             int batch_size,
             bool want_max)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    int og_len = len;
    int m = len;

    sint* val = new sint[batch_size * og_len];
    std::copy(begin, begin + batch_size * og_len, val);
    if (len == 1)
    {
        for (int b = 0; b < batch_size; b++)
            output[b] = val[0];
        return;
    }

    int log2m = std::ceil(std::log2(m));
    for (int i = 0; i < log2m; i++)
    {
        int counter = m / 2;  //
        int offset = m % 2;   // if m is odd, offset is 1
        int q = (m + 1) / 2;
        S* msb = new S[batch_size * q];
        /* for(int b = 0; b < og_len*batch_size; b++) */
        /*     val[b] = UINT_TYPE(0); */

        max_min_msb_range<bm, bk>(val, msb, og_len, m, batch_size, want_max);  // get msb and max of 0 -> counter

        delete[] msb;
        m = counter + offset;
    }
    for (int b = 0; b < batch_size; b++)
    {
        output[b] = val[b * og_len];
    }
    delete[] val;
}

template <int bm, int bk, typename Datatype, typename Share>
void argmax_argmin(const sint_t<Additive_Share<Datatype, Share>>* begin,
                   int len,
                   XOR_Share<Datatype, Share>* output,
                   int batch_size,
                   bool want_max)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    int m = len;
    const int og_len = m;
    if (m == 1)
    {
        for (int b = 0; b < batch_size; b++)
            output[b] = SET_ALL_ONE();
        return;
    }

    auto val = new sint[batch_size * og_len];
    std::copy(begin, begin + og_len * batch_size, val);
    /* #if PARTY == 2 */
    /* for(int i = 0; i < m; i++) */
    /* std::cout << "val: " << val[i].get_p1() << std::endl; */
    /* #endif */

    int log2m = std::ceil(std::log2(m));
    for (int i = 0; i < log2m; i++)
    {
        int counter = m / 2;  //
        int offset = m % 2;   // if m is odd, offset is 1
        int q = (m + 1) / 2;
        S* msb = new S[batch_size * q];
        max_min_msb_range<bm, bk>(val, msb, og_len, m, batch_size, want_max);  // get msb and max of 0 -> counter

        // update args
        if (i == 0)  // first round
        {
            for (int b = 0; b < batch_size; b++)
            {
                for (int j = 1; j < m; j += 2)
                {
                    output[b * og_len + j - 1] = msb[b * q + j / 2];
                    output[b * og_len + j] = !msb[b * q + j / 2];
                }
                if (offset == 1)
                {
                    output[b * og_len + m - 1] = SET_ALL_ONE();  // single element is always max
                }
            }
        }
        else
        {
            int jump = 1 << (i + 1);  // i = 1 -> jump = 4, 4 values are being compared in total
            for (int b = 0; b < batch_size; b++)
            {
                for (int j = 0; j < counter; j++)
                {
                    for (int k = 0; k < jump && j * jump + k < og_len; k++)
                    {
                        if (k < jump / 2)
                        {
                            output[b * og_len + j * jump + k] = output[b * og_len + j * jump + k] & msb[b * q + j];
                        }
                        else
                        {
                            output[b * og_len + j * jump + k] = output[b * og_len + j * jump + k] & !msb[b * q + j];
                        }
                    }
                }
            }
            Share::communicate();
            for (int b = 0; b < batch_size; b++)
            {
                for (int j = 0; j < counter; j++)
                {
                    for (int k = 0; k < jump && j * jump + k < og_len; k++)
                    {
                        output[b * og_len + j * jump + k].complete_and();
                    }
                }
            }
        }
        delete[] msb;
        m = counter + offset;
    }
    delete[] val;
}

template <int bm, int bk, typename Datatype, typename Share>
void argmax_argmin_sint(const sint_t<Additive_Share<Datatype, Share>>* begin,
                        int len,
                        sint_t<Additive_Share<Datatype, Share>>* output,
                        int batch_size,
                        bool want_max)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    auto tmp_output = new S[batch_size * len];
    argmax_argmin<bm, bk>(begin, len, tmp_output, batch_size, want_max);
    bit2A_range(tmp_output, len * batch_size, output);  // TODO: remove this overhead somehow?
    delete[] tmp_output;
}

template <int bm, int bk, typename Datatype, typename Share>
void max_min_sint(const sint_t<Additive_Share<Datatype, Share>>* begin,
                  int len,
                  sint_t<Additive_Share<Datatype, Share>>* output,
                  int batch_size,
                  bool want_max)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    max_min<bm, bk>(begin, len, output, batch_size, want_max);
}
template <int bm, int bk, typename Datatype, typename Share>
void max_min_sint(const Additive_Share<Datatype, Share>* begin,
                  int len,
                  Additive_Share<Datatype, Share>* output,
                  int batch_size,
                  bool want_max)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    sint* tmp_output = new sint[((batch_size - 1) / BITLENGTH + 1)];
    sint* tmp_begin = new sint[len * ((batch_size - 1) / BITLENGTH + 1)];
    int k = 0;
    int c = 0;
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < batch_size; j++)
        {
            tmp_begin[c * len + i][k++] = begin[j * len + i];  // load all shares at position i into one sint
            if (k == BITLENGTH)
            {
                k = 0;
                c++;
            }
        }
        k = 0;
        c = 0;
    }

    max_min_sint<bm, bk>(tmp_begin,
                         len,
                         tmp_output,
                         (batch_size - 1) / BITLENGTH + 1,
                         want_max);  // TODO: Warning because of massive overhead
    for (int j = 0; j < batch_size; j++)
    {
        output[j] = tmp_output[c][k++];  // load result into output
        /* output[j] = tmp_begin[c][k++]; //load result into output */
        if (k == BITLENGTH)
        {
            k = 0;
            c++;
        }
    }
    delete[] tmp_output;
    delete[] tmp_begin;
}
template <int bm, int bk, typename Datatype, typename Share>
void argmax_argmin_sint(const Additive_Share<Datatype, Share>* begin,
                        int len,
                        Additive_Share<Datatype, Share>* output,
                        int batch_size,
                        bool want_max)
{
    using S = XOR_Share<Datatype, Share>;
    using A = Additive_Share<Datatype, Share>;
    using sint = sint_t<A>;
    sint* tmp_output = new sint[len * ((batch_size - 1) / BITLENGTH + 1)];
    sint* tmp_begin = new sint[len * ((batch_size - 1) / BITLENGTH + 1)];
    int k = 0;
    int c = 0;
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < batch_size; j++)
        {
            tmp_begin[c * len + i][k++] = begin[j * len + i];  // load all shares at position i into one sint
            if (k == BITLENGTH)
            {
                k = 0;
                c++;
            }
        }
        k = 0;
        c = 0;
    }

    argmax_argmin_sint<bm, bk>(tmp_begin,
                               len,
                               tmp_output,
                               (batch_size - 1) / BITLENGTH + 1,
                               want_max);  // TODO: Warning because of massive overhead
    for (int i = 0; i < len; i++)
    {
        for (int j = 0; j < batch_size; j++)
        {
            output[j * len + i] = tmp_output[c * len + i][k++];  // load result into output
            if (k == BITLENGTH)
            {
                k = 0;
                c++;
            }
        }
        k = 0;
        c = 0;
    }

    delete[] tmp_output;
    delete[] tmp_begin;
}
