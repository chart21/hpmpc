#pragma once

#include <vector>

using std::vector;

namespace IR {

/**
 * Taken from <https://github.com/data61/MP-SPDZ/blob/master/Processor/Conv2dTuple.h> but adapted
 * for HPMPC
 */
template <class sint>
class Conv2d {
  public:
    Conv2d(const vector<int>& args, int start);

    void pre(vector<sint>& S);
    void post(vector<sint>& S) const;

  private:
    int output_h, output_w;
    int inputs_h, inputs_w;
    int weights_h, weights_w;
    int stride_h, stride_w;
    int n_channels_in;
    int padding_h;
    int padding_w;
    int batch_size;
    size_t r0;
    size_t r1;
    int r2;
    vector<vector<vector<int>>> lengths;
    int filter_stride_h = 1;
    int filter_stride_w = 1;
};

template <class sint>
Conv2d<sint>::Conv2d(const vector<int>& arguments, int start) {
    assert(arguments.size() >= start + 15ul);
    auto args = arguments.data() + start + 3;
    output_h = args[0], output_w = args[1];
    inputs_h = args[2], inputs_w = args[3];
    weights_h = args[4], weights_w = args[5];
    stride_h = args[6], stride_w = args[7];
    n_channels_in = args[8];
    padding_h = args[9];
    padding_w = args[10];
    batch_size = args[11];
    r0 = arguments[start];
    r1 = arguments[start + 1];
    r2 = arguments[start + 2];
    lengths.resize(batch_size, vector<vector<int>>(output_h, vector<int>(output_w)));
    filter_stride_h = 1;
    filter_stride_w = 1;
    if (stride_h < 0) {
        filter_stride_h = -stride_h;
        stride_h = 1;
    }
    if (stride_w < 0) {
        filter_stride_w = -stride_w;
        stride_w = 1;
    }
}

template <class sint>
void Conv2d<sint>::pre(vector<sint>& S) {
    for (int i_batch = 0; i_batch < batch_size; i_batch++) {
        size_t base = r1 + i_batch * inputs_w * inputs_h * n_channels_in;
        sint* input_base = &S[base];

        size_t output = r0 + i_batch * output_h * output_w;

        if (output + (output_h - 1) * output_w + (output_w - 1) >= S.size()) { // greatest address
            S.resize(output + (output_h - 1) * output_w + (output_w - 1));
        }

        for (int out_y = 0; out_y < output_h; out_y++)
            for (int out_x = 0; out_x < output_w; out_x++) {
                int in_x_origin = (out_x * stride_w) - padding_w;
                int in_y_origin = (out_y * stride_h) - padding_h;

                sint* output_base = &S[output];

                for (int filter_y = 0; filter_y < weights_h; filter_y++) {
                    int in_y = in_y_origin + filter_y * filter_stride_h;
                    if ((0 <= in_y) and (in_y < inputs_h))
                        for (int filter_x = 0; filter_x < weights_w; filter_x++) {
                            int in_x = in_x_origin + filter_x * filter_stride_w;
                            if ((0 <= in_x) and (in_x < inputs_w)) {
                                sint* pixel_base =
                                    &input_base[(in_y * inputs_w + in_x) * n_channels_in];
                                sint* weight_base =
                                    &S[r2 + (filter_y * weights_w + filter_x) * n_channels_in];
                                for (int in_c = 0; in_c < n_channels_in; in_c++)
                                    output_base[out_y * output_w + out_x] =
                                        pixel_base[in_c].prepare_dot(weight_base[in_c]);
                                lengths[i_batch][out_y][out_x] += n_channels_in;
                            }
                        }
                }
                output_base[out_y * output_w + out_x].mask_and_send_dot_without_trunc();
            }
    }
}

template <class sint>
void Conv2d<sint>::post(vector<sint>& S) const {
    for (int i_batch = 0; i_batch < batch_size; i_batch++) {
        size_t base = r0 + i_batch * output_h * output_w;
        sint* output_base = &S[base];
        for (int out_y = 0; out_y < output_h; out_y++)
            for (int out_x = 0; out_x < output_w; out_x++) {
                output_base[out_y * output_w + out_x].complete_mult_without_trunc();
            }
    }
}

} // namespace IR