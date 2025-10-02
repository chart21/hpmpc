#pragma once
#include "include/pch.h"
#include "arch/DATATYPE.h"
#define FAKE_TRIPLES 0

// const ConvolutionParameter param(batchSize, inh, inw, din, dout, wh, ww, padding, stride, dilation);
struct ConvolutionParameter
{
    int batchSize;
    int inh;
    int inw;
    int din;
    int dout;
    int wh;
    int ww;
    int padding;
    int stride;
    int out_h;
    int out_w;
    int dilation;
    int x_size_per_batch;
    int w_size_per_batch;
    int y_size_per_batch;


    ConvolutionParameter(int batchSize,
                         int inh,
                         int inw,
                         int din,
                         int dout,
                         int wh,
                         int ww,
                         int padding,
                         int stride,
                         int oh,
                         int ow,
                         int dilation)
        : batchSize(batchSize),
          inh(inh),
          inw(inw),
          din(din),
          dout(dout),
          wh(wh),
          ww(ww),
          padding(padding),
          stride(stride),
          out_h(oh),
          out_w(ow),
          dilation(dilation)
    {
        x_size_per_batch = inh * inw * din;
        w_size_per_batch = wh * ww * dout;
        y_size_per_batch = out_h * out_w * dout;
    }
};

struct BatchNorm2DParameter
{
		int batchSize;
		int ch;
		int h;
		int w;
		int hw;
    int x_size_per_batch;
    int w_size_per_batch;
    int y_size_per_batch;
    BatchNorm2DParameter(int batchSize,
                         int ch,
                         int h,
                         int w)
        : batchSize(batchSize),
          ch(ch),
          h(h),
          w(w)
    {
        hw = h * w;
        x_size_per_batch = ch * h * w;
        w_size_per_batch = ch;
        y_size_per_batch = x_size_per_batch;
    }
};



struct FullyConnectedParameter
{
    int batchSize;
    int in_feat;
    int out_feat;
    int x_size_per_batch;
    int w_size_per_batch;
    int y_size_per_batch;
    FullyConnectedParameter(int batch,
                            int in_feat,
                            int out_feat)
        : batchSize(batch),
          in_feat(in_feat),
          out_feat(out_feat)
    {
        x_size_per_batch = in_feat;
        w_size_per_batch = in_feat * out_feat;
        y_size_per_batch = out_feat;
    }
};

#if FAKE_TRIPLES == 0
#define generateArithmeticTriples generateArithmeticDummyTriples
#define generateBooleanTriples generateBooleanDummyTriples
#define generateArithmeticAB2Triples generateArithmeticAB2DummyTriples
#define generateBooleanAB2Triples generateBooleanAB2DummyTriples
#define generateConvTriples generateLayerDummyTriples
#define generateFCTriples generateLayerDummyTriples
#define generateBatchNorm2DTriples generateLayerDummyTriples
#define generateBooleanAdditionTriples generateBooleanAdditionDummyTriples
#define generateMultiplexerTriples generateMultiplexerDummyTriples
#define generateCOTTriples generateCOTDummyTriples

#include <core/hpmpc_interface.hpp>


// Input: arrays of arithmetic triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateArithmeticDummyTriples(type a[],
                                    type b[],
                                    type c[],
                                    int bitlength,
                                    uint64_t num_triples,
                                    std::string ip,
                                    int port)
{
    std::cout << "ARITH AB\n";

    if (num_triples == 0)
        return;

    port += 10;

    //convert SIMD variables to regular uints
    const int vectorization_factor = DATTYPE / bitlength;

    UINT_TYPE* uint_a = new UINT_TYPE[num_triples];
    unorthogonalize_arithmetic(a, uint_a, num_triples / (DATTYPE / bitlength)); 
    UINT_TYPE* uint_b = new UINT_TYPE[num_triples];
    unorthogonalize_arithmetic(b, uint_b, num_triples / (DATTYPE / bitlength));
    UINT_TYPE* uint_c = new UINT_TYPE[num_triples];

    Iface::generateArithTriplesCheetah(uint_a, uint_b, uint_c, 1, num_triples, ip, port, PARTY + 1, 1);

    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
    DELETEARR(uint_a);
    DELETEARR(uint_b);
    DELETEARR(uint_c);
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateBooleanDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port,
                                 int cheetah_ot_type = CHEETAH_BOOL_OT_TYPE)
{
    std::cout << "ARITH AB\n";

    if(num_triples == 0) return;

    port += 10;

    //reinterpret SIMD bitstream as uint8 bitstream
    uint8_t* uint_a = (uint8_t*) a;
    uint8_t* uint_b = (uint8_t*) b;
    uint8_t* uint_c = (uint8_t*) c;

    Iface::generateBoolTriplesCheetah(uint_a, uint_b, uint_c, bitlength, num_triples / 8, ip, port, PARTY + 1, 1);
}



// Input: arrays of arithmetic triple shares [a], [b], [c] with size num_triples and ring size of bitlength
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateArithmeticAB2DummyTriples(type a[],
                                    type b[],
                                    type c[],
                                    int bitlength,
                                    uint64_t num_triples,
                                    std::string ip,
                                    int port)
{
    std::cout << "ARITH AB2\n";

    if(num_triples == 0) return;

    port += 10;

    //convert SIMD variables to regular uints
    const int vectorization_factor = DATTYPE / bitlength; 

    UINT_TYPE* uint_a;
    UINT_TYPE* uint_b;
    UINT_TYPE* uint_c;

    if(vectorization_factor == 1) // No need to unvectorize
    {
        uint_a = (UINT_TYPE*) a;
        uint_b = (UINT_TYPE*) b;
        uint_c = (UINT_TYPE*) c;
#if PARTY == 1
        uint_a = nullptr;
#else
        uint_b = nullptr;
#endif
    } else  {
#if PARTY == 0
    uint_a = NEW(UINT_TYPE[num_triples]);
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor)); 
#else
    uint_a = nullptr;
#endif
#if PARTY == 1
        uint_b = NEW(UINT_TYPE[num_triples]);
        unorthogonalize_arithmetic(b, uint_b, num_triples / (vectorization_factor));
#else
        uint_b = nullptr;
#endif // P_0 doesn't need b for AB2
        uint_c = NEW(UINT_TYPE[num_triples]);
    }

    Iface::generateArithTriplesCheetah(uint_a, uint_b, uint_c, 1, num_triples, ip, port, PARTY + 1, 1, Utils::PROTO::AB2);
    
    // convert UINT triple to SIMD type
    if (vectorization_factor != 1) {
        orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
#if PARTY == 0
        DELETEARR(uint_a);
#endif
#if PARTY == 1
        DELETEARR(uint_b);
#endif
        DELETEARR(uint_c);
    }
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with triples
template <typename type>
void generateBooleanAB2DummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port,
                                 int cheetah_ot_type = CHEETAH_BOOL_OT_TYPE)
{
    std::cout << "BOOL AB2\n";

    if (num_triples == 0) return;

    port += 10;
    
    //reinterpret SIMD bitstream as uint8 bitstream
#if PARTY == 0
    uint8_t* uint_a = (uint8_t*) a;
#else
    std::vector<uint8_t> zerosa(num_triples/8, 0);
    uint8_t* uint_a = zerosa.data();
#endif
#if PARTY == 1
    uint8_t* uint_b = (uint8_t*) b;
#else
    std::vector<uint8_t> zeros(num_triples/8, 0);
    uint8_t* uint_b = zeros.data();
#endif // P_0 doesn't need b for AB2
    uint8_t* uint_c = (uint8_t*) c;
    
    Iface::generateBoolTriplesCheetah(uint_a, uint_b, uint_c, bitlength,
            num_triples / 8, ip, port, PARTY + 1, 1);
}

// Input: array of boolean triple shares [a], [b], [c] with size num_triples
// Input: ip and port of the other party to connect to
// Output: [c] will be filled with shares of a + b
template <typename type>
void generateBooleanAdditionDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port,
                                 int cheetah_ot_type = CHEETAH_BOOL_OT_TYPE)
{
    constexpr int num_bits_per_input = REDUCED_BITLENGTH_k - REDUCED_BITLENGTH_m;
    if(num_triples == 0) return;
    if(num_bits_per_input <= 0) return;
    //reinterpret SIMD bitstream as uint8 bitstream
#if PARTY == 0
    auto av = reinterpret_cast<type (*)[num_bits_per_input]> (a); 
#else
    auto bv = reinterpret_cast<type (*)[num_bits_per_input]> (b);
#endif
    auto cv = reinterpret_cast<type (*)[num_bits_per_input]> (c);
    num_triples = num_triples / (num_bits_per_input * DATTYPE);
    type* carry_last = new type[num_triples];
    type* carry_this = new type[num_triples];
    type* ot_a = new type[num_triples];
    type* ot_b = new type[num_triples]; 
    int r = num_bits_per_input;
    const int k = num_bits_per_input;
    while(r > 0)
    {
        r--;
        switch(r)
        {
            case k - 1:
                for (uint64_t i = 0; i < num_triples ; i++)
                {
#if PARTY == 0
                    cv[i][r] = av[i][r];
                    ot_a[i] = av[i][r];
                    ot_b[i] = SET_ALL_ZERO();
#else
                    cv[i][r] = bv[i][r];
                    ot_b[i] = bv[i][r];
                    ot_a[i] = SET_ALL_ZERO();
#endif
                }
                generateBooleanAB2Triples(ot_a, ot_b, carry_last, bitlength, num_triples * DATTYPE, ip, port, cheetah_ot_type);
                break;
            case k - 2:
                for (uint64_t i = 0; i < num_triples ; i++)
                {
                    //update
#if PARTY == 0
                    cv[i][r] = av[i][r] ^ carry_last[i];
#else
                    cv[i][r] = bv[i][r] ^ carry_last[i];
#endif
                    //prepare
#if PARTY == 0 
                    ot_a[i] = av[i][r] ^ carry_last[i];
                    ot_b[i] = carry_last[i];
#else
                    ot_b[i] = bv[i][r] ^ carry_last[i];
                    ot_a[i] = carry_last[i];
#endif
                }
                generateBooleanTriples(ot_a, ot_b, carry_this, bitlength, num_triples * DATTYPE, ip, port, cheetah_ot_type);
                break;
            default:
                // complete_carry
                for (uint64_t i = 0; i < num_triples ; i++)
                {
                    carry_this[i] = carry_this[i] ^ carry_last[i];
                    carry_last[i] = carry_this[i];
                }
                // update result
                for (uint64_t i = 0; i < num_triples ; i++)
#if PARTY == 0
                        cv[i][r] = av[i][r] ^ carry_last[i];
#else
                        cv[i][r] = bv[i][r] ^ carry_last[i];
#endif

                // prepare_carry
                for (uint64_t i = 0; i < num_triples ; i++)
                {
#if PARTY == 0
                    ot_a[i] = av[i][r] ^ carry_last[i];
                    ot_b[i] = carry_last[i];
#else
                    ot_b[i] = bv[i][r] ^ carry_last[i];
                    ot_a[i] = carry_last[i];
#endif
                }
                generateBooleanTriples(ot_a, ot_b, carry_this, bitlength, num_triples * DATTYPE, ip, port, cheetah_ot_type);
                break;
            case 0:
                // complete_carry
                for (uint64_t i = 0; i < num_triples ; i++)
                {
                    carry_this[i] = carry_this[i] ^ carry_last[i];
                    carry_last[i] = carry_this[i];
                }
                // update result
                for (uint64_t i = 0; i < num_triples ; i++)
#if PARTY == 0
                    cv[i][r] = av[i][r] ^ carry_last[i];
#else
                    cv[i][r] = bv[i][r] ^ carry_last[i];
#endif
                delete[] carry_last;
                delete[] carry_this;
                delete[] ot_a;
                delete[] ot_b;
                return;
                    
        }
    }
}

// Input: For Party 0: Array of messages m0 stored in a[]
// Input: For Party 1: Array of selection bits stored in a[]
// Output: [c] will be filled with shares of mb, i.e. P0 holds -r and P1 holds m0/m1 + r
template <typename type>
void generateCOTDummyTriples(type a[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port)
{
    if(num_triples == 0) return;
    const int vectorization_factor = DATTYPE / bitlength; 

    if(vectorization_factor == 1) // No need to unvectorize
        {
#if PARTY == 0
            UINT_TYPE* uint_a = (UINT_TYPE*) a;
#else
            uint8_t* uint_a = (uint8_t*) a;
#endif
            UINT_TYPE* uint_c = (UINT_TYPE*) c;
            return;
        }
    
#if PARTY == 0
    UINT_TYPE* uint_a = NEW(UINT_TYPE[num_triples]); //stores m0
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor)); 
    
#else  //PARTY 1
    uint8_t* uint_a = (uint8_t*) a; //stores choice bit (packed)
#endif


    UINT_TYPE* uint_c = NEW(UINT_TYPE[num_triples]);
    for (uint64_t i = 0; i < num_triples; i++)
    {
       uint_c[i] = 0; // dummy assignment, replace with actual triple generation
    }
    
    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
#if PARTY == 0
    DELETEARR(uint_a);
#endif
    DELETEARR(uint_c);
}

// Input: Arithmetic shares stored in a[] and shared bits stored in b[]
// Output: [c] will be filled with shares ab
template <typename type>
void generateMultiplexerDummyTriples(type a[],
                                 type b[],
                                 type c[],
                                 int bitlength,
                                 uint64_t num_triples,
                                 std::string ip,
                                 int port)
{
    if(num_triples == 0) return;
    const int vectorization_factor = DATTYPE / bitlength; 
    
    uint8_t* uint_b = (uint8_t*) b; //stores choice bit (packed)

    if(vectorization_factor == 1) // No need to unvectorize
        {
            UINT_TYPE* uint_a = (UINT_TYPE*) a;
            UINT_TYPE* uint_c = (UINT_TYPE*) c;
            return;
        }
    
    UINT_TYPE* uint_a = NEW(UINT_TYPE[num_triples]); //stores arithmetic share
    unorthogonalize_arithmetic(a, uint_a, num_triples / (vectorization_factor)); 
    
    UINT_TYPE* uint_c = NEW(UINT_TYPE[num_triples]);
    for (uint64_t i = 0; i < num_triples; i++)
    {
       uint_c[i] = uint_a[i]; // dummy assignment, replace with actual triple generation
    }
    
    // convert UINT triple to SIMD type
    orthogonalize_arithmetic(uint_c, c, num_triples / (vectorization_factor));
    DELETEARR(uint_a);
    DELETEARR(uint_c);
}





//Input: arrays of layer triple shares [a], [b] with sizes predefined by convolution/Fc/Batchnorm params
//Output: Contigious array of clayer triple shares [c] storing the output
template <typename type, typename LayerParams>
void generateLayerDummyTriples(type** a,
                              type** b,
                              type c[],
                              int bitlength,
                              std::vector<LayerParams> params,
                              std::string ip,
                              int port)
{

    port += 10;
    const int factor = DATTYPE/BITLENGTH;
    if(factor == 1) // No need to unvectorize
        {
            UINT_TYPE** uint_w = (UINT_TYPE**) a;
            UINT_TYPE** uint_x = (UINT_TYPE**) b;
            UINT_TYPE* uint_y = (UINT_TYPE*) c;
            uint64_t y_index_counter = 0;
        for(int n = 0; n < params.size(); n++)
        {
            auto p = params[n];

            if constexpr (std::is_same_v<LayerParams, ConvolutionParameter>) {
                if (p.dilation != 1) {
                    std::cerr << "DILATION != 1 is not supported\n";
                }
                Utils::ConvParm conv{
                    .ic = (size_t)p.din,
                    .iw = (size_t)p.inw,
                    .ih = (size_t)p.inh,
                    .fc = (size_t)p.din,
                    .fw = (size_t)p.ww,
                    .fh = (size_t)p.wh,
                    .n_filters = (size_t)p.dout,
                    .stride = (size_t)p.stride,
                    .padding = (size_t)p.padding,
                };
                std::cout << "CONVOLUTION" << "\n";
                std::cout << conv.ih << " x ";
                std::cout << conv.iw << " x ";
                std::cout << conv.ic << ", ";
                std::cout << p.x_size_per_batch << ", ";
                std::cout << conv.fh << " x ";
                std::cout << conv.fw << " x ";
                std::cout << conv.fc << " x ";
                std::cout << conv.n_filters << ", ";
                std::cout << p.w_size_per_batch << "\n";

                Iface::generateConvTriplesCheetahWrapper(uint_w[n], uint_x[n], uint_y + y_index_counter,
                        conv, p.batchSize,
                        ip, port, PARTY + 1, 1,
                        AB2_TRIPLES == 0 ? Utils::PROTO::AB2 : Utils::PROTO::AB2);
            } else if constexpr (std::is_same_v<LayerParams, FullyConnectedParameter>) {
                std::cout << params.size() << "FullyConnected\n";
                std::cout << p.in_feat << ", ";
                std::cout << p.out_feat << ", ";
                std::cout << "x_size: " << p.x_size_per_batch << ", "; // smol
                std::cout << "w_size: " << p.w_size_per_batch << "\n"; // big

                Iface::generateFCTriplesCheetah(uint_w[n], uint_x[n], uint_y + y_index_counter,
                        p.batchSize, p.in_feat, p.out_feat,
                        PARTY + 1, ip, port, 1,
                        Utils::PROTO::AB2);

                if (p.out_feat == 10)
                    for (int i = 0; i < p.out_feat; ++i)
                        std::cout << PARTY << " " << *(uint_y + y_index_counter + i) << "\n";
            } else if constexpr (std::is_same_v<LayerParams, BatchNorm2DParameter>) {
                std::cout << params.size() << "BatchNorm\n";
            } else {
                std::cout << params.size() << "UNKNOWN\n";
            }
           // Layer(uint_w[i],uint_x[i],uint_y + y_index_counter, p); // calculate layer operation
            y_index_counter += p.y_size_per_batch * p.batchSize;
             
        } 
            return;
        }

        uint64_t c_index = 0;
        for(int n = 0; n < params.size(); n++)
        {
            auto p = params[n];
            const uint64_t x_size = p.x_size_per_batch * p.batchSize;
            const uint64_t w_size = p.w_size_per_batch;
            const uint64_t y_size = p.y_size_per_batch * p.batchSize;
       
           #if A_KNOWN == 0 || PARTY == 1
            UINT_TYPE* x = new UINT_TYPE[factor * x_size]; // Party1 holds X2 in plain in AB2 setting
            #else
            UINT_TYPE* x = nullptr;
            #endif
           #if A_KNOWN == 0 || PARTY == 0 
            UINT_TYPE* w = new UINT_TYPE[w_size];  // W is always constant
            #else 
            UINT_TYPE* w = nullptr; 
           #endif 
            UINT_TYPE* y = new UINT_TYPE[factor * y_size];

        #if A_KNOWN == 0 || PARTY == 1
        for (int i = 0; i < x_size; i++)
        {
            alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&b[n][i], temp, 1);
            for (int j = 0; j < factor; j++)
                x[j * x_size + i] = temp[j]; 
        }
        #endif
        #if A_KNOWN == 0 || PARTY == 0 
        for (int i = 0; i < w_size; i++)
        {
            alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
            unorthogonalize_arithmetic(&a[n][i], temp, 1);
            w[i] = temp[0];
        }
        #endif

        p.batchSize *= factor;
        // Conv2D(w,x,y, p) // calculate layer operation
        for (int i = 0; i < y_size; i++)
        {
            alignas(sizeof(DATATYPE)) UINT_TYPE temp[factor];
            for (int j = 0; j < factor; j++)
                temp[j] = y[j * y_size + i];
            orthogonalize_arithmetic(temp, c + c_index + i, 1);
        }
        delete[] x;
        delete[] w;
        delete[] y;
        c_index += y_size;
        }
}




#else

#define generateArithmeticTriples generateFakeArithmeticTriples
#define generateBooleanTriples generateFakeBooleanTriples
#define generateArithmeticAB2Triples generateFakeArithmeticTriples
#define generateBooleanAB2Triples generateFakeBooleanTriples
#define generateConvTriples generateFakeLayerTriples
#define generateFCTriples generateFakeLayerTriples
#define generateBatchNorm2DTriples generateFakeLayerTriples
#define generateBooleanAdditionTriples generateFakeBooleanAdditionTriples
#define generateMultiplexerTriples generateFakeMultiplexerTriples
#define generateCOTTriples generateCOTDummyTriples

template <typename type>
void generateFakeArithmeticTriples(type a[],
                                   type b[],
                                   type c[],
                                   int bitlength,
                                   uint64_t num_triples,
                                   std::string ip,
                                   int port)
{
}

template <typename type>
void generateFakeBooleanTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

    template <typename type>
void generateFakeAB2ArithmeticTriples(type a[],
                                   type b[],
                                   type c[],
                                   int bitlength,
                                   uint64_t num_triples,
                                   std::string ip,
                                   int port)
{
}

template <typename type>
void generateFakeAB2BooleanTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

    template <typename type>
void generateFakeBooleanAdditionTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

template <typename type>
void generateFakeMultiplexerTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

template <typename type>
void generateFakeCOTTriples(type a[],
                                type b[],
                                type c[],
                                int bitlength,
                                uint64_t num_triples,
                                std::string ip,
                                int port)
{
}

template <typename type, typename LayerParams>
void generateFakeLayerTriples(type** a,
                             type** b,
                             type c[],
                             int bitlength,
                             std::vector<LayerParams> params,
                             std::string ip,
                             int port)
{
}



#endif
