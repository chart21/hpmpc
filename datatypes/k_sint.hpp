#pragma once
#include <array>
#include <stdexcept>
#include "../protocols/Protocols.h"
template<typename Share>
class sint_t {
private:
    Share shares[BITLENGTH];
public:

    //temporary constructor
    sint_t() {
        }
    
    sint_t(UINT_TYPE value) {
        for (int i = 0; i < BITLENGTH; i++) 
          shares[i] = Share::public_val(PROMOTE(value));
        }

    template<int id>
    sint_t(UINT_TYPE value) {
        alignas(sizeof(DATATYPE)) UINT_TYPE temp_u[DATTYPE] = {value};
        init(temp_u);
        }


    template<int id>
    sint_t(UINT_TYPE value[DATTYPE]) {
                init(value);
    }
    
    template<int id>
    void prepare_receive_from() {
        for (int i = 0; i < BITLENGTH; i++) 
          shares[i].template prepare_receive_from<id>();
    }
    
    template<int id>
    void prepare_receive_from(DATATYPE values[BITLENGTH]) {
        for (int i = 0; i < BITLENGTH; i++) 
          shares[i].template prepare_receive_from<id>(values[i]);
    }

    template<int id>
    void prepare_receive_and_replicate(UINT_TYPE value) {
        if constexpr (id == PSELF || PROTOCOL == 13) {
          if (current_phase == 1) {
            /* alignas(sizeof(DATATYPE)) UINT_TYPE temp_u[DATTYPE] = {value}; */
            /* orthogonalize_arithmetic(temp_u, (DATATYPE*) temp_u); */
            /* prepare_receive_from<id>((DATATYPE*) temp_u); */
            DATATYPE temp_u[BITLENGTH];
            for (int i = 0; i < BITLENGTH; i++)
                shares[i].template prepare_receive_from<id>(PROMOTE(value));
            return;
            }
        }
            prepare_receive_from<id>();
    }

    template<int id>
    void complete_receive_from() {
        for (int i = 0; i < BITLENGTH; i++) 
          shares[i].template complete_receive_from<id>();
    }

    template <int id> void init(UINT_TYPE value[DATTYPE]) {
        if constexpr (id == PSELF) {
          if (current_phase == 1) {

            DATATYPE temp_d[BITLENGTH];
            orthogonalize_arithmetic(value, temp_d);
            for (int i = 0; i < BITLENGTH; i++) 
              shares[i].template prepare_receive_from<id>(temp_d[i]);
          }
        }

        for (int i = 0; i < BITLENGTH; i++) {
          shares[i].template prepare_receive_from<id>();
        }
    }

    Share& operator[](int idx) {
        return shares[idx];
    }

    const Share& operator[](int idx) const {
        return shares[idx];
    }

    
    sint_t operator+(const sint_t& other) const {
        sint_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = shares[i] + other[i];
        }
        return result;
    }

    sint_t operator-(const sint_t& other) const {
        sint_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = shares[i] - other[i];
        }
        return result;
    }

        sint_t operator*(const sint_t & other) const {
        sint_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = shares[i] * other[i];
        }
        return result;
    }
    
        void operator+=(const sint_t& other) {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i] = shares[i] + other[i];
        }
    }

        void operator*=(const UINT_TYPE other) {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i].mult_public_fixed(PROMOTE(other));
        }
        }

        sint_t mult_public(const UINT_TYPE other) {
        sint_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = shares[i].mult_public(PROMOTE(other));
        }
        return result;
        }
    
    bool operator==(const sint_t& b) const
    {
        return false; // Needed for Eigen optimizations
    }

    void operator*=(const sint_t& other) {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i] = shares[i] * other[i];
        }
    }


        void complete_mult() {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i].complete_mult();
        }
        }

        void complete_mult_without_trunc(){
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i].complete_mult_without_trunc();
        }

        }
       
        sint_t prepare_dot(const sint_t& other) const {
        sint_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = shares[i].prepare_dot(other[i]);
        }
        return result;
        }

        void complete_receive_from(int id) 
        {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i].template complete_receive_from<id>();
        }
        }

        void prepare_reveal_to_all() const {
            for(int i = 0; i < BITLENGTH; ++i) {
                shares[i].prepare_reveal_to_all();
            }
        }
        
        
        void complete_reveal_to_all(UINT_TYPE result[DATTYPE]) const {
            DATATYPE temp[BITLENGTH];
            for(int i = 0; i < BITLENGTH; ++i) {
               temp[i] = shares[i].complete_reveal_to_all();
            }
            unorthogonalize_arithmetic(temp, result);
        }

        UINT_TYPE complete_reveal_to_all_single() const {
            DATATYPE temp[BITLENGTH];
            alignas(sizeof(DATATYPE)) UINT_TYPE result[DATTYPE];
            for(int i = 0; i < BITLENGTH; ++i) {
               temp[i] = shares[i].complete_reveal_to_all();
            }
            unorthogonalize_arithmetic(temp, result);
            return result[0];
        }
       

        Share* get_share_pointer() {
            return shares;
        }

        static sint_t<Share> load_shares(Share shares[BITLENGTH]) {
            sint_t<Share> result;
            for(int i = 0; i < BITLENGTH; ++i) {
                result[i] = shares[i];
            }
            return result;
        }

        void prepare_XOR(const sint_t<Share> &a, const sint_t<Share> &b) {
            for(int i = 0; i < BITLENGTH; ++i) {
                shares[i] = a[i] * b[i];
            }
        }

        void complete_XOR(const sint_t<Share> &a, const sint_t<Share> &b) {
            for(int i = 0; i < BITLENGTH; ++i) {
                /* shares[i].complete_XOR(a[i], b[i]); */
                shares[i].complete_mult_without_trunc();
                shares[i] = a[i] + b[i] - shares[i] - shares[i];
            }
        }

        void complete_bit_injection_S1() {
            Share::complete_bit_injection_S1(shares);
        }

        void mask_and_send_dot()
        {
            for(int i = 0; i < BITLENGTH; ++i) 
                shares[i].mask_and_send_dot();
        }
        
        void mask_and_send_dot_without_trunc()
        {
            for(int i = 0; i < BITLENGTH; ++i) 
                shares[i].mask_and_send_dot_without_trunc();
        }

        void complete_bit_injection_S2() {
            Share::complete_bit_injection_S2(shares);
        }

        UINT_TYPE get_p1()
        {
            /* return shares[0].get_p1(); */
            return 0;
        }

        static void communicate()
        {
            Share::communicate();
        }

       
        sint_t relu() const
        {
            sint_t result;
            for(int i = 0; i < BITLENGTH; ++i) 
                result.shares[i] = shares[i].relu();
            return result;
        }

        void prepare_trunc_2k_inputs(sint_t& rmk2, sint_t& rmsb, sint_t& c, sint_t& c_prime)
        {
            for(int i = 0; i < BITLENGTH; ++i) 
                shares[i].prepare_trunc_2k_inputs(rmk2.shares[i], rmsb.shares[i], c.shares[i], c_prime.shares[i]);
        }

        void complete_trunc_2k_inputs(sint_t& rmk2, sint_t& rmsb, sint_t& c, sint_t& c_prime)
        {
            for(int i = 0; i < BITLENGTH; ++i) 
                shares[i].complete_trunc_2k_inputs(rmk2.shares[i], rmsb.shares[i], c.shares[i], c_prime.shares[i]);
        }

static void RELU(const sint_t* begin, const sint_t* end,  sint_t* output){
    /* int i = 0; */
    /* for (const sint_t* iter = begin; iter != end; ++iter) { */
    /*         output[i++] = iter->relu(); */
    /* } */
    int len = end - begin;
    std::copy(begin, end, output);
    for(int i = 0; i < len; ++i) {
        output[i] = output[i].relu();
    }
}


};



