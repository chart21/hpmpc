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

    template<int id>
    sint_t(UINT_TYPE value) {
        UINT_TYPE temp_u[DATTYPE] = {value};
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
              shares[i] = Share(temp_d[i]);
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

        void complete_mult() {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i].complete_mult();
        }
        }

        void complete_receive_from(int id) {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i].template complete_receive_from<id>();
        }
        }

        void prepare_reveal_to_all() {
            for(int i = 0; i < BITLENGTH; ++i) {
                shares[i].prepare_reveal_to_all();
            }
        }

        void complete_reveal_to_all(UINT_TYPE result[DATTYPE]) {
            
            for(int i = 0; i < BITLENGTH; ++i) {
               ((DATATYPE*) result)[i] = shares[i].complete_reveal_to_all();
            unorthogonalize_arithmetic((DATATYPE*) result, result);
            }
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

};



