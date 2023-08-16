#pragma once
#include <array>
#include <stdexcept>
#include "../protocols/Protocols.h"
template<typename Share>
class sbitset_t {
private:
    Share shares[BITLENGTH];
public:

    //temporary constructor
    sbitset_t() {
        }

    template<int id>
    sbitset_t(UINT_TYPE value) {
        UINT_TYPE temp_u[DATTYPE] = {value};
        init(temp_u);
        }

    template<int id>
    sbitset_t(UINT_TYPE value[DATTYPE]) {
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
            orthogonalize_boolean(value, temp_d);
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

    
    sbitset_t operator^(const sbitset_t& other) const {
        sbitset_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = shares[i] ^ other[i];
        }
        return result;
    }

    sbitset_t operator~() const {
        sbitset_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = ~shares[i];
        }
        return result;
    }

        sbitset_t operator&(const sbitset_t & other) const {
        sbitset_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            result[i] = shares[i] & other[i];
        }
        return result;
    }

        void complete_and() {
        for(int i = 0; i < BITLENGTH; ++i) {
            shares[i].complete_and();
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
           DATATYPE temp[BITLENGTH];
            for(int i = 0; i < BITLENGTH; ++i) {
               temp[i] = shares[i].complete_reveal_to_all();
            }
            unorthogonalize_boolean((DATATYPE*) temp, result);
        }
       

        Share* get_share_pointer() {
            return shares;
        }

        static sbitset_t<Share> load_shares(Share shares[BITLENGTH]) {
            sbitset_t<Share> result;
            for(int i = 0; i < BITLENGTH; ++i) {
                result[i] = shares[i];
            }
            return result;
        }

        static sbitset_t prepare_A2B_S1(Share s[BITLENGTH])
        {
            sbitset_t<Share> result;
            Share::prepare_A2B_S1(s,result.get_share_pointer());
            return result;
        }

        static sbitset_t prepare_A2B_S2(Share s[BITLENGTH])
        {
            sbitset_t<Share> result;
            Share::prepare_A2B_S2(s,result.get_share_pointer());
            return result;
        }
        
        void complete_A2B_S1()
        {
            Share::complete_A2B_S1(shares);
        }

        void complete_A2B_S2()
        {
            Share::complete_A2B_S2(shares);
        }

        void complete_bit_injection_S1()
        {
            Share::complete_bit_injection_S1(shares);
        }

        void complete_bit_injection_S2()
        {
            Share::complete_bit_injection_S2(shares);
        }

};


