#include <array>
#include <stdexcept>
#include "../protocols/Protocols.h"
template<typename Pr>
class sbitset_t {
private:
    Pr P; //protocol
    DATATYPE shares[BITLENGTH][Pr::VALS_PER_SHARE];
    sbitset_t(UINT_TYPE value, int id, Pr protocol) {
        P = protocol;
        if(id == PSELF && current_phase == 1) //read value
        {
            #if VALS_PER_SHARE == 1
            UINT_TYPE* temp_u = (UINT_TYPE*) shares;
            std::fill_n(temp_u, DATTYPE, value);
            orthogonalize_boolean(temp_u, (DATATYPE*) shares);
            #else
            UINT_TYPE temp_u[DATTYPE] = {value};
            DATATYPE* temp_d = (DATATYPE*) temp_u;
            orthogonalize_boolean(temp_u, temp_d);
            for(int i = 0; i < BITLENGTH; i++) {
                shares[i][0] = temp_d[i];
            }
            #endif
        }
            P.prepare_receive_from(shares, id, BITLENGTH, OP_ADD, OP_SUB);
    }

    sbitset_t(UINT_TYPE values[DATTYPE], int id, Pr protocol) {
        P = protocol;
        if(id == PSELF && current_phase == 1)
        {
            #if VALS_PER_SHARE == 1
            UINT_TYPE* temp_u = (UINT_TYPE*) shares;
            memcpy(temp_u, values, DATTYPE * sizeof(UINT_TYPE));
            orthogonalize_arithmetic(temp_u, (DATATYPE*) shares);
            #else
            DATATYPE* temp_d = (DATATYPE*) values;
            orthogonalize_arithmetic(values, temp_d);
            for(int i = 0; i < BITLENGTH; i++) {
                shares[i][0] = temp_d[i];
            }
            #endif
        }
            P.prepare_receive_from(shares, id, BITLENGTH, OP_ADD, OP_SUB);
    }
    
    DATATYPE* get_shares() {
        return (DATATYPE*) shares;
    }


    DATATYPE* operator[](std::size_t idx) {
        return shares[idx];
    }

    const DATATYPE* operator[](std::size_t idx) const {
        return shares[idx];
    }

    
    sbitset_t operator|(const sbitset_t& other) const {
        sbitset_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            P.Add(shares[i], other[i], result[i], FUNC_XOR);
        }
        return result;
    }

    sbitset_t operator!() const {
        sbitset_t result;
        for(int i = 0; i < BITLENGTH; ++i) {
            P.Not(shares[i],result[i]);
        }
        return result;
    }

        sbitset_t operator&(const sbitset_t & other) const {
        sbitset_t result;
        for(std::size_t i = 0; i < BITLENGTH; ++i) {
            P.prepareMult(shares[i], other[i],result[i], FUNC_XOR, FUNC_XOR, FUNC_AND);
        }
        return result;
    }

        void completeAND() {
        for(std::size_t i = 0; i < BITLENGTH; ++i) {
            P.completeMult(shares[i], FUNC_XOR, FUNC_XOR);
        }
        }

        void complete_receive_from(int id) {
        P.complete_receive_from(shares, id, BITLENGTH, FUNC_XOR, FUNC_XOR);
        }

        void prepare_reveal_to_all() {
            for(int i = 0; i < BITLENGTH; ++i) {
                P.prepare_reveal_to_all(shares[i]);
            }
        }

        void complete_reveal_to_all(UINT_TYPE result[DATTYPE]) {
            
            for(int i = 0; i < BITLENGTH; ++i) {
               ((DATATYPE*) result)[i] = P.complete_reveal_to_all(shares[i], OP_ADD, OP_SUB);
            }
            unorthogonalize_boolean((DATATYPE*) result, result);
        }
        


    // ... You can define other operations similarly

};


