#include <array>
#include <stdexcept>
#include "../arch/DATATYPE.h"

template<typename Int_type, typename Pr>
class sint_t {
private:
    std::array<DATATYPE, sizeof(Int_type)> arr;
    size_t k = sizeof(Int_type)*8;
    Pr P; //protocol
    sint_t(Int_type value, Pr protocol) {
        P = protocol;
        Int_type* ptr = reinterpret_cast<Int_type*>(arr);
        for(std::size_t i = 0; i < k*sizeof(DATATYPE)/sizeof(Int_type); ++i) {
            ptr[i] = value;
        }
        for(std::size_t i = 0; i < k; ++i) {
            orthogonalize(arr[i], arr[i]);
            
    }
    }


    int& operator[](std::size_t idx) {
        if (idx >= k) {
            throw std::out_of_range("Index out of range");
        }
        return arr[idx];
    }

    const int& operator[](std::size_t idx) const {
        if (idx >= k) {
            throw std::out_of_range("Index out of range");
        }
        return arr[idx];
    }

    
    sint_t operator+(const sint_t& other) const {
        sint_t result;
        for(std::size_t i = 0; i < k; ++i) {
            result[i] = P.Add(arr[i] + other[i]);
        }
        return result;
    }

    sint_t operator-(const sint_t & other) const {
        sint_t result;
        for(std::size_t i = 0; i < k; ++i) {
            result[i] = P.Sub(arr[i] - other[i]);
        }
        return result;
    }

        sint_t operator*(const sint_t & other) const {
        sint_t result;
        for(std::size_t i = 0; i < k; ++i) {
            P.prepareMult(arr[i], other[i],result[i]);
        }
        return result;
    }

        void completeMult() {
        for(std::size_t i = 0; i < k; ++i) {
            P.completeMult(arr[i]);
        }
        }
        


    // ... You can define other operations similarly

};

