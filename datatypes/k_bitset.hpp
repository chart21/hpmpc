#include <array>
#include <stdexcept>
#include "../arch/DATATYPE.h"

template<typename Int_type, typename Pr>
class sbitset {
private:
    std::array<DATATYPE, sizeof(Int_type)> arr;
    size_t k = sizeof(Int_type)*8;
    Pr P; //protocol
    sbitset(Int_type value, Pr protocol) {
        P = protocol;
        Int_type* ptr = reinterpret_cast<Int_type*>(arr);
        for(std::size_t i = 0; i < k*sizeof(DATATYPE)/sizeof(Int_type); ++i) {
            ptr[i] = value;
        }
            orthogonalize(ptr, ptr);
            
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

    
    sbitset operator&&(const sbitset& other) const {
        sbitset result;
            result = P.Add(arr + other, func_XOR);
        return result;
    }

    sbitset operator-(const sbitset & other) const {
        sbitset result;
            result = P.Add(arr + other, func_XOR);
        return result;
    }

        sbitset operator*(const sbitset & other) const {
        sbitset result;
            P.prepareMult(arr, other,result, func_XOR, func_XOR, func_MUL);
        return result;
    }

        void completeMult() {
            P.completeMult(arr, func_XOR, func_XOR);
        }
        


    // ... You can define other operations similarly

};

