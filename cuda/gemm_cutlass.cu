#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>

#include <cutlass/util/host_tensor.h>


#include <iostream>
#include <gemm.cuh>

#include <cstdint>

class Int64Wrapper {
public:
    int64_t value;   // Main storage for the int64 value
    int64_t dummy;   // Dummy variable, always zero

    // Constructor to initialize the main value and dummy
    Int64Wrapper(int64_t v = 0) : value(v), dummy(0) {}

    // Arithmetic operators
    Int64Wrapper operator+(const Int64Wrapper& other) const {
        // Include dummy in the operation; dummy is always zero
        return Int64Wrapper(value + other.value + dummy + other.dummy);
    }

    Int64Wrapper operator-(const Int64Wrapper& other) const {
        return Int64Wrapper(value - other.value);
    }

    Int64Wrapper operator*(const Int64Wrapper& other) const {
        // Include dummy in the operation; dummy is always zero
        return Int64Wrapper(value * other.value + dummy + other.dummy);
    }

    Int64Wrapper operator/(const Int64Wrapper& other) const {
        return Int64Wrapper(value / other.value);
    }

    // Compound assignment operators
    Int64Wrapper& operator+=(const Int64Wrapper& other) {
        value += other.value;
        return *this;
    }

    Int64Wrapper& operator-=(const Int64Wrapper& other) {
        value -= other.value;
        return *this;
    }

    Int64Wrapper& operator*=(const Int64Wrapper& other) {
        value *= other.value;
        return *this;
    }

    Int64Wrapper& operator/=(const Int64Wrapper& other) {
        value /= other.value;
        return *this;
    }

    // Comparison operators
    bool operator==(const Int64Wrapper& other) const {
        return value == other.value;
    }

    bool operator!=(const Int64Wrapper& other) const {
        return value != other.value;
    }

    bool operator<(const Int64Wrapper& other) const {
        return value < other.value;
    }

    bool operator>(const Int64Wrapper& other) const {
        return value > other.value;
    }

    bool operator<=(const Int64Wrapper& other) const {
        return value <= other.value;
    }

    bool operator>=(const Int64Wrapper& other) const {
        return value >= other.value;
    }

    // Stream operators
    friend std::ostream& operator<<(std::ostream& os, const Int64Wrapper& obj) {
        os << obj.value;
        return os;
    }

    friend std::istream& operator>>(std::istream& is, Int64Wrapper& obj) {
        is >> obj.value;
        return is;
    }
};



int main() {
    typedef int64_t Type;
   
    //int M = 512;
    //int N = 1024;
    //int K = 2048;
    int M = 2;
    int N = 2;
    int K = 2;

    auto X = new Type[M * N];
    auto W = new Type[N * K];
    auto Y = new Type[M * K];

    for (int i = 0; i < M * N; i++) {
        X[i] = i;
    }
    for (int i = 0; i < N * K; i++) {
        W[i] = i;
    }
    for (int i = 0; i < M * K; i++) {
        Y[i] = 0;
    }

    Type *x, *w, *y;

    cudaMalloc((void **)&x, M * N * sizeof(Type));
    cudaMalloc((void **)&w, N * K * sizeof(Type));
    cudaMalloc((void **)&y, M * K * sizeof(Type));

    cudaMemcpy(x, X, M * N * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(w, W, N * K * sizeof(Type), cudaMemcpyHostToDevice);
    cudaMemcpy(y, Y, M * K * sizeof(Type), cudaMemcpyHostToDevice);

    gpu::gemm<Type>(M, N, K, x, false, w, false, y, false);

    cudaMemcpy(Y, y, M * K * sizeof(Type), cudaMemcpyDeviceToHost);

    for (int i = 0; i < M * K; i++) {
        std::cout << Y[i] << " ";
    }
    std::cout << std::endl;



    return EXIT_SUCCESS;
}
