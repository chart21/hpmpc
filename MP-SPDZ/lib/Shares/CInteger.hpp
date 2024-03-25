#ifndef CINTEGER_H
#define CINTEGER_H

#include <cassert>
#include <compare>
#include <cstdint>
#include <utility>
#include <vector>

#include "../Constants.hpp"
#include "../Shares/Integer.hpp"
#include "../help/Input.hpp"
#include "../help/Util.hpp"

namespace IR {

// template <class int_t, class uint_t>
// class Integer;

template <class int_t, class uint_t>
class CInteger {
  public:
    using Base = DATATYPE;
    // using UBase = uint_t;

    CInteger() : nums(PROMOTE(0)) {}
    CInteger(int_t a) : nums(PROMOTE(a)) {}
    CInteger(DATATYPE a) : nums(a) {
        // nums.resize(SIZE_VEC);
        // unorthogonalize_arithmetic(&a, (UBase*)nums.data(), 1);
    }
    // CInteger(const std::vector<Base>& a) : nums(a) {}
    CInteger(const CInteger& other) : nums(other.nums) {}                // copy
    CInteger(CInteger&& other) noexcept : nums(std::move(other.nums)) {} // move

    CInteger& operator=(const CInteger& other) { // copy
        nums = other.nums;

        return *this;
    }

    CInteger& operator=(CInteger&& other) noexcept { // move
        if (this == &other)
            return *this;

        nums = std::move(other.nums);
        return *this;
    }

    CInteger& operator=(DATATYPE other) { // DATATYPE -> CINT
        // nums.resize(SIZE_VEC);
        // unorthogonalize_arithmetic(&other, (UBase*)nums.data(), 1);
        nums = other;

        return *this;
    }

    CInteger& operator=(const Integer<int64_t, uint64_t>& other) { // INT -> CINT
        const auto& all = other.get_all();

        if (all.size() == 1) {
            nums = PROMOTE(all[0]);
            return *this;
        }

        assert(all.size() == SIZE_VEC);

        orthogonalize_arithmetic((uint_t*)all.data(), &nums, 1);
        return *this;
    }

    CInteger operator+(const CInteger& other) const;
    CInteger operator-(const CInteger& other) const;

    CInteger operator-() const;

    CInteger operator*(const CInteger& other) const;
    CInteger operator/(const CInteger& other) const;

    CInteger operator%(const CInteger& other) const;
    CInteger operator~() const;

    CInteger operator&(const CInteger& other) const;
    CInteger& operator&=(const CInteger& other);

    CInteger operator^(const CInteger& other) const;
    CInteger& operator^=(const CInteger& other);

    CInteger operator|(const CInteger& other) const;
    CInteger& operator|=(const CInteger& other);

    CInteger operator<<(const CInteger& other) const;
    CInteger operator>>(const CInteger& other) const;

    int_t get() const { return get_all()[0]; }
    DATATYPE get_type() const;
    std::vector<int_t> get_all() const {
        std::vector<int_t> res;
        res.resize(SIZE_VEC);

        DATATYPE tmp = nums;
        unorthogonalize_arithmetic(&tmp, (uint_t*)(res.data()), 1);
        return res;
    }

    size_t size() const { return SIZE_VEC; }

  private:
    Base nums;

    // Base plus(const UBase& a, const UBase& other) const;
    // Base minus(const Base& a, const Base& other) const;
};

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator+(const CInteger<int_t, uint_t>& other) const {
    return CInteger(OP_ADD(nums, other.nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t> CInteger<int_t, uint_t>::operator-() const {
    return CInteger(OP_SUB(ZERO, nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator-(const CInteger<int_t, uint_t>& other) const {
    return CInteger(OP_SUB(nums, other.nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator*(const CInteger<int_t, uint_t>& other) const {
    return CInteger(OP_MULT(nums, other.nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator/(const CInteger<int_t, uint_t>& other) const {
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res;
    for (size_t i = 0; i < vec1.size(); ++i)
        res.push_back(INT_TYPE(vec1[i]) / vec2[i]);

    DATATYPE dt;
    orthogonalize_arithmetic(res.data(), &dt, 1);
    // UBase ua = a;
    // return Integer{Base(ua / other.a)};
    return CInteger(dt);
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator%(const CInteger<int_t, uint_t>& other) const {
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec2[i] == 0)
            res.push_back(0u);
        else
            res.push_back(INT_TYPE(vec1[i]) % vec2[i]);
    }

    DATATYPE dt;
    orthogonalize_arithmetic(res.data(), &dt, 1);
    // UBase ua = a;
    // return Integer{Base(ua % other.a)};
    return CInteger(dt);
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t> CInteger<int_t, uint_t>::operator~() const {
    // UBase b(a);
    // UBase c(-1);
    // return CInteger{Base(~b)};
    return CInteger(NOT(nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator&(const CInteger<int_t, uint_t>& other) const {
    // return CInteger{a & other.a};
    return CInteger(AND(nums, other.nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>& CInteger<int_t, uint_t>::operator&=(const CInteger<int_t, uint_t>& other) {
    nums = AND(nums, other.nums);
    return *this;
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator^(const CInteger<int_t, uint_t>& other) const {
    // return CInteger{a ^ other.a};
    return CInteger(XOR(nums, other.nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>& CInteger<int_t, uint_t>::operator^=(const CInteger<int_t, uint_t>& other) {
    nums = XOR(nums, other.nums);
    return *this;
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator|(const CInteger<int_t, uint_t>& other) const {
    // return CInteger{a | other.a};
    return CInteger(OR(nums, other.nums));
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>& CInteger<int_t, uint_t>::operator|=(const CInteger<int_t, uint_t>& other) {
    nums = OR(nums, other.nums);
    return *this;
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator<<(const CInteger<int_t, uint_t>& other) const {
    // return CInteger{a << other.a};
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res;
    for (size_t i = 0; i < vec1.size(); ++i) {
        res.push_back(UINT_TYPE(vec1[i]) << vec2[i]);
    }

    DATATYPE dt;
    orthogonalize_arithmetic(res.data(), &dt, 1);

    return CInteger(dt);
}

template <class int_t, class uint_t>
CInteger<int_t, uint_t>
CInteger<int_t, uint_t>::operator>>(const CInteger<int_t, uint_t>& other) const {
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec2[i] >= sizeof(UINT_TYPE) * 8)
            res.push_back(0);
        else
            res.push_back(UINT_TYPE(vec1[i]) >> vec2[i]);
    }

    DATATYPE dt;
    orthogonalize_arithmetic(res.data(), &dt, 1);
    return CInteger(dt);
}

template <class int_t, class uint_t>
DATATYPE CInteger<int_t, uint_t>::get_type() const {
    return nums;
}

} // namespace IR

#endif
