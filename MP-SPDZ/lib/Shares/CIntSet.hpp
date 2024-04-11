#ifndef CINTSET_H
#define CINTSET_H

#include <vector>

#include "Integer.hpp"

using std::vector;

namespace IR {

template <class cint, size_t LENGTH = BITLENGTH>
class CIntSet {
  public:
    CIntSet() : nums(LENGTH) {}

    CIntSet(const vector<DATATYPE>& vec) {
        nums.reserve(vec.size());

        assert(vec.size() == LENGTH);

        for (const auto& ele : vec)
            nums.emplace_back(ele);
    }

    CIntSet(const DATATYPE& a) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums.push_back(a);
    }

    CIntSet(const cint::IBase& a) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums.push_back(a);
    }

    const vector<DATATYPE> get_type() const {
        vector<DATATYPE> res;
        for (const auto& ele : nums)
            res.push_back(ele.get_type());
        return res;
    }

    vector<INT_TYPE> get_all() const {
        vector<INT_TYPE> tmp(DATTYPE);
        tmp.resize(DATTYPE); // TODO
        unorthogonalize_arithmetic((DATATYPE*)(get_type().data()), (UINT_TYPE*)tmp.data());
        return tmp;
    }

    cint::IBase get() const { return nums[0].get(); }

    CIntSet& operator=(const typename cint::IBase& other) {
        nums.clear();
        nums.reserve(LENGTH);
        for (size_t i = 0; i < LENGTH; ++i)
            nums.push_back(other);

        return *this;
    }

    CIntSet& operator=(const vector<UINT_TYPE>& other) {
        nums.clear();
        nums.reserve(LENGTH);

        DATATYPE tmp[LENGTH];
        orthogonalize_arithmetic((UINT_TYPE*)other.data(), tmp);

        for (size_t i = 0; i < LENGTH; ++i)
            nums.push_back(tmp[i]);

        return *this;
    }

    CIntSet& operator=(const Integer<int64_t, uint64_t>& other) {
        nums.clear();
        vector<DATATYPE> tmp;
        tmp.resize(LENGTH);

        orthogonalize_arithmetic((UINT_TYPE*)other.get_all().data(), tmp.data());

        for (auto& ele : tmp)
            nums.push_back(ele);
        // nums.clear();
        // nums.reserve(LENGTH);
        //
        // for (size_t i = 0; i < LENGTH; ++i)
        //     nums.push_back(other);
        //
        return *this;
    }

    cint& operator[](const size_t& index) { return nums[index]; }

    CIntSet operator+(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] + other.nums[i];
        return set;
    }
    CIntSet operator-(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] - other.nums[i];
        return set;
    }

    CIntSet operator-() const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = -nums[i];
        return set;
    }

    CIntSet operator*(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] * other.nums[i];
        return set;
    }
    CIntSet operator/(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] / other.nums[i];
        return set;
    }

    CIntSet operator%(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] % other.nums[i];
        return set;
    }
    CIntSet operator~() const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = ~nums[i];
        return set;
    }

    CIntSet operator&(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] & other.nums[i];
        return set;
    }
    CIntSet& operator&=(const CIntSet& other) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] &= other.nums[i];
        return *this;
    }

    CIntSet operator^(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] ^ other.nums[i];
        return set;
    }
    CIntSet& operator^=(const CIntSet& other) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] ^= other.nums[i];
        return *this;
    }

    CIntSet operator|(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] | other.nums[i];
        return set;
    }
    CIntSet& operator|=(const CIntSet& other) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] |= other.nums[i];
        return *this;
    }

    CIntSet operator<<(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] << other.nums[i];
        return set;
    }
    CIntSet operator>>(const CIntSet& other) const {
        CIntSet set;
        for (size_t i = 0; i < LENGTH; ++i)
            set[i] = nums[i] >> other.nums[i];
        return set;
    }

    CIntSet operator==(const CIntSet& other) const {
        CIntSet res;

        for (size_t i = 0; i < BITLENGTH; ++i) {
            res[i] = nums[i] == other.nums[i];
        }

        return res;
    }
    CIntSet operator<(const CIntSet& other) const {
        CIntSet res;

        for (size_t i = 0; i < BITLENGTH; ++i) {
            res[i] = nums[i] < other.nums[i];
        }

        return res;
    }
    CIntSet operator>(const CIntSet& other) const {
        CIntSet res;

        for (size_t i = 0; i < BITLENGTH; ++i) {
            res[i] = nums[i] > other.nums[i];
        }

        return res;
    }
    CIntSet operator<(const cint::IBase& other) const {
        CIntSet res;

        for (size_t i = 0; i < BITLENGTH; ++i) {
            res[i] = nums[i] < other;
        }

        return res;
    }

    CIntSet operator==(const cint::IBase& other) const {
        CIntSet res;

        for (size_t i = 0; i < BITLENGTH; ++i) {
            res[i] = nums[i] == other;
        }

        return res;
    }

  private:
    vector<cint> nums;
};

} // namespace IR

#endif