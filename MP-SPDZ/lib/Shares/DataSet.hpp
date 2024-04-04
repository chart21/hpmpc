#ifndef DATASET_H
#define DATASET_H

#include <cstddef>
#include <cstdint>
#include <vector>

namespace IR {

template <class int_t, class uint_t, size_t LENGTH = BITLENGTH>
class DataSet {
  public:
    using Base = DATATYPE;
    using IBase = int_t;

    DataSet() {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = ZERO;
    }
    DataSet(int_t a) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = PROMOTE(a);
    }
    DataSet(DATATYPE a) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = a;
    }
    DataSet(const Integer<int64_t, uint64_t>& other) {
        const auto& all = other.get_all();

        if (all.size() == 1) {
            for (size_t i = 0; i < LENGTH; ++i)
                nums[i] = PROMOTE(all[0]);
        } else {
            assert(all.size() == DATTYPE);

            orthogonalize_arithmetic((uint_t*)all.data(), nums);
        }
    }
    DataSet(const DataSet& other) {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = other.nums[i];
    } // copy
    DataSet(DataSet&& other) noexcept {
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = other.nums[i];
    } // move

    DataSet& operator=(const DataSet& other) { // copy
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = other.nums[i];

        return *this;
    }

    DataSet& operator=(DataSet&& other) noexcept { // move
        if (this == &other)
            return *this;

        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = other.nums[i];
        return *this;
    }

    DataSet& operator=(DATATYPE other) { // DATATYPE -> CINT
        for (size_t i = 0; i < LENGTH; ++i)
            nums[i] = other;

        return *this;
    }

    DataSet& operator=(const std::vector<UINT_TYPE>& other) {
        assert(other.size() == DATTYPE);
        orthogonalize_arithmetic((UINT_TYPE*)other.data(), nums);

        return *this;
    }

    DataSet& operator=(const Integer<int64_t, uint64_t>& other) { // INT -> CINT
        const auto& all = other.get_all();

        if (all.size() == 1) {
            for (size_t i = 0; i < LENGTH; ++i)
                nums[i] = PROMOTE(all[0]);

            return *this;
        }

        assert(all.size() == DATTYPE);

        orthogonalize_arithmetic((uint_t*)all.data(), nums);
        return *this;
    }

    DataSet operator+(const DataSet& other) const;
    DataSet operator-(const DataSet& other) const;

    DataSet operator-() const;

    DataSet operator*(const DataSet& other) const;
    DataSet operator/(const DataSet& other) const;

    DataSet operator%(const DataSet& other) const;
    DataSet operator~() const;

    DataSet operator&(const DataSet& other) const;
    DataSet& operator&=(const DataSet& other);

    DataSet operator^(const DataSet& other) const;
    DataSet& operator^=(const DataSet& other);

    DataSet operator|(const DataSet& other) const;
    DataSet& operator|=(const DataSet& other);

    DataSet operator<<(const DataSet& other) const;
    DataSet operator>>(const DataSet& other) const;

    int_t get() const { return get_all()[0]; }
    DATATYPE* get_type();
    std::vector<int_t> get_all() const {
        std::vector<int_t> res;
        res.resize(DATTYPE);

        unorthogonalize_arithmetic((DATATYPE*)nums, (uint_t*)(res.data()));
        return res;
    }

    size_t size() const { return LENGTH; }

    DataSet operator==(const DataSet& other) const {
        log(Level::ERROR, "unsupported operation");
        return DataSet();
    }
    DataSet operator<(const DataSet& other) const {
        log(Level::ERROR, "unsupported operation");
        return DataSet();
    }
    DataSet operator>(const DataSet& other) const {
        log(Level::ERROR, "unsupported operation");
        return DataSet();
    }
    DataSet operator<(const IBase& other) const {
        log(Level::ERROR, "unsupported operation");
        return DataSet();
    }
    DataSet operator==(const IBase& other) const {
        log(Level::ERROR, "unsupported operation");
        return DataSet();
    }

    DATATYPE& operator[](const size_t& i) { return nums[i]; }

  private:
    Base nums[LENGTH];

    // Base plus(const UBase& a, const UBase& other) const;
    // Base minus(const Base& a, const Base& other) const;
};

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator+(const DataSet<int_t, uint_t, LENGTH>& other) const {
    DataSet res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = OP_ADD(nums[i], other.nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH> DataSet<int_t, uint_t, LENGTH>::operator-() const {
    DataSet res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = OP_SUB(ZERO, nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator-(const DataSet<int_t, uint_t, LENGTH>& other) const {
    DataSet<int_t, uint_t, LENGTH> res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = OP_SUB(nums[i], other.nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator*(const DataSet<int_t, uint_t, LENGTH>& other) const {
    DataSet<int_t, uint_t, LENGTH> res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = OP_MULT(nums[i], other.nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator/(const DataSet<int_t, uint_t, LENGTH>& other) const {
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res_u;
    for (size_t i = 0; i < vec1.size(); ++i)
        res_u.push_back(INT_TYPE(vec1[i]) / vec2[i]);

    DataSet<int_t, uint_t, LENGTH> res;
    orthogonalize_arithmetic(res_u.data(), res.get_type());
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator%(const DataSet<int_t, uint_t, LENGTH>& other) const {
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res_u;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec2[i] == 0)
            res_u.push_back(0u);
        else
            res_u.push_back(INT_TYPE(vec1[i]) % vec2[i]);
    }

    DataSet<int_t, uint_t, LENGTH> res;
    orthogonalize_arithmetic(res_u.data(), res.get_type());
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH> DataSet<int_t, uint_t, LENGTH>::operator~() const {
    DataSet<int_t, uint_t, LENGTH> res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = NOT(nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator&(const DataSet<int_t, uint_t, LENGTH>& other) const {
    DataSet<int_t, uint_t, LENGTH> res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = AND(nums[i], other.nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>&
DataSet<int_t, uint_t, LENGTH>::operator&=(const DataSet<int_t, uint_t, LENGTH>& other) {
    for (size_t i = 0; i < LENGTH; ++i)
        nums[i] = AND(nums[i], other.nums[i]);
    return *this;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator^(const DataSet<int_t, uint_t, LENGTH>& other) const {
    DataSet<int_t, uint_t, LENGTH> res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = XOR(nums[i], other.nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>&
DataSet<int_t, uint_t, LENGTH>::operator^=(const DataSet<int_t, uint_t, LENGTH>& other) {
    for (size_t i = 0; i < LENGTH; ++i)
        nums[i] = XOR(nums[i], other.nums[i]);
    return *this;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator|(const DataSet<int_t, uint_t, LENGTH>& other) const {
    DataSet<int_t, uint_t, LENGTH> res;
    for (size_t i = 0; i < LENGTH; ++i)
        res[i] = OR(nums[i], other.nums[i]);
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>&
DataSet<int_t, uint_t, LENGTH>::operator|=(const DataSet<int_t, uint_t, LENGTH>& other) {
    for (size_t i = 0; i < LENGTH; ++i)
        nums[i] = OR(nums[i], other.nums[i]);
    return *this;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator<<(const DataSet<int_t, uint_t, LENGTH>& other) const {
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res_u;
    for (size_t i = 0; i < vec1.size(); ++i) {
        res_u.push_back(UINT_TYPE(vec1[i]) << vec2[i]);
    }

    DataSet<int_t, uint_t, LENGTH> res;
    orthogonalize_arithmetic(res_u.data(), res.get_type());

    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DataSet<int_t, uint_t, LENGTH>
DataSet<int_t, uint_t, LENGTH>::operator>>(const DataSet<int_t, uint_t, LENGTH>& other) const {
    auto vec1 = get_all();
    auto vec2 = other.get_all();

    std::vector<UINT_TYPE> res_u;
    for (size_t i = 0; i < vec1.size(); ++i) {
        if (vec2[i] >= sizeof(UINT_TYPE) * 8)
            res_u.push_back(0);
        else
            res_u.push_back(UINT_TYPE(vec1[i]) >> vec2[i]);
    }

    DataSet<int_t, uint_t, LENGTH> res;
    orthogonalize_arithmetic(res_u.data(), res.get_type());
    return res;
}

template <class int_t, class uint_t, size_t LENGTH>
DATATYPE* DataSet<int_t, uint_t, LENGTH>::get_type() {
    return nums;
}

} // namespace IR

#endif