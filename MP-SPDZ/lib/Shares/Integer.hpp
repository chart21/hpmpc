#ifndef INTEGER_H
#define INTEGER_H

#include <cassert>
#include <compare>
#include <cstdint>
#include <utility>
#include <vector>

#include "../Constants.hpp"
#include "../help/Input.hpp"
#include "../help/Util.hpp"

namespace IR {

template <class int_t, class uint_t>
class Integer {
  public:
    using Base = int_t;
    using UBase = uint_t;

    Integer() : nums{0} {}
    Integer(Base a) : nums{a} {}
    Integer(DATATYPE a) : nums() {
        std::vector<UINT_TYPE> tmp;
        tmp.resize(SIZE_VEC);
        unorthogonalize_arithmetic(&a, tmp.data(), 1);
        for (const auto& ele : tmp)
            nums.push_back(INT_TYPE(ele));
    }
    Integer(const std::vector<Base>& a) : nums(a) {}
    Integer(const Integer& other) : nums(other.nums) {}                // copy
    Integer(Integer&& other) noexcept : nums(std::move(other.nums)) {} // move

    Integer& operator=(const Integer& other) { // copy
        nums = other.nums;

        return *this;
    }

    Integer& operator=(Integer&& other) noexcept { // move
        if (this == &other)
            return *this;

        nums = std::move(other.nums);
        return *this;
    }

    Integer& operator=(const Base& other) { // Base assignable
        nums.clear();
        nums.push_back(other);

        return *this;
    }

    Integer operator+(const Integer& other) const;
    Integer operator-(const Integer& other) const;

    Integer operator-() const;

    Integer operator*(const Integer& other) const;
    Integer operator/(const Integer& other) const;

    Integer operator%(const Integer& other) const;
    Integer operator~() const;

    Integer operator==(const Integer& other) const;
    Integer operator<(const Integer& other) const;
    Integer operator>(const Integer& other) const;
    Integer operator<(const Base& other) const;
    Integer operator==(const Base& other) const;

    Integer operator&(const Integer& other) const;
    Integer& operator&=(const Integer& other);

    Integer operator^(const Integer& other) const;
    Integer& operator^=(const Integer& other);

    Integer operator|(const Integer& other) const;
    Integer& operator|=(const Integer& other);

    Integer operator<<(const Integer& other) const;
    Integer operator>>(const Integer& other) const;

    inline Base& get() { return nums[0]; }
    std::vector<Base> get_all() const { return nums; }
    DATATYPE get_type() const {
        if (nums.size() == SIZE_VEC) {
            std::vector<UINT_TYPE> vec;
            for (const auto& ele : nums) {
                vec.emplace_back(ele);
            }
            DATATYPE a;
            orthogonalize_arithmetic(vec.data(), &a, 1);
            return a;
        } else {
            return PROMOTE(UINT_TYPE(nums[0]));
        }
    }

    void add(const Base& a) { nums.push_back(a); }

    size_t size() const { return nums.size(); }

  private:
    alignas(DATATYPE) std::vector<Base> nums;

    Base plus(const UBase& a, const UBase& other) const;
    Base minus(const Base& a, const Base& other) const;
};

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator+(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));

    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(plus(nums[i], other.nums[i]));
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(plus(ele, other.nums[0]));
    } else {
        for (const auto& ele : other.nums)
            res.add(plus(nums[0], ele));
    }
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>::Base Integer<int_t, uint_t>::plus(const UBase& a,
                                                          const UBase& other) const {
    if (a >= 0) {
        Base diff = std::numeric_limits<Base>::max() - a;
        if (diff < other) {
            return std::numeric_limits<Base>::min() + other - 1 - diff;
        }
    } else {
        Base diff = std::numeric_limits<Base>::min() - a;
        if (other < diff) {
            return std::numeric_limits<Base>::max() + other + 1 - diff;
        }
    }
    return a + other;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t> Integer<int_t, uint_t>::operator-() const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    for (const auto& ele : nums)
        res.add(-ele);
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator-(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(minus(nums[i], other.nums[i]));
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(minus(ele, other.nums[0]));
    } else {
        for (const auto& ele : other.nums)
            res.add(minus(nums[0], ele));
    }
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>::Base Integer<int_t, uint_t>::minus(const Base& a, const Base& other) const {
    if ((other > 0 && a < std::numeric_limits<Base>::min() + other) ||
        (other < 0 && a > std::numeric_limits<Base>::max() + other)) {
        if (other > 0) { // underflow
            Base diff = std::numeric_limits<Base>::min() - a;
            return std::numeric_limits<Base>::max() - (other - 1 + diff);
        } else { // overflow
            Base diff = std::numeric_limits<Base>::max() - a;
            return std::numeric_limits<Base>::min() - (other + 1 + diff);
        }
    }
    return a - other;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator*(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(UBase(nums[i]) * other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const UBase& ele : nums)
            res.add(ele * other.nums[0]);
    } else {
        for (const UBase& ele : other.nums)
            res.add(ele * nums[0]);
    }
    // UBase ua = a;
    // return Integer{Base(ua * other.a)};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator/(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i) {
            if (other.nums[i] == 0) {
                res.add(0);
            } else {
                res.add(nums[i] / other.nums[i]);
            }
        }
    } else if (nums.size() > other.nums.size()) {
        if (other.nums[0] == 0) {
            res.add(0);
        } else {
            for (const Base& ele : nums)
                res.add(ele / other.nums[0]);
        }
    } else {
        for (const Base& ele : other.nums) {
            if (ele == 0) {
                res.add(0);
            } else {
                res.add(nums[0] / ele);
            }
        }
    }
    // UBase ua = a;
    // return Integer{Base(ua / other.a)};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator%(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i) {
            if (other.nums[i] == 0) {
                res.add(0);
            } else {
                res.add(nums[i] % other.nums[i]);
            }
        }
    } else if (nums.size() > other.nums.size()) {
        if (other.nums[0] == 0) {
            res.add(0);
        } else {
            for (const auto& ele : nums)
                res.add(ele % other.nums[0]);
        }
    } else {
        for (const auto& ele : other.nums) {
            if (ele == 0) {
                res.add(0);
            } else {
                res.add(nums[0] % ele);
            }
        }
    }
    // if (other.a == 0)
    //     return Integer();
    // return Integer{a % other.a};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t> Integer<int_t, uint_t>::operator~() const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    for (const UBase& ele : nums)
        res.add(~ele);
    // UBase b(a);
    // UBase c(-1);
    // return Integer{Base(~b)};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator&(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(nums[i] & other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(ele & other.nums[0]);
    } else {
        for (const auto& ele : other.nums)
            res.add(nums[0] & ele);
    }

    // return Integer{a & other.a};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>& Integer<int_t, uint_t>::operator&=(const Integer<int_t, uint_t>& other) {
    int diff = other.nums.size() - nums.size();
    for (int i = 0; i < diff; ++i)
        nums.emplace_back(nums[0]);

    if (other.nums.size() == nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            nums[i] &= other.nums[i];
    } else { // (nums.size() > other.nums.size())
        assert(other.nums.size() <= 1);
        for (size_t i = 0; i < nums.size(); ++i)
            nums[i] &= other.nums[0];
    }

    return *this;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator^(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));

    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(nums[i] ^ other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(ele ^ other.nums[0]);
    } else {
        for (const auto& ele : other.nums)
            res.add(nums[0] ^ ele);
    }

    // return Integer{a ^ other.a};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>& Integer<int_t, uint_t>::operator^=(const Integer<int_t, uint_t>& other) {
    int diff = (other.nums.size() - nums.size());
    for (int i = 0; i < diff; ++i)
        nums.emplace_back(nums[0]);

    if (other.nums.size() == nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            nums[i] ^= other.nums[i];
    } else { // (nums.size() > other.nums.size())
        assert(other.nums.size() <= 1);
        for (size_t i = 0; i < nums.size(); ++i)
            nums[i] ^= other.nums[0];
    }
    return *this;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator|(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; nums.size(); ++i)
            res.add(nums[i] | other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(ele | other.nums[0]);
    } else {
        for (const auto& ele : other.nums)
            res.add(nums[0] | ele);
    }
    // return Integer{a | other.a};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>& Integer<int_t, uint_t>::operator|=(const Integer<int_t, uint_t>& other) {
    int diff = other.nums.size() - nums.size();
    for (int i = 0; i < diff; ++i)
        nums.emplace_back(nums[0]);

    if (other.nums.size() == nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            nums[i] |= other.nums[i];
    } else { // (nums.size() > other.nums.size())
        assert(other.nums.size() <= 1);
        for (size_t i = 0; i < nums.size(); ++i)
            nums[i] |= other.nums[0];
    }

    return *this;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator<<(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(nums[i] << other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(ele << other.nums[0]);
    } else {
        for (const auto& ele : other.nums)
            res.add(nums[0] << ele);
    }
    // return Integer{a << other.a};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator>>(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(UBase(nums[i]) >> other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const UBase& ele : nums)
            res.add(ele >> other.nums[0]);
    } else {
        for (const UBase& ele : other.nums)
            res.add(nums[0] >> ele);
    }
    // UBase nums(a);
    // if (other.a >= sizeof(Base) * 8)
    //     return Integer{0};
    // return Integer{Base(nums >> UBase(other.a))};
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator==(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(nums[i] == other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(ele == other.nums[0]);
    } else {
        for (const auto& ele : other.nums)
            res.add(nums[0] == ele);
    }
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator<(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(nums[i] < other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(ele < other.nums[0]);
    } else {
        for (const auto& ele : other.nums)
            res.add(nums[0] < ele);
    }
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t>
Integer<int_t, uint_t>::operator>(const Integer<int_t, uint_t>& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    if (nums.size() == other.nums.size()) {
        for (size_t i = 0; i < nums.size(); ++i)
            res.add(nums[i] > other.nums[i]);
    } else if (nums.size() > other.nums.size()) {
        for (const auto& ele : nums)
            res.add(ele > other.nums[0]);
    } else {
        for (const auto& ele : other.nums)
            res.add(nums[0] > ele);
    }
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t> Integer<int_t, uint_t>::operator<(const Base& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    for (const auto& ele : nums)
        res.add(ele < other);
    return res;
}

template <class int_t, class uint_t>
Integer<int_t, uint_t> Integer<int_t, uint_t>::operator==(const Base& other) const {
    Integer<int_t, uint_t> res(std::vector<Base>(0));
    for (const auto& ele : nums)
        res.add(ele == other);
    return res;
}

} // namespace IR

#endif
