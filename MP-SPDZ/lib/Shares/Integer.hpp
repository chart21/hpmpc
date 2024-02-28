#ifndef INTEGER_H
#define INTEGER_H

#include <cassert>
#include <compare>
#include <cstdint>
#include <utility>

#include "../Constants.hpp"
#include "../help/Input.hpp"
#include "../help/Util.hpp"

namespace IR {

template <class int_t, class uint_t>
class Integer {
  public:
    using Base = int_t;
    using UBase = uint_t;

  private:
    Base a;

  public:
    Integer() : a(0) {}
    Integer(Base a) : a(a) {}
    Integer(const Integer& other) : a(other.a) {}                // copy
    Integer(Integer&& other) noexcept : a(std::move(other.a)) {} // move

    Integer& operator=(const Integer& other) { // copy
        a = other.a;
        return *this;
    }

    Integer& operator=(Integer&& other) noexcept { // move
        if (this == &other)
            return *this;

        a = std::move(other.a);
        return *this;
    }

    Integer& operator=(const Base& other) { // Base assignable
        a = other;
        return *this;
    }

    Integer operator+(const Integer& other) const {
        if (a >= 0) {
            Base diff = std::numeric_limits<Base>::max() - a;
            if (diff < other.a) {
                return Integer{std::numeric_limits<Base>::min() + other.a - 1 - diff};
            }
        } else {
            Base diff = std::numeric_limits<Base>::min() - a;
            if (other.a < diff) {
                return Integer{std::numeric_limits<Base>::max() + other.a + 1 - diff};
            }
        }
        return Integer{a + other.a};
    }

    Integer operator-(const Integer& other) const {
        if ((other.a > 0 && a < std::numeric_limits<Base>::min() + other.a) ||
            (other.a < 0 && a > std::numeric_limits<Base>::max() + other.a)) {
            if (other.a > 0) { // underflow
                Base diff = std::numeric_limits<Base>::min() - a;
                return Integer{std::numeric_limits<Base>::max() - (other.a - 1 + diff)};
            } else { // overflow
                Base diff = std::numeric_limits<Base>::max() - a;
                return Integer{std::numeric_limits<Base>::min() - (other.a + 1 + diff)};
            }
        }
        return Integer{a - other.a};
    }

    Integer operator-() const { return Integer{-a}; }
    Integer operator*(const Integer& other) const {
        UBase ua = a;
        return Integer{Base(ua * other.a)};
    }
    Integer operator~() const {
        UBase b(a);
        UBase c(-1);
        return Integer{Base(c + b)};
    }

    bool operator<(const Base& other) const { return a < other; }
    bool operator<(const Integer& other) const { return a < other.a; }
    bool operator!=(const Base& other) const { return a != other; }
    std::strong_ordering operator<=>(const Base& other) const { return a <=> other; }

    Integer operator&(const Integer& other) const { return Integer{a & other.a}; }
    Integer& operator&=(const Integer& other) {
        a &= other.a;
        return *this;
    }

    Integer operator^(const Integer& other) const { return Integer{a ^ other.a}; }
    Integer& operator^=(const Integer& other) {
        a ^= other.a;
        return *this;
    }

    Integer operator|(const Integer& other) const { return Integer{a | other.a}; }
    Integer& operator|=(const Integer& other) {
        a |= other.a;
        return *this;
    }

    Integer operator<<(const Integer& other) const { return Integer{a << other.a}; }
    Integer operator>>(const Integer& other) const {
        UBase tmp(a);
        if (other.a >= sizeof(Base) * 8)
            return Integer{0};
        return Integer{Base(tmp >> UBase(other.a))};
    }

    inline Base get() const { return a; }
    inline Base& get() { return a; }
    inline Integer abs() const { return a < 0 ? -a : a; }
};

} // namespace IR

#endif
