#ifndef MATRIX_H
#define MATRIX_H

#include <cassert>
#include <deque>

using std::deque;

namespace IR { // namespace IR

template <class sint>
class MatrixCalculus101 {
  public:
    explicit MatrixCalculus101() : dot_prods() {}

    void next_dotprod();

    void add_mul(const sint& a, const sint& b);

    sint get_next();

  private:
    deque<deque<sint>> dot_prods;
};

template <class sint>
void MatrixCalculus101<sint>::next_dotprod() {
    dot_prods.emplace_back(0);
}

template <class sint>
void MatrixCalculus101<sint>::add_mul(const sint& a, const sint& b) {
    auto& last = dot_prods.back();
    last.push_back(a.prepare_mult(b));
}

template <class sint>
sint MatrixCalculus101<sint>::get_next() {
    assert(dot_prods.size() > 0);
    auto& last = dot_prods.front();
    sint res(0);

    for (auto& a : last) {
        a.complete_mult_without_trunc();
        res = res + a;
    }

    dot_prods.pop_front();

    return res;
}

} // namespace IR

#endif