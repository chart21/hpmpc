#include "Integer.hpp"

#include <limits>

namespace IR {

// Integer Integer::operator-(const Integer& other) const {
//     if (a >= 0) {
//         long diff = std::numeric_limits<Base>::max() - a;
//         if (diff > other.a) {
//             std::cout << a << " - " << other.a << "\n";
//             return Integer{std::numeric_limits<Base>::min() - other.a - 1 -
//             diff};
//         }
//     } else {
//         long diff = std::numeric_limits<Base>::min() - a;
//         if (other.a > diff) {
//             return Integer{std::numeric_limits<Base>::max() - (other.a - 1) -
//             diff};
//         }
//     }
//     return Integer{a - other.a};
// }

} // namespace IR
