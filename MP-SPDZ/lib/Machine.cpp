#include "Machine.hpp"

#include "Constants.hpp"
#include "Shares/ASetShare.hpp"
#include "Shares/AShare.hpp"
#include "Shares/BSetShare.hpp"
#include "Shares/Integer.hpp"

template class IR::Machine<sint_t<Additive_Share<DATATYPE, IR::PlainShare<DATATYPE>>>, sbitset_t,
                           IR::PBInteger<DATATYPE>>;