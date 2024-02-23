#include "Program.hpp"

#include "Machine.hpp"
#include "Shares/ASetShare.hpp"
#include "Shares/AShare.hpp"
#include "Shares/BSetShare.hpp"
#include "Shares/Integer.hpp"

template class IR::Program<sint_t<Additive_Share<DATATYPE, IR::PlainShare<DATATYPE>>>, sbitset_t,
                           IR::PBInteger<DATATYPE>>;