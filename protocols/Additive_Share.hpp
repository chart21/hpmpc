#pragma once
/* #include "Share.hpp" */
#include <functional>
#include "../arch/DATATYPE.h"

template <typename Datatype, typename Share_Type>
class Additive_Share : public Share_Type
{
public:
    Additive_Share() {}

    Additive_Share(const Share_Type& s) : Share_Type(s) {}

    Additive_Share operator+(const Additive_Share<Datatype,Share_Type>& b) const
    {
        return Additive_Share(Share_Type::Add(b, OP_ADD));
    }
    
    Additive_Share operator-(const Additive_Share<Datatype,Share_Type>& b) const
    {
        return Additive_Share(Share_Type::Add(b, OP_SUB));
    }

    Additive_Share operator*(const Additive_Share<Datatype, Share_Type>& b) const
    {
        return Additive_Share(Share_Type::prepare_mult(b, OP_ADD, OP_SUB, OP_MULT));
    }

    void complete_mult()
    {
    #if PROTOCOL == 1 // Sharemind needs custom overload
        Share_Type::complete_mult(OP_ADD, OP_SUB, OP_MULT);
    #else
        Share_Type::complete_mult(OP_ADD, OP_SUB);
    #endif
    }

    template <int id>
    void prepare_receive_from()
    {
        Share_Type::template prepare_receive_from<id>(OP_ADD, OP_SUB);
    }

    template <int id>
    void complete_receive_from()
    {
        Share_Type::template complete_receive_from<id>(OP_ADD, OP_SUB);
    }
    
    void prepare_reveal_to_all()
    {
        Share_Type::prepare_reveal_to_all();
    }

    Datatype complete_reveal_to_all()
    {
        return Share_Type::complete_Reveal(OP_ADD, OP_SUB);
    }
};


